from io import BytesIO, StringIO
import base64
from copy import deepcopy
from datetime import datetime
import html
import json
import math
import os
from pathlib import Path
import re
from secrets import randbelow
from threading import Lock
import time
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingRegressor,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.inspection import permutation_importance
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Lasso, LinearRegression, LogisticRegression, PoissonRegressor, QuantileRegressor, Ridge, RidgeClassifier, SGDClassifier, SGDRegressor, TweedieRegressor
from sklearn.metrics import accuracy_score, average_precision_score, auc, precision_recall_curve, roc_curve
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

from auth import (
    authenticate_user,
    create_user,
    current_user,
    current_user_id,
    is_user_authenticated,
    log_user_in,
    user_has_subscription,
)
from db import get_db_connection, init_auth_db
from email_notifications import send_subscription_success_email
from payments import PaymentService, SUBSCRIPTION_AMOUNT, SUBSCRIPTION_CURRENCY, SUBSCRIPTION_INTERVAL
from reports import ReportRenderer

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier = None
    XGBRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:
    CatBoostClassifier = None
    CatBoostRegressor = None

try:
    import shap
except ImportError:
    shap = None


try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        env_path = Path(".env")
        if not env_path.exists():
            return
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv()


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = "dev-secret-change-me"

ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
DATASETS = {}
DOWNLOADS = {}
RUN_PROGRESS = {}
RUN_PROGRESS_LOCK = Lock()


def cleanup_run_progress():
    cutoff = time.time() - 60 * 60
    with RUN_PROGRESS_LOCK:
        expired = [job_id for job_id, progress in RUN_PROGRESS.items() if progress.get("updated_at", 0) < cutoff]
        for job_id in expired:
            RUN_PROGRESS.pop(job_id, None)


def active_run_progress_id():
    return request.form.get("run_progress_id", "").strip()


def initialize_run_progress(job_id, total):
    if not job_id:
        return
    cleanup_run_progress()
    total = max(int(total or 0), 0)
    with RUN_PROGRESS_LOCK:
        RUN_PROGRESS[job_id] = {
            "completed": 0,
            "total": total,
            "percent": 0,
            "status": "running",
            "label": "",
            "labels": [],
            "updated_at": time.time(),
        }


def advance_run_progress(job_id, label=""):
    if not job_id:
        return
    with RUN_PROGRESS_LOCK:
        progress = RUN_PROGRESS.get(job_id)
        if progress is None:
            return
        total = max(progress.get("total", 0), 0)
        completed = min(progress.get("completed", 0) + 1, total) if total else progress.get("completed", 0) + 1
        progress["completed"] = completed
        progress["label"] = label
        if label:
            progress.setdefault("labels", []).append(label)
        progress["percent"] = int((completed / total) * 100) if total else 100
        progress["status"] = "complete" if total and completed >= total else "running"
        progress["updated_at"] = time.time()


def complete_run_progress(job_id):
    if not job_id:
        return
    with RUN_PROGRESS_LOCK:
        progress = RUN_PROGRESS.get(job_id)
        if progress is None:
            return
        total = max(progress.get("total", 0), 0)
        progress["completed"] = total
        progress["percent"] = 100
        progress["status"] = "complete"
        progress["updated_at"] = time.time()
DISPLAY_DECIMALS = 3
DISPLAY_FLOAT_FORMAT = f"{{:.{DISPLAY_DECIMALS}f}}".format
MAX_SPLIT_SEED = 999999


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


init_auth_db()


def read_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.filename).suffix.lower()
    file_bytes = uploaded_file.read()

    if suffix == ".csv":
        return pd.read_csv(StringIO(file_bytes.decode("utf-8-sig")))

    return pd.read_excel(BytesIO(file_bytes))


def simulate_test_data(row_count=1000):
    rng = np.random.default_rng(42)
    features = rng.normal(size=(row_count, 4))
    x1, x2, x3, x4 = features.T

    class_score = 0.9 * x1 - 0.7 * x2 + 0.35 * x3 + rng.normal(scale=0.8, size=row_count)
    class_probability = sigmoid(class_score)
    classification_target = rng.binomial(1, class_probability)

    regression_target = 12 + 2.4 * x1 - 1.6 * x2 + 0.8 * x3 + 0.25 * x4 + rng.normal(scale=1.5, size=row_count)

    return pd.DataFrame(
        {
            "classification_target": classification_target,
            "regression_target": regression_target,
            "feature_1": x1,
            "feature_2": x2,
            "feature_3": x3,
            "feature_4": x4,
        }
    )


payment_service = PaymentService(get_db_connection, url_for, send_subscription_success_email)


def mollie_configured():
    return payment_service.configured()


def create_mollie_first_payment(user):
    return payment_service.create_first_payment(user)


def sync_mollie_payment(payment_id):
    return payment_service.sync_payment(payment_id)


def cancel_mollie_subscription(user):
    return payment_service.cancel_subscription(user)


def load_dataset_from_db(current_id):
    user_id = current_user_id()
    if not user_id:
        return None

    with get_db_connection() as connection:
        row = connection.execute(
            "SELECT filename, csv_data FROM datasets WHERE id = ? AND user_id = ?",
            (current_id, user_id),
        ).fetchone()
    if row is None:
        return None

    dataset = {"data": pd.read_csv(StringIO(row["csv_data"])), "filename": row["filename"]}
    DATASETS[current_id] = dataset
    return dataset


def current_dataset():
    current_id = session.get("dataset_id")
    if not current_id:
        return None
    return DATASETS.get(current_id) or load_dataset_from_db(current_id)


def display_value(value):
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        return DISPLAY_FLOAT_FORMAT(float(value))
    return value


def display_frame(frame):
    if frame is None:
        return frame
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(display_value)
        elif pd.api.types.is_object_dtype(display[column]):
            display[column] = display[column].map(display_value)
    return display


def display_table(frame, **kwargs):
    return display_frame(frame).to_html(**kwargs)


def save_dataset(data, filename):
    current_id = str(uuid4())
    DATASETS[current_id] = {"data": data, "filename": filename}
    session["dataset_id"] = current_id
    session["selected_pro_run_ids"] = {}
    session["selected_pro_compare_ids"] = {}

    user_id = current_user_id()
    if user_id:
        with get_db_connection() as connection:
            connection.execute(
                "INSERT OR REPLACE INTO datasets (id, user_id, filename, csv_data) VALUES (?, ?, ?, ?)",
                (current_id, user_id, filename, data.to_csv(index=False)),
            )


def preview_table(data):
    preview = data.head(25)
    return display_table(
        preview,
        classes="preview-table",
        index=False,
        border=0,
        na_rep="",
        escape=True,
    )


def sigmoid(values):
    values = np.clip(values, -500, 500)
    return 1 / (1 + np.exp(-values))


def normal_two_sided_pvalue(z_value):
    return float(math.erfc(abs(z_value) / math.sqrt(2)))


def preprocessing_options(tab):
    split_seed = tab.get("selected_split_seed")
    if split_seed is None:
        split_seed = random_split_seed()
    return {
        "missing_values": tab.get("selected_missing_values", "drop"),
        "categorical_encoding": tab.get("selected_categorical_encoding", "one_hot_drop_first"),
        "scaling": tab.get("selected_scaling", "on"),
        "split_seed": split_seed,
        "outlier_handling": tab.get("selected_outlier_handling", "none"),
        "calibration": tab.get("selected_calibration", "off"),
        "tuned_params": tab.get("tuned_params", {}),
    }


def tuned_param(options, model_name, param_name, default=None):
    params = (options or {}).get("tuned_params", {}).get(model_name, {})
    return params.get(param_name, default)


def tuned_float(options, model_name, param_name, default):
    try:
        return float(tuned_param(options, model_name, param_name, default))
    except (TypeError, ValueError):
        return default


def tuned_int(options, model_name, param_name, default):
    try:
        return int(tuned_param(options, model_name, param_name, default))
    except (TypeError, ValueError):
        return default


def preprocessing_seed(options):
    options = options or {}
    return int(options.get("split_seed", options.get("selected_split_seed", random_split_seed())))


def scaling_enabled(options):
    options = options or {}
    return options.get("scaling", options.get("selected_scaling", "on")) == "on"


def categorical_encoding_mode(options):
    options = options or {}
    return options.get("categorical_encoding", options.get("selected_categorical_encoding", "one_hot_drop_first"))


def missing_value_mode(options):
    options = options or {}
    return options.get("missing_values", options.get("selected_missing_values", "drop"))


def outlier_mode(options):
    options = options or {}
    return options.get("outlier_handling", options.get("selected_outlier_handling", "none"))


def preprocessing_details(options):
    return {
        "Missing values": {
            "drop": "Drop incomplete rows",
            "impute_mean_mode": "Impute mean/mode",
            "impute_median_mode": "Impute median/mode",
        }.get(missing_value_mode(options), "Drop incomplete rows"),
        "Categorical encoding": {
            "one_hot_drop_first": "One-hot, drop first",
            "one_hot_full": "One-hot, all levels",
            "ordinal": "Ordinal codes",
        }.get(categorical_encoding_mode(options), "One-hot, drop first"),
        "Feature scaling": "On" if scaling_enabled(options) else "Off",
        "Outlier handling": {
            "none": "None",
            "winsorize": "Winsorize numeric columns",
            "remove_iqr": "Remove IQR outliers",
        }.get(outlier_mode(options), "None"),
        "Random seed": preprocessing_seed(options),
    }


def fill_missing_predictors(model_data, predictors, options):
    if missing_value_mode(options) == "drop":
        return model_data.dropna()

    model_data = model_data.copy()
    for column in predictors:
        if pd.api.types.is_numeric_dtype(model_data[column]):
            if missing_value_mode(options) == "impute_median_mode":
                fill_value = model_data[column].median()
            else:
                fill_value = model_data[column].mean()
            if pd.isna(fill_value):
                fill_value = 0
        else:
            modes = model_data[column].mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else "Missing"
        model_data[column] = model_data[column].fillna(fill_value)
    return model_data


def apply_outlier_handling(model_data, target, predictors, task, options):
    mode = outlier_mode(options)
    if mode == "none":
        return model_data

    model_data = model_data.copy()
    numeric_columns = [column for column in predictors if pd.api.types.is_numeric_dtype(model_data[column])]
    if task == "regression" and pd.api.types.is_numeric_dtype(model_data[target]):
        numeric_columns.append(target)
    numeric_columns = list(dict.fromkeys(numeric_columns))
    if not numeric_columns:
        return model_data

    if mode == "winsorize":
        for column in numeric_columns:
            lower = model_data[column].quantile(0.01)
            upper = model_data[column].quantile(0.99)
            if pd.notna(lower) and pd.notna(upper) and lower < upper:
                model_data[column] = model_data[column].clip(lower, upper)
        return model_data

    if mode == "remove_iqr":
        keep = pd.Series(True, index=model_data.index)
        for column in numeric_columns:
            q1 = model_data[column].quantile(0.25)
            q3 = model_data[column].quantile(0.75)
            iqr = q3 - q1
            if pd.notna(iqr) and iqr > 0:
                keep &= model_data[column].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        return model_data.loc[keep]

    return model_data


def safe_feature_columns(columns):
    safe_columns = []
    seen = set()
    for column in columns:
        base = re.sub(r"[\[\]<>]", "_", str(column)).strip()
        base = re.sub(r"\s+", " ", base) or "feature"
        candidate = base
        suffix = 2
        while candidate in seen:
            candidate = f"{base}_{suffix}"
            suffix += 1
        safe_columns.append(candidate)
        seen.add(candidate)
    return safe_columns


def encode_predictors(model_data, predictors, options):
    x_raw = model_data[predictors]
    if categorical_encoding_mode(options) == "ordinal":
        encoded = pd.DataFrame(index=x_raw.index)
        for column in predictors:
            if pd.api.types.is_numeric_dtype(x_raw[column]):
                encoded[column] = pd.to_numeric(x_raw[column], errors="coerce")
            else:
                encoded[column] = pd.Categorical(x_raw[column]).codes.astype(float)
        encoded = encoded.astype(float)
        encoded.columns = safe_feature_columns(encoded.columns)
        return encoded

    drop_first = categorical_encoding_mode(options) == "one_hot_drop_first"
    encoded = pd.get_dummies(x_raw, drop_first=drop_first, dtype=float)
    encoded.columns = safe_feature_columns(encoded.columns)
    return encoded


def scaled_frames(split, options):
    if not scaling_enabled(options):
        return split["x_train"], split["x_test"], "Off"
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(split["x_train"])
    scaled_test = scaler.transform(split["x_test"])
    return scaled_train, scaled_test, "StandardScaler"


def add_preprocessing_details(details, options):
    if options is None:
        return details
    details.update(preprocessing_details(options))
    if options.get("detail_tuned_params_label"):
        details["Tuned detail params"] = options["detail_tuned_params_label"]
    return details

def prepare_classification_data(data, target, predictors, binary_only, options=None):
    if target in predictors:
        raise ValueError("The target column cannot also be used as a predictor.")

    selected = [target] + predictors
    model_data = data[selected].copy()
    model_data = model_data.loc[model_data[target].notna()].copy()
    model_data = fill_missing_predictors(model_data, predictors, options)
    model_data = apply_outlier_handling(model_data, target, predictors, "classification", options)
    if len(model_data) < 5:
        raise ValueError("At least five usable rows are required after preprocessing.")

    target_values = model_data[target]
    classes = list(pd.unique(target_values))
    if binary_only and len(classes) != 2:
        raise ValueError("Logistic regression requires a target column with exactly two classes.")
    if len(classes) < 2:
        raise ValueError("Classification requires a target column with at least two classes.")

    x_encoded = encode_predictors(model_data, predictors, options)
    if x_encoded.empty:
        raise ValueError("At least one usable predictor is required.")
    if x_encoded.isna().any().any():
        raise ValueError("Preprocessing left missing predictor values. Choose imputation or drop incomplete rows.")

    return model_data, target_values, classes, x_encoded


def parse_test_size(value):
    try:
        test_size = float(value)
    except (TypeError, ValueError):
        test_size = 0.2
    return min(max(test_size, 0.2), 0.8)


def parse_cv_folds(value):
    try:
        folds = int(value)
    except (TypeError, ValueError):
        return 0
    return folds if folds in {3, 5, 10} else 0


def parse_threshold(value):
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return 0.5
    return min(max(threshold, 0.05), 0.95)


def parse_choice(value, allowed, default):
    return value if value in allowed else default


def random_split_seed():
    return randbelow(MAX_SPLIT_SEED + 1)


def parse_split_seed(value):
    try:
        seed = int(value)
    except (TypeError, ValueError):
        return random_split_seed()
    return min(max(seed, 0), MAX_SPLIT_SEED)

def parse_tuning_iterations(value):
    try:
        iterations = int(value)
    except (TypeError, ValueError):
        return 10
    return min(max(iterations, 3), 30)

def dataset_id():
    return session.get("dataset_id")


def selected_pro_run_key(tab_name):
    current_id = dataset_id()
    if not current_id:
        return None
    return f"{current_id}:{tab_name}"


def selected_pro_run_id(tab_name):
    key = selected_pro_run_key(tab_name)
    if not key:
        return None
    selected_runs = session.get("selected_pro_run_ids", {})
    try:
        return int(selected_runs.get(key))
    except (TypeError, ValueError):
        return None


def set_selected_pro_run_id(tab_name, run_id):
    key = selected_pro_run_key(tab_name)
    if not key:
        return
    selected_runs = dict(session.get("selected_pro_run_ids", {}))
    selected_runs[key] = int(run_id)
    session["selected_pro_run_ids"] = selected_runs


def selected_compare_run_ids(tab_name):
    key = selected_pro_run_key(tab_name)
    if not key:
        return []
    compare_runs = session.get("selected_pro_compare_ids", {})
    values = compare_runs.get(key, [])
    ids = []
    for value in values:
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    return ids[:2]


def set_selected_compare_run_ids(tab_name, run_ids):
    key = selected_pro_run_key(tab_name)
    if not key:
        return
    compare_runs = dict(session.get("selected_pro_compare_ids", {}))
    compare_runs[key] = [int(run_id) for run_id in run_ids[:2]]
    session["selected_pro_compare_ids"] = compare_runs


def metric_rows(metrics):
    return pd.DataFrame([{"Metric": metric["label"], "Value": metric["value"]} for metric in metrics])


def details_frame(details):
    return pd.DataFrame([{"Setting": key, "Value": value} for key, value in details.items()])


def register_downloads(result_type, result):
    current_id = dataset_id()
    if not current_id:
        result["downloads"] = []
        return result

    downloads = {
        "metrics": metric_rows(result.get("metrics", [])).to_csv(index=False),
    }
    for key, value in result.pop("download_data", {}).items():
        downloads[key] = value

    DOWNLOADS.setdefault(current_id, {})[result_type] = downloads
    result["downloads"] = [
        {
            "label": f"Download {key.replace('_', ' ')}",
            "href": f"/download/{result_type}/{key}",
        }
        for key in downloads
    ]
    return result


def calibration_method(options):
    return (options or {}).get("calibration", "off")


def calibration_enabled(options):
    return calibration_method(options) in {"sigmoid", "isotonic"}


def with_probability_calibration(estimator, options):
    if not calibration_enabled(options):
        return estimator
    return CalibratedClassifierCV(estimator=estimator, method=calibration_method(options), cv=3)


def classification_estimator(model_name, options=None):
    if model_name == "logistic":
        model = LogisticRegression(
            C=tuned_float(options, "logistic", "C", 1.0),
            max_iter=1000,
            random_state=preprocessing_seed(options),
        )
        estimator = make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
        return with_probability_calibration(estimator, options)
    if model_name == "elastic_net_logistic":
        model = LogisticRegression(
            C=tuned_float(options, "elastic_net_logistic", "C", 1.0),
            l1_ratio=tuned_float(options, "elastic_net_logistic", "l1_ratio", 0.5),
            max_iter=3000,
            random_state=preprocessing_seed(options),
            solver="saga",
        )
        estimator = make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
        return with_probability_calibration(estimator, options)
    if model_name == "tree":
        return with_probability_calibration(DecisionTreeClassifier(
            max_depth=tuned_param(options, "tree", "max_depth", 4),
            min_samples_leaf=tuned_int(options, "tree", "min_samples_leaf", 1),
            random_state=preprocessing_seed(options),
        ), options)
    if model_name == "random_forest":
        return with_probability_calibration(RandomForestClassifier(
            n_estimators=tuned_int(options, "random_forest", "n_estimators", 200),
            max_depth=tuned_param(options, "random_forest", "max_depth", None),
            min_samples_leaf=tuned_int(options, "random_forest", "min_samples_leaf", 1),
            random_state=preprocessing_seed(options),
            n_jobs=-1,
        ), options)
    if model_name == "extra_trees":
        return with_probability_calibration(ExtraTreesClassifier(
            n_estimators=tuned_int(options, "extra_trees", "n_estimators", 200),
            max_depth=tuned_param(options, "extra_trees", "max_depth", None),
            min_samples_leaf=tuned_int(options, "extra_trees", "min_samples_leaf", 1),
            random_state=preprocessing_seed(options),
            n_jobs=-1,
        ), options)
    if model_name == "gradient_boosting":
        return with_probability_calibration(GradientBoostingClassifier(
            n_estimators=tuned_int(options, "gradient_boosting", "n_estimators", 100),
            learning_rate=tuned_float(options, "gradient_boosting", "learning_rate", 0.1),
            max_depth=tuned_int(options, "gradient_boosting", "max_depth", 3),
            random_state=preprocessing_seed(options),
        ), options)
    if model_name == "adaboost":
        return with_probability_calibration(AdaBoostClassifier(
            n_estimators=tuned_int(options, "adaboost", "n_estimators", 100),
            learning_rate=tuned_float(options, "adaboost", "learning_rate", 0.5),
            random_state=preprocessing_seed(options),
        ), options)
    if model_name == "hist_gradient_boosting":
        return with_probability_calibration(HistGradientBoostingClassifier(
            max_iter=tuned_int(options, "hist_gradient_boosting", "max_iter", 100),
            learning_rate=tuned_float(options, "hist_gradient_boosting", "learning_rate", 0.1),
            max_leaf_nodes=tuned_int(options, "hist_gradient_boosting", "max_leaf_nodes", 31),
            l2_regularization=tuned_float(options, "hist_gradient_boosting", "l2_regularization", 0.0),
            random_state=preprocessing_seed(options),
        ), options)
    if model_name == "xgboost":
        if XGBClassifier is None:
            raise RuntimeError("XGBoost is not installed. Run pip install xgboost or install requirements.txt.")
        return with_probability_calibration(XGBClassifier(
            n_estimators=tuned_int(options, "xgboost", "n_estimators", 200),
            learning_rate=tuned_float(options, "xgboost", "learning_rate", 0.05),
            max_depth=tuned_int(options, "xgboost", "max_depth", 4),
            subsample=tuned_float(options, "xgboost", "subsample", 0.9),
            colsample_bytree=tuned_float(options, "xgboost", "colsample_bytree", 0.9),
            random_state=preprocessing_seed(options),
            n_jobs=-1,
            verbosity=0,
        ), options)
    if model_name == "catboost":
        if CatBoostClassifier is None:
            raise RuntimeError("CatBoost is not installed. Run pip install catboost or install requirements.txt.")
        return with_probability_calibration(CatBoostClassifier(
            iterations=tuned_int(options, "catboost", "iterations", 200),
            learning_rate=tuned_float(options, "catboost", "learning_rate", 0.05),
            depth=tuned_int(options, "catboost", "depth", 6),
            random_seed=preprocessing_seed(options),
            verbose=False,
        ), options)
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM is not installed. Run pip install lightgbm or install requirements.txt.")
        return with_probability_calibration(LGBMClassifier(
            n_estimators=tuned_int(options, "lightgbm", "n_estimators", 200),
            learning_rate=tuned_float(options, "lightgbm", "learning_rate", 0.05),
            num_leaves=tuned_int(options, "lightgbm", "num_leaves", 31),
            max_depth=tuned_int(options, "lightgbm", "max_depth", -1),
            min_child_samples=tuned_int(options, "lightgbm", "min_child_samples", 20),
            random_state=preprocessing_seed(options),
            n_jobs=-1,
            verbosity=-1,
        ), options)
    if model_name == "naive_bayes":
        return with_probability_calibration(GaussianNB(
            var_smoothing=tuned_float(options, "naive_bayes", "var_smoothing", 1e-9),
        ), options)
    if model_name == "lda":
        return with_probability_calibration(LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage=tuned_param(options, "lda", "shrinkage", "auto"),
        ), options)
    if model_name == "qda":
        return with_probability_calibration(QuadraticDiscriminantAnalysis(
            reg_param=tuned_float(options, "qda", "reg_param", 0.1),
        ), options)
    if model_name == "gaussian_process":
        model = GaussianProcessClassifier(
            max_iter_predict=tuned_int(options, "gaussian_process", "max_iter_predict", 100),
            random_state=preprocessing_seed(options),
            n_jobs=-1,
        )
        estimator = make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
        return with_probability_calibration(estimator, options)
    if model_name == "mlp":
        hidden_layer_sizes = tuned_param(options, "mlp", "hidden_layer_sizes", (50,))
        if isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = tuple(hidden_layer_sizes)
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=tuned_float(options, "mlp", "alpha", 0.0001),
            learning_rate_init=tuned_float(options, "mlp", "learning_rate_init", 0.001),
            max_iter=1000,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=preprocessing_seed(options),
        )
        estimator = make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
        return with_probability_calibration(estimator, options)
    if model_name == "passive_aggressive":
        model = SGDClassifier(
            loss="hinge",
            penalty=None,
            learning_rate="pa1",
            eta0=tuned_float(options, "passive_aggressive", "eta0", 1.0),
            max_iter=1000,
            random_state=preprocessing_seed(options),
            tol=1e-3,
        )
        estimator = make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
        return with_probability_calibration(estimator, options)
    if model_name == "ridge_classifier":
        model = RidgeClassifier(
            alpha=tuned_float(options, "ridge_classifier", "alpha", 1.0),
        )
        estimator = make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
        return with_probability_calibration(estimator, options)
    if model_name == "linear_svm":
        model = LinearSVC(
            C=tuned_float(options, "linear_svm", "C", 1.0),
            max_iter=5000,
            random_state=preprocessing_seed(options),
        )
        estimator = make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
        return with_probability_calibration(estimator, options)
    if model_name == "svm":
        model = SVC(
            kernel="rbf",
            C=tuned_float(options, "svm", "C", 1.0),
            gamma=tuned_param(options, "svm", "gamma", "scale"),
            probability=True,
            random_state=preprocessing_seed(options),
        )
        estimator = make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
        return with_probability_calibration(estimator, options)
    if model_name == "voting":
        logistic_base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=preprocessing_seed(options)))
        knn_base = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7, weights="distance"))
        base_estimators = [
            ("logistic", logistic_base),
            ("random_forest", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=preprocessing_seed(options), n_jobs=-1)),
            ("gradient_boosting", GradientBoostingClassifier(n_estimators=75, learning_rate=0.05, max_depth=3, random_state=preprocessing_seed(options))),
            ("knn", knn_base),
        ]
        return with_probability_calibration(VotingClassifier(
            estimators=base_estimators,
            voting=tuned_param(options, "voting", "voting", "soft"),
            weights=tuned_param(options, "voting", "weights", None),
            n_jobs=-1,
        ), options)
    if model_name == "stacking":
        logistic_base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=preprocessing_seed(options)))
        linear_svm_base = make_pipeline(StandardScaler(), LinearSVC(C=1.0, max_iter=5000, random_state=preprocessing_seed(options)))
        base_estimators = [
            ("logistic", logistic_base),
            ("random_forest", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=preprocessing_seed(options), n_jobs=-1)),
            ("gradient_boosting", GradientBoostingClassifier(n_estimators=75, learning_rate=0.05, max_depth=3, random_state=preprocessing_seed(options))),
            ("linear_svm", linear_svm_base),
        ]
        return with_probability_calibration(StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(
                C=tuned_float(options, "stacking", "C", 1.0),
                max_iter=1000,
                random_state=preprocessing_seed(options),
            ),
            passthrough=bool(tuned_param(options, "stacking", "passthrough", False)),
            cv=3,
            n_jobs=-1,
        ), options)
    if model_name == "dummy":
        return DummyClassifier(
            strategy=tuned_param(options, "dummy", "strategy", "most_frequent"),
            random_state=preprocessing_seed(options),
        )
    if model_name == "knn":
        model = KNeighborsClassifier(
            n_neighbors=tuned_int(options, "knn", "n_neighbors", 5),
            weights=tuned_param(options, "knn", "weights", "distance"),
        )
        estimator = make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
        return with_probability_calibration(estimator, options)
    raise ValueError("Unknown classification model.")


def regression_estimator(model_name, options=None):
    if model_name == "linear":
        return LinearRegression()
    if model_name == "ridge":
        model = Ridge(alpha=tuned_float(options, "ridge", "alpha", 1.0))
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    if model_name == "lasso":
        model = Lasso(
            alpha=tuned_float(options, "lasso", "alpha", 0.1),
            max_iter=10000,
            random_state=preprocessing_seed(options),
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    if model_name == "elastic_net":
        model = ElasticNet(
            alpha=tuned_float(options, "elastic_net", "alpha", 0.1),
            l1_ratio=tuned_float(options, "elastic_net", "l1_ratio", 0.5),
            max_iter=10000,
            random_state=preprocessing_seed(options),
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    if model_name == "huber":
        model = HuberRegressor(
            epsilon=tuned_float(options, "huber", "epsilon", 1.35),
            alpha=tuned_float(options, "huber", "alpha", 0.0001),
            max_iter=1000,
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    if model_name == "bayesian_ridge":
        model = BayesianRidge(
            alpha_1=tuned_float(options, "bayesian_ridge", "alpha_1", 1e-6),
            lambda_1=tuned_float(options, "bayesian_ridge", "lambda_1", 1e-6),
            max_iter=1000,
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    if model_name == "quantile":
        model = QuantileRegressor(
            quantile=tuned_float(options, "quantile", "quantile", 0.5),
            alpha=tuned_float(options, "quantile", "alpha", 0.0001),
            solver="highs",
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    if model_name == "passive_aggressive":
        model = SGDRegressor(
            loss="epsilon_insensitive",
            penalty=None,
            learning_rate="pa1",
            eta0=tuned_float(options, "passive_aggressive", "eta0", 1.0),
            epsilon=tuned_float(options, "passive_aggressive", "epsilon", 0.1),
            max_iter=1000,
            random_state=preprocessing_seed(options),
            tol=1e-3,
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    if model_name == "kernel_ridge":
        model = KernelRidge(
            alpha=tuned_float(options, "kernel_ridge", "alpha", 1.0),
            kernel=tuned_param(options, "kernel_ridge", "kernel", "rbf"),
            gamma=tuned_param(options, "kernel_ridge", "gamma", None),
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    if model_name == "pls":
        model = PLSRegression(
            n_components=tuned_int(options, "pls", "n_components", 1),
            scale=False,
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    if model_name == "tweedie":
        model = TweedieRegressor(
            power=tuned_float(options, "tweedie", "power", 1.5),
            alpha=tuned_float(options, "tweedie", "alpha", 0.1),
            link=tuned_param(options, "tweedie", "link", "auto"),
            max_iter=1000,
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    if model_name == "poisson":
        model = PoissonRegressor(
            alpha=tuned_float(options, "poisson", "alpha", 0.1),
            max_iter=1000,
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=tuned_int(options, "random_forest", "n_estimators", 200),
            max_depth=tuned_param(options, "random_forest", "max_depth", None),
            min_samples_leaf=tuned_int(options, "random_forest", "min_samples_leaf", 1),
            random_state=preprocessing_seed(options),
            n_jobs=-1,
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=tuned_int(options, "extra_trees", "n_estimators", 200),
            max_depth=tuned_param(options, "extra_trees", "max_depth", None),
            min_samples_leaf=tuned_int(options, "extra_trees", "min_samples_leaf", 1),
            random_state=preprocessing_seed(options),
            n_jobs=-1,
        )
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=tuned_int(options, "gradient_boosting", "n_estimators", 100),
            learning_rate=tuned_float(options, "gradient_boosting", "learning_rate", 0.1),
            max_depth=tuned_int(options, "gradient_boosting", "max_depth", 3),
            random_state=preprocessing_seed(options),
        )
    if model_name == "lightgbm":
        if LGBMRegressor is None:
            raise RuntimeError("LightGBM is not installed. Run pip install lightgbm or install requirements.txt.")
        return LGBMRegressor(
            n_estimators=tuned_int(options, "lightgbm", "n_estimators", 200),
            learning_rate=tuned_float(options, "lightgbm", "learning_rate", 0.05),
            num_leaves=tuned_int(options, "lightgbm", "num_leaves", 31),
            max_depth=tuned_int(options, "lightgbm", "max_depth", -1),
            min_child_samples=tuned_int(options, "lightgbm", "min_child_samples", 20),
            random_state=preprocessing_seed(options),
            n_jobs=-1,
            verbosity=-1,
        )
    if model_name == "xgboost":
        if XGBRegressor is None:
            raise RuntimeError("XGBoost is not installed. Run pip install xgboost or install requirements.txt.")
        return XGBRegressor(
            n_estimators=tuned_int(options, "xgboost", "n_estimators", 200),
            learning_rate=tuned_float(options, "xgboost", "learning_rate", 0.05),
            max_depth=tuned_int(options, "xgboost", "max_depth", 4),
            subsample=tuned_float(options, "xgboost", "subsample", 0.9),
            colsample_bytree=tuned_float(options, "xgboost", "colsample_bytree", 0.9),
            objective="reg:squarederror",
            random_state=preprocessing_seed(options),
            n_jobs=-1,
            verbosity=0,
        )
    if model_name == "catboost":
        if CatBoostRegressor is None:
            raise RuntimeError("CatBoost is not installed. Run pip install catboost or install requirements.txt.")
        return CatBoostRegressor(
            iterations=tuned_int(options, "catboost", "iterations", 200),
            learning_rate=tuned_float(options, "catboost", "learning_rate", 0.05),
            depth=tuned_int(options, "catboost", "depth", 6),
            loss_function="RMSE",
            random_seed=preprocessing_seed(options),
            verbose=False,
        )
    if model_name == "bagging":
        return BaggingRegressor(
            estimator=DecisionTreeRegressor(
                max_depth=tuned_param(options, "bagging", "max_depth", None),
                min_samples_leaf=tuned_int(options, "bagging", "min_samples_leaf", 1),
                random_state=preprocessing_seed(options),
            ),
            n_estimators=tuned_int(options, "bagging", "n_estimators", 100),
            random_state=preprocessing_seed(options),
            n_jobs=-1,
        )
    if model_name == "voting":
        ridge_base = make_pipeline(StandardScaler(), Ridge(alpha=1.0)) if scaling_enabled(options) else Ridge(alpha=1.0)
        svr_base = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1))
        base_estimators = [
            ("ridge", ridge_base),
            ("random_forest", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=preprocessing_seed(options), n_jobs=-1)),
            ("gradient_boosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=preprocessing_seed(options))),
            ("svr", svr_base),
        ]
        return VotingRegressor(
            estimators=base_estimators,
            weights=tuned_param(options, "voting", "weights", None),
            n_jobs=-1,
        )
    if model_name == "stacking":
        ridge_base = make_pipeline(StandardScaler(), Ridge(alpha=1.0)) if scaling_enabled(options) else Ridge(alpha=1.0)
        svr_base = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.1))
        base_estimators = [
            ("ridge", ridge_base),
            ("random_forest", RandomForestRegressor(n_estimators=100, max_depth=8, random_state=preprocessing_seed(options), n_jobs=-1)),
            ("gradient_boosting", GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=preprocessing_seed(options))),
            ("svr", svr_base),
        ]
        return StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=tuned_float(options, "stacking", "alpha", 1.0)),
            passthrough=bool(tuned_param(options, "stacking", "passthrough", False)),
            n_jobs=-1,
        )
    if model_name == "svr":
        model = SVR(
            kernel="rbf",
            C=tuned_float(options, "svr", "C", 1.0),
            epsilon=tuned_float(options, "svr", "epsilon", 0.1),
            gamma=tuned_param(options, "svr", "gamma", "scale"),
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    if model_name == "knn":
        model = KNeighborsRegressor(
            n_neighbors=tuned_int(options, "knn", "n_neighbors", 5),
            weights=tuned_param(options, "knn", "weights", "distance"),
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    raise ValueError("Unknown regression model.")


def append_classification_cv(metrics, model_name, x_encoded, target_values, folds, options=None):
    if folds <= 1:
        return metrics

    y_codes, _ = encode_target(target_values)
    min_class_count = int(pd.Series(y_codes).value_counts().min())
    actual_folds = min(folds, min_class_count)
    if actual_folds < 2:
        metrics.append({"label": "CV accuracy", "value": "Not enough class balance"})
        return metrics

    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=preprocessing_seed(options))
    scores = cross_val_score(classification_estimator(model_name, options), x_encoded, y_codes, cv=cv, scoring="accuracy")
    metrics.extend([
        {"label": "CV folds", "value": actual_folds},
        {"label": "CV accuracy mean", "value": f"{scores.mean():.3f}"},
        {"label": "CV accuracy SD", "value": f"{scores.std():.3f}"},
    ])
    return metrics


def append_regression_cv(metrics, model_name, x_encoded, y, folds, options=None):
    if folds <= 1:
        return metrics

    actual_folds = min(folds, len(y))
    if actual_folds < 2:
        metrics.append({"label": "CV R squared", "value": "Not enough rows"})
        return metrics

    cv = KFold(n_splits=actual_folds, shuffle=True, random_state=preprocessing_seed(options))
    estimator = regression_estimator(model_name, options)
    r2_scores = cross_val_score(estimator, x_encoded, y, cv=cv, scoring="r2")
    rmse_scores = -cross_val_score(estimator, x_encoded, y, cv=cv, scoring="neg_root_mean_squared_error")
    metrics.extend([
        {"label": "CV folds", "value": actual_folds},
        {"label": "CV R squared mean", "value": f"{r2_scores.mean():.3f}"},
        {"label": "CV RMSE mean", "value": f"{rmse_scores.mean():.3f}"},
    ])
    return metrics


def encode_target(target_values):
    encoded_target = pd.Categorical(target_values)
    return encoded_target.codes, encoded_target.categories


def split_classification_data(target_values, x_encoded, test_size, options=None):
    y_codes, class_names = encode_target(target_values)
    indices = np.arange(len(y_codes))
    class_counts = pd.Series(y_codes).value_counts()
    stratify = y_codes if class_counts.min() >= 2 else None

    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=preprocessing_seed(options),
            stratify=stratify,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=preprocessing_seed(options))

    if len(np.unique(y_codes[train_idx])) < 2:
        raise ValueError("The training split must contain at least two target classes.")

    return {
        "x_train": x_encoded.iloc[train_idx],
        "x_test": x_encoded.iloc[test_idx],
        "target_train": target_values.iloc[train_idx],
        "target_test": target_values.iloc[test_idx],
        "y_train": y_codes[train_idx],
        "y_test": y_codes[test_idx],
        "class_names": class_names,
    }


def confusion_table(actual, predicted):
    return pd.crosstab(
        pd.Series(actual, name="Actual"),
        pd.Series(predicted, name="Predicted"),
        dropna=False,
    )


def details_table(details):
    return display_table(details_frame(details), index=False, border=0, classes="model-details")


def importance_frame(feature_names, importances):
    importances = pd.DataFrame(
        {
            "Predictor": feature_names,
            "Importance": importances,
        }
    )
    importances = importances.sort_values("Importance", ascending=False)
    importances = importances[importances["Importance"] > 0]
    if importances.empty:
        importances = pd.DataFrame({"Predictor": ["None"], "Importance": [0.0]})
    return importances


def importance_table(feature_names, importances):
    return display_table(importance_frame(feature_names, importances), index=False, border=0, classes="importances")


def permutation_importance_frame(model, x_test, y_test, feature_names, scoring, options):
    result = permutation_importance(
        model,
        x_test,
        y_test,
        n_repeats=5,
        random_state=preprocessing_seed(options),
        scoring=scoring,
    )
    importances = pd.DataFrame(
        {
            "Predictor": feature_names,
            "Importance": result.importances_mean,
            "Importance SD": result.importances_std,
        }
    )
    importances = importances.sort_values("Importance", ascending=False)
    importances = importances[importances["Importance"] > 0]
    if importances.empty:
        importances = pd.DataFrame({"Predictor": ["None"], "Importance": [0.0], "Importance SD": [0.0]})
    return importances


def permutation_importance_html(importances):
    return display_table(importances, index=False, border=0, classes="importances")


def tree_plot_image(tree, feature_names, class_names):
    fig_width = max(11, min(22, len(feature_names) * 1.8))
    fig_height = max(6, min(14, tree.get_depth() * 2.6 + 2))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    plot_tree(
        tree,
        feature_names=list(feature_names),
        class_names=[str(class_name) for class_name in class_names],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=True,
        fontsize=8,
        ax=ax,
    )
    ax.set_title("Classification tree")
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def calibration_label(options):
    labels = {"off": "Off", "sigmoid": "Sigmoid", "isotonic": "Isotonic"}
    return labels.get(calibration_method(options), "Off")


def fit_generic_classification_model(data, target, predictors, test_size, cv_folds, model_name, model_label, options=None, binary_only=False):
    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=binary_only, options=options)
    split = split_classification_data(target_values, x_encoded, test_size, options)
    class_names = split["class_names"]
    estimator = classification_estimator(model_name, options)
    estimator.fit(split["x_train"], split["y_train"])
    predicted_codes = estimator.predict(split["x_test"])
    predicted_labels = class_names[predicted_codes]
    accuracy = accuracy_score(split["target_test"], predicted_labels)
    confusion = confusion_table(split["target_test"].to_numpy(), predicted_labels)
    importances = permutation_importance_frame(estimator, split["x_test"], split["y_test"], x_encoded.columns, "accuracy", options)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
        {"label": "Test accuracy", "value": f"{accuracy:.3f}"},
        {"label": "Classes", "value": len(class_names)},
    ]
    metrics = append_classification_cv(metrics, model_name, x_encoded, target_values, cv_folds, options)
    details = {
        "Model": model_label,
        "Probability calibration": calibration_label(options),
        "Importance method": "Permutation importance on held-out test set",
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }

    return {
        "title": f"{model_label} results",
        "description": f"Target: {target}. Test-set metrics are shown. Classes: {', '.join(str(value) for value in classes)}.",
        "target": target,
        "positive_class": str(classes[1]) if len(classes) == 2 else None,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": permutation_importance_html(importances),
        "details_html": details_table(add_preprocessing_details(details, options)),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_logistic_regression(data, target, predictors, test_size, cv_folds, options=None):
    if calibration_enabled(options):
        return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "logistic", "Logistic regression", options, binary_only=True)

    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=True, options=options)
    split = split_classification_data(target_values, x_encoded, test_size, options)
    y = (split["target_train"] == classes[1]).astype(float).to_numpy()
    x_matrix = np.column_stack([np.ones(len(split["x_train"])), split["x_train"].to_numpy(dtype=float)])
    feature_names = ["Intercept"] + list(x_encoded.columns)
    beta = np.zeros(x_matrix.shape[1])
    c_value = tuned_param(options, "logistic", "C")
    ridge = 1 / max(float(c_value), 1e-8) if c_value is not None else 1e-8

    for _ in range(100):
        probabilities = np.clip(sigmoid(x_matrix @ beta), 1e-8, 1 - 1e-8)
        weights = probabilities * (1 - probabilities)
        gradient = x_matrix.T @ (y - probabilities) - ridge * beta
        hessian = (x_matrix.T * weights) @ x_matrix + ridge * np.eye(x_matrix.shape[1])
        step = np.linalg.solve(hessian, gradient)
        beta_next = beta + step
        if np.max(np.abs(step)) < 1e-7:
            beta = beta_next
            break
        beta = beta_next

    probabilities = np.clip(sigmoid(x_matrix @ beta), 1e-8, 1 - 1e-8)
    log_likelihood = float(np.sum(y * np.log(probabilities) + (1 - y) * np.log(1 - probabilities)))
    null_probability = np.clip(np.mean(y), 1e-8, 1 - 1e-8)
    null_log_likelihood = float(np.sum(y * np.log(null_probability) + (1 - y) * np.log(1 - null_probability)))
    pseudo_r2 = 1 - (log_likelihood / null_log_likelihood)
    parameter_count = len(beta)
    aic = 2 * parameter_count - 2 * log_likelihood
    bic = np.log(len(y)) * parameter_count - 2 * log_likelihood

    final_weights = probabilities * (1 - probabilities)
    information = (x_matrix.T * final_weights) @ x_matrix + ridge * np.eye(x_matrix.shape[1])
    covariance = np.linalg.pinv(information)
    standard_errors = np.sqrt(np.diag(covariance))
    z_scores = beta / standard_errors
    p_values = [normal_two_sided_pvalue(z_score) for z_score in z_scores]
    odds_ratios = np.exp(np.clip(beta, -50, 50))

    coefficients = pd.DataFrame(
        {
            "Term": feature_names,
            "Coefficient": beta,
            "Std. Error": standard_errors,
            "z value": z_scores,
            "p value": p_values,
            "Odds Ratio": odds_ratios,
        }
    )

    test_matrix = np.column_stack([np.ones(len(split["x_test"])), split["x_test"].to_numpy(dtype=float)])
    test_probabilities = np.clip(sigmoid(test_matrix @ beta), 1e-8, 1 - 1e-8)
    predicted = np.where(test_probabilities >= 0.5, classes[1], classes[0])
    confusion = confusion_table(split["target_test"].to_numpy(), predicted)
    accuracy = float(np.mean(predicted == split["target_test"].to_numpy()))

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
        {"label": "Test accuracy", "value": f"{accuracy:.3f}"},
        {"label": "Train AIC", "value": f"{aic:.3f}"},
        {"label": "Train McFadden R2", "value": f"{pseudo_r2:.3f}"},
        {"label": "Train log likelihood", "value": f"{log_likelihood:.3f}"},
    ]
    metrics = append_classification_cv(metrics, "logistic", x_encoded, target_values, cv_folds, options)
    details = {
        "Model": "Logistic regression",
        "C": c_value if c_value is not None else "Default",
        "Decision threshold": "0.5",
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }

    return {
        "title": "Logistic regression results",
        "description": f"Target: {target}. Positive class: {classes[1]}. Metrics are computed on the held-out test set.",
        "target": target,
        "positive_class": str(classes[1]),
        "metrics": metrics,
        "coefficients_html": display_table(coefficients, index=False, border=0, classes="coefficients"),
        "importances_html": None,
        "details_html": details_table(add_preprocessing_details(details, options)),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
        "download_data": {
            "coefficients": coefficients.to_csv(index=False),
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_tree_model(data, target, predictors, test_size, cv_folds, options=None):
    if calibration_enabled(options):
        return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "tree", "Tree model", options)

    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False, options=options)
    split = split_classification_data(target_values, x_encoded, test_size, options)
    class_names = split["class_names"]
    min_samples_leaf = tuned_int(options, "tree", "min_samples_leaf", max(1, int(len(split["y_train"]) * 0.02)))

    tree = DecisionTreeClassifier(
        max_depth=tuned_param(options, "tree", "max_depth", 4),
        min_samples_leaf=min_samples_leaf,
        random_state=preprocessing_seed(options),
    )
    tree.fit(split["x_train"], split["y_train"])
    predicted_codes = tree.predict(split["x_test"])
    predicted_labels = class_names[predicted_codes]
    accuracy = accuracy_score(split["target_test"], predicted_labels)

    confusion = confusion_table(split["target_test"].to_numpy(), predicted_labels)
    tree_plot = tree_plot_image(tree, x_encoded.columns, class_names)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
        {"label": "Test accuracy", "value": f"{accuracy:.3f}"},
        {"label": "Tree depth", "value": tree.get_depth()},
        {"label": "Terminal nodes", "value": tree.get_n_leaves()},
    ]
    metrics = append_classification_cv(metrics, "tree", x_encoded, target_values, cv_folds, options)
    details = {
        "Model": "Decision tree classifier",
        "Max depth": tree.max_depth,
        "Minimum samples per leaf": min_samples_leaf,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }
    importances_html = importance_table(x_encoded.columns, tree.feature_importances_)
    importances = pd.DataFrame({"Predictor": x_encoded.columns, "Importance": tree.feature_importances_})

    return {
        "title": "Tree model results",
        "description": f"Target: {target}. Test-set metrics are shown. Classes: {', '.join(str(value) for value in classes)}.",
        "target": target,
        "positive_class": None,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": importances_html,
        "details_html": details_table(add_preprocessing_details(details, options)),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": tree_plot,
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_random_forest_model(data, target, predictors, test_size, cv_folds, options=None):
    if calibration_enabled(options):
        return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "random_forest", "Random Forest", options)

    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False, options=options)
    split = split_classification_data(target_values, x_encoded, test_size, options)
    class_names = split["class_names"]
    min_samples_leaf = tuned_int(options, "random_forest", "min_samples_leaf", max(1, int(len(split["y_train"]) * 0.01)))
    model = RandomForestClassifier(
        n_estimators=tuned_int(options, "random_forest", "n_estimators", 200),
        max_depth=tuned_param(options, "random_forest", "max_depth", None),
        min_samples_leaf=min_samples_leaf,
        random_state=preprocessing_seed(options),
        n_jobs=-1,
    )
    model.fit(split["x_train"], split["y_train"])
    predicted_labels = class_names[model.predict(split["x_test"])]
    accuracy = accuracy_score(split["target_test"], predicted_labels)
    confusion = confusion_table(split["target_test"].to_numpy(), predicted_labels)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
        {"label": "Test accuracy", "value": f"{accuracy:.3f}"},
        {"label": "Trees", "value": model.n_estimators},
        {"label": "Classes", "value": len(class_names)},
    ]
    metrics = append_classification_cv(metrics, "random_forest", x_encoded, target_values, cv_folds, options)
    details = {
        "Model": "RandomForestClassifier",
        "Trees": model.n_estimators,
        "Maximum depth": model.max_depth if model.max_depth is not None else "None",
        "Minimum samples per leaf": min_samples_leaf,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }
    importances = pd.DataFrame({"Predictor": x_encoded.columns, "Importance": model.feature_importances_})

    return {
        "title": "Random Forest results",
        "description": f"Target: {target}. Test-set metrics are shown. Classes: {', '.join(str(value) for value in classes)}.",
        "target": target,
        "positive_class": None,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, model.feature_importances_),
        "details_html": details_table(add_preprocessing_details(details, options)),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_gradient_boosting_model(data, target, predictors, test_size, cv_folds, options=None):
    if calibration_enabled(options):
        return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "gradient_boosting", "Gradient Boosting", options)

    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False, options=options)
    split = split_classification_data(target_values, x_encoded, test_size, options)
    class_names = split["class_names"]
    model = GradientBoostingClassifier(
        n_estimators=tuned_int(options, "gradient_boosting", "n_estimators", 100),
        learning_rate=tuned_float(options, "gradient_boosting", "learning_rate", 0.1),
        max_depth=tuned_int(options, "gradient_boosting", "max_depth", 3),
        random_state=preprocessing_seed(options),
    )
    model.fit(split["x_train"], split["y_train"])
    predicted_labels = class_names[model.predict(split["x_test"])]
    accuracy = accuracy_score(split["target_test"], predicted_labels)
    confusion = confusion_table(split["target_test"].to_numpy(), predicted_labels)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
        {"label": "Test accuracy", "value": f"{accuracy:.3f}"},
        {"label": "Boosting stages", "value": model.n_estimators},
        {"label": "Learning rate", "value": model.learning_rate},
    ]
    metrics = append_classification_cv(metrics, "gradient_boosting", x_encoded, target_values, cv_folds, options)
    details = {
        "Model": "GradientBoostingClassifier",
        "Boosting stages": model.n_estimators,
        "Learning rate": model.learning_rate,
        "Maximum tree depth": model.max_depth,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }
    importances = pd.DataFrame({"Predictor": x_encoded.columns, "Importance": model.feature_importances_})

    return {
        "title": "Gradient Boosting results",
        "description": f"Target: {target}. Test-set metrics are shown. Classes: {', '.join(str(value) for value in classes)}.",
        "target": target,
        "positive_class": None,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, model.feature_importances_),
        "details_html": details_table(add_preprocessing_details(details, options)),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_svm_model(data, target, predictors, test_size, cv_folds, options=None):
    if calibration_enabled(options):
        return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "svm", "Support Vector Machine", options)

    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False, options=options)
    split = split_classification_data(target_values, x_encoded, test_size, options)
    class_names = split["class_names"]
    scaled_train, scaled_test, scaling_label = scaled_frames(split, options)
    model = SVC(
        kernel="rbf",
        C=tuned_float(options, "svm", "C", 1.0),
        gamma=tuned_param(options, "svm", "gamma", "scale"),
        random_state=preprocessing_seed(options),
    )
    model.fit(scaled_train, split["y_train"])
    predicted_labels = class_names[model.predict(scaled_test)]
    accuracy = accuracy_score(split["target_test"], predicted_labels)
    confusion = confusion_table(split["target_test"].to_numpy(), predicted_labels)
    importances = permutation_importance_frame(model, scaled_test, split["y_test"], x_encoded.columns, "accuracy", options)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
        {"label": "Test accuracy", "value": f"{accuracy:.3f}"},
        {"label": "Kernel", "value": model.kernel},
        {"label": "Support vectors", "value": int(np.sum(model.n_support_))},
    ]
    metrics = append_classification_cv(metrics, "svm", x_encoded, target_values, cv_folds, options)
    details = {
        "Model": "SVC",
        "Kernel": model.kernel,
        "C": model.C,
        "Gamma": model.gamma,
        "Support vectors by class": ", ".join(str(value) for value in model.n_support_),
        "Importance method": "Permutation importance on held-out test set",
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
    }

    return {
        "title": "Support Vector Machine results",
        "description": f"Target: {target}. Features were standardized using the training split; test-set metrics are shown.",
        "target": target,
        "positive_class": None,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": permutation_importance_html(importances),
        "details_html": details_table(add_preprocessing_details(details, options)),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_knn_model(data, target, predictors, test_size, cv_folds, options=None):
    if calibration_enabled(options):
        return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "knn", "kNN", options)

    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False, options=options)
    split = split_classification_data(target_values, x_encoded, test_size, options)
    class_names = split["class_names"]
    scaled_train, scaled_test, scaling_label = scaled_frames(split, options)
    neighbors = min(tuned_int(options, "knn", "n_neighbors", 5), len(split["y_train"]))
    model = KNeighborsClassifier(n_neighbors=neighbors, weights=tuned_param(options, "knn", "weights", "distance"))
    model.fit(scaled_train, split["y_train"])
    predicted_labels = class_names[model.predict(scaled_test)]
    accuracy = accuracy_score(split["target_test"], predicted_labels)
    confusion = confusion_table(split["target_test"].to_numpy(), predicted_labels)
    importances = permutation_importance_frame(model, scaled_test, split["y_test"], x_encoded.columns, "accuracy", options)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
        {"label": "Test accuracy", "value": f"{accuracy:.3f}"},
        {"label": "Neighbors", "value": neighbors},
        {"label": "Weights", "value": model.weights},
    ]
    metrics = append_classification_cv(metrics, "knn", x_encoded, target_values, cv_folds, options)
    details = {
        "Model": "KNeighborsClassifier",
        "Neighbors": neighbors,
        "Weights": model.weights,
        "Distance metric": model.metric,
        "Importance method": "Permutation importance on held-out test set",
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
    }

    return {
        "title": "kNN results",
        "description": f"Target: {target}. Features were standardized using the training split; test-set metrics are shown.",
        "target": target,
        "positive_class": None,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": permutation_importance_html(importances),
        "details_html": details_table(add_preprocessing_details(details, options)),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_extra_trees_model(data, target, predictors, test_size, cv_folds, options=None):
    if calibration_enabled(options):
        return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "extra_trees", "Extra Trees", options)

    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False, options=options)
    split = split_classification_data(target_values, x_encoded, test_size, options)
    class_names = split["class_names"]
    min_samples_leaf = tuned_int(options, "extra_trees", "min_samples_leaf", max(1, int(len(split["y_train"]) * 0.01)))
    model = ExtraTreesClassifier(
        n_estimators=tuned_int(options, "extra_trees", "n_estimators", 200),
        max_depth=tuned_param(options, "extra_trees", "max_depth", None),
        min_samples_leaf=min_samples_leaf,
        random_state=preprocessing_seed(options),
        n_jobs=-1,
    )
    model.fit(split["x_train"], split["y_train"])
    predicted_labels = class_names[model.predict(split["x_test"])]
    accuracy = accuracy_score(split["target_test"], predicted_labels)
    confusion = confusion_table(split["target_test"].to_numpy(), predicted_labels)
    importances = pd.DataFrame({"Predictor": x_encoded.columns, "Importance": model.feature_importances_})

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
        {"label": "Test accuracy", "value": f"{accuracy:.3f}"},
        {"label": "Trees", "value": model.n_estimators},
        {"label": "Classes", "value": len(class_names)},
    ]
    metrics = append_classification_cv(metrics, "extra_trees", x_encoded, target_values, cv_folds, options)
    details = {
        "Model": "ExtraTreesClassifier",
        "Trees": model.n_estimators,
        "Maximum depth": model.max_depth if model.max_depth is not None else "None",
        "Minimum samples per leaf": min_samples_leaf,
        "Probability calibration": calibration_label(options),
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }

    return {
        "title": "Extra Trees results",
        "description": f"Target: {target}. Extremely randomized tree ensemble; test-set metrics are shown.",
        "target": target,
        "positive_class": str(classes[1]) if len(classes) == 2 else None,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, model.feature_importances_),
        "details_html": details_table(add_preprocessing_details(details, options)),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_naive_bayes_model(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "naive_bayes", "Naive Bayes", options)


def fit_lightgbm_model(data, target, predictors, test_size, cv_folds, options=None):
    if LGBMClassifier is None:
        raise RuntimeError("LightGBM is not installed. Run pip install lightgbm or install requirements.txt.")
    if calibration_enabled(options):
        return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "lightgbm", "LightGBM", options)

    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False, options=options)
    split = split_classification_data(target_values, x_encoded, test_size, options)
    class_names = split["class_names"]
    model = LGBMClassifier(
        n_estimators=tuned_int(options, "lightgbm", "n_estimators", 200),
        learning_rate=tuned_float(options, "lightgbm", "learning_rate", 0.05),
        num_leaves=tuned_int(options, "lightgbm", "num_leaves", 31),
        max_depth=tuned_int(options, "lightgbm", "max_depth", -1),
        min_child_samples=tuned_int(options, "lightgbm", "min_child_samples", 20),
        random_state=preprocessing_seed(options),
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(split["x_train"], split["y_train"])
    predicted_labels = class_names[model.predict(split["x_test"])]
    accuracy = accuracy_score(split["target_test"], predicted_labels)
    confusion = confusion_table(split["target_test"].to_numpy(), predicted_labels)
    importances = pd.DataFrame({"Predictor": x_encoded.columns, "Importance": model.feature_importances_})

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
        {"label": "Test accuracy", "value": f"{accuracy:.3f}"},
        {"label": "Boosting rounds", "value": model.n_estimators},
        {"label": "Classes", "value": len(class_names)},
    ]
    metrics = append_classification_cv(metrics, "lightgbm", x_encoded, target_values, cv_folds, options)
    details = {
        "Model": "LGBMClassifier",
        "Boosting rounds": model.n_estimators,
        "Learning rate": model.learning_rate,
        "Leaves": model.num_leaves,
        "Maximum depth": model.max_depth,
        "Minimum child samples": model.min_child_samples,
        "Probability calibration": calibration_label(options),
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }

    return {
        "title": "LightGBM results",
        "description": f"Target: {target}. Gradient boosting decision tree model; test-set metrics are shown.",
        "target": target,
        "positive_class": str(classes[1]) if len(classes) == 2 else None,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, model.feature_importances_),
        "details_html": details_table(add_preprocessing_details(details, options)),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_adaboost_model(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "adaboost", "AdaBoost", options)


def fit_hist_gradient_boosting_model(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "hist_gradient_boosting", "Hist Gradient Boosting", options)


def fit_xgboost_classification(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "xgboost", "XGBoost", options)


def fit_catboost_classification(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "catboost", "CatBoost", options)


def fit_lda_model(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "lda", "Linear Discriminant Analysis", options)


def fit_qda_model(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "qda", "Quadratic Discriminant Analysis", options)


def fit_gaussian_process_model(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "gaussian_process", "Gaussian Process", options)


def fit_mlp_model(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "mlp", "MLP Neural Network", options)


def fit_passive_aggressive_model(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "passive_aggressive", "Passive Aggressive", options)


def fit_ridge_classifier_model(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "ridge_classifier", "Ridge Classifier", options)


def fit_linear_svm_model(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "linear_svm", "Linear SVM", options)


def fit_voting_model(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "voting", "Voting Classifier", options)


def fit_stacking_model(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "stacking", "Stacking Classifier", options)


def fit_dummy_classifier_model(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "dummy", "Dummy Classifier", options)


def fit_elastic_net_logistic_regression(data, target, predictors, test_size, cv_folds, options=None):
    if calibration_enabled(options):
        return fit_generic_classification_model(data, target, predictors, test_size, cv_folds, "elastic_net_logistic", "Elastic Net Logistic Regression", options, binary_only=True)

    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=True, options=options)
    split = split_classification_data(target_values, x_encoded, test_size, options)
    scaled_train, scaled_test, scaling_label = scaled_frames(split, options)
    model = LogisticRegression(
        C=tuned_float(options, "elastic_net_logistic", "C", 1.0),
        l1_ratio=tuned_float(options, "elastic_net_logistic", "l1_ratio", 0.5),
        max_iter=3000,
        random_state=preprocessing_seed(options),
        solver="saga",
    )
    model.fit(scaled_train, split["y_train"])
    predicted_codes = model.predict(scaled_test)
    predicted_labels = split["class_names"][predicted_codes]
    accuracy = accuracy_score(split["target_test"], predicted_labels)
    confusion = confusion_table(split["target_test"].to_numpy(), predicted_labels)
    coefficients = pd.DataFrame(
        {
            "Term": ["Intercept"] + list(x_encoded.columns),
            "Coefficient": [model.intercept_[0]] + list(model.coef_[0]),
        }
    )
    coefficients["Odds Ratio"] = np.exp(np.clip(coefficients["Coefficient"], -50, 50))

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
        {"label": "Test accuracy", "value": f"{accuracy:.3f}"},
        {"label": "Non-zero coefficients", "value": int(np.sum(np.abs(model.coef_[0]) > 1e-8))},
    ]
    metrics = append_classification_cv(metrics, "elastic_net_logistic", x_encoded, target_values, cv_folds, options)
    details = {
        "Model": "Elastic Net Logistic Regression",
        "C": model.C,
        "L1 ratio": model.l1_ratio,
        "Feature scaling": scaling_label,
        "Probability calibration": calibration_label(options),
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }

    return {
        "title": "Elastic Net Logistic Regression results",
        "description": f"Target: {target}. Positive class: {classes[1]}. Metrics are computed on the held-out test set.",
        "target": target,
        "positive_class": str(classes[1]),
        "metrics": metrics,
        "coefficients_html": display_table(coefficients, index=False, border=0, classes="coefficients"),
        "importances_html": None,
        "details_html": details_table(add_preprocessing_details(details, options)),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
        "download_data": {
            "coefficients": coefficients.to_csv(index=False),
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def prepare_regression_data(data, target, predictors, options=None):
    if target in predictors:
        raise ValueError("The target column cannot also be used as a predictor.")

    selected = [target] + predictors
    model_data = data[selected].copy()
    y_series = pd.to_numeric(model_data[target], errors="coerce")
    model_data = model_data.loc[y_series.notna()].copy()
    model_data[target] = y_series.loc[y_series.notna()].to_numpy(dtype=float)
    model_data = fill_missing_predictors(model_data, predictors, options)
    model_data = apply_outlier_handling(model_data, target, predictors, "regression", options)
    y = model_data[target].to_numpy(dtype=float)
    if len(y) < 3:
        raise ValueError("The regression target must contain at least three numeric values after preprocessing.")

    x_encoded = encode_predictors(model_data, predictors, options)
    if x_encoded.empty:
        raise ValueError("At least one usable predictor is required.")
    if x_encoded.isna().any().any():
        raise ValueError("Preprocessing left missing predictor values. Choose imputation or drop incomplete rows.")

    return y, x_encoded


def split_regression_data(y, x_encoded, test_size, options=None):
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=preprocessing_seed(options))
    if len(train_idx) < 2 or len(test_idx) < 1:
        raise ValueError("The train/test split leaves too few rows for regression.")

    return {
        "x_train": x_encoded.iloc[train_idx],
        "x_test": x_encoded.iloc[test_idx],
        "y_train": y[train_idx],
        "y_test": y[test_idx],
    }


def regression_metric_list(y, predictions, parameter_count=None):
    residuals = y - predictions
    sse = float(np.sum(residuals**2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - (sse / tss) if tss > 0 else 0.0
    rmse = math.sqrt(sse / len(y))
    mae = float(np.mean(np.abs(residuals)))
    metrics = [
        {"label": "Observations", "value": len(y)},
        {"label": "Test R squared", "value": f"{r_squared:.3f}"},
        {"label": "Test RMSE", "value": f"{rmse:.3f}"},
        {"label": "Test MAE", "value": f"{mae:.3f}"},
    ]

    if parameter_count is not None and len(y) > parameter_count:
        residual_df = len(y) - parameter_count
        adj_r_squared = 1 - ((1 - r_squared) * (len(y) - 1) / residual_df)
        metrics.append({"label": "Test adj. R squared", "value": f"{adj_r_squared:.3f}"})

    return metrics


def regression_coefficient_table(feature_names, coefficients):
    coefficients = pd.DataFrame(
        {
            "Term": feature_names,
            "Coefficient": coefficients,
        }
    )
    return display_table(coefficients, index=False, border=0, classes="coefficients")


def fit_linear_regression(data, target, predictors, test_size, cv_folds, options=None):
    y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
    split = split_regression_data(y, x_encoded, test_size, options)
    x_train = np.column_stack([np.ones(len(split["x_train"])), split["x_train"].to_numpy(dtype=float)])
    x_test = np.column_stack([np.ones(len(split["x_test"])), split["x_test"].to_numpy(dtype=float)])
    feature_names = ["Intercept"] + list(x_encoded.columns)
    rank = np.linalg.matrix_rank(x_train)
    if len(split["y_train"]) <= rank:
        raise ValueError("Not enough training rows for the selected predictors.")

    beta = np.linalg.pinv(x_train) @ split["y_train"]
    train_fitted = x_train @ beta
    train_residuals = split["y_train"] - train_fitted
    train_sse = float(np.sum(train_residuals**2))
    train_residual_df = len(split["y_train"]) - rank
    train_mse = train_sse / train_residual_df
    covariance = train_mse * np.linalg.pinv(x_train.T @ x_train)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_values = np.divide(beta, standard_errors, out=np.zeros_like(beta), where=standard_errors > 0)
    p_values = [normal_two_sided_pvalue(t_value) for t_value in t_values]
    test_predictions = x_test @ beta

    coefficients = pd.DataFrame(
        {
            "Term": feature_names,
            "Coefficient": beta,
            "Std. Error": standard_errors,
            "t value": t_values,
            "Approx. p value": p_values,
        }
    )

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], test_predictions, parameter_count=rank)[1:]
    metrics = append_regression_cv(metrics, "linear", x_encoded, y, cv_folds, options)
    details = {
        "Model": "Ordinary least squares",
        "Train residual df": train_residual_df,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }

    return {
        "title": "Linear Regression results",
        "description": f"Target: {target}. Ordinary least squares fit; metrics are computed on the held-out test set.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": display_table(coefficients, index=False, border=0, classes="coefficients"),
        "importances_html": None,
        "details_html": details_table(add_preprocessing_details(details, options)),
        "download_data": {
            "coefficients": coefficients.to_csv(index=False),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_ridge_regression(data, target, predictors, test_size, cv_folds, options=None):
    y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
    split = split_regression_data(y, x_encoded, test_size, options)
    scaled_train, scaled_test, scaling_label = scaled_frames(split, options)
    model = Ridge(alpha=tuned_float(options, "ridge", "alpha", 1.0))
    model.fit(scaled_train, split["y_train"])
    predictions = model.predict(scaled_test)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions, parameter_count=scaled_train.shape[1] + 1)[1:]
    metrics = append_regression_cv(metrics, "ridge", x_encoded, y, cv_folds, options)
    details = {
        "Model": "Ridge",
        "Alpha": model.alpha,
        "Feature scaling": scaling_label,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }
    coefficients = pd.DataFrame({"Term": ["Intercept"] + list(x_encoded.columns), "Coefficient": [model.intercept_] + list(model.coef_)})

    return {
        "title": "Ridge Regression results",
        "description": f"Target: {target}. Predictors were standardized using the training split; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": display_table(coefficients, index=False, border=0, classes="coefficients"),
        "importances_html": None,
        "details_html": details_table(add_preprocessing_details(details, options)),
        "download_data": {
            "coefficients": coefficients.to_csv(index=False),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_lasso_regression(data, target, predictors, test_size, cv_folds, options=None):
    y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
    split = split_regression_data(y, x_encoded, test_size, options)
    scaled_train, scaled_test, scaling_label = scaled_frames(split, options)
    model = Lasso(
        alpha=tuned_float(options, "lasso", "alpha", 0.1),
        max_iter=10000,
        random_state=preprocessing_seed(options),
    )
    model.fit(scaled_train, split["y_train"])
    predictions = model.predict(scaled_test)
    nonzero_count = int(np.sum(np.abs(model.coef_) > 1e-8))

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions, parameter_count=nonzero_count + 1)[1:]
    metrics = append_regression_cv(metrics, "lasso", x_encoded, y, cv_folds, options)
    details = {
        "Model": "Lasso",
        "Alpha": model.alpha,
        "Non-zero coefficients": nonzero_count,
        "Feature scaling": scaling_label,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }
    coefficients = pd.DataFrame({"Term": ["Intercept"] + list(x_encoded.columns), "Coefficient": [model.intercept_] + list(model.coef_)})

    return {
        "title": "Lasso Regression results",
        "description": f"Target: {target}. Predictors were standardized using the training split; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": display_table(coefficients, index=False, border=0, classes="coefficients"),
        "importances_html": None,
        "details_html": details_table(add_preprocessing_details(details, options)),
        "download_data": {
            "coefficients": coefficients.to_csv(index=False),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_elastic_net_regression(data, target, predictors, test_size, cv_folds, options=None):
    y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
    split = split_regression_data(y, x_encoded, test_size, options)
    scaled_train, scaled_test, scaling_label = scaled_frames(split, options)
    model = ElasticNet(
        alpha=tuned_float(options, "elastic_net", "alpha", 0.1),
        l1_ratio=tuned_float(options, "elastic_net", "l1_ratio", 0.5),
        max_iter=10000,
        random_state=preprocessing_seed(options),
    )
    model.fit(scaled_train, split["y_train"])
    predictions = model.predict(scaled_test)
    nonzero_count = int(np.sum(np.abs(model.coef_) > 1e-8))

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions, parameter_count=nonzero_count + 1)[1:]
    metrics = append_regression_cv(metrics, "elastic_net", x_encoded, y, cv_folds, options)
    details = {
        "Model": "ElasticNet",
        "Alpha": model.alpha,
        "L1 ratio": model.l1_ratio,
        "Non-zero coefficients": nonzero_count,
        "Feature scaling": scaling_label,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }
    coefficients = pd.DataFrame({"Term": ["Intercept"] + list(x_encoded.columns), "Coefficient": [model.intercept_] + list(model.coef_)})

    return {
        "title": "Elastic Net Regression results",
        "description": f"Target: {target}. Regularized regression blending Ridge and Lasso penalties; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": display_table(coefficients, index=False, border=0, classes="coefficients"),
        "importances_html": None,
        "details_html": details_table(add_preprocessing_details(details, options)),
        "download_data": {
            "coefficients": coefficients.to_csv(index=False),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_huber_regression(data, target, predictors, test_size, cv_folds, options=None):
    y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
    split = split_regression_data(y, x_encoded, test_size, options)
    scaled_train, scaled_test, scaling_label = scaled_frames(split, options)
    model = HuberRegressor(
        epsilon=tuned_float(options, "huber", "epsilon", 1.35),
        alpha=tuned_float(options, "huber", "alpha", 0.0001),
        max_iter=1000,
    )
    model.fit(scaled_train, split["y_train"])
    predictions = model.predict(scaled_test)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions, parameter_count=scaled_train.shape[1] + 1)[1:]
    metrics = append_regression_cv(metrics, "huber", x_encoded, y, cv_folds, options)
    details = {
        "Model": "HuberRegressor",
        "Epsilon": model.epsilon,
        "Alpha": model.alpha,
        "Estimated outliers": int(np.sum(model.outliers_)),
        "Feature scaling": scaling_label,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }
    coefficients = pd.DataFrame({"Term": ["Intercept"] + list(x_encoded.columns), "Coefficient": [model.intercept_] + list(model.coef_)})

    return {
        "title": "Huber Regression results",
        "description": f"Target: {target}. Robust linear regression that reduces the impact of large residuals; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": display_table(coefficients, index=False, border=0, classes="coefficients"),
        "importances_html": None,
        "details_html": details_table(add_preprocessing_details(details, options)),
        "download_data": {
            "coefficients": coefficients.to_csv(index=False),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_random_forest_regression(data, target, predictors, test_size, cv_folds, options=None):
    y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
    split = split_regression_data(y, x_encoded, test_size, options)
    min_samples_leaf = tuned_int(options, "random_forest", "min_samples_leaf", max(1, int(len(split["y_train"]) * 0.01)))
    model = RandomForestRegressor(
        n_estimators=tuned_int(options, "random_forest", "n_estimators", 200),
        max_depth=tuned_param(options, "random_forest", "max_depth", None),
        min_samples_leaf=min_samples_leaf,
        random_state=preprocessing_seed(options),
        n_jobs=-1,
    )
    model.fit(split["x_train"], split["y_train"])
    predictions = model.predict(split["x_test"])

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions)[1:]
    metrics = append_regression_cv(metrics, "random_forest", x_encoded, y, cv_folds, options)
    details = {
        "Model": "RandomForestRegressor",
        "Trees": model.n_estimators,
        "Maximum depth": model.max_depth if model.max_depth is not None else "None",
        "Minimum samples per leaf": min_samples_leaf,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }
    importances = pd.DataFrame({"Predictor": x_encoded.columns, "Importance": model.feature_importances_})

    return {
        "title": "Random Forest Regression results",
        "description": f"Target: {target}. Ensemble of regression trees; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, model.feature_importances_),
        "details_html": details_table(add_preprocessing_details(details, options)),
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_extra_trees_regression(data, target, predictors, test_size, cv_folds, options=None):
    y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
    split = split_regression_data(y, x_encoded, test_size, options)
    min_samples_leaf = tuned_int(options, "extra_trees", "min_samples_leaf", 1)
    model = ExtraTreesRegressor(
        n_estimators=tuned_int(options, "extra_trees", "n_estimators", 200),
        max_depth=tuned_param(options, "extra_trees", "max_depth", None),
        min_samples_leaf=min_samples_leaf,
        random_state=preprocessing_seed(options),
        n_jobs=-1,
    )
    model.fit(split["x_train"], split["y_train"])
    predictions = model.predict(split["x_test"])

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions)[1:]
    metrics = append_regression_cv(metrics, "extra_trees", x_encoded, y, cv_folds, options)
    details = {
        "Model": "ExtraTreesRegressor",
        "Trees": model.n_estimators,
        "Maximum depth": model.max_depth if model.max_depth is not None else "None",
        "Minimum samples per leaf": min_samples_leaf,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }
    importances = pd.DataFrame({"Predictor": x_encoded.columns, "Importance": model.feature_importances_})

    return {
        "title": "Extra Trees Regression results",
        "description": f"Target: {target}. Highly randomized tree ensemble; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, model.feature_importances_),
        "details_html": details_table(add_preprocessing_details(details, options)),
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_gradient_boosting_regression(data, target, predictors, test_size, cv_folds, options=None):
    y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
    split = split_regression_data(y, x_encoded, test_size, options)
    model = GradientBoostingRegressor(
        n_estimators=tuned_int(options, "gradient_boosting", "n_estimators", 100),
        learning_rate=tuned_float(options, "gradient_boosting", "learning_rate", 0.1),
        max_depth=tuned_int(options, "gradient_boosting", "max_depth", 3),
        random_state=preprocessing_seed(options),
    )
    model.fit(split["x_train"], split["y_train"])
    predictions = model.predict(split["x_test"])

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions)[1:]
    metrics = append_regression_cv(metrics, "gradient_boosting", x_encoded, y, cv_folds, options)
    details = {
        "Model": "GradientBoostingRegressor",
        "Boosting stages": model.n_estimators,
        "Learning rate": model.learning_rate,
        "Maximum tree depth": model.max_depth,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }
    importances = pd.DataFrame({"Predictor": x_encoded.columns, "Importance": model.feature_importances_})

    return {
        "title": "Gradient Boosting Regression results",
        "description": f"Target: {target}. Sequential boosted-tree regression model; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, model.feature_importances_),
        "details_html": details_table(add_preprocessing_details(details, options)),
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_lightgbm_regression(data, target, predictors, test_size, cv_folds, options=None):
    if LGBMRegressor is None:
        raise RuntimeError("LightGBM is not installed. Run pip install lightgbm or install requirements.txt.")
    y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
    split = split_regression_data(y, x_encoded, test_size, options)
    model = LGBMRegressor(
        n_estimators=tuned_int(options, "lightgbm", "n_estimators", 200),
        learning_rate=tuned_float(options, "lightgbm", "learning_rate", 0.05),
        num_leaves=tuned_int(options, "lightgbm", "num_leaves", 31),
        max_depth=tuned_int(options, "lightgbm", "max_depth", -1),
        min_child_samples=tuned_int(options, "lightgbm", "min_child_samples", 20),
        random_state=preprocessing_seed(options),
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(split["x_train"], split["y_train"])
    predictions = model.predict(split["x_test"])

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions)[1:]
    metrics = append_regression_cv(metrics, "lightgbm", x_encoded, y, cv_folds, options)
    details = {
        "Model": "LGBMRegressor",
        "Boosting stages": model.n_estimators,
        "Learning rate": model.learning_rate,
        "Leaves": model.num_leaves,
        "Maximum depth": model.max_depth,
        "Minimum child samples": model.min_child_samples,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }
    importances = pd.DataFrame({"Predictor": x_encoded.columns, "Importance": model.feature_importances_})

    return {
        "title": "LightGBM Regression results",
        "description": f"Target: {target}. Gradient boosted decision-tree regression using LightGBM; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, model.feature_importances_),
        "details_html": details_table(add_preprocessing_details(details, options)),
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_svr_regression(data, target, predictors, test_size, cv_folds, options=None):
    y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
    split = split_regression_data(y, x_encoded, test_size, options)
    scaled_train, scaled_test, scaling_label = scaled_frames(split, options)
    model = SVR(
        kernel="rbf",
        C=tuned_float(options, "svr", "C", 1.0),
        epsilon=tuned_float(options, "svr", "epsilon", 0.1),
        gamma=tuned_param(options, "svr", "gamma", "scale"),
    )
    model.fit(scaled_train, split["y_train"])
    predictions = model.predict(scaled_test)
    importances = permutation_importance_frame(model, scaled_test, split["y_test"], x_encoded.columns, "neg_root_mean_squared_error", options)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions)[1:]
    metrics = append_regression_cv(metrics, "svr", x_encoded, y, cv_folds, options)
    details = {
        "Model": "SVR",
        "Kernel": model.kernel,
        "C": model.C,
        "Epsilon": model.epsilon,
        "Gamma": model.gamma,
        "Support vectors": len(model.support_),
        "Feature scaling": scaling_label,
        "Importance method": "Permutation importance on held-out test set",
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }

    return {
        "title": "Support Vector Regression results",
        "description": f"Target: {target}. Predictors were standardized using the training split; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": permutation_importance_html(importances),
        "details_html": details_table(add_preprocessing_details(details, options)),
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def fit_knn_regression(data, target, predictors, test_size, cv_folds, options=None):
    y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
    split = split_regression_data(y, x_encoded, test_size, options)
    scaled_train, scaled_test, scaling_label = scaled_frames(split, options)
    neighbors = min(tuned_int(options, "knn", "n_neighbors", 5), len(split["y_train"]))
    model = KNeighborsRegressor(n_neighbors=neighbors, weights=tuned_param(options, "knn", "weights", "distance"))
    model.fit(scaled_train, split["y_train"])
    predictions = model.predict(scaled_test)
    importances = permutation_importance_frame(model, scaled_test, split["y_test"], x_encoded.columns, "neg_root_mean_squared_error", options)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions)[1:]
    metrics = append_regression_cv(metrics, "knn", x_encoded, y, cv_folds, options)
    details = {
        "Model": "KNeighborsRegressor",
        "Neighbors": neighbors,
        "Weights": model.weights,
        "Distance metric": model.metric,
        "Feature scaling": scaling_label,
        "Importance method": "Permutation importance on held-out test set",
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }

    return {
        "title": "kNN Regression results",
        "description": f"Target: {target}. Predictors were standardized using the training split; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": permutation_importance_html(importances),
        "details_html": details_table(add_preprocessing_details(details, options)),
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False),
        },
    }


def estimator_leaf(estimator):
    if hasattr(estimator, "named_steps"):
        return list(estimator.named_steps.values())[-1]
    return estimator


def fitted_regression_coefficients(estimator, feature_names):
    leaf = estimator_leaf(estimator)
    if not hasattr(leaf, "coef_"):
        return None
    coefficients = np.ravel(leaf.coef_)
    if len(coefficients) != len(feature_names):
        return None
    intercept = getattr(leaf, "intercept_", None)
    if intercept is None:
        return pd.DataFrame({"Term": feature_names, "Coefficient": coefficients})
    return pd.DataFrame(
        {
            "Term": ["Intercept"] + list(feature_names),
            "Coefficient": [float(np.ravel(intercept)[0])] + list(coefficients),
        }
    )


def fitted_regression_importances(estimator, x_test, y_test, feature_names, options):
    leaf = estimator_leaf(estimator)
    if hasattr(leaf, "feature_importances_"):
        return importance_frame(feature_names, leaf.feature_importances_)
    return permutation_importance_frame(estimator, x_test, y_test, feature_names, "neg_root_mean_squared_error", options)


def fit_generic_regression_model(data, target, predictors, test_size, cv_folds, model_name, model_label, description, options=None):
    y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
    split = split_regression_data(y, x_encoded, test_size, options)
    estimator = regression_estimator(model_name, options)
    estimator.fit(split["x_train"], split["y_train"])
    predictions = np.ravel(estimator.predict(split["x_test"]))

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions)[1:]
    metrics = append_regression_cv(metrics, model_name, x_encoded, y, cv_folds, options)

    details = {
        "Model": model_label,
        "Estimator": estimator_leaf(estimator).__class__.__name__,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": preprocessing_seed(options),
    }
    coefficients = fitted_regression_coefficients(estimator, x_encoded.columns)
    importances = None if coefficients is not None else fitted_regression_importances(
        estimator,
        split["x_test"],
        split["y_test"],
        x_encoded.columns,
        options,
    )

    download_data = {"details": details_frame(add_preprocessing_details(details, options)).to_csv(index=False)}
    if coefficients is not None:
        download_data["coefficients"] = coefficients.to_csv(index=False)
    if importances is not None:
        download_data["variable_importance"] = importances.to_csv(index=False)

    return {
        "title": f"{model_label} results",
        "description": f"Target: {target}. {description}",
        "target": target,
        "metrics": metrics,
        "coefficients_html": display_table(coefficients, index=False, border=0, classes="coefficients") if coefficients is not None else None,
        "importances_html": display_table(importances, index=False, border=0, classes="importances") if importances is not None else None,
        "details_html": details_table(add_preprocessing_details(details, options)),
        "download_data": download_data,
    }


def fit_bayesian_ridge_regression(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_regression_model(
        data,
        target,
        predictors,
        test_size,
        cv_folds,
        "bayesian_ridge",
        "Bayesian Ridge Regression",
        "Regularized linear regression with Bayesian shrinkage; test-set metrics are shown.",
        options,
    )


def fit_quantile_regression(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_regression_model(
        data,
        target,
        predictors,
        test_size,
        cv_folds,
        "quantile",
        "Quantile Regression",
        "Median-oriented regression that can be tuned to other quantiles; test-set metrics are shown.",
        options,
    )


def fit_passive_aggressive_regression(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_regression_model(
        data,
        target,
        predictors,
        test_size,
        cv_folds,
        "passive_aggressive",
        "Passive Aggressive Regression",
        "Fast linear regression for larger datasets; test-set metrics are shown.",
        options,
    )


def fit_kernel_ridge_regression(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_regression_model(
        data,
        target,
        predictors,
        test_size,
        cv_folds,
        "kernel_ridge",
        "Kernel Ridge Regression",
        "Regularized nonlinear regression using kernel features; test-set metrics are shown.",
        options,
    )


def fit_pls_regression(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_regression_model(
        data,
        target,
        predictors,
        test_size,
        cv_folds,
        "pls",
        "PLS Regression",
        "Latent-component regression for correlated predictors; test-set metrics are shown.",
        options,
    )


def fit_tweedie_regression(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_regression_model(
        data,
        target,
        predictors,
        test_size,
        cv_folds,
        "tweedie",
        "Tweedie Regression",
        "Generalized linear regression for skewed positive targets; test-set metrics are shown.",
        options,
    )


def fit_poisson_regression(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_regression_model(
        data,
        target,
        predictors,
        test_size,
        cv_folds,
        "poisson",
        "Poisson Regression",
        "Generalized linear regression for nonnegative count-like targets; test-set metrics are shown.",
        options,
    )


def fit_xgboost_regression(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_regression_model(
        data,
        target,
        predictors,
        test_size,
        cv_folds,
        "xgboost",
        "XGBoost Regression",
        "Gradient boosted tree regression using XGBoost; test-set metrics are shown.",
        options,
    )


def fit_catboost_regression(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_regression_model(
        data,
        target,
        predictors,
        test_size,
        cv_folds,
        "catboost",
        "CatBoost Regression",
        "Ordered boosted-tree regression using CatBoost; test-set metrics are shown.",
        options,
    )


def fit_bagging_regression(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_regression_model(
        data,
        target,
        predictors,
        test_size,
        cv_folds,
        "bagging",
        "Bagging Regressor",
        "Bootstrap ensemble regression that stabilizes tree predictions; test-set metrics are shown.",
        options,
    )


def fit_voting_regression(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_regression_model(
        data,
        target,
        predictors,
        test_size,
        cv_folds,
        "voting",
        "Voting Regressor",
        "Averaged ensemble regression across several base models; test-set metrics are shown.",
        options,
    )


def fit_stacking_regression(data, target, predictors, test_size, cv_folds, options=None):
    return fit_generic_regression_model(
        data,
        target,
        predictors,
        test_size,
        cv_folds,
        "stacking",
        "Stacking Regressor",
        "Ensemble regression that blends several base models through a final meta-model; test-set metrics are shown.",
        options,
    )


CLASSIFICATION_MODEL_FITTERS = {
    "logistic": fit_logistic_regression,
    "tree": fit_tree_model,
    "random_forest": fit_random_forest_model,
    "extra_trees": fit_extra_trees_model,
    "gradient_boosting": fit_gradient_boosting_model,
    "adaboost": fit_adaboost_model,
    "hist_gradient_boosting": fit_hist_gradient_boosting_model,
    "xgboost": fit_xgboost_classification,
    "catboost": fit_catboost_classification,
    "lightgbm": fit_lightgbm_model,
    "naive_bayes": fit_naive_bayes_model,
    "lda": fit_lda_model,
    "qda": fit_qda_model,
    "gaussian_process": fit_gaussian_process_model,
    "mlp": fit_mlp_model,
    "passive_aggressive": fit_passive_aggressive_model,
    "ridge_classifier": fit_ridge_classifier_model,
    "linear_svm": fit_linear_svm_model,
    "voting": fit_voting_model,
    "stacking": fit_stacking_model,
    "dummy": fit_dummy_classifier_model,
    "elastic_net_logistic": fit_elastic_net_logistic_regression,
    "svm": fit_svm_model,
    "knn": fit_knn_model,
}

REGRESSION_MODEL_FITTERS = {
    "linear": fit_linear_regression,
    "ridge": fit_ridge_regression,
    "lasso": fit_lasso_regression,
    "elastic_net": fit_elastic_net_regression,
    "huber": fit_huber_regression,
    "bayesian_ridge": fit_bayesian_ridge_regression,
    "quantile": fit_quantile_regression,
    "passive_aggressive": fit_passive_aggressive_regression,
    "kernel_ridge": fit_kernel_ridge_regression,
    "pls": fit_pls_regression,
    "tweedie": fit_tweedie_regression,
    "poisson": fit_poisson_regression,
    "random_forest": fit_random_forest_regression,
    "extra_trees": fit_extra_trees_regression,
    "gradient_boosting": fit_gradient_boosting_regression,
    "lightgbm": fit_lightgbm_regression,
    "xgboost": fit_xgboost_regression,
    "catboost": fit_catboost_regression,
    "bagging": fit_bagging_regression,
    "voting": fit_voting_regression,
    "stacking": fit_stacking_regression,
    "svr": fit_svr_regression,
    "knn": fit_knn_regression,
}

CLASSIFICATION_TAB_CONFIGS = {
    "classification": {
        "id": "classification",
        "form_name": "classification",
        "model_field": "classification_model",
        "target_field": "target",
        "predictors_field": "predictors",
        "test_size_field": "classification_test_size",
        "cv_folds_field": "classification_cv_folds",
        "default_model": "logistic",
    },
    "pro_classification": {
        "id": "pro_classification",
        "form_name": "pro_classification",
        "model_field": "pro_classification_model",
        "detail_model_field": "pro_classification_detail_model",
        "target_field": "pro_classification_target",
        "predictors_field": "pro_classification_predictors",
        "test_size_field": "pro_classification_test_size",
        "cv_folds_field": "pro_classification_cv_folds",
        "missing_values_field": "pro_classification_missing_values",
        "categorical_encoding_field": "pro_classification_categorical_encoding",
        "scaling_field": "pro_classification_scaling",
        "split_seed_field": "pro_classification_split_seed",
        "outlier_handling_field": "pro_classification_outlier_handling",
        "calibration_field": "pro_classification_calibration",
        "tuning_mode_field": "pro_classification_tuning_mode",
        "tuning_iterations_field": "pro_classification_tuning_iterations",
        "threshold_field": "pro_classification_threshold",
        "default_model": "logistic",
        "allow_model_comparison": True,
    },
}

REGRESSION_TAB_CONFIGS = {
    "regression": {
        "id": "regression",
        "form_name": "regression",
        "model_field": "regression_model",
        "target_field": "regression_target",
        "predictors_field": "regression_predictors",
        "test_size_field": "regression_test_size",
        "cv_folds_field": "regression_cv_folds",
        "default_model": "linear",
    },
    "pro_regression": {
        "id": "pro_regression",
        "form_name": "pro_regression",
        "model_field": "pro_regression_model",
        "detail_model_field": "pro_regression_detail_model",
        "target_field": "pro_regression_target",
        "predictors_field": "pro_regression_predictors",
        "test_size_field": "pro_regression_test_size",
        "cv_folds_field": "pro_regression_cv_folds",
        "missing_values_field": "pro_regression_missing_values",
        "categorical_encoding_field": "pro_regression_categorical_encoding",
        "scaling_field": "pro_regression_scaling",
        "split_seed_field": "pro_regression_split_seed",
        "outlier_handling_field": "pro_regression_outlier_handling",
        "tuning_mode_field": "pro_regression_tuning_mode",
        "tuning_iterations_field": "pro_regression_tuning_iterations",
        "default_model": "linear",
        "allow_model_comparison": True,
    },
}


def make_model_tab(config):
    tab = config.copy()
    tab.update(
        {
            "selected_model": config["default_model"],
            "selected_models": [config["default_model"]],
            "selected_detail_model": "best",
            "allow_model_comparison": config.get("allow_model_comparison", False),
            "selected_test_size": 0.2,
            "selected_cv_folds": 0,
            "selected_missing_values": "drop",
            "selected_categorical_encoding": "one_hot_drop_first",
            "selected_scaling": "on",
            "selected_split_seed": random_split_seed(),
            "selected_outlier_handling": "none",
            "selected_calibration": "off",
            "selected_tuning_mode": "off",
            "selected_tuning_iterations": 10,
            "selected_threshold": 0.5,
            "run_name": "",
            "run_notes": "",
            "threshold_field": config.get("threshold_field"),
            "calibration_field": config.get("calibration_field"),
            "selected_target": None,
            "selected_predictors": [],
            "error": None,
            "output": None,
            "comparison_html": None,
            "detail_metric_comparison_html": None,
            "recommendation": None,
            "comparison_download": None,
            "comparison_pdf_download": None,
            "report_download": None,
            "report_pdf_download": None,
            "run_history": [],
            "run_comparison": None,
        }
    )
    return tab

def make_model_tabs():
    return {
        "classification": make_model_tab(CLASSIFICATION_TAB_CONFIGS["classification"]),
        "regression": make_model_tab(REGRESSION_TAB_CONFIGS["regression"]),
        "pro_classification": make_model_tab(CLASSIFICATION_TAB_CONFIGS["pro_classification"]),
        "pro_regression": make_model_tab(REGRESSION_TAB_CONFIGS["pro_regression"]),
    }


def apply_classification_defaults(tab, columns):
    if tab["selected_target"] is None and columns:
        tab["selected_target"] = columns[0]
        tab["selected_predictors"] = columns[1:]


def apply_regression_defaults(tab, data, columns):
    if tab["selected_target"] is not None or not columns:
        return

    numeric_columns = list(data.select_dtypes(include="number").columns)
    if numeric_columns:
        tab["selected_target"] = "regression_target" if "regression_target" in numeric_columns else numeric_columns[0]
        tab["selected_predictors"] = [column for column in numeric_columns if column != tab["selected_target"]]
    else:
        tab["selected_target"] = columns[0]
        tab["selected_predictors"] = columns[1:]


def available_model_names(tab):
    if tab["form_name"] in REGRESSION_TAB_CONFIGS:
        return set(REGRESSION_MODEL_FITTERS)
    return set(CLASSIFICATION_MODEL_FITTERS)


def populate_tab_from_request(tab):
    if tab.get("allow_model_comparison"):
        selected_models = request.form.getlist(tab["model_field"])
        allowed_models = available_model_names(tab)
        tab["selected_models"] = [model for model in selected_models if model in allowed_models]
        tab["selected_model"] = tab["selected_models"][0] if tab["selected_models"] else tab["default_model"]
        selected_detail_model = request.form.get(tab.get("detail_model_field", ""), "best")
        tab["selected_detail_model"] = selected_detail_model if selected_detail_model == "best" or selected_detail_model in allowed_models else "best"
        tab["run_name"] = request.form.get("run_name", "").strip()[:120]
        tab["run_notes"] = request.form.get("run_notes", "").strip()[:2000]
    else:
        tab["selected_model"] = request.form.get(tab["model_field"], tab["default_model"])
        tab["selected_models"] = [tab["selected_model"]]
        tab["selected_detail_model"] = "best"

    tab["selected_test_size"] = parse_test_size(request.form.get(tab["test_size_field"]))
    tab["selected_cv_folds"] = parse_cv_folds(request.form.get(tab["cv_folds_field"]))
    if tab.get("allow_model_comparison"):
        tab["selected_missing_values"] = parse_choice(
            request.form.get(tab["missing_values_field"]),
            {"drop", "impute_mean_mode", "impute_median_mode"},
            "drop",
        )
        tab["selected_categorical_encoding"] = parse_choice(
            request.form.get(tab["categorical_encoding_field"]),
            {"one_hot_drop_first", "one_hot_full", "ordinal"},
            "one_hot_drop_first",
        )
        tab["selected_scaling"] = parse_choice(request.form.get(tab["scaling_field"]), {"on", "off"}, "on")
        tab["selected_split_seed"] = parse_split_seed(request.form.get(tab["split_seed_field"]))
        tab["selected_outlier_handling"] = parse_choice(
            request.form.get(tab["outlier_handling_field"]),
            {"none", "winsorize", "remove_iqr"},
            "none",
        )
        if tab.get("calibration_field"):
            tab["selected_calibration"] = parse_choice(
                request.form.get(tab["calibration_field"]),
                {"off", "sigmoid", "isotonic"},
                "off",
            )
        tab["selected_tuning_mode"] = parse_choice(
            request.form.get(tab["tuning_mode_field"]),
            {"off", "grid", "random"},
            "off",
        )
        tab["selected_tuning_iterations"] = parse_tuning_iterations(request.form.get(tab["tuning_iterations_field"]))
        if tab.get("threshold_field"):
            tab["selected_threshold"] = parse_threshold(request.form.get(tab["threshold_field"]))
    tab["selected_target"] = request.form.get(tab["target_field"])
    tab["selected_predictors"] = request.form.getlist(tab["predictors_field"])

CLASSIFICATION_MODEL_LABELS = {
    "logistic": "Logistic regression",
    "tree": "Tree model",
    "random_forest": "Random Forest",
    "extra_trees": "Extra Trees",
    "gradient_boosting": "Gradient Boosting",
    "adaboost": "AdaBoost",
    "catboost": "CatBoost",
    "dummy": "Dummy Classifier",
    "hist_gradient_boosting": "Hist Gradient Boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "naive_bayes": "Naive Bayes",
    "lda": "Linear Discriminant Analysis",
    "linear_svm": "Linear SVM",
    "qda": "Quadratic Discriminant Analysis",
    "gaussian_process": "Gaussian Process",
    "mlp": "MLP Neural Network",
    "passive_aggressive": "Passive Aggressive",
    "ridge_classifier": "Ridge Classifier",
    "stacking": "Stacking Classifier",
    "voting": "Voting Classifier",
    "elastic_net_logistic": "Elastic Net Logistic Regression",
    "svm": "Support Vector Machine",
    "knn": "kNN",
}


def tuning_mode(tab):
    return tab.get("selected_tuning_mode", "off")


def tuning_enabled(tab):
    return tuning_mode(tab) in {"grid", "random"}


def tuning_iterations(tab):
    return tab.get("selected_tuning_iterations", 10)


def estimator_step_name(estimator, estimator_class):
    if isinstance(estimator, CalibratedClassifierCV):
        return f"estimator__{estimator_step_name(estimator.estimator, estimator_class)}"
    if hasattr(estimator, "named_steps"):
        for name, step in estimator.named_steps.items():
            if isinstance(step, estimator_class):
                return f"{name}__"
    return ""


def classification_tuning_grid(model_name, estimator):
    if model_name == "logistic":
        prefix = estimator_step_name(estimator, LogisticRegression)
        return {f"{prefix}C": [0.1, 1.0, 10.0]}
    if model_name == "elastic_net_logistic":
        prefix = estimator_step_name(estimator, LogisticRegression)
        return {f"{prefix}C": [0.1, 1.0, 10.0], f"{prefix}l1_ratio": [0.15, 0.5, 0.85]}
    if model_name == "tree":
        prefix = estimator_step_name(estimator, DecisionTreeClassifier)
        return {f"{prefix}max_depth": [3, 4, 6, None], f"{prefix}min_samples_leaf": [1, 5, 10]}
    if model_name == "random_forest":
        prefix = estimator_step_name(estimator, RandomForestClassifier)
        return {f"{prefix}n_estimators": [100, 200], f"{prefix}max_depth": [None, 6, 10], f"{prefix}min_samples_leaf": [1, 5]}
    if model_name == "extra_trees":
        prefix = estimator_step_name(estimator, ExtraTreesClassifier)
        return {f"{prefix}n_estimators": [100, 200], f"{prefix}max_depth": [None, 6, 10], f"{prefix}min_samples_leaf": [1, 5]}
    if model_name == "gradient_boosting":
        prefix = estimator_step_name(estimator, GradientBoostingClassifier)
        return {f"{prefix}n_estimators": [50, 100], f"{prefix}learning_rate": [0.05, 0.1, 0.2], f"{prefix}max_depth": [2, 3]}
    if model_name == "adaboost":
        prefix = estimator_step_name(estimator, AdaBoostClassifier)
        return {f"{prefix}n_estimators": [50, 100, 200], f"{prefix}learning_rate": [0.25, 0.5, 1.0]}
    if model_name == "hist_gradient_boosting":
        prefix = estimator_step_name(estimator, HistGradientBoostingClassifier)
        return {
            f"{prefix}max_iter": [50, 100, 200],
            f"{prefix}learning_rate": [0.05, 0.1],
            f"{prefix}max_leaf_nodes": [15, 31],
        }
    if model_name == "xgboost":
        if XGBClassifier is None:
            return {}
        prefix = estimator_step_name(estimator, XGBClassifier)
        return {
            f"{prefix}n_estimators": [100, 200],
            f"{prefix}learning_rate": [0.03, 0.05, 0.1],
            f"{prefix}max_depth": [3, 4, 6],
            f"{prefix}subsample": [0.8, 1.0],
        }
    if model_name == "catboost":
        if CatBoostClassifier is None:
            return {}
        prefix = estimator_step_name(estimator, CatBoostClassifier)
        return {
            f"{prefix}iterations": [100, 200],
            f"{prefix}learning_rate": [0.03, 0.05, 0.1],
            f"{prefix}depth": [4, 6, 8],
        }
    if model_name == "lightgbm":
        if LGBMClassifier is None:
            return {}
        prefix = estimator_step_name(estimator, LGBMClassifier)
        return {
            f"{prefix}n_estimators": [100, 200],
            f"{prefix}learning_rate": [0.03, 0.05, 0.1],
            f"{prefix}num_leaves": [15, 31, 63],
            f"{prefix}min_child_samples": [10, 20, 40],
        }
    if model_name == "naive_bayes":
        prefix = estimator_step_name(estimator, GaussianNB)
        return {f"{prefix}var_smoothing": [1e-9, 1e-8, 1e-7]}
    if model_name == "lda":
        prefix = estimator_step_name(estimator, LinearDiscriminantAnalysis)
        return {f"{prefix}shrinkage": ["auto", None, 0.25, 0.5]}
    if model_name == "qda":
        prefix = estimator_step_name(estimator, QuadraticDiscriminantAnalysis)
        return {f"{prefix}reg_param": [0.0, 0.1, 0.25, 0.5]}
    if model_name == "gaussian_process":
        prefix = estimator_step_name(estimator, GaussianProcessClassifier)
        return {f"{prefix}max_iter_predict": [50, 100, 200]}
    if model_name == "mlp":
        prefix = estimator_step_name(estimator, MLPClassifier)
        return {
            f"{prefix}hidden_layer_sizes": [(50,), (100,), (50, 25)],
            f"{prefix}alpha": [0.0001, 0.001],
            f"{prefix}learning_rate_init": [0.001, 0.01],
        }
    if model_name == "passive_aggressive":
        prefix = estimator_step_name(estimator, SGDClassifier)
        return {f"{prefix}eta0": [0.1, 1.0, 10.0]}
    if model_name == "ridge_classifier":
        prefix = estimator_step_name(estimator, RidgeClassifier)
        return {f"{prefix}alpha": [0.1, 1.0, 10.0]}
    if model_name == "linear_svm":
        prefix = estimator_step_name(estimator, LinearSVC)
        return {f"{prefix}C": [0.1, 1.0, 10.0]}
    if model_name == "svm":
        prefix = estimator_step_name(estimator, SVC)
        return {f"{prefix}C": [0.5, 1.0, 2.0], f"{prefix}gamma": ["scale", "auto"]}
    if model_name == "voting":
        prefix = estimator_step_name(estimator, VotingClassifier)
        return {f"{prefix}voting": ["soft", "hard"]}
    if model_name == "stacking":
        prefix = estimator_step_name(estimator, StackingClassifier)
        return {f"{prefix}final_estimator__C": [0.1, 1.0, 10.0], f"{prefix}passthrough": [False, True]}
    if model_name == "dummy":
        prefix = estimator_step_name(estimator, DummyClassifier)
        return {f"{prefix}strategy": ["most_frequent", "stratified", "prior"]}
    if model_name == "knn":
        prefix = estimator_step_name(estimator, KNeighborsClassifier)
        return {f"{prefix}n_neighbors": [3, 5, 9], f"{prefix}weights": ["uniform", "distance"]}
    return {}


def regression_tuning_grid(model_name, estimator):
    if model_name == "linear":
        return {}
    if model_name == "ridge":
        prefix = estimator_step_name(estimator, Ridge)
        return {f"{prefix}alpha": [0.1, 1.0, 10.0]}
    if model_name == "lasso":
        prefix = estimator_step_name(estimator, Lasso)
        return {f"{prefix}alpha": [0.01, 0.1, 1.0]}
    if model_name == "elastic_net":
        prefix = estimator_step_name(estimator, ElasticNet)
        return {f"{prefix}alpha": [0.01, 0.1, 1.0], f"{prefix}l1_ratio": [0.15, 0.5, 0.85]}
    if model_name == "huber":
        prefix = estimator_step_name(estimator, HuberRegressor)
        return {f"{prefix}epsilon": [1.1, 1.35, 1.75], f"{prefix}alpha": [0.0001, 0.001, 0.01]}
    if model_name == "bayesian_ridge":
        prefix = estimator_step_name(estimator, BayesianRidge)
        return {f"{prefix}alpha_1": [1e-7, 1e-6, 1e-5], f"{prefix}lambda_1": [1e-7, 1e-6, 1e-5]}
    if model_name == "quantile":
        prefix = estimator_step_name(estimator, QuantileRegressor)
        return {f"{prefix}alpha": [0.0, 0.0001, 0.001], f"{prefix}quantile": [0.25, 0.5, 0.75]}
    if model_name == "passive_aggressive":
        prefix = estimator_step_name(estimator, SGDRegressor)
        return {f"{prefix}eta0": [0.1, 1.0, 10.0], f"{prefix}epsilon": [0.05, 0.1, 0.2]}
    if model_name == "kernel_ridge":
        prefix = estimator_step_name(estimator, KernelRidge)
        return {f"{prefix}alpha": [0.1, 1.0, 10.0], f"{prefix}kernel": ["rbf", "linear"], f"{prefix}gamma": [0.01, 0.1, 1.0]}
    if model_name == "pls":
        prefix = estimator_step_name(estimator, PLSRegression)
        return {f"{prefix}n_components": [1, 2, 3]}
    if model_name == "tweedie":
        prefix = estimator_step_name(estimator, TweedieRegressor)
        return {f"{prefix}power": [0.0, 1.0, 1.5], f"{prefix}alpha": [0.01, 0.1, 1.0]}
    if model_name == "poisson":
        prefix = estimator_step_name(estimator, PoissonRegressor)
        return {f"{prefix}alpha": [0.01, 0.1, 1.0]}
    if model_name == "random_forest":
        return {"n_estimators": [100, 200], "max_depth": [None, 6, 10], "min_samples_leaf": [1, 5]}
    if model_name == "extra_trees":
        return {"n_estimators": [100, 200], "max_depth": [None, 6, 10], "min_samples_leaf": [1, 5]}
    if model_name == "gradient_boosting":
        return {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [2, 3]}
    if model_name == "lightgbm":
        if LGBMRegressor is None:
            return {}
        return {
            "n_estimators": [100, 200],
            "learning_rate": [0.03, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "min_child_samples": [10, 20, 40],
        }
    if model_name == "xgboost":
        if XGBRegressor is None:
            return {}
        return {
            "n_estimators": [100, 200],
            "learning_rate": [0.03, 0.05, 0.1],
            "max_depth": [3, 4, 6],
            "subsample": [0.8, 1.0],
        }
    if model_name == "catboost":
        if CatBoostRegressor is None:
            return {}
        return {
            "iterations": [100, 200],
            "learning_rate": [0.03, 0.05, 0.1],
            "depth": [4, 6, 8],
        }
    if model_name == "bagging":
        return {
            "n_estimators": [50, 100, 200],
            "estimator__max_depth": [None, 4, 8],
            "estimator__min_samples_leaf": [1, 5],
        }
    if model_name == "voting":
        return {"weights": [None, [2, 1, 1, 1], [1, 2, 2, 1]]}
    if model_name == "stacking":
        return {"final_estimator__alpha": [0.1, 1.0, 10.0], "passthrough": [False, True]}
    if model_name == "svr":
        prefix = estimator_step_name(estimator, SVR)
        return {f"{prefix}C": [0.5, 1.0, 2.0], f"{prefix}epsilon": [0.05, 0.1, 0.2], f"{prefix}gamma": ["scale", "auto"]}
    if model_name == "knn":
        prefix = estimator_step_name(estimator, KNeighborsRegressor)
        return {f"{prefix}n_neighbors": [3, 5, 9], f"{prefix}weights": ["uniform", "distance"]}
    return {}


def search_cv_for_mode(estimator, param_grid, mode, iterations, scoring, cv, seed):
    if mode == "grid":
        return GridSearchCV(estimator, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
    return RandomizedSearchCV(
        estimator,
        param_grid,
        n_iter=min(iterations, max(1, math.prod(len(values) for values in param_grid.values()))),
        cv=cv,
        scoring=scoring,
        random_state=seed,
        n_jobs=-1,
    )


def formatted_best_params(params):
    if not params:
        return "-"
    cleaned = {key.split("__")[-1]: value for key, value in params.items()}
    return ", ".join(f"{key}={value}" for key, value in cleaned.items())


def normalized_search_params(params):
    return {key.split("__")[-1]: value for key, value in params.items()}


def tune_classification_model(model_name, data, target, predictors, test_size, cv_folds, options, mode, iterations):
    _, target_values, _, x_encoded = prepare_classification_data(data, target, predictors, binary_only=(model_name in {"logistic", "elastic_net_logistic"}), options=options)
    split = split_classification_data(target_values, x_encoded, test_size, options)
    min_class_count = int(pd.Series(split["y_train"]).value_counts().min())
    actual_folds = min(cv_folds if cv_folds > 1 else 3, min_class_count)
    if actual_folds < 2:
        return {"error": "Not enough class balance"}

    estimator = classification_estimator(model_name, options)
    param_grid = classification_tuning_grid(model_name, estimator)
    if not param_grid:
        return {"error": "No tuning grid"}

    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=preprocessing_seed(options))
    search = search_cv_for_mode(estimator, param_grid, mode, iterations, "accuracy", cv, preprocessing_seed(options))
    search.fit(split["x_train"], split["y_train"])
    predictions = search.best_estimator_.predict(split["x_test"])
    return {
        "test_accuracy": float(accuracy_score(split["y_test"], predictions)),
        "cv_accuracy": float(search.best_score_),
        "params": formatted_best_params(search.best_params_),
        "raw_params": normalized_search_params(search.best_params_),
    }


def tune_regression_model(model_name, data, target, predictors, test_size, cv_folds, options, mode, iterations):
    y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
    split = split_regression_data(y, x_encoded, test_size, options)
    actual_folds = min(cv_folds if cv_folds > 1 else 3, len(split["y_train"]))
    if actual_folds < 2:
        return {"error": "Not enough rows"}

    estimator = regression_estimator(model_name, options)
    param_grid = regression_tuning_grid(model_name, estimator)
    if not param_grid:
        return {"error": "No tuning grid"}

    cv = KFold(n_splits=actual_folds, shuffle=True, random_state=preprocessing_seed(options))
    search = search_cv_for_mode(estimator, param_grid, mode, iterations, "neg_root_mean_squared_error", cv, preprocessing_seed(options))
    search.fit(split["x_train"], split["y_train"])
    predictions = np.ravel(search.best_estimator_.predict(split["x_test"]))
    rmse = math.sqrt(float(np.mean((split["y_test"] - predictions) ** 2)))
    return {
        "test_rmse": rmse,
        "cv_rmse": float(-search.best_score_),
        "params": formatted_best_params(search.best_params_),
        "raw_params": normalized_search_params(search.best_params_),
    }

def metric_value(output, label):
    for metric in output.get("metrics", []):
        if metric.get("label") == label:
            return metric.get("value")
    return None


def metric_float(output, label):
    value = metric_value(output, label)
    try:
        return float(str(value).replace("%", ""))
    except (TypeError, ValueError):
        return None


def format_optional_metric(value):
    if value is None:
        return "-"
    return f"{value:.3f}"


def weighted_precision_recall_f1(output):
    confusion_csv = output.get("download_data", {}).get("confusion_matrix")
    if not confusion_csv:
        return None, None, None

    confusion = pd.read_csv(StringIO(confusion_csv), index_col=0)
    confusion.index = confusion.index.map(str)
    confusion.columns = confusion.columns.map(str)
    labels = sorted(set(confusion.index) | set(confusion.columns))
    confusion = confusion.reindex(index=labels, columns=labels, fill_value=0).astype(float)
    supports = confusion.sum(axis=1)
    total = float(supports.sum())
    if total <= 0:
        return None, None, None

    precision_values = []
    recall_values = []
    f1_values = []
    weights = []
    for label in labels:
        support = float(supports.loc[label])
        if support <= 0:
            continue
        true_positive = float(confusion.loc[label, label])
        predicted_positive = float(confusion[label].sum())
        precision = true_positive / predicted_positive if predicted_positive else 0.0
        recall = true_positive / support
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
        weights.append(support / total)

    if not weights:
        return None, None, None
    return (
        float(np.average(precision_values, weights=weights)),
        float(np.average(recall_values, weights=weights)),
        float(np.average(f1_values, weights=weights)),
    )



def line_plot_image(x_values, y_values, xlabel, ylabel, title, annotation=None, baseline=None):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(x_values, y_values, color="#176b87", linewidth=2.3)
    if baseline is not None:
        ax.plot(baseline["x"], baseline["y"], color="#94a3b8", linestyle="--", linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    if annotation:
        ax.text(
            0.98,
            0.04,
            annotation,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d8dee8"},
        )
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def classification_score_values(estimator, x_test):
    if hasattr(estimator, "predict_proba"):
        probabilities = estimator.predict_proba(x_test)
        classes = list(estimator.classes_)
        positive_index = classes.index(1) if 1 in classes else len(classes) - 1
        return probabilities[:, positive_index]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(x_test)
        if np.ndim(scores) > 1:
            return scores[:, -1]
        return scores
    raise ValueError("The selected model does not expose probability or decision scores.")


def threshold_metrics(y_true, scores, threshold):
    predicted = scores >= threshold
    true_positive = int(np.sum((predicted == 1) & (y_true == 1)))
    true_negative = int(np.sum((predicted == 0) & (y_true == 0)))
    false_positive = int(np.sum((predicted == 1) & (y_true == 0)))
    false_negative = int(np.sum((predicted == 0) & (y_true == 1)))
    total = len(y_true)
    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    specificity = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "predicted_positive_rate": float(np.mean(predicted)) if total else 0.0,
        "predicted": predicted,
    }


def threshold_metrics_frame(metrics):
    return pd.DataFrame(
        [
            {"Metric": "Threshold", "Value": f"{metrics['threshold']:.3f}"},
            {"Metric": "Accuracy", "Value": f"{metrics['accuracy']:.3f}"},
            {"Metric": "Precision", "Value": f"{metrics['precision']:.3f}"},
            {"Metric": "Recall", "Value": f"{metrics['recall']:.3f}"},
            {"Metric": "Specificity", "Value": f"{metrics['specificity']:.3f}"},
            {"Metric": "F1", "Value": f"{metrics['f1']:.3f}"},
            {"Metric": "Predicted positive", "Value": f"{metrics['predicted_positive_rate']:.1%}"},
        ]
    )


def threshold_analysis_table(y_true, scores, selected_threshold=0.5):
    rows = []
    thresholds = sorted({round(float(threshold), 3) for threshold in np.arange(0.1, 1.0, 0.1)} | {round(float(selected_threshold), 3)})
    threshold_results = [threshold_metrics(y_true, scores, threshold) for threshold in thresholds]
    best_f1 = max((result["f1"] for result in threshold_results), default=0.0)
    for result in threshold_results:
        threshold = result["threshold"]
        is_selected = math.isclose(threshold, selected_threshold, abs_tol=0.0005)
        is_best = math.isclose(result["f1"], best_f1, abs_tol=1e-12)
        rows.append(
            {
                "Threshold": f"{threshold:.3f}",
                "Accuracy": f"{result['accuracy']:.3f}",
                "Precision": f"{result['precision']:.3f}",
                "Recall": f"{result['recall']:.3f}",
                "Specificity": f"{result['specificity']:.3f}",
                "F1": f"{result['f1']:.3f}",
                "Predicted positive": f"{result['predicted_positive_rate']:.1%}",
                "Status": "Selected / Best F1" if is_selected and is_best else "Selected" if is_selected else "Best F1" if is_best else "",
            }
        )
    return pd.DataFrame(rows)


def add_binary_classification_analytics(output, data, target, predictors, model_name, test_size, threshold=0.5, options=None):
    try:
        _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=True, options=options)
        split = split_classification_data(target_values, x_encoded, test_size, options)
        estimator = classification_estimator(model_name, options)
        estimator.fit(split["x_train"], split["y_train"])
        scores = classification_score_values(estimator, split["x_test"])
        y_true = split["y_test"]
        threshold = parse_threshold(threshold)

        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(y_true, scores)
        average_precision = average_precision_score(y_true, scores)
        selected_threshold_metrics = threshold_metrics(y_true, scores, threshold)
        threshold_table = threshold_analysis_table(y_true, scores, threshold)
        selected_labels = np.where(selected_threshold_metrics["predicted"], classes[1], classes[0])
        selected_confusion = confusion_table(split["target_test"].to_numpy(), selected_labels)
        selected_threshold_frame = threshold_metrics_frame(selected_threshold_metrics)

        output["roc_plot"] = line_plot_image(
            fpr,
            tpr,
            "False positive rate",
            "True positive rate",
            "ROC curve",
            annotation=f"AUC = {roc_auc:.3f}",
            baseline={"x": [0, 1], "y": [0, 1]},
        )
        output["pr_plot"] = line_plot_image(
            recall,
            precision,
            "Recall",
            "Precision",
            "Precision-recall curve",
            annotation=f"Avg. precision = {average_precision:.3f}",
        )
        output["confusion_html"] = selected_confusion.to_html(border=0, classes="confusion")
        output["selected_threshold_html"] = display_table(selected_threshold_frame, index=False, border=0, classes="selected-threshold")
        output["threshold_html"] = display_table(threshold_table, index=False, border=0, classes="threshold-analysis")
        output.setdefault("download_data", {})["confusion_matrix"] = selected_confusion.to_csv()
        output.setdefault("download_data", {})["selected_threshold_metrics"] = selected_threshold_frame.to_csv(index=False)
        output.setdefault("download_data", {})["threshold_analysis"] = threshold_table.to_csv(index=False)
        output.setdefault("metrics", []).extend(
            [
                {"label": "Decision threshold", "value": f"{threshold:.3f}"},
                {"label": "Threshold accuracy", "value": f"{selected_threshold_metrics['accuracy']:.3f}"},
                {"label": "Threshold precision", "value": f"{selected_threshold_metrics['precision']:.3f}"},
                {"label": "Threshold recall", "value": f"{selected_threshold_metrics['recall']:.3f}"},
                {"label": "Threshold specificity", "value": f"{selected_threshold_metrics['specificity']:.3f}"},
                {"label": "Threshold F1", "value": f"{selected_threshold_metrics['f1']:.3f}"},
                {"label": "ROC AUC", "value": f"{roc_auc:.3f}"},
                {"label": "Avg. precision", "value": f"{average_precision:.3f}"},
            ]
        )
    except Exception:
        output["roc_plot"] = None
        output["pr_plot"] = None
        output["selected_threshold_html"] = None
        output["threshold_html"] = None
    return output

def scatter_plot_image(x_values, y_values, xlabel, ylabel, title, identity_line=False, zero_line=False):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.scatter(x_values, y_values, color="#176b87", alpha=0.72, edgecolor="#0f4f63", linewidth=0.35)
    if identity_line:
        lower = min(float(np.min(x_values)), float(np.min(y_values)))
        upper = max(float(np.max(x_values)), float(np.max(y_values)))
        ax.plot([lower, upper], [lower, upper], color="#94a3b8", linestyle="--", linewidth=1.4)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
    if zero_line:
        ax.axhline(0, color="#94a3b8", linestyle="--", linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def histogram_plot_image(values, xlabel, title):
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bins = min(30, max(8, int(math.sqrt(len(values)))))
    ax.hist(values, bins=bins, color="#176b87", alpha=0.78, edgecolor="#0f4f63")
    ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1.4)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, axis="y", color="#e2e8f0", linewidth=0.8)
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def figure_to_image(fig):
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def cv_fold_plot_image(fold_frame, metric_column, title):
    if fold_frame is None or fold_frame.empty or metric_column not in fold_frame:
        return None

    x_values = fold_frame["Fold"].astype(int).to_numpy()
    y_values = fold_frame[metric_column].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(x_values, y_values, color="#176b87", linewidth=2.1, marker="o", markersize=6)
    ax.axhline(float(np.mean(y_values)), color="#94a3b8", linestyle="--", linewidth=1.3, label="Mean")
    ax.set_xlabel("Fold")
    ax.set_ylabel(metric_column)
    ax.set_title(title)
    ax.grid(True, color="#e2e8f0", linewidth=0.8)
    ax.legend(loc="best")
    buffer = BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


TREE_SHAP_MODELS = {"random_forest", "extra_trees", "xgboost", "lightgbm", "catboost"}


def top_explainability_features(output, feature_names, limit=2):
    feature_set = set(feature_names)
    for artifact_name, column_name in [("variable_importance", "Predictor"), ("coefficients", "Term")]:
        csv_data = (output.get("download_data") or {}).get(artifact_name)
        if not csv_data:
            continue
        try:
            frame = pd.read_csv(StringIO(csv_data))
        except Exception:
            continue
        if column_name not in frame.columns:
            continue
        candidates = [
            str(value)
            for value in frame[column_name].dropna().tolist()
            if str(value) in feature_set and str(value).lower() not in {"none", "intercept"}
        ]
        if candidates:
            return candidates[:limit]
    return list(feature_names[:limit])


def feature_effect_grid(values):
    clean = pd.Series(values).dropna().astype(float)
    if clean.empty or clean.nunique() < 2:
        return None
    unique_values = np.sort(clean.unique())
    if len(unique_values) <= 10:
        return unique_values
    low, high = np.quantile(clean, [0.05, 0.95])
    if math.isclose(float(low), float(high)):
        low, high = float(clean.min()), float(clean.max())
    if math.isclose(float(low), float(high)):
        return None
    return np.linspace(low, high, 20)


def model_response_values(estimator, x_values, task):
    if task == "classification":
        return np.ravel(classification_score_values(estimator, x_values))
    return np.ravel(estimator.predict(x_values))


def pdp_ice_plot_and_frame(estimator, x_values, features, task, seed):
    features = [feature for feature in features if feature in x_values.columns]
    if not features:
        return None, None

    sample_size = min(25, len(x_values))
    sample = x_values.sample(sample_size, random_state=seed) if len(x_values) > sample_size else x_values.copy()
    rows = []
    fig, axes = plt.subplots(len(features), 1, figsize=(8.5, max(4.2, 3.2 * len(features))), squeeze=False)
    for axis, feature in zip(axes.ravel(), features):
        grid = feature_effect_grid(x_values[feature])
        if grid is None:
            axis.axis("off")
            continue
        ice_values = []
        for _, row in sample.iterrows():
            repeated = pd.DataFrame([row.to_dict()] * len(grid), columns=x_values.columns)
            repeated[feature] = grid
            responses = model_response_values(estimator, repeated, task)
            ice_values.append(responses)
            axis.plot(grid, responses, color="#cbd5e1", linewidth=0.8, alpha=0.55)
        pdp_values = np.mean(np.vstack(ice_values), axis=0)
        axis.plot(grid, pdp_values, color="#176b87", linewidth=2.4)
        axis.set_title(feature)
        axis.set_xlabel(feature)
        axis.set_ylabel("Model score" if task == "classification" else "Prediction")
        axis.grid(True, color="#e2e8f0", linewidth=0.8)
        for grid_value, pdp_value in zip(grid, pdp_values):
            rows.append({"Feature": feature, "Value": grid_value, "Partial dependence": pdp_value})

    if not rows:
        plt.close(fig)
        return None, None
    fig.suptitle("Partial dependence and ICE")
    return figure_to_image(fig), pd.DataFrame(rows)


def shap_values_matrix(estimator, x_values):
    if shap is None:
        return None
    explainer = shap.TreeExplainer(estimator)
    values = explainer.shap_values(x_values)
    if isinstance(values, list):
        return np.asarray(values[-1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, -1]
    return values


def shap_summary_plot_and_frame(estimator, x_values, model_name):
    if shap is None or model_name not in TREE_SHAP_MODELS:
        return None, None
    try:
        sample_size = min(200, len(x_values))
        sample = x_values.sample(sample_size, random_state=0) if len(x_values) > sample_size else x_values.copy()
        values = shap_values_matrix(estimator, sample)
        if values is None:
            return None, None
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(sample.columns):
            return None, None
        summary = pd.DataFrame(
            {
                "Feature": sample.columns,
                "Mean absolute SHAP": np.mean(np.abs(values), axis=0),
            }
        ).sort_values("Mean absolute SHAP", ascending=False)
        summary = summary[summary["Mean absolute SHAP"] > 0].head(15)
        if summary.empty:
            return None, None

        fig, ax = plt.subplots(figsize=(8.5, max(4.2, len(summary) * 0.35 + 1.2)))
        ordered = summary.iloc[::-1]
        ax.barh(ordered["Feature"], ordered["Mean absolute SHAP"], color="#176b87", alpha=0.86)
        ax.set_xlabel("Mean absolute SHAP value")
        ax.set_title("SHAP feature impact")
        ax.grid(True, axis="x", color="#e2e8f0", linewidth=0.8)
        return figure_to_image(fig), summary
    except Exception:
        return None, None


def add_model_explainability(output, task, data, target, predictors, model_name, test_size, options=None):
    try:
        if task == "classification":
            _, target_values, _, x_encoded = prepare_classification_data(
                data,
                target,
                predictors,
                binary_only=(model_name in {"logistic", "elastic_net_logistic"}),
                options=options,
            )
            split = split_classification_data(target_values, x_encoded, test_size, options)
            estimator = classification_estimator(model_name, options)
        else:
            y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
            split = split_regression_data(y, x_encoded, test_size, options)
            estimator = regression_estimator(model_name, options)

        estimator.fit(split["x_train"], split["y_train"])
        features = top_explainability_features(output, x_encoded.columns, limit=2)
        pdp_image, pdp_frame = pdp_ice_plot_and_frame(
            estimator,
            split["x_test"],
            features,
            task,
            preprocessing_seed(options),
        )
        output["pdp_ice_plot"] = pdp_image
        if pdp_frame is not None:
            output.setdefault("download_data", {})["partial_dependence"] = pdp_frame.to_csv(index=False)

        shap_image, shap_frame = shap_summary_plot_and_frame(estimator_leaf(estimator), split["x_test"], model_name)
        output["shap_summary_plot"] = shap_image
        if shap_frame is not None:
            output.setdefault("download_data", {})["shap_summary"] = shap_frame.to_csv(index=False)
    except Exception:
        output["pdp_ice_plot"] = None
        output["shap_summary_plot"] = None
    return output


def cv_summary_frame(metric_scores):
    rows = []
    for metric, scores in metric_scores.items():
        values = np.asarray(scores, dtype=float)
        rows.append(
            {
                "Metric": metric,
                "Mean": f"{np.mean(values):.3f}",
                "SD": f"{np.std(values):.3f}",
                "Min": f"{np.min(values):.3f}",
                "Max": f"{np.max(values):.3f}",
            }
        )
    return pd.DataFrame(rows)


def classification_cv_diagnostics(data, target, predictors, model_name, folds, options=None):
    if folds <= 1:
        return None

    _, target_values, _, x_encoded = prepare_classification_data(
        data,
        target,
        predictors,
        binary_only=(model_name == "logistic"),
        options=options,
    )
    y_codes, _ = encode_target(target_values)
    min_class_count = int(pd.Series(y_codes).value_counts().min())
    actual_folds = min(folds, min_class_count)
    if actual_folds < 2:
        return None

    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=preprocessing_seed(options))
    scores = cross_val_score(classification_estimator(model_name, options), x_encoded, y_codes, cv=cv, scoring="accuracy")
    folds_frame = pd.DataFrame(
        {
            "Fold": np.arange(1, len(scores) + 1),
            "Accuracy": scores,
        }
    )
    summary = cv_summary_frame({"Accuracy": scores})
    return {
        "folds": folds_frame,
        "summary": summary,
        "plot": cv_fold_plot_image(folds_frame, "Accuracy", "Cross-validation accuracy by fold"),
        "stats": {
            "accuracy_sd": float(np.std(scores)),
            "accuracy_min": float(np.min(scores)),
            "accuracy_max": float(np.max(scores)),
        },
    }


def regression_cv_diagnostics(data, target, predictors, model_name, folds, options=None):
    if folds <= 1:
        return None

    y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
    actual_folds = min(folds, len(y))
    if actual_folds < 2:
        return None

    cv = KFold(n_splits=actual_folds, shuffle=True, random_state=preprocessing_seed(options))
    estimator = regression_estimator(model_name, options)
    r2_scores = cross_val_score(estimator, x_encoded, y, cv=cv, scoring="r2")
    rmse_scores = -cross_val_score(estimator, x_encoded, y, cv=cv, scoring="neg_root_mean_squared_error")
    folds_frame = pd.DataFrame(
        {
            "Fold": np.arange(1, len(rmse_scores) + 1),
            "R squared": r2_scores,
            "RMSE": rmse_scores,
        }
    )
    summary = cv_summary_frame({"R squared": r2_scores, "RMSE": rmse_scores})
    return {
        "folds": folds_frame,
        "summary": summary,
        "plot": cv_fold_plot_image(folds_frame, "RMSE", "Cross-validation RMSE by fold"),
        "stats": {
            "r2_sd": float(np.std(r2_scores)),
            "r2_min": float(np.min(r2_scores)),
            "r2_max": float(np.max(r2_scores)),
            "rmse_sd": float(np.std(rmse_scores)),
            "rmse_min": float(np.min(rmse_scores)),
            "rmse_max": float(np.max(rmse_scores)),
        },
    }


def add_cv_diagnostics(output, tab, data, model_name, options=None):
    try:
        if tab["form_name"] == "pro_classification":
            diagnostics = classification_cv_diagnostics(
                data,
                tab["selected_target"],
                tab["selected_predictors"],
                model_name,
                tab["selected_cv_folds"],
                options,
            )
        else:
            diagnostics = regression_cv_diagnostics(
                data,
                tab["selected_target"],
                tab["selected_predictors"],
                model_name,
                tab["selected_cv_folds"],
                options,
            )
    except Exception:
        diagnostics = None

    if not diagnostics:
        output["cv_summary_html"] = None
        output["cv_diagnostics_html"] = None
        output["cv_plot"] = None
        return output

    summary = diagnostics["summary"]
    folds_frame = diagnostics["folds"]
    output["cv_summary_html"] = display_table(summary, index=False, border=0, classes="cv-summary")
    output["cv_diagnostics_html"] = display_table(folds_frame, index=False, border=0, classes="cv-diagnostics")
    output["cv_plot"] = diagnostics["plot"]
    output.setdefault("download_data", {})["cv_summary"] = summary.to_csv(index=False)
    output.setdefault("download_data", {})["cv_diagnostics"] = folds_frame.to_csv(index=False)

    stats = diagnostics["stats"]
    if tab["form_name"] == "pro_classification":
        output.setdefault("metrics", []).extend(
            [
                {"label": "CV accuracy min", "value": f"{stats['accuracy_min']:.3f}"},
                {"label": "CV accuracy max", "value": f"{stats['accuracy_max']:.3f}"},
            ]
        )
    else:
        output.setdefault("metrics", []).extend(
            [
                {"label": "CV R squared SD", "value": f"{stats['r2_sd']:.3f}"},
                {"label": "CV R squared min", "value": f"{stats['r2_min']:.3f}"},
                {"label": "CV R squared max", "value": f"{stats['r2_max']:.3f}"},
                {"label": "CV RMSE SD", "value": f"{stats['rmse_sd']:.3f}"},
                {"label": "CV RMSE min", "value": f"{stats['rmse_min']:.3f}"},
                {"label": "CV RMSE max", "value": f"{stats['rmse_max']:.3f}"},
            ]
        )
    return output


def add_regression_analytics(output, data, target, predictors, model_name, test_size, options=None):
    try:
        y, x_encoded = prepare_regression_data(data, target, predictors, options=options)
        split = split_regression_data(y, x_encoded, test_size, options)
        estimator = regression_estimator(model_name, options)
        estimator.fit(split["x_train"], split["y_train"])
        predictions = np.ravel(estimator.predict(split["x_test"]))
        residuals = split["y_test"] - predictions
        diagnostics = pd.DataFrame(
            {
                "Actual": split["y_test"],
                "Predicted": predictions,
                "Residual": residuals,
            }
        )

        output["predicted_actual_plot"] = scatter_plot_image(
            split["y_test"],
            predictions,
            "Actual",
            "Predicted",
            "Predicted vs actual",
            identity_line=True,
        )
        output["residuals_fitted_plot"] = scatter_plot_image(
            predictions,
            residuals,
            "Fitted value",
            "Residual",
            "Residuals vs fitted",
            zero_line=True,
        )
        output["residual_distribution_plot"] = histogram_plot_image(
            residuals,
            "Residual",
            "Residual distribution",
        )
        output["residual_diagnostics_html"] = display_table(diagnostics.head(25), index=False, border=0, classes="residual-diagnostics")
        output.setdefault("download_data", {})["residual_diagnostics"] = diagnostics.to_csv(index=False)
        output.setdefault("metrics", []).extend(
            [
                {"label": "Residual mean", "value": f"{np.mean(residuals):.3f}"},
                {"label": "Residual SD", "value": f"{np.std(residuals, ddof=1):.3f}" if len(residuals) > 1 else "0.000"},
            ]
        )
    except Exception:
        output["predicted_actual_plot"] = None
        output["residuals_fitted_plot"] = None
        output["residual_distribution_plot"] = None
        output["residual_diagnostics_html"] = None
    return output
def register_comparison_download(tab, comparison):
    current_id = dataset_id()
    if not current_id:
        return None

    DOWNLOADS.setdefault(current_id, {}).setdefault(tab["form_name"], {})["model_comparison"] = comparison.to_csv(index=False)
    return {
        "href": url_for("download_result", result_type=tab["form_name"], artifact="model_comparison"),
        "label": "Download model comparison CSV",
    }


def model_comparison_pdf_download(tab):
    return {
        "href": url_for("download_result", result_type=tab["form_name"], artifact="model_comparison_pdf"),
        "label": "Download model comparison PDF",
    }


def status_badges(row):
    badges = []
    if row.get("_is_best"):
        badges.append('<span class="status-badge best">Best</span>')
    if row.get("_is_detail"):
        badges.append('<span class="status-badge selected">Selected</span>')
    status = row.get("Status", "Fit")
    if status and status not in {"Fit", "Best", "Detailed", "Best / Detailed"}:
        badges.append(f'<span class="status-badge fit">{html.escape(str(status))}</span>')
    if not badges:
        badges.append('<span class="status-badge fit">Fit</span>')
    return f'<span class="status-badges">{"".join(badges)}</span>'


def comparison_html(rows, display_columns):
    comparison = pd.DataFrame([{key: value for key, value in row.items() if not key.startswith("_")} for row in rows])
    comparison = comparison[display_columns]
    header_cells = "".join(f"<th>{html.escape(column)}</th>" for column in display_columns)
    body_rows = []
    for row in rows:
        row_classes = ["best-row" if row.get("_is_best") else "", "selected-row" if row.get("_is_detail") else ""]
        class_attr = " ".join(row_class for row_class in row_classes if row_class)
        class_attr = f' class="{class_attr}"' if class_attr else ""
        model_attr = html.escape(str(row.get("_model_name", "")), quote=True)
        cells = []
        for column in display_columns:
            if column == "Status":
                cells.append(f"<td>{status_badges(row)}</td>")
            else:
                cells.append(f"<td>{html.escape(str(row.get(column, '')))}</td>")
        body_rows.append(f'<tr{class_attr} data-model-name="{model_attr}">{"".join(cells)}</tr>')
    table_html = f'<table class="dataframe model-comparison"><thead><tr>{header_cells}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>'
    return table_html, comparison


def detail_metric_comparison_html(tab, rows, model_name, display_columns):
    if not tuning_enabled(tab):
        return None

    selected_row = next((row for row in rows if row.get("_model_name") == model_name), None)
    if not selected_row:
        return None

    if tab["form_name"] == "pro_classification":
        metric_rows = [
            {"Metric": "Accuracy", "Default": selected_row.get("Default accuracy", "-"), "Tuned": selected_row.get("Tuned accuracy", "-")},
            {"Metric": "CV accuracy", "Default": selected_row.get("Default CV accuracy", "-"), "Tuned": selected_row.get("Tuned CV accuracy", "-")},
            {"Metric": "Precision", "Default": selected_row.get("Precision", "-"), "Tuned": "-"},
            {"Metric": "Recall", "Default": selected_row.get("Recall", "-"), "Tuned": "-"},
            {"Metric": "F1", "Default": selected_row.get("F1", "-"), "Tuned": "-"},
        ]
    else:
        metric_rows = [
            {"Metric": "RMSE", "Default": selected_row.get("Default RMSE", "-"), "Tuned": selected_row.get("Tuned RMSE", "-")},
            {"Metric": "CV RMSE", "Default": selected_row.get("Default CV RMSE", "-"), "Tuned": selected_row.get("Tuned CV RMSE", "-")},
            {"Metric": "R squared", "Default": selected_row.get("Test R squared", "-"), "Tuned": "-"},
            {"Metric": "CV R squared", "Default": selected_row.get("CV R squared", "-"), "Tuned": "-"},
            {"Metric": "MAE", "Default": selected_row.get("Test MAE", "-"), "Tuned": "-"},
        ]

    metric_rows.append({"Metric": "Best params", "Default": "-", "Tuned": selected_row.get("Best params", "-")})
    return display_table(pd.DataFrame(metric_rows)[display_columns], index=False, border=0, classes="detail-metric-comparison")


def parse_display_metric(value):
    try:
        return float(str(value).replace("%", "").strip())
    except (TypeError, ValueError):
        return None


def recommendation_label(tab, model_name):
    return model_label_for_run(tab, model_name) if model_name else "Selected model"


def recommendation_report_frame(recommendation):
    if not recommendation:
        return pd.DataFrame()
    rows = [
        {"Section": "Recommendation", "Item": recommendation.get("title", "")},
        {"Section": "Summary", "Item": recommendation.get("summary", "")},
    ]
    section_items = [
        ("Why this model", recommendation.get("why_best") or recommendation.get("evidence") or []),
        ("Feature interpretation", recommendation.get("feature_interpretation") or []),
        ("Warnings", recommendation.get("warnings") or recommendation.get("concerns") or []),
        ("Next actions", recommendation.get("actions") or []),
    ]
    for section_label, items in section_items:
        for item in items:
            rows.append({"Section": section_label, "Item": item})
    return pd.DataFrame(rows)


def feature_frame_from_output(output, artifact_name):
    csv_data = (output.get("download_data") or {}).get(artifact_name)
    if not csv_data:
        return None
    try:
        return pd.read_csv(StringIO(csv_data))
    except Exception:
        return None


def top_feature_interpretations(output, task_type):
    coefficients = feature_frame_from_output(output, "coefficients")
    if coefficients is not None and "Coefficient" in coefficients.columns:
        term_column = "Term" if "Term" in coefficients.columns else coefficients.columns[0]
        frame = coefficients.copy()
        frame = frame[frame[term_column].astype(str).str.lower() != "intercept"]
        frame["abs_coefficient"] = frame["Coefficient"].abs()
        frame = frame.sort_values("abs_coefficient", ascending=False).head(3)
        items = []
        for _, row in frame.iterrows():
            term = str(row[term_column])
            coefficient = float(row["Coefficient"])
            direction = "positive" if coefficient >= 0 else "negative"
            if task_type == "classification" and "Odds Ratio" in frame.columns:
                odds_ratio = float(row["Odds Ratio"])
                effect = "higher odds of the positive class" if coefficient >= 0 else "lower odds of the positive class"
                items.append(f"{term} has the strongest {direction} coefficient; higher values are associated with {effect} (odds ratio {odds_ratio:.3f}).")
            else:
                effect = "higher predicted values" if coefficient >= 0 else "lower predicted values"
                items.append(f"{term} has the strongest {direction} coefficient ({coefficient:.3f}), so higher values are associated with {effect}.")
        if items:
            return items

    importances = feature_frame_from_output(output, "variable_importance")
    if importances is not None and "Importance" in importances.columns:
        name_column = "Predictor" if "Predictor" in importances.columns else importances.columns[0]
        frame = importances.copy()
        frame = frame[frame["Importance"] > 0].sort_values("Importance", ascending=False).head(3)
        items = []
        for _, row in frame.iterrows():
            predictor = str(row[name_column])
            importance = float(row["Importance"])
            if "Importance SD" in frame.columns:
                items.append(f"{predictor} is one of the strongest drivers by permutation importance ({importance:.3f}, SD {float(row['Importance SD']):.3f}).")
            else:
                items.append(f"{predictor} is one of the strongest drivers by model importance ({importance:.3f}).")
        if items:
            return items

    return ["No coefficient or feature-importance table is available for this selected model."]


def classification_balance_warning(data, target):
    if data is None or target not in data:
        return None
    counts = data[target].dropna().value_counts()
    total = int(counts.sum())
    if total == 0 or counts.empty:
        return None
    minority_share = float(counts.min() / total)
    if minority_share < 0.2:
        return f"The target is imbalanced: the smallest class is {minority_share:.3f} of rows, so accuracy may overstate model quality."
    return None


def build_classification_recommendation(tab, rows, best_model_name, detail_model_name, output, data=None):
    selected_row = next((row for row in rows if row.get("_model_name") == detail_model_name), {})
    best_label = recommendation_label(tab, best_model_name)
    detail_label = recommendation_label(tab, detail_model_name)
    accuracy = parse_display_metric(selected_row.get("Tuned accuracy")) or parse_display_metric(selected_row.get("Default accuracy"))
    cv_accuracy = parse_display_metric(selected_row.get("Tuned CV accuracy")) or parse_display_metric(selected_row.get("Default CV accuracy"))
    cv_sd = parse_display_metric(selected_row.get("CV accuracy SD"))
    precision = metric_float(output, "Threshold precision") or parse_display_metric(selected_row.get("Precision"))
    recall = metric_float(output, "Threshold recall") or parse_display_metric(selected_row.get("Recall"))
    f1 = metric_float(output, "Threshold F1") or parse_display_metric(selected_row.get("F1"))
    specificity = metric_float(output, "Threshold specificity")
    threshold = metric_value(output, "Decision threshold") or f"{tab.get('selected_threshold', 0.5):.3f}"
    tuned_accuracy = parse_display_metric(selected_row.get("Tuned accuracy"))
    default_accuracy = parse_display_metric(selected_row.get("Default accuracy"))
    train_rows = metric_float(output, "Train rows")
    test_rows = metric_float(output, "Test rows")

    evidence = []
    if accuracy is not None:
        evidence.append(f"{detail_label} has test accuracy {accuracy:.3f}.")
    if cv_accuracy is not None:
        evidence.append(f"Cross-validation accuracy is {cv_accuracy:.3f}.")
    if cv_sd is not None:
        evidence.append(f"CV accuracy SD is {cv_sd:.3f}, indicating {'stable' if cv_sd < 0.04 else 'variable'} fold performance.")
    if f1 is not None:
        evidence.append(f"At threshold {threshold}, F1 is {f1:.3f}.")
    if precision is not None and recall is not None:
        evidence.append(f"Threshold tradeoff: precision {precision:.3f}, recall {recall:.3f}.")

    why_best = []
    if detail_model_name == best_model_name:
        if tuned_accuracy is not None:
            why_best.append(f"{detail_label} ranked highest after tuning with test accuracy {tuned_accuracy:.3f}.")
        elif accuracy is not None:
            why_best.append(f"{detail_label} ranked highest on held-out test accuracy ({accuracy:.3f}).")
        if cv_accuracy is not None:
            why_best.append(f"Its cross-validation accuracy is {cv_accuracy:.3f}, giving a broader check beyond the single test split.")
    else:
        why_best.append(f"{best_label} is still the top-ranked model; this panel explains the selected detail model, {detail_label}.")

    concerns = []
    if best_model_name and detail_model_name != best_model_name:
        concerns.append(f"The selected detail model is not the top-ranked model; {best_label} ranked best in the comparison.")
    if accuracy is not None and cv_accuracy is not None and accuracy - cv_accuracy >= 0.08:
        concerns.append("Test accuracy is meaningfully higher than CV accuracy, which can indicate overfitting or a lucky test split.")
    if cv_sd is not None and cv_sd >= 0.06:
        concerns.append("CV accuracy varies noticeably across folds, so the result may be split-sensitive.")
    if tuned_accuracy is not None and default_accuracy is not None and tuned_accuracy <= default_accuracy + 0.001:
        concerns.append("Hyperparameter tuning did not materially improve test accuracy.")
    if precision is not None and precision < 0.65:
        concerns.append("Precision is weak, so many positive predictions may be false positives.")
    if recall is not None and recall < 0.65:
        concerns.append("Recall is weak, so the model may miss many true positives.")
    if f1 is not None and f1 < 0.65:
        concerns.append("F1 is weak, suggesting the precision-recall balance may need work.")
    if precision is not None and recall is not None and abs(precision - recall) >= 0.15:
        higher = "precision" if precision > recall else "recall"
        concerns.append(f"The selected threshold is {higher}-heavy; review whether that matches the decision cost.")
    if specificity is not None and recall is not None and abs(specificity - recall) >= 0.2:
        concerns.append("Recall and specificity are far apart, suggesting asymmetric errors at the selected threshold.")
    balance_warning = classification_balance_warning(data, tab.get("selected_target"))
    if balance_warning:
        concerns.append(balance_warning)
    if test_rows is not None and test_rows < 20:
        concerns.append("The held-out test set is small, so one or two rows can noticeably move the reported metrics.")
    if not concerns:
        concerns.append("No major stability or threshold concerns are visible from the current diagnostics.")

    actions = []
    if best_model_name and detail_model_name != best_model_name:
        actions.append(f"Inspect {best_label} as the detail model before finalizing.")
    if cv_sd is not None and cv_sd >= 0.06:
        actions.append("Try more folds, more data, or simpler models to confirm stability.")
    if precision is not None and recall is not None:
        actions.append("Adjust the decision threshold to balance precision and recall for the use case.")
    if balance_warning:
        actions.append("Consider stratified sampling, class weighting, or resampling before relying on accuracy alone.")
    if train_rows is not None and train_rows < 100:
        actions.append("Collect more labeled rows before treating the comparison as final.")
    if not tuning_enabled(tab):
        actions.append("Enable hyperparameter tuning for the strongest candidate models.")
    if tab.get("selected_scaling") == "off" and detail_model_name in {"logistic", "elastic_net_logistic", "svm", "knn"}:
        actions.append("Try feature scaling for models that depend on coefficient size, margins, or distance.")
    if any("accuracy is meaningfully higher than CV accuracy" in concern for concern in concerns):
        actions.append("Compare a simpler model and check for leakage-prone columns that encode the outcome.")
    actions.append("Review feature importance or coefficients for plausibility before sharing the report.")

    title = f"Recommend {best_label}"
    if detail_model_name != best_model_name:
        summary = f"{best_label} is recommended by the comparison ranking; the panel below currently explains selected detail model {detail_label}."
    else:
        summary = f"{detail_label} performed best among the compared classifiers, with the selection based on test accuracy, CV stability, and threshold diagnostics."

    return {
        "title": title,
        "summary": summary,
        "why_best": why_best or evidence[:2],
        "feature_interpretation": top_feature_interpretations(output, "classification"),
        "warnings": concerns,
        "evidence": evidence,
        "concerns": concerns,
        "actions": actions,
    }


def build_regression_recommendation(tab, rows, best_model_name, detail_model_name, output, data=None):
    selected_row = next((row for row in rows if row.get("_model_name") == detail_model_name), {})
    best_label = recommendation_label(tab, best_model_name)
    detail_label = recommendation_label(tab, detail_model_name)
    rmse = parse_display_metric(selected_row.get("Tuned RMSE")) or parse_display_metric(selected_row.get("Default RMSE"))
    cv_rmse = parse_display_metric(selected_row.get("Tuned CV RMSE")) or parse_display_metric(selected_row.get("Default CV RMSE"))
    cv_rmse_sd = parse_display_metric(selected_row.get("CV RMSE SD"))
    r_squared = parse_display_metric(selected_row.get("Test R squared"))
    cv_r_squared = parse_display_metric(selected_row.get("CV R squared"))
    residual_mean = metric_float(output, "Residual mean")
    residual_sd = metric_float(output, "Residual SD")
    tuned_rmse = parse_display_metric(selected_row.get("Tuned RMSE"))
    default_rmse = parse_display_metric(selected_row.get("Default RMSE"))
    train_rows = metric_float(output, "Train rows")
    test_rows = metric_float(output, "Test rows")

    evidence = []
    if rmse is not None:
        evidence.append(f"{detail_label} has test RMSE {rmse:.3f}.")
    if r_squared is not None:
        evidence.append(f"Test R squared is {r_squared:.3f}.")
    if cv_rmse is not None:
        evidence.append(f"Cross-validation RMSE is {cv_rmse:.3f}.")
    if cv_rmse_sd is not None:
        evidence.append(f"CV RMSE SD is {cv_rmse_sd:.3f}.")
    if residual_mean is not None and residual_sd is not None:
        evidence.append(f"Residual mean is {residual_mean:.3f} with residual SD {residual_sd:.3f}.")

    why_best = []
    if detail_model_name == best_model_name:
        if tuned_rmse is not None:
            why_best.append(f"{detail_label} ranked best after tuning with test RMSE {tuned_rmse:.3f}.")
        elif rmse is not None:
            why_best.append(f"{detail_label} ranked best by the lowest held-out test RMSE ({rmse:.3f}).")
        if cv_rmse is not None:
            why_best.append(f"Its cross-validation RMSE is {cv_rmse:.3f}, giving a stability check across folds.")
    else:
        why_best.append(f"{best_label} is still the top-ranked model by RMSE; this panel explains the selected detail model, {detail_label}.")

    concerns = []
    if best_model_name and detail_model_name != best_model_name:
        concerns.append(f"The selected detail model is not the top-ranked model; {best_label} ranked best by RMSE.")
    if cv_rmse is not None and rmse is not None and cv_rmse > rmse * 1.25:
        concerns.append("CV RMSE is substantially higher than test RMSE, suggesting possible split optimism.")
    if cv_rmse_sd is not None and cv_rmse is not None and cv_rmse_sd > cv_rmse * 0.2:
        concerns.append("CV RMSE varies meaningfully across folds.")
    if r_squared is not None and r_squared < 0.2:
        concerns.append("Test R squared is low, so the model explains only a small share of target variation.")
    if residual_mean is not None and residual_sd is not None and residual_sd > 0 and abs(residual_mean) > residual_sd * 0.1:
        concerns.append("Residual mean is not close to zero relative to residual spread, suggesting possible bias.")
    if residual_sd is not None and rmse is not None and residual_sd > rmse * 1.5:
        concerns.append("Residual spread is high relative to RMSE, suggesting noisy predictions or uneven errors.")
    if tuned_rmse is not None and default_rmse is not None and tuned_rmse >= default_rmse - 0.001:
        concerns.append("Hyperparameter tuning did not materially reduce RMSE.")
    if cv_r_squared is not None and cv_r_squared < 0:
        concerns.append("CV R squared is below zero, so the model may generalize poorly.")
    if test_rows is not None and test_rows < 20:
        concerns.append("The held-out test set is small, so the regression metrics may be sensitive to individual rows.")
    if not concerns:
        concerns.append("No major residual or cross-validation concerns are visible from the current diagnostics.")

    actions = []
    if best_model_name and detail_model_name != best_model_name:
        actions.append(f"Inspect {best_label} as the detail model before finalizing.")
    if cv_rmse_sd is not None and cv_rmse is not None and cv_rmse_sd > cv_rmse * 0.2:
        actions.append("Compare simpler models or add data to reduce fold-to-fold variability.")
    if residual_mean is not None and residual_sd is not None and residual_sd > 0 and abs(residual_mean) > residual_sd * 0.1:
        actions.append("Inspect residual plots for systematic under- or over-prediction.")
    if train_rows is not None and train_rows < 100:
        actions.append("Collect more rows before treating the ranking as final.")
    if r_squared is not None and r_squared < 0.2:
        actions.append("Add more predictive features, transform the target, or segment the problem if the relationship is weak.")
    if any("CV RMSE is substantially higher" in concern for concern in concerns):
        actions.append("Check for leakage-prone columns and compare simpler regularized models.")
    if not tuning_enabled(tab):
        actions.append("Enable hyperparameter tuning for regularized and tree-based candidates.")
    if tab.get("selected_scaling") == "off" and detail_model_name in {"ridge", "lasso", "svr", "knn"}:
        actions.append("Try feature scaling for regularized, margin-based, or distance-based models.")
    actions.append("Review feature importance or coefficients for domain plausibility.")

    title = f"Recommend {best_label}"
    if detail_model_name != best_model_name:
        summary = f"{best_label} is recommended by the comparison ranking; the panel below currently explains selected detail model {detail_label}."
    else:
        summary = f"{detail_label} performed best among the compared regressors, with the selection based on RMSE ranking, CV stability, and residual diagnostics."

    return {
        "title": title,
        "summary": summary,
        "why_best": why_best or evidence[:2],
        "feature_interpretation": top_feature_interpretations(output, "regression"),
        "warnings": concerns,
        "evidence": evidence,
        "concerns": concerns,
        "actions": actions,
    }



def choose_detail_model(tab, successful_outputs, best_model_name):
    output_by_model = {model_name: output for model_name, output, _ in successful_outputs}
    selected_detail_model = tab.get("selected_detail_model", "best")
    if selected_detail_model != "best" and selected_detail_model in output_by_model:
        return selected_detail_model, output_by_model[selected_detail_model]
    tab["selected_detail_model"] = "best"
    return best_model_name, output_by_model[best_model_name]


def detail_options_with_tuning(options, model_name, tuned_results):
    tuned = tuned_results.get(model_name)
    if not tuned or tuned.get("error") or not tuned.get("raw_params"):
        return options

    tuned_options = deepcopy(options)
    tuned_options.setdefault("tuned_params", {})[model_name] = tuned["raw_params"]
    tuned_options["detail_tuned_params_label"] = tuned.get("params", "-")
    return tuned_options


def refit_tuned_detail_output(tab, dataset, model_name, options, fitter_map, default_fitter):
    fit_model = fitter_map.get(model_name, default_fitter)
    return fit_model(
        dataset["data"],
        tab["selected_target"],
        tab["selected_predictors"],
        tab["selected_test_size"],
        tab["selected_cv_folds"],
        options,
    )


def comparison_row_status(model_name, best_model_name, detail_model_name):
    labels = []
    if model_name == best_model_name:
        labels.append("Best")
    if model_name == detail_model_name:
        labels.append("Detailed")
    return " / ".join(labels) if labels else "Fit"


def fit_failure_summary(rows):
    failures = [
        f"{row.get('Model', 'Model')}: {row.get('Status')}"
        for row in rows
        if row.get("Status") and row.get("Status") != "Fit"
    ]
    if not failures:
        return "No selected model could be fit."
    return "No selected model could be fit. " + " | ".join(failures[:3])


def handle_classification_comparison_submission(tab, dataset):
    options = preprocessing_options(tab)
    progress_job_id = active_run_progress_id()
    initialize_run_progress(progress_job_id, len(tab["selected_models"]))
    successful_outputs = []
    tuned_results = {}
    rows = []

    for model_name in tab["selected_models"]:
        model_label = CLASSIFICATION_MODEL_LABELS.get(model_name, model_name)
        try:
            fit_model = CLASSIFICATION_MODEL_FITTERS.get(model_name, fit_logistic_regression)
            output = fit_model(
                dataset["data"],
                tab["selected_target"],
                tab["selected_predictors"],
                tab["selected_test_size"],
                tab["selected_cv_folds"],
                options,
            )
            accuracy = metric_float(output, "Test accuracy")
            cv_accuracy = metric_float(output, "CV accuracy mean")
            cv_diagnostics = classification_cv_diagnostics(
                dataset["data"],
                tab["selected_target"],
                tab["selected_predictors"],
                model_name,
                tab["selected_cv_folds"],
                options,
            )
            cv_stats = cv_diagnostics["stats"] if cv_diagnostics else {}
            precision, recall, f1 = weighted_precision_recall_f1(output)
            tuned = tune_classification_model(
                model_name,
                dataset["data"],
                tab["selected_target"],
                tab["selected_predictors"],
                tab["selected_test_size"],
                tab["selected_cv_folds"],
                options,
                tuning_mode(tab),
                tuning_iterations(tab),
            ) if tuning_enabled(tab) else None
            if tuned:
                tuned_results[model_name] = tuned
            tuned_accuracy = tuned.get("test_accuracy") if tuned and not tuned.get("error") else None
            tuned_cv_accuracy = tuned.get("cv_accuracy") if tuned and not tuned.get("error") else None
            ranking_accuracy = tuned_accuracy if tuned_accuracy is not None else accuracy
            rows.append(
                {
                    "Model": model_label,
                    "Default accuracy": format_optional_metric(accuracy),
                    "Default CV accuracy": format_optional_metric(cv_accuracy),
                    "CV accuracy SD": format_optional_metric(cv_stats.get("accuracy_sd")),
                    "CV accuracy min": format_optional_metric(cv_stats.get("accuracy_min")),
                    "CV accuracy max": format_optional_metric(cv_stats.get("accuracy_max")),
                    "Tuned accuracy": format_optional_metric(tuned_accuracy),
                    "Tuned CV accuracy": format_optional_metric(tuned_cv_accuracy),
                    "Precision": format_optional_metric(precision),
                    "Recall": format_optional_metric(recall),
                    "F1": format_optional_metric(f1),
                    "Best params": tuned.get("params", "-") if tuned else "-",
                    "Status": tuned.get("error", "Fit") if tuned and tuned.get("error") else "Fit",
                    "_accuracy": ranking_accuracy if ranking_accuracy is not None else -1.0,
                    "_model_name": model_name,
                }
            )
            successful_outputs.append((model_name, output, ranking_accuracy if ranking_accuracy is not None else -1.0))
        except Exception as exc:
            rows.append(
                {
                    "Model": model_label,
                    "Default accuracy": "-",
                    "Default CV accuracy": "-",
                    "CV accuracy SD": "-",
                    "CV accuracy min": "-",
                    "CV accuracy max": "-",
                    "Tuned accuracy": "-",
                    "Tuned CV accuracy": "-",
                    "Precision": "-",
                    "Recall": "-",
                    "F1": "-",
                    "Best params": "-",
                    "Status": str(exc),
                    "_accuracy": -1.0,
                    "_model_name": model_name,
                }
            )
        finally:
            advance_run_progress(progress_job_id, model_label)

    if not rows:
        tab["error"] = "Select at least one model."
        complete_run_progress(progress_job_id)
        return

    best_model_name = None
    if successful_outputs:
        best_model_name, _, _ = max(successful_outputs, key=lambda item: item[2])
        detail_model_name, detail_output = choose_detail_model(tab, successful_outputs, best_model_name)
        detail_options = detail_options_with_tuning(options, detail_model_name, tuned_results)
        if detail_options is not options:
            detail_output = refit_tuned_detail_output(
                tab,
                dataset,
                detail_model_name,
                detail_options,
                CLASSIFICATION_MODEL_FITTERS,
                fit_logistic_regression,
            )
        tab["selected_model"] = detail_model_name
        tab["selected_models"] = [row["_model_name"] for row in rows]
        detail_output = add_binary_classification_analytics(
            detail_output,
            dataset["data"],
            tab["selected_target"],
            tab["selected_predictors"],
            detail_model_name,
            tab["selected_test_size"],
            tab.get("selected_threshold", 0.5),
            detail_options,
        )
        detail_output = add_cv_diagnostics(detail_output, tab, dataset["data"], detail_model_name, detail_options)
        detail_output = add_model_explainability(
            detail_output,
            "classification",
            dataset["data"],
            tab["selected_target"],
            tab["selected_predictors"],
            detail_model_name,
            tab["selected_test_size"],
            detail_options,
        )
        for row in rows:
            row["_is_best"] = row["_model_name"] == best_model_name
            row["_is_detail"] = row["_model_name"] == detail_model_name
            row["Status"] = comparison_row_status(row["_model_name"], best_model_name, detail_model_name)
        tab["detail_metric_comparison_html"] = detail_metric_comparison_html(
            tab,
            rows,
            detail_model_name,
            ["Metric", "Default", "Tuned"],
        )
        tab["recommendation"] = build_classification_recommendation(tab, rows, best_model_name, detail_model_name, detail_output, dataset["data"])
        tab["output"] = register_downloads(tab["form_name"], detail_output)
    else:
        tab["error"] = fit_failure_summary(rows)
        tab["detail_metric_comparison_html"] = None
        tab["recommendation"] = None

    tab["comparison_html"], comparison = comparison_html(rows, ["Model", "Default accuracy", "Default CV accuracy", "CV accuracy SD", "CV accuracy min", "CV accuracy max", "Tuned accuracy", "Tuned CV accuracy", "Precision", "Recall", "F1", "Best params", "Status"])
    tab["comparison_download"] = register_comparison_download(tab, comparison)
    tab["comparison_pdf_download"] = model_comparison_pdf_download(tab)

PRO_TAB_NAMES = {"pro_classification", "pro_regression"}


def model_label_for_run(tab, model_name):
    if tab["form_name"] in REGRESSION_TAB_CONFIGS:
        return REGRESSION_MODEL_LABELS.get(model_name, model_name)
    return CLASSIFICATION_MODEL_LABELS.get(model_name, model_name)


report_renderer = ReportRenderer(display_table, display_frame, model_label_for_run, preprocessing_details, recommendation_report_frame)


def output_metric_summary(tab):
    output = tab.get("output") or {}
    if tab["form_name"] == "pro_classification":
        accuracy = metric_value(output, "Test accuracy")
        cv_accuracy = metric_value(output, "CV accuracy mean")
        parts = []
        if accuracy is not None:
            parts.append(f"accuracy {accuracy}")
        if cv_accuracy is not None:
            parts.append(f"CV {cv_accuracy}")
        return ", ".join(parts) if parts else "Classification run"

    r_squared = metric_value(output, "Test R squared")
    rmse = metric_value(output, "Test RMSE")
    parts = []
    if r_squared is not None:
        parts.append(f"R2 {r_squared}")
    if rmse is not None:
        parts.append(f"RMSE {rmse}")
    return ", ".join(parts) if parts else "Regression run"


def clean_run_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() == "none" else text


def default_run_name(timestamp):
    timestamp_text = clean_run_text(timestamp)
    return f"Run {timestamp_text}".strip() if timestamp_text else "Run"


def run_history_entry(tab):
    detail_model = tab.get("selected_model", tab["default_model"])
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name = clean_run_text(tab.get("run_name")) or default_run_name(timestamp)
    run_notes = clean_run_text(tab.get("run_notes"))
    return {
        "timestamp": timestamp,
        "run_name": run_name,
        "run_notes": run_notes,
        "target": tab.get("selected_target") or "-",
        "models": ", ".join(model_label_for_run(tab, model) for model in tab.get("selected_models", [])),
        "detail_model": model_label_for_run(tab, detail_model),
        "summary": output_metric_summary(tab),
    }


def pro_run_json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

def current_pro_run_scope():
    user_id = current_user_id()
    current_id = dataset_id()
    if not user_id or not current_id:
        return None, None
    return user_id, current_id


def pro_report_download(tab_name):
    if tab_name not in PRO_TAB_NAMES:
        return None
    return {
        "href": url_for("download_result", result_type=tab_name, artifact="pro_report"),
        "label": "Download Pro report",
    }


def pro_report_pdf_download(tab_name):
    if tab_name not in PRO_TAB_NAMES:
        return None
    return {
        "href": url_for("download_result", result_type=tab_name, artifact="pro_report_pdf"),
        "label": "Download Pro report PDF",
    }


def tab_download_artifacts(tab):
    current_id = dataset_id()
    if not current_id:
        return {}
    return deepcopy(DOWNLOADS.get(current_id, {}).get(tab["form_name"], {}))


def load_pro_runs_from_db(tab_name):
    user_id, current_id = current_pro_run_scope()
    if not user_id or not current_id:
        return []

    runs = []
    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, snapshot_json
            FROM pro_runs
            WHERE user_id = ? AND dataset_id = ? AND tab_name = ?
            ORDER BY id DESC
            LIMIT 5
            """,
            (user_id, current_id, tab_name),
        ).fetchall()
    for row in rows:
        try:
            snapshot = json.loads(row["snapshot_json"])
        except json.JSONDecodeError:
            continue
        snapshot["run_id"] = row["id"]
        runs.append(snapshot)
    return runs


def load_pro_run_from_db(tab_name, run_id):
    user_id, current_id = current_pro_run_scope()
    if not user_id or not current_id or tab_name not in PRO_TAB_NAMES:
        return None

    with get_db_connection() as connection:
        row = connection.execute(
            """
            SELECT id, snapshot_json
            FROM pro_runs
            WHERE id = ? AND user_id = ? AND dataset_id = ? AND tab_name = ?
            """,
            (run_id, user_id, current_id, tab_name),
        ).fetchone()
    if row is None:
        return None
    try:
        snapshot = json.loads(row["snapshot_json"])
    except json.JSONDecodeError:
        return None
    snapshot["run_id"] = row["id"]
    return snapshot


def update_pro_run_snapshot(tab_name, run_id, updater):
    user_id, current_id = current_pro_run_scope()
    if not user_id or not current_id or tab_name not in PRO_TAB_NAMES:
        return False

    with get_db_connection() as connection:
        row = connection.execute(
            """
            SELECT snapshot_json
            FROM pro_runs
            WHERE id = ? AND user_id = ? AND dataset_id = ? AND tab_name = ?
            """,
            (run_id, user_id, current_id, tab_name),
        ).fetchone()
        if row is None:
            return False
        try:
            snapshot = json.loads(row["snapshot_json"])
        except json.JSONDecodeError:
            return False
        updater(snapshot)
        connection.execute(
            "UPDATE pro_runs SET snapshot_json = ? WHERE id = ?",
            (json.dumps(snapshot, default=pro_run_json_default), run_id),
        )
    return True


def update_run_history(tab, runs, active_run_id=None):
    history = []
    compare_ids = set(selected_compare_run_ids(tab["form_name"]))
    for run in runs:
        entry = run.get("history_entry")
        if not entry:
            continue
        entry = deepcopy(entry)
        entry["run_name"] = clean_run_text(entry.get("run_name")) or default_run_name(entry.get("timestamp", ""))
        entry["run_notes"] = clean_run_text(entry.get("run_notes"))
        entry["run_id"] = run.get("run_id")
        entry["is_selected"] = active_run_id is not None and entry["run_id"] == active_run_id
        entry["is_compare_selected"] = entry["run_id"] in compare_ids
        history.append(entry)
    tab["run_history"] = history


def snapshot_metric_map(snapshot):
    output = snapshot.get("output") or {}
    return {metric.get("label"): metric.get("value") for metric in output.get("metrics", []) if metric.get("label")}


def compact_snapshot_settings(snapshot, tab_name):
    return {
        "Target": snapshot.get("selected_target") or "-",
        "Predictors": ", ".join(snapshot.get("selected_predictors") or []) or "-",
        "Models": ", ".join(model_label_for_run({"form_name": tab_name}, model) for model in snapshot.get("selected_models") or []) or "-",
        "Detail model": model_label_for_run({"form_name": tab_name}, snapshot.get("selected_model") or snapshot.get("selected_detail_model") or "-"),
        "Test split": f"{float(snapshot.get('selected_test_size') or 0):.0%}",
        "CV": f"{snapshot.get('selected_cv_folds')} folds" if snapshot.get("selected_cv_folds") else "Off",
        "Tuning": {"off": "Off", "grid": "Grid search", "random": "Random search"}.get(snapshot.get("selected_tuning_mode", "off"), "Off"),
        "Scaling": "On" if snapshot.get("selected_scaling") == "on" else "Off",
        "Split seed": snapshot.get("selected_split_seed", "-"),
    }


def recommendation_compare_summary(snapshot):
    recommendation = snapshot.get("recommendation") or {}
    return recommendation.get("summary") or "-"


def build_run_comparison(tab_name, run_ids):
    snapshots = [load_pro_run_from_db(tab_name, run_id) for run_id in run_ids[:2]]
    snapshots = [snapshot for snapshot in snapshots if snapshot]
    if len(snapshots) != 2:
        return None

    metric_names = []
    for snapshot in snapshots:
        for name in snapshot_metric_map(snapshot):
            if name not in metric_names:
                metric_names.append(name)
    setting_names = []
    settings = [compact_snapshot_settings(snapshot, tab_name) for snapshot in snapshots]
    for setting in settings:
        for name in setting:
            if name not in setting_names:
                setting_names.append(name)

    def run_label(snapshot):
        entry = snapshot.get("history_entry") or {}
        return clean_run_text(entry.get("run_name")) or f"Run {snapshot.get('run_id')}"

    metric_maps = [snapshot_metric_map(snapshot) for snapshot in snapshots]
    return {
        "runs": [
            {
                "id": snapshot.get("run_id"),
                "label": run_label(snapshot),
                "timestamp": (snapshot.get("history_entry") or {}).get("timestamp", "-"),
                "notes": clean_run_text((snapshot.get("history_entry") or {}).get("run_notes")),
            }
            for snapshot in snapshots
        ],
        "settings": [
            {"name": name, "left": settings[0].get(name, "-"), "right": settings[1].get(name, "-")}
            for name in setting_names
        ],
        "metrics": [
            {"name": name, "left": metric_maps[0].get(name, "-"), "right": metric_maps[1].get(name, "-")}
            for name in metric_names
        ],
        "recommendations": [
            {"label": run_label(snapshot), "summary": recommendation_compare_summary(snapshot)}
            for snapshot in snapshots
        ],
    }


def prune_old_pro_runs(connection, user_id, current_id, tab_name):
    old_rows = connection.execute(
        """
        SELECT id
        FROM pro_runs
        WHERE user_id = ? AND dataset_id = ? AND tab_name = ?
        ORDER BY id DESC
        LIMIT -1 OFFSET 5
        """,
        (user_id, current_id, tab_name),
    ).fetchall()
    if old_rows:
        connection.executemany("DELETE FROM pro_runs WHERE id = ?", [(row["id"],) for row in old_rows])


def save_pro_run(tab):
    if tab["form_name"] not in PRO_TAB_NAMES or tab.get("output") is None:
        return

    user_id, current_id = current_pro_run_scope()
    if not user_id or not current_id:
        return

    snapshot = {
        "selected_model": tab.get("selected_model"),
        "selected_models": list(tab.get("selected_models", [])),
        "selected_detail_model": tab.get("selected_detail_model", "best"),
        "selected_test_size": tab.get("selected_test_size"),
        "selected_cv_folds": tab.get("selected_cv_folds"),
        "selected_missing_values": tab.get("selected_missing_values"),
        "selected_categorical_encoding": tab.get("selected_categorical_encoding"),
        "selected_scaling": tab.get("selected_scaling"),
        "selected_split_seed": tab.get("selected_split_seed"),
        "selected_outlier_handling": tab.get("selected_outlier_handling"),
        "selected_calibration": tab.get("selected_calibration", "off"),
        "selected_tuning_mode": tab.get("selected_tuning_mode"),
        "selected_tuning_iterations": tab.get("selected_tuning_iterations"),
        "selected_threshold": tab.get("selected_threshold", 0.5),
        "run_name": tab.get("run_name", ""),
        "run_notes": tab.get("run_notes", ""),
        "selected_target": tab.get("selected_target"),
        "selected_predictors": list(tab.get("selected_predictors", [])),
        "comparison_html": tab.get("comparison_html"),
        "detail_metric_comparison_html": tab.get("detail_metric_comparison_html"),
        "recommendation": deepcopy(tab.get("recommendation")),
        "comparison_download": deepcopy(tab.get("comparison_download")),
        "comparison_pdf_download": deepcopy(tab.get("comparison_pdf_download")),
        "output": deepcopy(tab.get("output")),
        "download_artifacts": tab_download_artifacts(tab),
        "history_entry": run_history_entry(tab),
    }
    with get_db_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO pro_runs (user_id, dataset_id, tab_name, snapshot_json)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, current_id, tab["form_name"], json.dumps(snapshot, default=pro_run_json_default)),
        )
        current_run_id = cursor.lastrowid
        prune_old_pro_runs(connection, user_id, current_id, tab["form_name"])
    set_selected_pro_run_id(tab["form_name"], current_run_id)
    update_run_history(tab, load_pro_runs_from_db(tab["form_name"]), current_run_id)


def restore_download_artifacts(tab, snapshot):
    current_id = dataset_id()
    downloads = snapshot.get("download_artifacts") or {}
    if current_id and downloads:
        DOWNLOADS.setdefault(current_id, {})[tab["form_name"]] = deepcopy(downloads)


def restore_pro_run(tab, snapshot):
    for key in [
        "selected_model",
        "selected_models",
        "selected_detail_model",
        "selected_test_size",
        "selected_cv_folds",
        "selected_missing_values",
        "selected_categorical_encoding",
        "selected_scaling",
        "selected_split_seed",
        "selected_outlier_handling",
        "selected_calibration",
        "selected_tuning_mode",
        "selected_tuning_iterations",
        "selected_threshold",
        "run_name",
        "run_notes",
        "selected_target",
        "selected_predictors",
        "comparison_html",
        "detail_metric_comparison_html",
        "recommendation",
        "comparison_download",
        "comparison_pdf_download",
        "output",
    ]:
        tab[key] = deepcopy(snapshot.get(key))
    tab["run_name"] = clean_run_text(tab.get("run_name"))
    tab["run_notes"] = clean_run_text(tab.get("run_notes"))
    if tab.get("selected_threshold") is None:
        tab["selected_threshold"] = 0.5
    if tab.get("selected_calibration") is None:
        tab["selected_calibration"] = "off"
    restore_download_artifacts(tab, snapshot)
    if (snapshot.get("download_artifacts") or {}).get("model_comparison"):
        tab["comparison_pdf_download"] = model_comparison_pdf_download(tab)
    tab["report_download"] = pro_report_download(tab["form_name"])
    tab["report_pdf_download"] = pro_report_pdf_download(tab["form_name"])
    tab["selected_run_id"] = snapshot.get("run_id")


def restore_pro_runs(model_tabs, exclude=None):
    exclude = exclude or set()
    for tab_name in PRO_TAB_NAMES:
        runs = load_pro_runs_from_db(tab_name)
        tab = model_tabs[tab_name]
        active_run = None
        selected_id = selected_pro_run_id(tab_name)
        if runs:
            active_run = next((run for run in runs if run.get("run_id") == selected_id), None) or runs[0]
        if tab_name not in exclude and active_run:
            restore_pro_run(tab, active_run)
        update_run_history(tab, runs, active_run.get("run_id") if active_run else None)
        tab["run_comparison"] = build_run_comparison(tab_name, selected_compare_run_ids(tab_name))

def handle_classification_submission(tab, dataset):
    populate_tab_from_request(tab)
    progress_job_id = active_run_progress_id()

    if dataset is None:
        tab["error"] = "Upload a dataset on the Data tab before running classification."
    elif not tab["selected_target"] or not tab["selected_predictors"]:
        tab["error"] = "Select a target column and at least one predictor."
    elif tab.get("allow_model_comparison"):
        handle_classification_comparison_submission(tab, dataset)
    else:
        initialize_run_progress(progress_job_id, 1)
        try:
            fit_model = CLASSIFICATION_MODEL_FITTERS.get(tab["selected_model"], fit_logistic_regression)
            output = fit_model(
                dataset["data"],
                tab["selected_target"],
                tab["selected_predictors"],
                tab["selected_test_size"],
                tab["selected_cv_folds"],
            )
            tab["output"] = register_downloads(tab["form_name"], output)
        except Exception as exc:
            tab["error"] = str(exc)
        finally:
            advance_run_progress(progress_job_id, tab.get("selected_model", tab["default_model"]))


REGRESSION_MODEL_LABELS = {
    "bagging": "Bagging Regressor",
    "bayesian_ridge": "Bayesian Ridge Regression",
    "catboost": "CatBoost Regression",
    "elastic_net": "Elastic Net Regression",
    "extra_trees": "Extra Trees Regression",
    "gradient_boosting": "Gradient Boosting Regression",
    "huber": "Huber Regression",
    "kernel_ridge": "Kernel Ridge Regression",
    "knn": "kNN Regression",
    "linear": "Linear Regression",
    "lasso": "Lasso Regression",
    "lightgbm": "LightGBM Regression",
    "passive_aggressive": "Passive Aggressive Regression",
    "pls": "PLS Regression",
    "poisson": "Poisson Regression",
    "quantile": "Quantile Regression",
    "random_forest": "Random Forest Regression",
    "ridge": "Ridge Regression",
    "stacking": "Stacking Regressor",
    "svr": "Support Vector Regression",
    "tweedie": "Tweedie Regression",
    "voting": "Voting Regressor",
    "xgboost": "XGBoost Regression",
}


def handle_regression_comparison_submission(tab, dataset):
    options = preprocessing_options(tab)
    progress_job_id = active_run_progress_id()
    initialize_run_progress(progress_job_id, len(tab["selected_models"]))
    successful_outputs = []
    tuned_results = {}
    rows = []

    for model_name in tab["selected_models"]:
        model_label = REGRESSION_MODEL_LABELS.get(model_name, model_name)
        try:
            fit_model = REGRESSION_MODEL_FITTERS.get(model_name, fit_linear_regression)
            output = fit_model(
                dataset["data"],
                tab["selected_target"],
                tab["selected_predictors"],
                tab["selected_test_size"],
                tab["selected_cv_folds"],
                options,
            )
            r_squared = metric_float(output, "Test R squared")
            rmse = metric_float(output, "Test RMSE")
            mae = metric_float(output, "Test MAE")
            cv_r_squared = metric_float(output, "CV R squared mean")
            cv_rmse = metric_float(output, "CV RMSE mean")
            cv_diagnostics = regression_cv_diagnostics(
                dataset["data"],
                tab["selected_target"],
                tab["selected_predictors"],
                model_name,
                tab["selected_cv_folds"],
                options,
            )
            cv_stats = cv_diagnostics["stats"] if cv_diagnostics else {}
            tuned = tune_regression_model(
                model_name,
                dataset["data"],
                tab["selected_target"],
                tab["selected_predictors"],
                tab["selected_test_size"],
                tab["selected_cv_folds"],
                options,
                tuning_mode(tab),
                tuning_iterations(tab),
            ) if tuning_enabled(tab) else None
            if tuned:
                tuned_results[model_name] = tuned
            tuned_rmse = tuned.get("test_rmse") if tuned and not tuned.get("error") else None
            tuned_cv_rmse = tuned.get("cv_rmse") if tuned and not tuned.get("error") else None
            ranking_rmse = tuned_rmse if tuned_rmse is not None else rmse
            rows.append(
                {
                    "Model": model_label,
                    "Test R squared": format_optional_metric(r_squared),
                    "Default RMSE": format_optional_metric(rmse),
                    "Default CV RMSE": format_optional_metric(cv_rmse),
                    "CV RMSE SD": format_optional_metric(cv_stats.get("rmse_sd")),
                    "CV RMSE min": format_optional_metric(cv_stats.get("rmse_min")),
                    "CV RMSE max": format_optional_metric(cv_stats.get("rmse_max")),
                    "Tuned RMSE": format_optional_metric(tuned_rmse),
                    "Tuned CV RMSE": format_optional_metric(tuned_cv_rmse),
                    "Test MAE": format_optional_metric(mae),
                    "CV R squared": format_optional_metric(cv_r_squared),
                    "Best params": tuned.get("params", "-") if tuned else "-",
                    "Status": tuned.get("error", "Fit") if tuned and tuned.get("error") else "Fit",
                    "_rmse": ranking_rmse if ranking_rmse is not None else float("inf"),
                    "_model_name": model_name,
                }
            )
            successful_outputs.append((model_name, output, ranking_rmse if ranking_rmse is not None else float("inf")))
        except Exception as exc:
            rows.append(
                {
                    "Model": model_label,
                    "Test R squared": "-",
                    "Default RMSE": "-",
                    "Default CV RMSE": "-",
                    "CV RMSE SD": "-",
                    "CV RMSE min": "-",
                    "CV RMSE max": "-",
                    "Tuned RMSE": "-",
                    "Tuned CV RMSE": "-",
                    "Test MAE": "-",
                    "CV R squared": "-",
                    "Best params": "-",
                    "Status": str(exc),
                    "_rmse": float("inf"),
                    "_model_name": model_name,
                }
            )
        finally:
            advance_run_progress(progress_job_id, model_label)

    if not rows:
        tab["error"] = "Select at least one model."
        complete_run_progress(progress_job_id)
        return

    if successful_outputs:
        best_model_name, _, _ = min(successful_outputs, key=lambda item: item[2])
        detail_model_name, detail_output = choose_detail_model(tab, successful_outputs, best_model_name)
        detail_options = detail_options_with_tuning(options, detail_model_name, tuned_results)
        if detail_options is not options:
            detail_output = refit_tuned_detail_output(
                tab,
                dataset,
                detail_model_name,
                detail_options,
                REGRESSION_MODEL_FITTERS,
                fit_linear_regression,
            )
        tab["selected_model"] = detail_model_name
        tab["selected_models"] = [row["_model_name"] for row in rows]
        detail_output = add_regression_analytics(
            detail_output,
            dataset["data"],
            tab["selected_target"],
            tab["selected_predictors"],
            detail_model_name,
            tab["selected_test_size"],
            detail_options,
        )
        detail_output = add_cv_diagnostics(detail_output, tab, dataset["data"], detail_model_name, detail_options)
        detail_output = add_model_explainability(
            detail_output,
            "regression",
            dataset["data"],
            tab["selected_target"],
            tab["selected_predictors"],
            detail_model_name,
            tab["selected_test_size"],
            detail_options,
        )
        for row in rows:
            row["_is_best"] = row["_model_name"] == best_model_name
            row["_is_detail"] = row["_model_name"] == detail_model_name
            row["Status"] = comparison_row_status(row["_model_name"], best_model_name, detail_model_name)
        tab["detail_metric_comparison_html"] = detail_metric_comparison_html(
            tab,
            rows,
            detail_model_name,
            ["Metric", "Default", "Tuned"],
        )
        tab["recommendation"] = build_regression_recommendation(tab, rows, best_model_name, detail_model_name, detail_output, dataset["data"])
        tab["output"] = register_downloads(tab["form_name"], detail_output)
    else:
        tab["error"] = fit_failure_summary(rows)
        tab["detail_metric_comparison_html"] = None
        tab["recommendation"] = None

    tab["comparison_html"], comparison = comparison_html(
        rows,
        ["Model", "Test R squared", "Default RMSE", "Default CV RMSE", "CV RMSE SD", "CV RMSE min", "CV RMSE max", "Tuned RMSE", "Tuned CV RMSE", "Test MAE", "CV R squared", "Best params", "Status"],
    )
    tab["comparison_download"] = register_comparison_download(tab, comparison)
    tab["comparison_pdf_download"] = model_comparison_pdf_download(tab)

def handle_regression_submission(tab, dataset):
    populate_tab_from_request(tab)
    progress_job_id = active_run_progress_id()

    if dataset is None:
        tab["error"] = "Upload a dataset on the Data tab before running regression."
    elif not tab["selected_target"] or not tab["selected_predictors"]:
        tab["error"] = "Select a target column and at least one predictor."
    elif tab.get("allow_model_comparison"):
        handle_regression_comparison_submission(tab, dataset)
    else:
        initialize_run_progress(progress_job_id, 1)
        try:
            fit_model = REGRESSION_MODEL_FITTERS.get(tab["selected_model"], fit_linear_regression)
            output = fit_model(
                dataset["data"],
                tab["selected_target"],
                tab["selected_predictors"],
                tab["selected_test_size"],
                tab["selected_cv_folds"],
            )
            tab["output"] = register_downloads(tab["form_name"], output)
        except Exception as exc:
            tab["error"] = str(exc)
        finally:
            advance_run_progress(progress_job_id, tab.get("selected_model", tab["default_model"]))

@app.route("/", methods=["GET", "POST"])
def index():
    active_tab = request.form.get("active_tab", "data")
    auth_error = None
    auth_mode = None
    data_error = None
    form_name = request.form.get("form_name")
    model_tabs = make_model_tabs()

    if request.method == "POST" and form_name == "signup":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        auth_mode = "signup"
        try:
            if password != confirm_password:
                raise ValueError("Passwords do not match.")
            user_id = create_user(username, password)
            log_user_in(user_id, username.strip())
            return redirect(url_for("index"))
        except ValueError as exc:
            auth_error = str(exc)

    if request.method == "POST" and form_name == "login":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        user = authenticate_user(username, password)
        if user is None:
            auth_mode = "login"
            auth_error = "Username or password is incorrect."
        else:
            log_user_in(user["id"], user["username"])
            return redirect(url_for("index"))

    authenticated = is_user_authenticated()
    user = current_user() if authenticated else None
    has_subscription = user_has_subscription(user)

    if authenticated and request.method == "POST" and form_name == "load_pro_run":
        tab_name = request.form.get("tab_name", "")
        try:
            run_id = int(request.form.get("run_id", ""))
        except (TypeError, ValueError):
            return Response("Invalid Pro run.", status=400, mimetype="text/plain")
        if tab_name not in PRO_TAB_NAMES:
            return Response("Unknown Pro tab.", status=404, mimetype="text/plain")
        if load_pro_run_from_db(tab_name, run_id) is None:
            return Response("Pro run not found.", status=404, mimetype="text/plain")
        set_selected_pro_run_id(tab_name, run_id)
        return redirect(url_for("index") + f"#{tab_name}_recent_runs")

    if authenticated and request.method == "POST" and form_name == "update_pro_run_notes":
        tab_name = request.form.get("tab_name", "")
        try:
            run_id = int(request.form.get("run_id", ""))
        except (TypeError, ValueError):
            return Response("Invalid Pro run.", status=400, mimetype="text/plain")
        if tab_name not in PRO_TAB_NAMES:
            return Response("Unknown Pro tab.", status=404, mimetype="text/plain")
        run_name = clean_run_text(request.form.get("run_name"))[:120]
        run_notes = clean_run_text(request.form.get("run_notes"))[:2000]

        def apply_notes(snapshot):
            entry = snapshot.setdefault("history_entry", {})
            entry["run_name"] = run_name or clean_run_text(entry.get("run_name")) or default_run_name(entry.get("timestamp", ""))
            entry["run_notes"] = run_notes
            snapshot["run_name"] = entry["run_name"]
            snapshot["run_notes"] = run_notes

        if not update_pro_run_snapshot(tab_name, run_id, apply_notes):
            return Response("Pro run not found.", status=404, mimetype="text/plain")
        return redirect(url_for("index") + f"#{tab_name}_recent_runs")

    if authenticated and request.method == "POST" and form_name == "compare_pro_runs":
        tab_name = request.form.get("tab_name", "")
        if tab_name not in PRO_TAB_NAMES:
            return Response("Unknown Pro tab.", status=404, mimetype="text/plain")
        run_ids = []
        for value in request.form.getlist("compare_run_ids"):
            try:
                run_id = int(value)
            except (TypeError, ValueError):
                continue
            if load_pro_run_from_db(tab_name, run_id) is not None:
                run_ids.append(run_id)
        set_selected_compare_run_ids(tab_name, run_ids[:2])
        return redirect(url_for("index") + f"#{tab_name}_recent_runs")

    if authenticated and request.method == "POST" and form_name == "upload":
        uploaded_file = request.files.get("data_file")

        if uploaded_file is None or uploaded_file.filename == "":
            data_error = "Please choose a CSV or Excel file."
        elif not allowed_file(uploaded_file.filename):
            data_error = "Unsupported file type. Upload a .csv, .xls, or .xlsx file."
        else:
            try:
                data = read_uploaded_file(uploaded_file)
                save_dataset(data, uploaded_file.filename)
            except UnicodeDecodeError:
                data_error = "Could not decode the CSV file. Please upload a UTF-8 encoded CSV."
            except Exception as exc:
                data_error = f"Could not read the uploaded file: {exc}"

    if authenticated and request.method == "POST" and form_name == "test_data":
        active_tab = "data"
        save_dataset(simulate_test_data(), "simulated_test_data.csv")

    dataset = current_dataset() if authenticated else None

    if authenticated and request.method == "POST" and form_name in PRO_TAB_NAMES and not has_subscription:
        active_tab = form_name
        model_tabs[form_name]["error"] = "A Pro subscription is required to run this analysis."
    elif authenticated and request.method == "POST" and form_name in CLASSIFICATION_TAB_CONFIGS:
        active_tab = form_name
        handle_classification_submission(model_tabs[form_name], dataset)

    if authenticated and request.method == "POST" and form_name in PRO_TAB_NAMES and not has_subscription:
        active_tab = form_name
    elif authenticated and request.method == "POST" and form_name in REGRESSION_TAB_CONFIGS:
        active_tab = form_name
        handle_regression_submission(model_tabs[form_name], dataset)

    has_data = dataset is not None
    data = dataset["data"] if has_data else None
    filename = dataset["filename"] if has_data else None
    columns = list(data.columns) if has_data else []

    if has_data:
        for tab_name in CLASSIFICATION_TAB_CONFIGS:
            apply_classification_defaults(model_tabs[tab_name], columns)
        for tab_name in REGRESSION_TAB_CONFIGS:
            apply_regression_defaults(model_tabs[tab_name], data, columns)

        current_pro_tab = form_name if request.method == "POST" and form_name in PRO_TAB_NAMES else None
        if current_pro_tab and model_tabs[current_pro_tab].get("output") is not None:
            save_pro_run(model_tabs[current_pro_tab])
            restore_pro_runs(model_tabs)
        elif current_pro_tab:
            restore_pro_runs(model_tabs, exclude={current_pro_tab})
        else:
            restore_pro_runs(model_tabs)

    table_html = preview_table(data) if has_data else None
    row_count = min(25, len(data)) if has_data else 0
    total_rows = len(data) if has_data else 0

    return render_template(
        "index.html",
        active_tab=active_tab,
        auth_error=auth_error,
        auth_mode=auth_mode,
        is_authenticated=authenticated,
        current_username=session.get("username"),
        has_subscription=has_subscription,
        mollie_configured=mollie_configured(),
        subscription_price=f"{SUBSCRIPTION_CURRENCY} {SUBSCRIPTION_AMOUNT} / {SUBSCRIPTION_INTERVAL}",
        subscription_date=user["subscription_updated_at"] if user and user["subscription_updated_at"] else "-",
        subscription_fee=f"{SUBSCRIPTION_CURRENCY} {SUBSCRIPTION_AMOUNT} per month",
        data_error=data_error,
        table_html=table_html,
        filename=filename,
        row_count=row_count,
        total_rows=total_rows,
        has_data=has_data,
        columns=columns,
        classification_tab=model_tabs["classification"],
        regression_tab=model_tabs["regression"],
        pro_classification_tab=model_tabs["pro_classification"],
        pro_regression_tab=model_tabs["pro_regression"],
    )


@app.route("/run-progress/<job_id>")
def run_progress(job_id):
    if not is_user_authenticated():
        return jsonify({"status": "unauthorized", "completed": 0, "total": 0, "percent": 0}), 401
    cleanup_run_progress()
    with RUN_PROGRESS_LOCK:
        progress = deepcopy(RUN_PROGRESS.get(job_id))
    if progress is None:
        progress = {"completed": 0, "total": 0, "percent": 0, "status": "pending", "label": ""}
    progress.pop("updated_at", None)
    return jsonify(progress)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/subscribe/start", methods=["POST"])
def start_subscription():
    if not is_user_authenticated():
        return Response("Authentication required.", status=401, mimetype="text/plain")
    user = current_user()
    if user_has_subscription(user):
        return redirect(url_for("index"))
    try:
        checkout_url = create_mollie_first_payment(user)
    except RuntimeError as exc:
        return Response(str(exc), status=503, mimetype="text/plain")

    with get_db_connection() as connection:
        payment = connection.execute(
            """
            SELECT mollie_payment_id
            FROM subscription_payments
            WHERE user_id = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (user["id"],),
        ).fetchone()
    if payment:
        session["pending_mollie_payment_id"] = payment["mollie_payment_id"]
    return redirect(checkout_url)


@app.route("/subscribe/return")
def subscription_return():
    payment_id = session.pop("pending_mollie_payment_id", None)
    if payment_id:
        try:
            sync_mollie_payment(payment_id)
        except RuntimeError:
            pass
    return redirect(url_for("index") + "#pro_classification")


@app.route("/subscribe/cancel", methods=["POST"])
def cancel_subscription():
    if not is_user_authenticated():
        return Response("Authentication required.", status=401, mimetype="text/plain")
    user = current_user()
    if not user_has_subscription(user):
        return redirect(url_for("index"))
    try:
        cancel_mollie_subscription(user)
    except RuntimeError as exc:
        return Response(str(exc), status=503, mimetype="text/plain")
    return redirect(url_for("index"))


@app.route("/mollie/webhook", methods=["POST"])
def mollie_webhook():
    payment_id = request.form.get("id")
    if not payment_id:
        return Response("Missing payment id.", status=400, mimetype="text/plain")
    try:
        sync_mollie_payment(payment_id)
    except RuntimeError:
        return Response("Could not process webhook.", status=503, mimetype="text/plain")
    return Response("OK", mimetype="text/plain")


@app.route("/pro-runs/<tab_name>/<int:run_id>/load", methods=["POST"])
def load_pro_run(tab_name, run_id):
    if not is_user_authenticated():
        return Response("Authentication required.", status=401, mimetype="text/plain")
    if tab_name not in PRO_TAB_NAMES:
        return Response("Unknown Pro tab.", status=404, mimetype="text/plain")
    if load_pro_run_from_db(tab_name, run_id) is None:
        return Response("Pro run not found.", status=404, mimetype="text/plain")
    set_selected_pro_run_id(tab_name, run_id)
    return redirect(url_for("index") + f"#{tab_name}_recent_runs")


def selected_pro_run_snapshot(tab_name):
    selected_id = selected_pro_run_id(tab_name)
    if selected_id is not None:
        snapshot = load_pro_run_from_db(tab_name, selected_id)
        if snapshot:
            return snapshot
    runs = load_pro_runs_from_db(tab_name)
    return runs[0] if runs else None


def durable_pro_report(result_type):
    if result_type not in PRO_TAB_NAMES:
        return None
    snapshot = selected_pro_run_snapshot(result_type)
    if not snapshot:
        return None
    return report_renderer.pro_report_html(result_type, snapshot, current_dataset())


def durable_pro_report_pdf(result_type):
    if result_type not in PRO_TAB_NAMES:
        return None
    snapshot = selected_pro_run_snapshot(result_type)
    if not snapshot:
        return None
    return report_renderer.pro_report_pdf_bytes(result_type, snapshot, current_dataset())


def durable_download_artifact(result_type, artifact):
    user_id, current_id = current_pro_run_scope()
    if not user_id or not current_id or result_type not in PRO_TAB_NAMES:
        return None

    selected_snapshot = selected_pro_run_snapshot(result_type)
    if selected_snapshot:
        csv_data = (selected_snapshot.get("download_artifacts") or {}).get(artifact)
        if csv_data is not None:
            DOWNLOADS.setdefault(current_id, {}).setdefault(result_type, {})[artifact] = csv_data
            return csv_data

    with get_db_connection() as connection:
        rows = connection.execute(
            """
            SELECT snapshot_json
            FROM pro_runs
            WHERE user_id = ? AND dataset_id = ? AND tab_name = ?
            ORDER BY id DESC
            LIMIT 5
            """,
            (user_id, current_id, result_type),
        ).fetchall()
    for row in rows:
        try:
            snapshot = json.loads(row["snapshot_json"])
        except json.JSONDecodeError:
            continue
        csv_data = (snapshot.get("download_artifacts") or {}).get(artifact)
        if csv_data is not None:
            DOWNLOADS.setdefault(current_id, {}).setdefault(result_type, {})[artifact] = csv_data
            return csv_data
    return None


def model_comparison_pdf_artifact(result_type):
    current_id = dataset_id()
    csv_data = DOWNLOADS.get(current_id, {}).get(result_type, {}).get("model_comparison") if current_id else None
    if csv_data is None:
        csv_data = durable_download_artifact(result_type, "model_comparison")
    if csv_data is None:
        return None
    return report_renderer.model_comparison_pdf_bytes(csv_data)


@app.route("/download/<result_type>/<artifact>")
def download_result(result_type, artifact):
    if not is_user_authenticated():
        return Response("Authentication required.", status=401, mimetype="text/plain")

    if artifact == "pro_report":
        report_html = durable_pro_report(result_type)
        if report_html is None:
            return Response("No Pro report is available for this selection.", status=404, mimetype="text/plain")
        filename = f"{result_type}_report.html"
        return Response(
            report_html,
            mimetype="text/html",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    if artifact == "pro_report_pdf":
        report_pdf = durable_pro_report_pdf(result_type)
        if report_pdf is None:
            return Response("No Pro report PDF is available for this selection.", status=404, mimetype="text/plain")
        filename = f"{result_type}_report.pdf"
        return Response(
            report_pdf,
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    if artifact == "model_comparison_pdf":
        comparison_pdf = model_comparison_pdf_artifact(result_type)
        if comparison_pdf is None:
            return Response("No model comparison PDF is available for this selection.", status=404, mimetype="text/plain")
        filename = f"{result_type}_model_comparison.pdf"
        return Response(
            comparison_pdf,
            mimetype="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    current_id = dataset_id()
    csv_data = DOWNLOADS.get(current_id, {}).get(result_type, {}).get(artifact)
    if csv_data is None:
        csv_data = durable_download_artifact(result_type, artifact)
    if csv_data is None:
        return Response("No downloadable result is available for this selection.", status=404, mimetype="text/plain")

    filename = f"{result_type}_{artifact}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


if __name__ == "__main__":
    app.run(debug=True)


