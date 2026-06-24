from io import BytesIO, StringIO
import base64
from copy import deepcopy
from datetime import datetime
import html
import json
import math
import os
import sqlite3
import textwrap
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from flask import Flask, Response, redirect, render_template_string, request, session, url_for
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, average_precision_score, auc, precision_recall_curve, roc_curve
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, plot_tree
from werkzeug.security import check_password_hash, generate_password_hash


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
DB_PATH = Path(__file__).with_name("modelmetrica_users.sqlite3")
DATASETS = {}
DOWNLOADS = {}
DISPLAY_DECIMALS = 3
DISPLAY_FLOAT_FORMAT = f"{{:.{DISPLAY_DECIMALS}f}}".format
MOLLIE_API_BASE = "https://api.mollie.com/v2"
MOLLIE_API_KEY = os.environ.get("MOLLIE_API_KEY", "")
MOLLIE_BASE_URL = os.environ.get("MOLLIE_BASE_URL", "")
SUBSCRIPTION_AMOUNT = os.environ.get("MODELMETRICA_SUBSCRIPTION_AMOUNT", "9.99")
SUBSCRIPTION_CURRENCY = os.environ.get("MODELMETRICA_SUBSCRIPTION_CURRENCY", "EUR")
SUBSCRIPTION_INTERVAL = os.environ.get("MODELMETRICA_SUBSCRIPTION_INTERVAL", "1 month")
SUBSCRIPTION_DESCRIPTION = os.environ.get("MODELMETRICA_SUBSCRIPTION_DESCRIPTION", "ModelMetrica Pro subscription")

PAGE_TEMPLATE = """
{% macro render_classification_tab(tab, active_tab, has_data, columns) %}
      <section id="{{ tab.id }}" class="tab-panel {{ 'active' if active_tab == tab.id else '' }}">
        <div class="panel">
          {% if not has_data %}
            <p class="error">Upload a dataset on the Data tab before running classification.</p>
          {% else %}
            <form method="post" data-run-form {% if tab.allow_model_comparison %}data-pro-run-form{% endif %}>
              <input type="hidden" name="form_name" value="{{ tab.form_name }}">
              <input type="hidden" name="active_tab" value="{{ tab.id }}">
              <div>
                <label for="{{ tab.model_field }}">Model type</label>
                <select id="{{ tab.model_field }}" name="{{ tab.model_field }}" {% if tab.allow_model_comparison %}multiple{% endif %} required>
                  <option value="logistic" {{ 'selected' if 'logistic' in tab.selected_models else '' }}>Logistic regression</option>
                  <option value="tree" {{ 'selected' if 'tree' in tab.selected_models else '' }}>Tree model</option>
                  <option value="random_forest" {{ 'selected' if 'random_forest' in tab.selected_models else '' }}>Random Forest</option>
                  <option value="gradient_boosting" {{ 'selected' if 'gradient_boosting' in tab.selected_models else '' }}>Gradient Boosting</option>
                  <option value="svm" {{ 'selected' if 'svm' in tab.selected_models else '' }}>Support Vector Machine</option>
                  <option value="knn" {{ 'selected' if 'knn' in tab.selected_models else '' }}>kNN</option>
                </select>
              </div>
              {% if tab.allow_model_comparison %}
              <div>
                <label for="{{ tab.detail_model_field }}">Detail model</label>
                <select id="{{ tab.detail_model_field }}" name="{{ tab.detail_model_field }}" required>
                  <option value="best" {{ 'selected' if tab.selected_detail_model == 'best' else '' }}>Auto best</option>
                  <option value="logistic" {{ 'selected' if tab.selected_detail_model == 'logistic' else '' }}>Logistic regression</option>
                  <option value="tree" {{ 'selected' if tab.selected_detail_model == 'tree' else '' }}>Tree model</option>
                  <option value="random_forest" {{ 'selected' if tab.selected_detail_model == 'random_forest' else '' }}>Random Forest</option>
                  <option value="gradient_boosting" {{ 'selected' if tab.selected_detail_model == 'gradient_boosting' else '' }}>Gradient Boosting</option>
                  <option value="svm" {{ 'selected' if tab.selected_detail_model == 'svm' else '' }}>Support Vector Machine</option>
                  <option value="knn" {{ 'selected' if tab.selected_detail_model == 'knn' else '' }}>kNN</option>
                </select>
              </div>
              {% endif %}
              <div>
                <label for="{{ tab.target_field }}">Target column</label>
                <select id="{{ tab.target_field }}" name="{{ tab.target_field }}" required>
                  {% for column in columns %}
                    <option value="{{ column }}" {{ 'selected' if column == tab.selected_target else '' }}>{{ column }}</option>
                  {% endfor %}
                </select>
              </div>
              <div>
                <label for="{{ tab.predictors_field }}">Predictor columns</label>
                <select id="{{ tab.predictors_field }}" name="{{ tab.predictors_field }}" multiple required>
                  {% for column in columns %}
                    <option value="{{ column }}" {{ 'selected' if column in tab.selected_predictors else '' }}>{{ column }}</option>
                  {% endfor %}
                </select>
                <p>Select one or more predictors. Logistic regression requires exactly two target classes.</p>
              </div>
              <div>
                <label for="{{ tab.test_size_field }}">Test set size</label>
                <select id="{{ tab.test_size_field }}" name="{{ tab.test_size_field }}" required>
                  <option value="0.2" {{ 'selected' if tab.selected_test_size == 0.2 else '' }}>20%</option>
                  <option value="0.25" {{ 'selected' if tab.selected_test_size == 0.25 else '' }}>25%</option>
                  <option value="0.3" {{ 'selected' if tab.selected_test_size == 0.3 else '' }}>30%</option>
                  <option value="0.4" {{ 'selected' if tab.selected_test_size == 0.4 else '' }}>40%</option>
                </select>
              </div>
              <div>
                <label for="{{ tab.cv_folds_field }}">Cross-validation</label>
                <select id="{{ tab.cv_folds_field }}" name="{{ tab.cv_folds_field }}" required>
                  <option value="0" {{ 'selected' if tab.selected_cv_folds == 0 else '' }}>Off</option>
                  <option value="3" {{ 'selected' if tab.selected_cv_folds == 3 else '' }}>3 folds</option>
                  <option value="5" {{ 'selected' if tab.selected_cv_folds == 5 else '' }}>5 folds</option>
                  <option value="10" {{ 'selected' if tab.selected_cv_folds == 10 else '' }}>10 folds</option>
                </select>
              </div>
              {% if tab.allow_model_comparison %}
              <div>
                <label for="{{ tab.missing_values_field }}">Missing values</label>
                <select id="{{ tab.missing_values_field }}" name="{{ tab.missing_values_field }}" required>
                  <option value="drop" {{ 'selected' if tab.selected_missing_values == 'drop' else '' }}>Drop incomplete rows</option>
                  <option value="impute_mean_mode" {{ 'selected' if tab.selected_missing_values == 'impute_mean_mode' else '' }}>Impute mean/mode</option>
                  <option value="impute_median_mode" {{ 'selected' if tab.selected_missing_values == 'impute_median_mode' else '' }}>Impute median/mode</option>
                </select>
              </div>
              <div>
                <label for="{{ tab.categorical_encoding_field }}">Categorical encoding</label>
                <select id="{{ tab.categorical_encoding_field }}" name="{{ tab.categorical_encoding_field }}" required>
                  <option value="one_hot_drop_first" {{ 'selected' if tab.selected_categorical_encoding == 'one_hot_drop_first' else '' }}>One-hot, drop first</option>
                  <option value="one_hot_full" {{ 'selected' if tab.selected_categorical_encoding == 'one_hot_full' else '' }}>One-hot, all levels</option>
                  <option value="ordinal" {{ 'selected' if tab.selected_categorical_encoding == 'ordinal' else '' }}>Ordinal codes</option>
                </select>
              </div>
              <div>
                <label for="{{ tab.scaling_field }}">Feature scaling</label>
                <select id="{{ tab.scaling_field }}" name="{{ tab.scaling_field }}" required>
                  <option value="on" {{ 'selected' if tab.selected_scaling == 'on' else '' }}>On</option>
                  <option value="off" {{ 'selected' if tab.selected_scaling == 'off' else '' }}>Off</option>
                </select>
              </div>
              <div>
                <label for="{{ tab.split_seed_field }}">Split seed</label>
                <input id="{{ tab.split_seed_field }}" name="{{ tab.split_seed_field }}" type="number" min="0" max="999999" step="1" value="{{ tab.selected_split_seed }}" required>
              </div>
              <div>
                <label for="{{ tab.outlier_handling_field }}">Outliers</label>
                <select id="{{ tab.outlier_handling_field }}" name="{{ tab.outlier_handling_field }}" required>
                  <option value="none" {{ 'selected' if tab.selected_outlier_handling == 'none' else '' }}>No handling</option>
                  <option value="winsorize" {{ 'selected' if tab.selected_outlier_handling == 'winsorize' else '' }}>Winsorize numeric columns</option>
                  <option value="remove_iqr" {{ 'selected' if tab.selected_outlier_handling == 'remove_iqr' else '' }}>Remove IQR outliers</option>
                </select>
              </div>
              <div>
                <label for="{{ tab.tuning_mode_field }}">Hyperparameter tuning</label>
                <select id="{{ tab.tuning_mode_field }}" name="{{ tab.tuning_mode_field }}" required>
                  <option value="off" {{ 'selected' if tab.selected_tuning_mode == 'off' else '' }}>Off</option>
                  <option value="grid" {{ 'selected' if tab.selected_tuning_mode == 'grid' else '' }}>Grid search</option>
                  <option value="random" {{ 'selected' if tab.selected_tuning_mode == 'random' else '' }}>Random search</option>
                </select>
              </div>
              <div>
                <label for="{{ tab.tuning_iterations_field }}">Random search iterations</label>
                <input id="{{ tab.tuning_iterations_field }}" name="{{ tab.tuning_iterations_field }}" type="number" min="3" max="30" step="1" value="{{ tab.selected_tuning_iterations }}" required>
              </div>
              {% if tab.threshold_field %}
              <div>
                <label for="{{ tab.threshold_field }}">Decision threshold</label>
                <div class="threshold-control">
                  <input id="{{ tab.threshold_field }}" name="{{ tab.threshold_field }}" type="range" min="0.05" max="0.95" step="0.001" value="{{ '%.3f'|format(tab.selected_threshold) }}" data-threshold-input>
                  <output for="{{ tab.threshold_field }}" data-threshold-output>{{ '%.3f'|format(tab.selected_threshold) }}</output>
                </div>
              </div>
              {% endif %}
              {% endif %}
              <div>
                <button type="submit">Run</button>
              </div>
            </form>
          {% endif %}
          {% if tab.error %}
            <p class="error">{{ tab.error }}</p>
          {% endif %}
        </div>

        {% if tab.run_history %}
          <div class="panel">
            <h2>Recent Pro runs</h2>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Target</th>
                    <th>Models</th>
                    <th>Detail</th>
                    <th>Summary</th>
                  </tr>
                </thead>
                <tbody>
                  {% for run in tab.run_history %}
                    <tr>
                      <td>{{ run.timestamp }}</td>
                      <td>{{ run.target }}</td>
                      <td>{{ run.models }}</td>
                      <td>{{ run.detail_model }}</td>
                      <td>{{ run.summary }}</td>
                    </tr>
                  {% endfor %}
                </tbody>
              </table>
            </div>
          </div>
        {% endif %}
        {% if tab.comparison_html %}
          <div class="panel">
            <h2>Model comparison</h2>
            <p>Detailed results below show the selected detail model, or the best model when Auto best is chosen.</p>
            {% if tab.comparison_download %}
              <div class="download-links">
                <a href="{{ tab.comparison_download.href }}">{{ tab.comparison_download.label }}</a>
              </div>
            {% endif %}
            <div class="table-wrap">
              {{ tab.comparison_html|safe }}
            </div>
            {% if tab.detail_metric_comparison_html %}
              <h3>Selected model tuning comparison</h3>
              <div class="table-wrap">
                {{ tab.detail_metric_comparison_html|safe }}
              </div>
            {% endif %}
          </div>
        {% endif %}
        {% if tab.output %}
          {% if tab.recommendation %}
            <div class="panel recommendation-panel">
              <h2>Model recommendation</h2>
              <p><strong>{{ tab.recommendation.title }}</strong></p>
              <p>{{ tab.recommendation.summary }}</p>
              <div class="recommendation-grid">
                <div>
                  <h3>Evidence</h3>
                  <ul>
                    {% for item in tab.recommendation.evidence %}
                      <li>{{ item }}</li>
                    {% endfor %}
                  </ul>
                </div>
                <div>
                  <h3>Concerns</h3>
                  <ul>
                    {% for item in tab.recommendation.concerns %}
                      <li>{{ item }}</li>
                    {% endfor %}
                  </ul>
                </div>
                <div>
                  <h3>Next actions</h3>
                  <ul>
                    {% for item in tab.recommendation.actions %}
                      <li>{{ item }}</li>
                    {% endfor %}
                  </ul>
                </div>
              </div>
            </div>
          {% endif %}
          <div class="panel">
            <h2>{{ tab.output.title }}</h2>
            <p>{{ tab.output.description }}</p>
            {% if tab.output.downloads or tab.report_download or tab.report_pdf_download %}
              <div class="download-links">
                {% if tab.report_download %}
                  <a href="{{ tab.report_download.href }}">{{ tab.report_download.label }}</a>
                {% endif %}
                {% if tab.report_pdf_download %}
                  <a href="{{ tab.report_pdf_download.href }}">{{ tab.report_pdf_download.label }}</a>
                {% endif %}
                {% for download in tab.output.downloads %}
                  <a href="{{ download.href }}">{{ download.label }}</a>
                {% endfor %}
              </div>
            {% endif %}
            <div class="metric-row">
              {% for metric in tab.output.metrics %}
                <div class="metric"><span>{{ metric.label }}</span><strong>{{ metric.value }}</strong></div>
              {% endfor %}
            </div>
            {% if tab.output.coefficients_html %}
              <h3>Coefficients</h3>
              <div class="table-wrap">
                {{ tab.output.coefficients_html|safe }}
              </div>
            {% endif %}
            {% if tab.output.importances_html %}
              <h3>Variable importance</h3>
              <div class="table-wrap">
                {{ tab.output.importances_html|safe }}
              </div>
            {% endif %}
            {% if tab.output.details_html %}
              <h3>Model details</h3>
              <div class="table-wrap">
                {{ tab.output.details_html|safe }}
              </div>
            {% endif %}
            {% if tab.output.cv_summary_html or tab.output.cv_diagnostics_html or tab.output.cv_plot %}
              <h3>Cross-validation diagnostics</h3>
              {% if tab.output.cv_plot %}
                <div class="tree-plot">
                  <img src="data:image/png;base64,{{ tab.output.cv_plot }}" alt="Cross-validation fold score plot">
                </div>
              {% endif %}
              {% if tab.output.cv_summary_html %}
                <div class="table-wrap">
                  {{ tab.output.cv_summary_html|safe }}
                </div>
              {% endif %}
              {% if tab.output.cv_diagnostics_html %}
                <div class="table-wrap">
                  {{ tab.output.cv_diagnostics_html|safe }}
                </div>
              {% endif %}
            {% endif %}
            <h3>Confusion matrix</h3>
            <div class="table-wrap">
              {{ tab.output.confusion_html|safe }}
            </div>
            {% if tab.output.roc_plot %}
              <h3>ROC curve</h3>
              <div class="tree-plot">
                <img src="data:image/png;base64,{{ tab.output.roc_plot }}" alt="ROC curve">
              </div>
            {% endif %}
            {% if tab.output.pr_plot %}
              <h3>Precision-recall curve</h3>
              <div class="tree-plot">
                <img src="data:image/png;base64,{{ tab.output.pr_plot }}" alt="Precision-recall curve">
              </div>
            {% endif %}
            {% if tab.output.selected_threshold_html %}
              <h3>Selected threshold metrics</h3>
              <div class="table-wrap">
                {{ tab.output.selected_threshold_html|safe }}
              </div>
            {% endif %}
            {% if tab.output.threshold_html %}
              <h3>Threshold analysis</h3>
              <div class="table-wrap">
                {{ tab.output.threshold_html|safe }}
              </div>
            {% endif %}
            {% if tab.output.tree_plot %}
              <h3>Tree structure</h3>
              <div class="tree-plot">
                <img src="data:image/png;base64,{{ tab.output.tree_plot }}" alt="Classification tree plot">
              </div>
            {% endif %}
          </div>
        {% endif %}
      </section>
{% endmacro %}

{% macro render_regression_tab(tab, active_tab, has_data, columns) %}
      <section id="{{ tab.id }}" class="tab-panel {{ 'active' if active_tab == tab.id else '' }}">
        <div class="panel">
          {% if not has_data %}
            <p class="error">Upload a dataset on the Data tab before running regression.</p>
          {% else %}
            <form method="post" data-run-form {% if tab.allow_model_comparison %}data-pro-run-form{% endif %}>
              <input type="hidden" name="form_name" value="{{ tab.form_name }}">
              <input type="hidden" name="active_tab" value="{{ tab.id }}">
              <div>
                <label for="{{ tab.model_field }}">Model type</label>
                <select id="{{ tab.model_field }}" name="{{ tab.model_field }}" {% if tab.allow_model_comparison %}multiple{% endif %} required>
                  <option value="linear" {{ 'selected' if 'linear' in tab.selected_models else '' }}>Linear Regression</option>
                  <option value="ridge" {{ 'selected' if 'ridge' in tab.selected_models else '' }}>Ridge Regression</option>
                  <option value="lasso" {{ 'selected' if 'lasso' in tab.selected_models else '' }}>Lasso Regression</option>
                  <option value="random_forest" {{ 'selected' if 'random_forest' in tab.selected_models else '' }}>Random Forest Regression</option>
                  <option value="gradient_boosting" {{ 'selected' if 'gradient_boosting' in tab.selected_models else '' }}>Gradient Boosting Regression</option>
                  <option value="svr" {{ 'selected' if 'svr' in tab.selected_models else '' }}>Support Vector Regression</option>
                  <option value="knn" {{ 'selected' if 'knn' in tab.selected_models else '' }}>kNN Regression</option>
                </select>
              </div>
              {% if tab.allow_model_comparison %}
              <div>
                <label for="{{ tab.detail_model_field }}">Detail model</label>
                <select id="{{ tab.detail_model_field }}" name="{{ tab.detail_model_field }}" required>
                  <option value="best" {{ 'selected' if tab.selected_detail_model == 'best' else '' }}>Auto best</option>
                  <option value="linear" {{ 'selected' if tab.selected_detail_model == 'linear' else '' }}>Linear Regression</option>
                  <option value="ridge" {{ 'selected' if tab.selected_detail_model == 'ridge' else '' }}>Ridge Regression</option>
                  <option value="lasso" {{ 'selected' if tab.selected_detail_model == 'lasso' else '' }}>Lasso Regression</option>
                  <option value="random_forest" {{ 'selected' if tab.selected_detail_model == 'random_forest' else '' }}>Random Forest Regression</option>
                  <option value="gradient_boosting" {{ 'selected' if tab.selected_detail_model == 'gradient_boosting' else '' }}>Gradient Boosting Regression</option>
                  <option value="svr" {{ 'selected' if tab.selected_detail_model == 'svr' else '' }}>Support Vector Regression</option>
                  <option value="knn" {{ 'selected' if tab.selected_detail_model == 'knn' else '' }}>kNN Regression</option>
                </select>
              </div>
              {% endif %}
              <div>
                <label for="{{ tab.target_field }}">Numeric target column</label>
                <select id="{{ tab.target_field }}" name="{{ tab.target_field }}" required>
                  {% for column in columns %}
                    <option value="{{ column }}" {{ 'selected' if column == tab.selected_target else '' }}>{{ column }}</option>
                  {% endfor %}
                </select>
              </div>
              <div>
                <label for="{{ tab.predictors_field }}">Predictor columns</label>
                <select id="{{ tab.predictors_field }}" name="{{ tab.predictors_field }}" multiple required>
                  {% for column in columns %}
                    <option value="{{ column }}" {{ 'selected' if column in tab.selected_predictors else '' }}>{{ column }}</option>
                  {% endfor %}
                </select>
                <p>Select one or more predictors. Categorical predictors are automatically encoded.</p>
              </div>
              <div>
                <label for="{{ tab.test_size_field }}">Test set size</label>
                <select id="{{ tab.test_size_field }}" name="{{ tab.test_size_field }}" required>
                  <option value="0.2" {{ 'selected' if tab.selected_test_size == 0.2 else '' }}>20%</option>
                  <option value="0.25" {{ 'selected' if tab.selected_test_size == 0.25 else '' }}>25%</option>
                  <option value="0.3" {{ 'selected' if tab.selected_test_size == 0.3 else '' }}>30%</option>
                  <option value="0.4" {{ 'selected' if tab.selected_test_size == 0.4 else '' }}>40%</option>
                </select>
              </div>
              <div>
                <label for="{{ tab.cv_folds_field }}">Cross-validation</label>
                <select id="{{ tab.cv_folds_field }}" name="{{ tab.cv_folds_field }}" required>
                  <option value="0" {{ 'selected' if tab.selected_cv_folds == 0 else '' }}>Off</option>
                  <option value="3" {{ 'selected' if tab.selected_cv_folds == 3 else '' }}>3 folds</option>
                  <option value="5" {{ 'selected' if tab.selected_cv_folds == 5 else '' }}>5 folds</option>
                  <option value="10" {{ 'selected' if tab.selected_cv_folds == 10 else '' }}>10 folds</option>
                </select>
              </div>
              {% if tab.allow_model_comparison %}
              <div>
                <label for="{{ tab.missing_values_field }}">Missing values</label>
                <select id="{{ tab.missing_values_field }}" name="{{ tab.missing_values_field }}" required>
                  <option value="drop" {{ 'selected' if tab.selected_missing_values == 'drop' else '' }}>Drop incomplete rows</option>
                  <option value="impute_mean_mode" {{ 'selected' if tab.selected_missing_values == 'impute_mean_mode' else '' }}>Impute mean/mode</option>
                  <option value="impute_median_mode" {{ 'selected' if tab.selected_missing_values == 'impute_median_mode' else '' }}>Impute median/mode</option>
                </select>
              </div>
              <div>
                <label for="{{ tab.categorical_encoding_field }}">Categorical encoding</label>
                <select id="{{ tab.categorical_encoding_field }}" name="{{ tab.categorical_encoding_field }}" required>
                  <option value="one_hot_drop_first" {{ 'selected' if tab.selected_categorical_encoding == 'one_hot_drop_first' else '' }}>One-hot, drop first</option>
                  <option value="one_hot_full" {{ 'selected' if tab.selected_categorical_encoding == 'one_hot_full' else '' }}>One-hot, all levels</option>
                  <option value="ordinal" {{ 'selected' if tab.selected_categorical_encoding == 'ordinal' else '' }}>Ordinal codes</option>
                </select>
              </div>
              <div>
                <label for="{{ tab.scaling_field }}">Feature scaling</label>
                <select id="{{ tab.scaling_field }}" name="{{ tab.scaling_field }}" required>
                  <option value="on" {{ 'selected' if tab.selected_scaling == 'on' else '' }}>On</option>
                  <option value="off" {{ 'selected' if tab.selected_scaling == 'off' else '' }}>Off</option>
                </select>
              </div>
              <div>
                <label for="{{ tab.split_seed_field }}">Split seed</label>
                <input id="{{ tab.split_seed_field }}" name="{{ tab.split_seed_field }}" type="number" min="0" max="999999" step="1" value="{{ tab.selected_split_seed }}" required>
              </div>
              <div>
                <label for="{{ tab.outlier_handling_field }}">Outliers</label>
                <select id="{{ tab.outlier_handling_field }}" name="{{ tab.outlier_handling_field }}" required>
                  <option value="none" {{ 'selected' if tab.selected_outlier_handling == 'none' else '' }}>No handling</option>
                  <option value="winsorize" {{ 'selected' if tab.selected_outlier_handling == 'winsorize' else '' }}>Winsorize numeric columns</option>
                  <option value="remove_iqr" {{ 'selected' if tab.selected_outlier_handling == 'remove_iqr' else '' }}>Remove IQR outliers</option>
                </select>
              </div>
              <div>
                <label for="{{ tab.tuning_mode_field }}">Hyperparameter tuning</label>
                <select id="{{ tab.tuning_mode_field }}" name="{{ tab.tuning_mode_field }}" required>
                  <option value="off" {{ 'selected' if tab.selected_tuning_mode == 'off' else '' }}>Off</option>
                  <option value="grid" {{ 'selected' if tab.selected_tuning_mode == 'grid' else '' }}>Grid search</option>
                  <option value="random" {{ 'selected' if tab.selected_tuning_mode == 'random' else '' }}>Random search</option>
                </select>
              </div>
              <div>
                <label for="{{ tab.tuning_iterations_field }}">Random search iterations</label>
                <input id="{{ tab.tuning_iterations_field }}" name="{{ tab.tuning_iterations_field }}" type="number" min="3" max="30" step="1" value="{{ tab.selected_tuning_iterations }}" required>
              </div>
              {% endif %}
              <div>
                <button type="submit">Run</button>
              </div>
            </form>
          {% endif %}
          {% if tab.error %}
            <p class="error">{{ tab.error }}</p>
          {% endif %}
        </div>

        {% if tab.run_history %}
          <div class="panel">
            <h2>Recent Pro runs</h2>
            <div class="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Target</th>
                    <th>Models</th>
                    <th>Detail</th>
                    <th>Summary</th>
                  </tr>
                </thead>
                <tbody>
                  {% for run in tab.run_history %}
                    <tr>
                      <td>{{ run.timestamp }}</td>
                      <td>{{ run.target }}</td>
                      <td>{{ run.models }}</td>
                      <td>{{ run.detail_model }}</td>
                      <td>{{ run.summary }}</td>
                    </tr>
                  {% endfor %}
                </tbody>
              </table>
            </div>
          </div>
        {% endif %}
        {% if tab.comparison_html %}
          <div class="panel">
            <h2>Model comparison</h2>
            <p>Detailed results below show the selected detail model, or the best model when Auto best is chosen.</p>
            {% if tab.comparison_download %}
              <div class="download-links">
                <a href="{{ tab.comparison_download.href }}">{{ tab.comparison_download.label }}</a>
              </div>
            {% endif %}
            <div class="table-wrap">
              {{ tab.comparison_html|safe }}
            </div>
            {% if tab.detail_metric_comparison_html %}
              <h3>Selected model tuning comparison</h3>
              <div class="table-wrap">
                {{ tab.detail_metric_comparison_html|safe }}
              </div>
            {% endif %}
          </div>
        {% endif %}
        {% if tab.output %}
          {% if tab.recommendation %}
            <div class="panel recommendation-panel">
              <h2>Model recommendation</h2>
              <p><strong>{{ tab.recommendation.title }}</strong></p>
              <p>{{ tab.recommendation.summary }}</p>
              <div class="recommendation-grid">
                <div>
                  <h3>Evidence</h3>
                  <ul>
                    {% for item in tab.recommendation.evidence %}
                      <li>{{ item }}</li>
                    {% endfor %}
                  </ul>
                </div>
                <div>
                  <h3>Concerns</h3>
                  <ul>
                    {% for item in tab.recommendation.concerns %}
                      <li>{{ item }}</li>
                    {% endfor %}
                  </ul>
                </div>
                <div>
                  <h3>Next actions</h3>
                  <ul>
                    {% for item in tab.recommendation.actions %}
                      <li>{{ item }}</li>
                    {% endfor %}
                  </ul>
                </div>
              </div>
            </div>
          {% endif %}
          <div class="panel">
            <h2>{{ tab.output.title }}</h2>
            <p>{{ tab.output.description }}</p>
            {% if tab.output.downloads or tab.report_download or tab.report_pdf_download %}
              <div class="download-links">
                {% if tab.report_download %}
                  <a href="{{ tab.report_download.href }}">{{ tab.report_download.label }}</a>
                {% endif %}
                {% if tab.report_pdf_download %}
                  <a href="{{ tab.report_pdf_download.href }}">{{ tab.report_pdf_download.label }}</a>
                {% endif %}
                {% for download in tab.output.downloads %}
                  <a href="{{ download.href }}">{{ download.label }}</a>
                {% endfor %}
              </div>
            {% endif %}
            <div class="metric-row">
              {% for metric in tab.output.metrics %}
                <div class="metric"><span>{{ metric.label }}</span><strong>{{ metric.value }}</strong></div>
              {% endfor %}
            </div>
            {% if tab.output.coefficients_html %}
              <h3>Coefficients</h3>
              <div class="table-wrap">
                {{ tab.output.coefficients_html|safe }}
              </div>
            {% endif %}
            {% if tab.output.importances_html %}
              <h3>Variable importance</h3>
              <div class="table-wrap">
                {{ tab.output.importances_html|safe }}
              </div>
            {% endif %}
            {% if tab.output.details_html %}
              <h3>Model details</h3>
              <div class="table-wrap">
                {{ tab.output.details_html|safe }}
              </div>
            {% endif %}
            {% if tab.output.cv_summary_html or tab.output.cv_diagnostics_html or tab.output.cv_plot %}
              <h3>Cross-validation diagnostics</h3>
              {% if tab.output.cv_plot %}
                <div class="tree-plot">
                  <img src="data:image/png;base64,{{ tab.output.cv_plot }}" alt="Cross-validation fold score plot">
                </div>
              {% endif %}
              {% if tab.output.cv_summary_html %}
                <div class="table-wrap">
                  {{ tab.output.cv_summary_html|safe }}
                </div>
              {% endif %}
              {% if tab.output.cv_diagnostics_html %}
                <div class="table-wrap">
                  {{ tab.output.cv_diagnostics_html|safe }}
                </div>
              {% endif %}
            {% endif %}
            {% if tab.output.predicted_actual_plot %}
              <h3>Predicted vs actual</h3>
              <div class="tree-plot">
                <img src="data:image/png;base64,{{ tab.output.predicted_actual_plot }}" alt="Predicted vs actual plot">
              </div>
            {% endif %}
            {% if tab.output.residuals_fitted_plot %}
              <h3>Residuals vs fitted</h3>
              <div class="tree-plot">
                <img src="data:image/png;base64,{{ tab.output.residuals_fitted_plot }}" alt="Residuals vs fitted plot">
              </div>
            {% endif %}
            {% if tab.output.residual_distribution_plot %}
              <h3>Residual distribution</h3>
              <div class="tree-plot">
                <img src="data:image/png;base64,{{ tab.output.residual_distribution_plot }}" alt="Residual distribution plot">
              </div>
            {% endif %}
            {% if tab.output.residual_diagnostics_html %}
              <h3>Residual diagnostics</h3>
              <div class="table-wrap">
                {{ tab.output.residual_diagnostics_html|safe }}
              </div>
            {% endif %}
          </div>
        {% endif %}
      </section>
{% endmacro %}
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>ModelMetrica</title>
    <style>
      :root {
        --ink: #1f2933;
        --muted: #64748b;
        --line: #d8dee8;
        --surface: #ffffff;
        --page: #f5f7fa;
        --accent: #176b87;
        --accent-dark: #0f4f63;
        --danger: #b42318;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        background: var(--page);
        color: var(--ink);
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      header {
        background: var(--surface);
        border-bottom: 1px solid var(--line);
        padding: 20px 28px;
      }
      .header-content {
        align-items: center;
        display: flex;
        justify-content: space-between;
        gap: 16px;
      }
      .brand {
        font-size: 22px;
        font-weight: 750;
      }
      .user-actions {
        align-items: center;
        display: flex;
        gap: 12px;
      }
      .logout-link {
        border: 1px solid var(--line);
        border-radius: 6px;
        color: var(--accent-dark);
        font-weight: 650;
        padding: 9px 12px;
        text-decoration: none;
      }
      .logout-link:hover {
        border-color: var(--accent);
      }
      .subscription-status {
        background: #f8fafc;
        border: 1px solid var(--line);
        border-radius: 999px;
        color: var(--muted);
        font-size: 13px;
        font-weight: 750;
        padding: 7px 10px;
      }
      .subscription-status.active {
        background: #ccfbf1;
        border-color: #5eead4;
        color: #115e59;
      }
      .subscription-banner {
        align-items: center;
        background: var(--surface);
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        display: flex;
        justify-content: space-between;
        gap: 16px;
        margin: 20px 0 0;
        padding: 14px 16px;
      }
      .subscription-banner strong {
        display: block;
        margin-bottom: 3px;
      }
      .subscription-banner p {
        margin: 0;
      }
      .subscription-banner .subscription-status {
        flex: 0 0 auto;
      }
      main {
        width: min(1120px, calc(100% - 40px));
        margin: 0 auto;
        padding: 34px 0 48px;
      }
      h1 {
        margin: 0 0 10px;
        font-size: 34px;
        letter-spacing: 0;
      }
      p {
        color: var(--muted);
      }
      .tabs {
        display: flex;
        gap: 8px;
        border-bottom: 1px solid var(--line);
        margin-top: 24px;
      }
      .tab {
        border: 1px solid transparent;
        border-radius: 6px 6px 0 0;
        color: var(--muted);
        display: inline-block;
        font-weight: 650;
        padding: 11px 16px;
        text-decoration: none;
      }
      .tab.active {
        background: var(--surface);
        border-color: var(--line) var(--line) var(--surface);
        color: var(--accent-dark);
        margin-bottom: -1px;
      }
      .panel {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 22px;
        margin-top: 22px;
      }
      .tab-panel {
        display: none;
      }
      .tab-panel.active {
        display: block;
      }
      form {
        display: grid;
        gap: 14px;
      }
      .data-actions form + form {
        margin-top: 22px;
      }
      .form-row {
        display: flex;
        gap: 12px;
        align-items: center;
        flex-wrap: wrap;
      }
      label {
        display: block;
        font-weight: 650;
        margin-bottom: 6px;
      }
      input[type="file"],
      select {
        border: 1px solid var(--line);
        border-radius: 6px;
        background: #fbfcfe;
        min-width: 260px;
        padding: 10px;
      }
      select[multiple] {
        min-height: 150px;
      }
      button {
        border: 1px solid var(--accent);
        border-radius: 6px;
        background: var(--accent);
        color: #fff;
        cursor: pointer;
        font-weight: 650;
        padding: 11px 16px;
      }
      button:hover {
        background: var(--accent-dark);
        border-color: var(--accent-dark);
      }
      .auth-overlay {
        align-items: center;
        background: rgba(15, 23, 42, 0.62);
        display: flex;
        inset: 0;
        justify-content: center;
        padding: 24px;
        position: fixed;
        z-index: 1000;
      }
      .auth-card {
        background: var(--surface);
        border-radius: 8px;
        box-shadow: 0 24px 80px rgba(15, 23, 42, 0.28);
        max-width: 760px;
        padding: 26px;
        width: min(100%, 760px);
      }
      .auth-grid {
        display: grid;
        gap: 22px;
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .auth-card input {
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 10px;
        width: 100%;
      }
      .auth-card h2,
      .auth-card h3 {
        margin-top: 0;
      }
      .download-links {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 16px 0;
      }
      .download-links a {
        border: 1px solid var(--line);
        border-radius: 6px;
        color: var(--accent-dark);
        font-weight: 650;
        padding: 9px 12px;
        text-decoration: none;
      }
      .download-links a:hover {
        border-color: var(--accent);
      }
      .error {
        color: var(--danger);
        font-weight: 650;
      }
      .metric-row {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 12px;
        margin: 14px 0 18px;
      }
      .metric {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 14px;
        background: #fbfcfe;
      }
      .metric span {
        color: var(--muted);
        display: block;
        font-size: 12px;
        font-weight: 750;
        text-transform: uppercase;
      }
      .metric strong {
        display: block;
        font-size: 22px;
        margin-top: 5px;
      }
      .table-wrap {
        overflow-x: auto;
      }
      .tree-plot {
        overflow-x: auto;
      }
      .tree-plot img {
        display: block;
        max-width: 100%;
        min-width: 760px;
      }
      table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
      }
      th,
      td {
        border-bottom: 1px solid var(--line);
        padding: 10px 12px;
        text-align: left;
        vertical-align: top;
      }
      th {
        background: #f8fafc;
        font-weight: 750;
      }
      tr:nth-child(even) td {
        background: #fbfcfe;
      }
      .model-comparison tbody tr {
        cursor: pointer;
      }
      .model-comparison tbody tr:hover td {
        background: #eef6ff;
      }
      .model-comparison tbody tr.best-row td {
        border-left: 4px solid #0f766e;
      }
      .model-comparison tbody tr.selected-row td {
        background: #eef2ff;
        box-shadow: inset 0 1px 0 #c7d2fe, inset 0 -1px 0 #c7d2fe;
      }
      .model-comparison tbody tr.selected-row.best-row td {
        background: #ecfdf5;
      }
      .status-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
      }
      .status-badge {
        border: 1px solid var(--line);
        border-radius: 999px;
        color: var(--muted);
        display: inline-block;
        font-size: 12px;
        font-weight: 750;
        line-height: 1;
        padding: 5px 8px;
        white-space: nowrap;
      }
      .status-badge.best {
        background: #ccfbf1;
        border-color: #5eead4;
        color: #115e59;
      }
      .status-badge.selected {
        background: #e0e7ff;
        border-color: #a5b4fc;
        color: #3730a3;
      }
      .status-badge.fit {
        background: #f8fafc;
      }
      .recommendation-panel {
        border-left: 4px solid var(--accent);
      }
      .recommendation-grid {
        display: grid;
        gap: 18px;
        grid-template-columns: repeat(3, minmax(0, 1fr));
      }
      .recommendation-grid h3 {
        margin-bottom: 8px;
      }
      .recommendation-grid ul {
        margin: 0;
        padding-left: 18px;
      }
      .recommendation-grid li {
        margin-bottom: 7px;
      }
      .threshold-control {
        align-items: center;
        display: grid;
        gap: 12px;
        grid-template-columns: minmax(180px, 1fr) 64px;
      }
      .threshold-control input[type="range"] {
        accent-color: var(--accent);
      }
      .threshold-control output {
        background: #f8fafc;
        border: 1px solid var(--line);
        border-radius: 6px;
        font-variant-numeric: tabular-nums;
        font-weight: 750;
        padding: 8px 10px;
        text-align: center;
      }
      .processing-overlay {
        align-items: center;
        background: rgba(15, 23, 42, 0.56);
        bottom: 0;
        display: none;
        justify-content: center;
        left: 0;
        padding: 24px;
        position: fixed;
        right: 0;
        top: 0;
        z-index: 1000;
      }
      .processing-overlay.active {
        display: flex;
      }
      .subscription-overlay {
        align-items: center;
        background: rgba(15, 23, 42, 0.62);
        bottom: 0;
        display: none;
        justify-content: center;
        left: 0;
        padding: 24px;
        position: fixed;
        right: 0;
        top: 0;
        z-index: 1100;
      }
      .subscription-overlay.active {
        display: flex;
      }
      .processing-dialog {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 24px 70px rgba(15, 23, 42, 0.22);
        max-width: 360px;
        padding: 26px;
        text-align: center;
        width: min(100%, 360px);
      }
      .subscription-dialog {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 8px;
        box-shadow: 0 24px 70px rgba(15, 23, 42, 0.26);
        max-height: min(88vh, 760px);
        max-width: 680px;
        overflow-y: auto;
        padding: 34px;
        position: relative;
        width: min(100%, 680px);
      }
      .subscription-dialog h2 {
        margin: 0 0 8px;
      }
      .subscription-dialog p {
        color: var(--muted);
      }
      .subscription-price {
        color: var(--ink);
        font-size: 20px;
        font-weight: 800;
      }
      .subscription-benefits {
        color: var(--ink);
        display: grid;
        gap: 10px 20px;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        list-style-position: outside;
        margin: 18px 0;
        padding-left: 20px;
      }
      .subscription-benefits li {
        padding-right: 6px;
      }
      .modal-close {
        background: #f8fafc;
        border: 1px solid var(--line);
        border-radius: 999px;
        color: var(--muted);
        height: 32px;
        padding: 0;
        position: absolute;
        right: 14px;
        top: 14px;
        width: 32px;
      }
      .processing-spinner {
        animation: spin 0.85s linear infinite;
        border: 4px solid #d8dee8;
        border-top-color: var(--accent);
        border-radius: 999px;
        height: 44px;
        margin: 0 auto 16px;
        width: 44px;
      }
      .processing-dialog h2 {
        margin: 0 0 8px;
      }
      .processing-dialog p {
        color: var(--muted);
        margin: 0;
      }
      @keyframes spin {
        to {
          transform: rotate(360deg);
        }
      }
      @media (max-width: 760px) {
        .metric-row {
          grid-template-columns: 1fr;
        }
        .recommendation-grid {
          grid-template-columns: 1fr;
        }
        .auth-grid {
          grid-template-columns: 1fr;
        }
        .subscription-banner {
          align-items: flex-start;
          flex-direction: column;
        }
        .subscription-benefits {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <header>
      <div class="header-content">
        <div class="brand">ModelMetrica</div>
        {% if is_authenticated %}
          <div class="user-actions">
            <span>{{ current_username }}</span>
            <span class="subscription-status {{ 'active' if has_subscription else '' }}">{{ 'Pro active' if has_subscription else 'Free plan' }}</span>
            <a class="logout-link" href="{{ url_for('logout') }}">Log out</a>
          </div>
        {% endif %}
      </div>
    </header>
    {% if is_authenticated %}
    <main>
      <h1>Modeling workspace</h1>
      <p>Upload a CSV or Excel file, inspect the first rows, then run classification or regression analysis.</p>
      <div class="subscription-banner">
        <div>
          <strong>Subscription status</strong>
          <p>{{ 'Your Pro subscription is active. Pro classification and Pro regression are unlocked.' if has_subscription else 'You are on the free plan. Pro classification and Pro regression require a subscription.' }}</p>
        </div>
        <span class="subscription-status {{ 'active' if has_subscription else '' }}">{{ 'Pro active' if has_subscription else 'Free plan' }}</span>
      </div>

      <nav class="tabs">
        <a class="tab {{ 'active' if active_tab == 'data' else '' }}" href="#data">Data</a>
        <a class="tab {{ 'active' if active_tab == 'classification' else '' }}" href="#classification">Classification</a>
        <a class="tab {{ 'active' if active_tab == 'regression' else '' }}" href="#regression">Regression</a>
        <a class="tab {{ 'active' if active_tab == 'pro_classification' else '' }}" href="#pro_classification">Pro classification</a>
        <a class="tab {{ 'active' if active_tab == 'pro_regression' else '' }}" href="#pro_regression">Pro regression</a>
      </nav>

      <section id="data" class="tab-panel {{ 'active' if active_tab == 'data' else '' }}">
        <div class="panel data-actions">
          <form method="post" enctype="multipart/form-data">
            <input type="hidden" name="form_name" value="upload">
            <input type="hidden" name="active_tab" value="data">
            <div>
              <label for="data_file">Dataset</label>
              <div class="form-row">
                <input id="data_file" type="file" name="data_file" accept=".csv,.xls,.xlsx" required>
                <button type="submit">Upload and preview</button>
              </div>
            </div>
          </form>
          <form method="post">
            <input type="hidden" name="form_name" value="test_data">
            <input type="hidden" name="active_tab" value="data">
            <div>
              <label>Test data</label>
              <div class="form-row">
                <button type="submit">Load test data</button>
                <p>Simulates 1,000 rows with targets for classification and regression.</p>
              </div>
            </div>
          </form>
          {% if data_error %}
            <p class="error">{{ data_error }}</p>
          {% endif %}
        </div>

        {% if table_html %}
          <div class="panel">
            <h2>{{ filename }}</h2>
            <p>Showing the first {{ row_count }} rows from {{ total_rows }} total rows.</p>
            <div class="table-wrap">
              {{ table_html|safe }}
            </div>
          </div>
        {% endif %}
      </section>

      {{ render_classification_tab(classification_tab, active_tab, has_data, columns) }}
      {{ render_regression_tab(regression_tab, active_tab, has_data, columns) }}
      {{ render_classification_tab(pro_classification_tab, active_tab, has_data, columns) }}
      {{ render_regression_tab(pro_regression_tab, active_tab, has_data, columns) }}
    </main>
    {% endif %}
    {% if not is_authenticated %}
      <div class="auth-overlay" role="dialog" aria-modal="true" aria-labelledby="auth-title">
        <div class="auth-card">
          <h2 id="auth-title">Welcome to ModelMetrica</h2>
          <p>Log in or create an account to access the modeling workspace.</p>
          {% if auth_error %}
            <p class="error">{{ auth_error }}</p>
          {% endif %}
          <div class="auth-grid">
            <form method="post">
              <input type="hidden" name="form_name" value="login">
              <h3>Log in</h3>
              <div>
                <label for="login_username">Username</label>
                <input id="login_username" name="username" autocomplete="username" required>
              </div>
              <div>
                <label for="login_password">Password</label>
                <input id="login_password" name="password" type="password" autocomplete="current-password" required>
              </div>
              <button type="submit">Log in</button>
            </form>
            <form method="post">
              <input type="hidden" name="form_name" value="signup">
              <h3>Sign up</h3>
              <div>
                <label for="signup_username">Username</label>
                <input id="signup_username" name="username" autocomplete="username" required>
              </div>
              <div>
                <label for="signup_password">Password</label>
                <input id="signup_password" name="password" type="password" autocomplete="new-password" minlength="6" required>
              </div>
              <button type="submit">Create account</button>
            </form>
          </div>
        </div>
      </div>
    {% endif %}
    <div id="processing-overlay" class="processing-overlay" role="alertdialog" aria-modal="true" aria-labelledby="processing-title" aria-describedby="processing-description">
      <div class="processing-dialog">
        <div class="processing-spinner" aria-hidden="true"></div>
        <h2 id="processing-title">Processing</h2>
        <p id="processing-description">Running the model. This may take a moment.</p>
      </div>
    </div>
    <div id="subscription-overlay" class="subscription-overlay" role="dialog" aria-modal="true" aria-labelledby="subscription-title" aria-describedby="subscription-description">
      <div class="subscription-dialog">
        <button type="button" class="modal-close" data-close-subscription aria-label="Close subscription popup">x</button>
        <h2 id="subscription-title">Pro subscription required</h2>
        <p id="subscription-description">Pro classification and Pro regression require an active subscription.</p>
        <ul class="subscription-benefits">
          <li>Compare multiple models side by side.</li>
          <li>Select any fitted model for detailed drill-down.</li>
          <li>Use grid or random hyperparameter tuning.</li>
          <li>See default vs tuned model performance.</li>
          <li>Control missing values, encoding, scaling, seeds, and outliers.</li>
          <li>Review fold-by-fold cross-validation diagnostics.</li>
          <li>Inspect ROC and precision-recall curves for classification.</li>
          <li>Tune classification decision thresholds interactively.</li>
          <li>Analyze predicted vs actual and residual plots for regression.</li>
          <li>View coefficients and feature importance when available.</li>
          <li>Get guided recommendations with evidence and concerns.</li>
          <li>Keep recent Pro run history across refreshes.</li>
          <li>Export durable HTML and PDF Pro reports.</li>
        </ul>
        <p class="subscription-price">{{ subscription_price }}</p>
        {% if mollie_configured %}
          <form method="post" action="{{ url_for('start_subscription') }}">
            <button type="submit">Subscribe with Mollie</button>
          </form>
        {% else %}
          <p class="error">Mollie payments are not configured yet. Set MOLLIE_API_KEY and a public MOLLIE_BASE_URL to enable checkout.</p>
        {% endif %}
      </div>
    </div>
    <script>
      const tabs = document.querySelectorAll(".tab");
      const panels = document.querySelectorAll(".tab-panel");
      const processingOverlay = document.getElementById("processing-overlay");
      const subscriptionOverlay = document.getElementById("subscription-overlay");
      const hasSubscription = {{ 'true' if has_subscription else 'false' }};

      function showProcessingOverlay() {
        if (processingOverlay) {
          processingOverlay.classList.add("active");
        }
      }

      function showSubscriptionOverlay() {
        if (subscriptionOverlay) {
          subscriptionOverlay.classList.add("active");
        }
      }

      function hideSubscriptionOverlay() {
        if (subscriptionOverlay) {
          subscriptionOverlay.classList.remove("active");
        }
      }

      function activateTab(hash) {
        const validTabs = ["#classification", "#regression", "#pro_classification", "#pro_regression"];
        const target = validTabs.includes(hash) ? hash.slice(1) : "data";
        tabs.forEach((tab) => tab.classList.toggle("active", tab.getAttribute("href") === `#${target}`));
        panels.forEach((panel) => panel.classList.toggle("active", panel.id === target));
      }

      tabs.forEach((tab) => {
        tab.addEventListener("click", (event) => {
          event.preventDefault();
          history.replaceState(null, "", tab.getAttribute("href"));
          activateTab(tab.getAttribute("href"));
        });
      });

      document.querySelectorAll(".model-comparison tbody tr[data-model-name]").forEach((row) => {
        row.addEventListener("click", () => {
          const panel = row.closest(".tab-panel");
          if (!panel) {
            return;
          }
          const detailSelect = panel.querySelector("select[name$='_detail_model']");
          const form = panel.querySelector("form");
          if (!detailSelect || !form) {
            return;
          }
          detailSelect.value = row.dataset.modelName;
          form.requestSubmit ? form.requestSubmit() : form.submit();
        });
      });

      document.querySelectorAll("form[data-run-form]").forEach((form) => {
        form.addEventListener("submit", (event) => {
          if (!form.checkValidity()) {
            return;
          }
          if (form.hasAttribute("data-pro-run-form") && !hasSubscription) {
            event.preventDefault();
            showSubscriptionOverlay();
            return;
          }
          showProcessingOverlay();
        });
      });

      document.querySelectorAll("[data-close-subscription]").forEach((button) => {
        button.addEventListener("click", hideSubscriptionOverlay);
      });

      if (subscriptionOverlay) {
        subscriptionOverlay.addEventListener("click", (event) => {
          if (event.target === subscriptionOverlay) {
            hideSubscriptionOverlay();
          }
        });
      }

      document.querySelectorAll("[data-threshold-input]").forEach((input) => {
        const output = input.closest(".threshold-control")?.querySelector("[data-threshold-output]");
        const update = () => {
          if (output) {
            output.value = Number(input.value).toFixed(3);
            output.textContent = Number(input.value).toFixed(3);
          }
        };
        input.addEventListener("input", update);
        update();
      });

      if (window.location.hash) {
        activateTab(window.location.hash);
      }

      window.addEventListener("pageshow", () => {
        if (processingOverlay) {
          processingOverlay.classList.remove("active");
        }
      });
    </script>
  </body>
</html>
"""


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def get_db_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def ensure_column(connection, table_name, column_name, definition):
    columns = {row["name"] for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in columns:
        connection.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def init_auth_db():
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        ensure_column(connection, "users", "subscription_status", "TEXT NOT NULL DEFAULT 'inactive'")
        ensure_column(connection, "users", "mollie_customer_id", "TEXT")
        ensure_column(connection, "users", "mollie_subscription_id", "TEXT")
        ensure_column(connection, "users", "subscription_updated_at", "TEXT")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                csv_data TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS pro_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                dataset_id TEXT NOT NULL,
                tab_name TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (dataset_id) REFERENCES datasets(id)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_pro_runs_lookup ON pro_runs (user_id, dataset_id, tab_name, id DESC)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS subscription_payments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                mollie_payment_id TEXT NOT NULL UNIQUE,
                mollie_customer_id TEXT,
                mollie_subscription_id TEXT,
                status TEXT NOT NULL DEFAULT 'open',
                checkout_url TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
            """
        )


def find_user(username):
    with get_db_connection() as connection:
        return connection.execute(
            "SELECT id, username, password_hash FROM users WHERE lower(username) = lower(?)",
            (username,),
        ).fetchone()


def create_user(username, password):
    username = username.strip()
    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters.")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters.")
    if find_user(username):
        raise ValueError("That username is already taken.")

    password_hash = generate_password_hash(password)
    with get_db_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (username, password_hash),
        )
        return cursor.lastrowid


def authenticate_user(username, password):
    user = find_user(username.strip())
    if user is None or not check_password_hash(user["password_hash"], password):
        return None
    return user


def log_user_in(user_id, username):
    session["user_id"] = user_id
    session["username"] = username


def is_user_authenticated():
    return bool(session.get("user_id"))


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


def current_user_id():
    return session.get("user_id")


def current_user():
    user_id = current_user_id()
    if not user_id:
        return None
    with get_db_connection() as connection:
        return connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()


def user_has_subscription(user=None):
    user = user or current_user()
    return bool(user and user["subscription_status"] == "active")


def mollie_configured():
    return bool(MOLLIE_API_KEY) and mollie_api_key_valid() and mollie_webhook_url_valid()


def mollie_api_key_valid():
    return MOLLIE_API_KEY.startswith(("test_", "live_"))


def mollie_webhook_url_valid():
    webhook_url = external_url_for("mollie_webhook")
    parsed = urlparse(webhook_url)
    return parsed.scheme in {"http", "https"} and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}


def external_url_for(endpoint, **values):
    if MOLLIE_BASE_URL:
        return urljoin(MOLLIE_BASE_URL.rstrip("/") + "/", url_for(endpoint, **values).lstrip("/"))
    return url_for(endpoint, _external=True, **values)


def mollie_request(method, path, payload=None):
    if not MOLLIE_API_KEY:
        raise RuntimeError("Mollie API key is not configured.")
    if not mollie_api_key_valid():
        raise RuntimeError("MOLLIE_API_KEY must be a Mollie profile API key that starts with test_ or live_.")
    if not mollie_webhook_url_valid():
        raise RuntimeError("MOLLIE_BASE_URL must be a public URL that Mollie can reach for webhooks, for example an ngrok HTTPS URL.")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_obj = Request(
        f"{MOLLIE_API_BASE}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {MOLLIE_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urlopen(request_obj, timeout=20) as response:
            body = response.read().decode("utf-8")
            return json.loads(body) if body else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Mollie API error {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Mollie: {exc.reason}") from exc


def ensure_mollie_customer(user):
    if user["mollie_customer_id"]:
        return user["mollie_customer_id"]
    customer = mollie_request(
        "POST",
        "/customers",
        {
            "name": user["username"],
            "metadata": {"user_id": user["id"]},
        },
    )
    customer_id = customer["id"]
    with get_db_connection() as connection:
        connection.execute("UPDATE users SET mollie_customer_id = ? WHERE id = ?", (customer_id, user["id"]))
    return customer_id


def create_mollie_first_payment(user):
    customer_id = ensure_mollie_customer(user)
    payment = mollie_request(
        "POST",
        "/payments",
        {
            "amount": {"currency": SUBSCRIPTION_CURRENCY, "value": SUBSCRIPTION_AMOUNT},
            "customerId": customer_id,
            "sequenceType": "first",
            "description": SUBSCRIPTION_DESCRIPTION,
            "redirectUrl": external_url_for("subscription_return"),
            "webhookUrl": external_url_for("mollie_webhook"),
            "metadata": {"user_id": user["id"]},
        },
    )
    checkout_url = (payment.get("_links") or {}).get("checkout", {}).get("href")
    if not checkout_url:
        raise RuntimeError("Mollie did not return a checkout URL.")
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT OR REPLACE INTO subscription_payments
            (user_id, mollie_payment_id, mollie_customer_id, status, checkout_url, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            (user["id"], payment["id"], customer_id, payment.get("status", "open"), checkout_url),
        )
    return checkout_url


def create_mollie_subscription(user_id, customer_id):
    with get_db_connection() as connection:
        user = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if user and user["mollie_subscription_id"]:
        return user["mollie_subscription_id"]
    subscription = mollie_request(
        "POST",
        f"/customers/{customer_id}/subscriptions",
        {
            "amount": {"currency": SUBSCRIPTION_CURRENCY, "value": SUBSCRIPTION_AMOUNT},
            "interval": SUBSCRIPTION_INTERVAL,
            "description": SUBSCRIPTION_DESCRIPTION,
            "webhookUrl": external_url_for("mollie_webhook"),
            "metadata": {"user_id": user_id},
        },
    )
    subscription_id = subscription["id"]
    with get_db_connection() as connection:
        connection.execute(
            """
            UPDATE users
            SET subscription_status = 'active',
                mollie_subscription_id = ?,
                subscription_updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
            """,
            (subscription_id, user_id),
        )
    return subscription_id


def sync_mollie_payment(payment_id):
    payment = mollie_request("GET", f"/payments/{payment_id}")
    with get_db_connection() as connection:
        row = connection.execute(
            "SELECT * FROM subscription_payments WHERE mollie_payment_id = ?",
            (payment_id,),
        ).fetchone()
        if not row:
            return payment
        status = payment.get("status", "unknown")
        customer_id = payment.get("customerId") or row["mollie_customer_id"]
        subscription_id = row["mollie_subscription_id"]
        if status == "paid" and customer_id and not subscription_id:
            subscription_id = create_mollie_subscription(row["user_id"], customer_id)
        connection.execute(
            """
            UPDATE subscription_payments
            SET status = ?,
                mollie_customer_id = ?,
                mollie_subscription_id = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE mollie_payment_id = ?
            """,
            (status, customer_id, subscription_id, payment_id),
        )
        if status in {"failed", "canceled", "expired"}:
            connection.execute(
                """
                UPDATE users
                SET subscription_status = 'inactive',
                    subscription_updated_at = CURRENT_TIMESTAMP
                WHERE id = ? AND mollie_subscription_id IS NULL
                """,
                (row["user_id"],),
            )
    return payment


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
    return {
        "missing_values": tab.get("selected_missing_values", "drop"),
        "categorical_encoding": tab.get("selected_categorical_encoding", "one_hot_drop_first"),
        "scaling": tab.get("selected_scaling", "on"),
        "split_seed": tab.get("selected_split_seed", 42),
        "outlier_handling": tab.get("selected_outlier_handling", "none"),
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
    return int(options.get("split_seed", options.get("selected_split_seed", 42)))


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


def encode_predictors(model_data, predictors, options):
    x_raw = model_data[predictors]
    if categorical_encoding_mode(options) == "ordinal":
        encoded = pd.DataFrame(index=x_raw.index)
        for column in predictors:
            if pd.api.types.is_numeric_dtype(x_raw[column]):
                encoded[column] = pd.to_numeric(x_raw[column], errors="coerce")
            else:
                encoded[column] = pd.Categorical(x_raw[column]).codes.astype(float)
        return encoded.astype(float)

    drop_first = categorical_encoding_mode(options) == "one_hot_drop_first"
    return pd.get_dummies(x_raw, drop_first=drop_first, dtype=float)


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
    return min(max(test_size, 0.1), 0.5)


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


def parse_split_seed(value):
    try:
        seed = int(value)
    except (TypeError, ValueError):
        return 42
    return min(max(seed, 0), 999999)

def parse_tuning_iterations(value):
    try:
        iterations = int(value)
    except (TypeError, ValueError):
        return 10
    return min(max(iterations, 3), 30)

def dataset_id():
    return session.get("dataset_id")


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


def classification_estimator(model_name, options=None):
    if model_name == "logistic":
        model = LogisticRegression(
            C=tuned_float(options, "logistic", "C", 1.0),
            max_iter=1000,
            random_state=preprocessing_seed(options),
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    if model_name == "tree":
        return DecisionTreeClassifier(
            max_depth=tuned_param(options, "tree", "max_depth", 4),
            min_samples_leaf=tuned_int(options, "tree", "min_samples_leaf", 1),
            random_state=preprocessing_seed(options),
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=tuned_int(options, "random_forest", "n_estimators", 200),
            max_depth=tuned_param(options, "random_forest", "max_depth", None),
            min_samples_leaf=tuned_int(options, "random_forest", "min_samples_leaf", 1),
            random_state=preprocessing_seed(options),
            n_jobs=-1,
        )
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=tuned_int(options, "gradient_boosting", "n_estimators", 100),
            learning_rate=tuned_float(options, "gradient_boosting", "learning_rate", 0.1),
            max_depth=tuned_int(options, "gradient_boosting", "max_depth", 3),
            random_state=preprocessing_seed(options),
        )
    if model_name == "svm":
        model = SVC(
            kernel="rbf",
            C=tuned_float(options, "svm", "C", 1.0),
            gamma=tuned_param(options, "svm", "gamma", "scale"),
            probability=True,
            random_state=preprocessing_seed(options),
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
    if model_name == "knn":
        model = KNeighborsClassifier(
            n_neighbors=tuned_int(options, "knn", "n_neighbors", 5),
            weights=tuned_param(options, "knn", "weights", "distance"),
        )
        return make_pipeline(StandardScaler(), model) if scaling_enabled(options) else model
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
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=tuned_int(options, "random_forest", "n_estimators", 200),
            max_depth=tuned_param(options, "random_forest", "max_depth", None),
            min_samples_leaf=tuned_int(options, "random_forest", "min_samples_leaf", 1),
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


def fit_logistic_regression(data, target, predictors, test_size, cv_folds, options=None):
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


CLASSIFICATION_MODEL_FITTERS = {
    "logistic": fit_logistic_regression,
    "tree": fit_tree_model,
    "random_forest": fit_random_forest_model,
    "gradient_boosting": fit_gradient_boosting_model,
    "svm": fit_svm_model,
    "knn": fit_knn_model,
}

REGRESSION_MODEL_FITTERS = {
    "linear": fit_linear_regression,
    "ridge": fit_ridge_regression,
    "lasso": fit_lasso_regression,
    "random_forest": fit_random_forest_regression,
    "gradient_boosting": fit_gradient_boosting_regression,
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
            "selected_split_seed": 42,
            "selected_outlier_handling": "none",
            "selected_tuning_mode": "off",
            "selected_tuning_iterations": 10,
            "selected_threshold": 0.5,
            "threshold_field": config.get("threshold_field"),
            "selected_target": None,
            "selected_predictors": [],
            "error": None,
            "output": None,
            "comparison_html": None,
            "detail_metric_comparison_html": None,
            "recommendation": None,
            "comparison_download": None,
            "report_download": None,
            "report_pdf_download": None,
            "run_history": [],
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
        selected_models = request.form.getlist(tab["model_field"]) or [tab["default_model"]]
        allowed_models = available_model_names(tab)
        tab["selected_models"] = [model for model in selected_models if model in allowed_models]
        if not tab["selected_models"]:
            tab["selected_models"] = [tab["default_model"]]
        tab["selected_model"] = tab["selected_models"][0]
        selected_detail_model = request.form.get(tab.get("detail_model_field", ""), "best")
        tab["selected_detail_model"] = selected_detail_model if selected_detail_model == "best" or selected_detail_model in allowed_models else "best"
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
    "gradient_boosting": "Gradient Boosting",
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
    if hasattr(estimator, "named_steps"):
        for name, step in estimator.named_steps.items():
            if isinstance(step, estimator_class):
                return f"{name}__"
    return ""


def classification_tuning_grid(model_name, estimator):
    if model_name == "logistic":
        prefix = estimator_step_name(estimator, LogisticRegression)
        return {f"{prefix}C": [0.1, 1.0, 10.0]}
    if model_name == "tree":
        return {"max_depth": [3, 4, 6, None], "min_samples_leaf": [1, 5, 10]}
    if model_name == "random_forest":
        return {"n_estimators": [100, 200], "max_depth": [None, 6, 10], "min_samples_leaf": [1, 5]}
    if model_name == "gradient_boosting":
        return {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [2, 3]}
    if model_name == "svm":
        prefix = estimator_step_name(estimator, SVC)
        return {f"{prefix}C": [0.5, 1.0, 2.0], f"{prefix}gamma": ["scale", "auto"]}
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
    if model_name == "random_forest":
        return {"n_estimators": [100, 200], "max_depth": [None, 6, 10], "min_samples_leaf": [1, 5]}
    if model_name == "gradient_boosting":
        return {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [2, 3]}
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
    _, target_values, _, x_encoded = prepare_classification_data(data, target, predictors, binary_only=(model_name == "logistic"), options=options)
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
    predictions = search.best_estimator_.predict(split["x_test"])
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
        predictions = estimator.predict(split["x_test"])
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
    for section in ["evidence", "concerns", "actions"]:
        section_label = section.capitalize()
        for item in recommendation.get(section, []):
            rows.append({"Section": section_label, "Item": item})
    return pd.DataFrame(rows)


def recommendation_html(recommendation):
    frame = recommendation_report_frame(recommendation)
    if frame.empty:
        return ""
    return display_table(frame, index=False, border=0, classes="report-table")


def build_classification_recommendation(tab, rows, best_model_name, detail_model_name, output):
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

    concerns = []
    if best_model_name and detail_model_name != best_model_name:
        concerns.append(f"The selected detail model is not the top-ranked model; {best_label} ranked best in the comparison.")
    if cv_sd is not None and cv_sd >= 0.06:
        concerns.append("CV accuracy varies noticeably across folds, so the result may be split-sensitive.")
    if tuned_accuracy is not None and default_accuracy is not None and tuned_accuracy <= default_accuracy + 0.001:
        concerns.append("Hyperparameter tuning did not materially improve test accuracy.")
    if precision is not None and recall is not None and abs(precision - recall) >= 0.15:
        higher = "precision" if precision > recall else "recall"
        concerns.append(f"The selected threshold is {higher}-heavy; review whether that matches the decision cost.")
    if specificity is not None and recall is not None and abs(specificity - recall) >= 0.2:
        concerns.append("Recall and specificity are far apart, suggesting asymmetric errors at the selected threshold.")
    if not concerns:
        concerns.append("No major stability or threshold concerns are visible from the current diagnostics.")

    actions = []
    if best_model_name and detail_model_name != best_model_name:
        actions.append(f"Inspect {best_label} as the detail model before finalizing.")
    if cv_sd is not None and cv_sd >= 0.06:
        actions.append("Try more folds, more data, or simpler models to confirm stability.")
    if precision is not None and recall is not None:
        actions.append("Adjust the decision threshold to balance precision and recall for the use case.")
    if not tuning_enabled(tab):
        actions.append("Enable hyperparameter tuning for the strongest candidate models.")
    actions.append("Review feature importance or coefficients for plausibility before sharing the report.")

    title = f"Recommend {best_label}"
    if detail_model_name != best_model_name:
        summary = f"{best_label} is recommended by the comparison ranking; the panel below currently explains selected detail model {detail_label}."
    else:
        summary = f"{detail_label} is the strongest current candidate based on the comparison ranking and selected-threshold diagnostics."

    return {"title": title, "summary": summary, "evidence": evidence, "concerns": concerns, "actions": actions}


def build_regression_recommendation(tab, rows, best_model_name, detail_model_name, output):
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

    concerns = []
    if best_model_name and detail_model_name != best_model_name:
        concerns.append(f"The selected detail model is not the top-ranked model; {best_label} ranked best by RMSE.")
    if cv_rmse is not None and rmse is not None and cv_rmse > rmse * 1.25:
        concerns.append("CV RMSE is substantially higher than test RMSE, suggesting possible split optimism.")
    if cv_rmse_sd is not None and cv_rmse is not None and cv_rmse_sd > cv_rmse * 0.2:
        concerns.append("CV RMSE varies meaningfully across folds.")
    if residual_mean is not None and residual_sd is not None and residual_sd > 0 and abs(residual_mean) > residual_sd * 0.1:
        concerns.append("Residual mean is not close to zero relative to residual spread, suggesting possible bias.")
    if tuned_rmse is not None and default_rmse is not None and tuned_rmse >= default_rmse - 0.001:
        concerns.append("Hyperparameter tuning did not materially reduce RMSE.")
    if cv_r_squared is not None and cv_r_squared < 0:
        concerns.append("CV R squared is below zero, so the model may generalize poorly.")
    if not concerns:
        concerns.append("No major residual or cross-validation concerns are visible from the current diagnostics.")

    actions = []
    if best_model_name and detail_model_name != best_model_name:
        actions.append(f"Inspect {best_label} as the detail model before finalizing.")
    if cv_rmse_sd is not None and cv_rmse is not None and cv_rmse_sd > cv_rmse * 0.2:
        actions.append("Compare simpler models or add data to reduce fold-to-fold variability.")
    if residual_mean is not None and residual_sd is not None and residual_sd > 0 and abs(residual_mean) > residual_sd * 0.1:
        actions.append("Inspect residual plots for systematic under- or over-prediction.")
    if not tuning_enabled(tab):
        actions.append("Enable hyperparameter tuning for regularized and tree-based candidates.")
    actions.append("Review feature importance or coefficients for domain plausibility.")

    title = f"Recommend {best_label}"
    if detail_model_name != best_model_name:
        summary = f"{best_label} is recommended by the comparison ranking; the panel below currently explains selected detail model {detail_label}."
    else:
        summary = f"{detail_label} is the strongest current candidate based on RMSE ranking, CV stability, and residual diagnostics."

    return {"title": title, "summary": summary, "evidence": evidence, "concerns": concerns, "actions": actions}



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

def handle_classification_comparison_submission(tab, dataset):
    options = preprocessing_options(tab)
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

    if not rows:
        tab["error"] = "Select at least one model."
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
        tab["output"] = register_downloads(tab["form_name"], detail_output)
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
        tab["recommendation"] = build_classification_recommendation(tab, rows, best_model_name, detail_model_name, tab["output"])
    else:
        tab["error"] = "No selected model could be fit."
        tab["detail_metric_comparison_html"] = None
        tab["recommendation"] = None

    tab["comparison_html"], comparison = comparison_html(rows, ["Model", "Default accuracy", "Default CV accuracy", "CV accuracy SD", "CV accuracy min", "CV accuracy max", "Tuned accuracy", "Tuned CV accuracy", "Precision", "Recall", "F1", "Best params", "Status"])
    tab["comparison_download"] = register_comparison_download(tab, comparison)

PRO_TAB_NAMES = {"pro_classification", "pro_regression"}


def model_label_for_run(tab, model_name):
    if tab["form_name"] in REGRESSION_TAB_CONFIGS:
        return REGRESSION_MODEL_LABELS.get(model_name, model_name)
    return CLASSIFICATION_MODEL_LABELS.get(model_name, model_name)


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


def run_history_entry(tab):
    detail_model = tab.get("selected_model", tab["default_model"])
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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
            SELECT snapshot_json
            FROM pro_runs
            WHERE user_id = ? AND dataset_id = ? AND tab_name = ?
            ORDER BY id DESC
            LIMIT 5
            """,
            (user_id, current_id, tab_name),
        ).fetchall()
    for row in rows:
        try:
            runs.append(json.loads(row["snapshot_json"]))
        except json.JSONDecodeError:
            continue
    return runs


def update_run_history(tab, runs):
    tab["run_history"] = [run["history_entry"] for run in runs if run.get("history_entry")]


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
        "selected_tuning_mode": tab.get("selected_tuning_mode"),
        "selected_tuning_iterations": tab.get("selected_tuning_iterations"),
        "selected_threshold": tab.get("selected_threshold", 0.5),
        "selected_target": tab.get("selected_target"),
        "selected_predictors": list(tab.get("selected_predictors", [])),
        "comparison_html": tab.get("comparison_html"),
        "detail_metric_comparison_html": tab.get("detail_metric_comparison_html"),
        "recommendation": deepcopy(tab.get("recommendation")),
        "comparison_download": deepcopy(tab.get("comparison_download")),
        "output": deepcopy(tab.get("output")),
        "download_artifacts": tab_download_artifacts(tab),
        "history_entry": run_history_entry(tab),
    }
    with get_db_connection() as connection:
        connection.execute(
            """
            INSERT INTO pro_runs (user_id, dataset_id, tab_name, snapshot_json)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, current_id, tab["form_name"], json.dumps(snapshot, default=pro_run_json_default)),
        )
        prune_old_pro_runs(connection, user_id, current_id, tab["form_name"])
    update_run_history(tab, load_pro_runs_from_db(tab["form_name"]))


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
        "selected_tuning_mode",
        "selected_tuning_iterations",
        "selected_threshold",
        "selected_target",
        "selected_predictors",
        "comparison_html",
        "detail_metric_comparison_html",
        "recommendation",
        "comparison_download",
        "output",
    ]:
        tab[key] = deepcopy(snapshot.get(key))
    if tab.get("selected_threshold") is None:
        tab["selected_threshold"] = 0.5
    restore_download_artifacts(tab, snapshot)
    tab["report_download"] = pro_report_download(tab["form_name"])
    tab["report_pdf_download"] = pro_report_pdf_download(tab["form_name"])


def restore_pro_runs(model_tabs, exclude=None):
    exclude = exclude or set()
    for tab_name in PRO_TAB_NAMES:
        runs = load_pro_runs_from_db(tab_name)
        tab = model_tabs[tab_name]
        if tab_name not in exclude and runs:
            restore_pro_run(tab, runs[0])
        update_run_history(tab, runs)

def handle_classification_submission(tab, dataset):
    populate_tab_from_request(tab)

    if dataset is None:
        tab["error"] = "Upload a dataset on the Data tab before running classification."
    elif not tab["selected_target"] or not tab["selected_predictors"]:
        tab["error"] = "Select a target column and at least one predictor."
    elif tab.get("allow_model_comparison"):
        handle_classification_comparison_submission(tab, dataset)
    else:
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


REGRESSION_MODEL_LABELS = {
    "linear": "Linear Regression",
    "ridge": "Ridge Regression",
    "lasso": "Lasso Regression",
    "random_forest": "Random Forest Regression",
    "gradient_boosting": "Gradient Boosting Regression",
    "svr": "Support Vector Regression",
    "knn": "kNN Regression",
}


def handle_regression_comparison_submission(tab, dataset):
    options = preprocessing_options(tab)
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

    if not rows:
        tab["error"] = "Select at least one model."
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
        tab["output"] = register_downloads(tab["form_name"], detail_output)
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
        tab["recommendation"] = build_regression_recommendation(tab, rows, best_model_name, detail_model_name, tab["output"])
    else:
        tab["error"] = "No selected model could be fit."
        tab["detail_metric_comparison_html"] = None
        tab["recommendation"] = None

    tab["comparison_html"], comparison = comparison_html(
        rows,
        ["Model", "Test R squared", "Default RMSE", "Default CV RMSE", "CV RMSE SD", "CV RMSE min", "CV RMSE max", "Tuned RMSE", "Tuned CV RMSE", "Test MAE", "CV R squared", "Best params", "Status"],
    )
    tab["comparison_download"] = register_comparison_download(tab, comparison)

def handle_regression_submission(tab, dataset):
    populate_tab_from_request(tab)

    if dataset is None:
        tab["error"] = "Upload a dataset on the Data tab before running regression."
    elif not tab["selected_target"] or not tab["selected_predictors"]:
        tab["error"] = "Select a target column and at least one predictor."
    elif tab.get("allow_model_comparison"):
        handle_regression_comparison_submission(tab, dataset)
    else:
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

@app.route("/", methods=["GET", "POST"])
def index():
    active_tab = request.form.get("active_tab", "data")
    auth_error = None
    data_error = None
    form_name = request.form.get("form_name")
    model_tabs = make_model_tabs()

    if request.method == "POST" and form_name == "signup":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        try:
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
            auth_error = "Username or password is incorrect."
        else:
            log_user_in(user["id"], user["username"])
            return redirect(url_for("index"))

    authenticated = is_user_authenticated()
    user = current_user() if authenticated else None
    has_subscription = user_has_subscription(user)

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

    return render_template_string(
        PAGE_TEMPLATE,
        active_tab=active_tab,
        auth_error=auth_error,
        is_authenticated=authenticated,
        current_username=session.get("username"),
        has_subscription=has_subscription,
        mollie_configured=mollie_configured(),
        subscription_price=f"{SUBSCRIPTION_CURRENCY} {SUBSCRIPTION_AMOUNT} / {SUBSCRIPTION_INTERVAL}",
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


def latest_pro_run_snapshot(tab_name):
    runs = load_pro_runs_from_db(tab_name)
    return runs[0] if runs else None


def csv_report_table(csv_data, css_class="report-table"):
    if not csv_data:
        return ""
    try:
        return display_table(pd.read_csv(StringIO(csv_data)), index=False, border=0, classes=css_class)
    except Exception:
        return f"<pre>{html.escape(csv_data)}</pre>"


def metrics_report_table(metrics):
    if not metrics:
        return ""
    rows = [{"Metric": item.get("label", ""), "Value": item.get("value", "")} for item in metrics]
    return display_table(pd.DataFrame(rows), index=False, border=0, classes="report-table")


def report_model_labels(tab_name, models):
    tab = {"form_name": tab_name}
    return ", ".join(model_label_for_run(tab, model) for model in models)


def report_metadata_rows(tab_name, snapshot, dataset):
    data = dataset.get("data") if dataset else None
    history_entry = snapshot.get("history_entry") or {}
    detail_model = snapshot.get("selected_model") or snapshot.get("selected_detail_model") or "-"
    cv_folds = snapshot.get("selected_cv_folds")
    return [
        {"Field": "Generated", "Value": datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
        {"Field": "Run time", "Value": history_entry.get("timestamp", "-")},
        {"Field": "Report type", "Value": "Pro classification" if tab_name == "pro_classification" else "Pro regression"},
        {"Field": "Dataset", "Value": dataset.get("filename", "-") if dataset else "-"},
        {"Field": "Rows", "Value": len(data) if data is not None else "-"},
        {"Field": "Columns", "Value": len(data.columns) if data is not None else "-"},
        {"Field": "Target", "Value": snapshot.get("selected_target") or "-"},
        {"Field": "Predictors", "Value": ", ".join(snapshot.get("selected_predictors") or []) or "-"},
        {"Field": "Models compared", "Value": report_model_labels(tab_name, snapshot.get("selected_models") or []) or "-"},
        {"Field": "Detail model", "Value": model_label_for_run({"form_name": tab_name}, detail_model)},
        {"Field": "Test split", "Value": f"{float(snapshot.get('selected_test_size') or 0):.0%}"},
        {"Field": "Cross-validation", "Value": f"{cv_folds} folds" if cv_folds else "Off"},
        {"Field": "Missing values", "Value": preprocessing_details(snapshot).get("Missing values", "-")},
        {"Field": "Categorical encoding", "Value": preprocessing_details(snapshot).get("Categorical encoding", "-")},
        {"Field": "Feature scaling", "Value": preprocessing_details(snapshot).get("Feature scaling", "-")},
        {"Field": "Split seed", "Value": snapshot.get("selected_split_seed", 42)},
        {"Field": "Outlier handling", "Value": preprocessing_details(snapshot).get("Outlier handling", "-")},
        {"Field": "Hyperparameter tuning", "Value": {"off": "Off", "grid": "Grid search", "random": "Random search"}.get(snapshot.get("selected_tuning_mode", "off"), "Off")},
        {"Field": "Random search iterations", "Value": snapshot.get("selected_tuning_iterations", 10)},
        {"Field": "Decision threshold", "Value": f"{float(snapshot.get('selected_threshold') or 0.5):.3f}" if tab_name == "pro_classification" else "-"},
    ]


def report_metadata_table(tab_name, snapshot, dataset):
    rows = report_metadata_rows(tab_name, snapshot, dataset)
    return display_table(pd.DataFrame(rows), index=False, border=0, classes="report-table")


def report_image(title, image_data, alt_text):
    if not image_data:
        return ""
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f"<img class=\"report-plot\" src=\"data:image/png;base64,{html.escape(image_data)}\" "
        f"alt=\"{html.escape(alt_text)}\"></section>"
    )


def pro_report_html(tab_name, snapshot, dataset):
    output = snapshot.get("output") or {}
    artifacts = snapshot.get("download_artifacts") or {}
    recommendation = snapshot.get("recommendation")
    report_title = "Pro Classification Report" if tab_name == "pro_classification" else "Pro Regression Report"
    selected_title = output.get("title") or "Selected model results"
    selected_description = output.get("description") or ""
    sections = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        f"<title>{html.escape(report_title)}</title>",
        """
        <style>
          body { color: #1f2933; font-family: Arial, sans-serif; line-height: 1.45; margin: 32px; }
          h1 { font-size: 28px; margin-bottom: 6px; }
          h2 { border-bottom: 1px solid #d8dee8; font-size: 20px; margin-top: 30px; padding-bottom: 6px; }
          h3 { font-size: 16px; margin-top: 22px; }
          .muted { color: #64748b; }
          .report-table { border-collapse: collapse; margin: 10px 0 18px; width: 100%; }
          .report-table th, .report-table td { border: 1px solid #d8dee8; padding: 7px 9px; text-align: left; vertical-align: top; }
          .report-table th { background: #f5f7fa; }
          .report-plot { border: 1px solid #d8dee8; max-width: 100%; }
          pre { background: #f5f7fa; border: 1px solid #d8dee8; overflow: auto; padding: 12px; }
        </style></head><body>
        """,
        f"<h1>{html.escape(report_title)}</h1>",
        f"<p class=\"muted\">Single-run export for the selected Pro model comparison.</p>",
        "<section><h2>Dataset metadata</h2>",
        report_metadata_table(tab_name, snapshot, dataset),
        "</section>",
    ]

    comparison_table = csv_report_table(artifacts.get("model_comparison"))
    if comparison_table:
        sections.extend(["<section><h2>Model comparison</h2>", comparison_table, "</section>"])

    if recommendation:
        sections.extend(["<section><h2>Model recommendation</h2>", recommendation_html(recommendation), "</section>"])

    sections.extend([
        "<section><h2>Selected detail model</h2>",
        f"<h3>{html.escape(selected_title)}</h3>",
        f"<p>{html.escape(selected_description)}</p>",
    ])
    metric_table = metrics_report_table(output.get("metrics") or [])
    if metric_table:
        sections.extend(["<h3>Metrics</h3>", metric_table])
    cv_summary_table = csv_report_table(artifacts.get("cv_summary"))
    cv_diagnostics_table = csv_report_table(artifacts.get("cv_diagnostics"))
    if cv_summary_table or cv_diagnostics_table or output.get("cv_plot"):
        sections.extend(["<h3>Cross-validation diagnostics</h3>"])
        if output.get("cv_plot"):
            sections.append(report_image("Cross-validation fold scores", output.get("cv_plot"), "Cross-validation fold score plot"))
        if cv_summary_table:
            sections.extend(["<h4>Summary</h4>", cv_summary_table])
        if cv_diagnostics_table:
            sections.extend(["<h4>Fold scores</h4>", cv_diagnostics_table])
    for heading, key in [
        ("Coefficients", "coefficients_html"),
        ("Variable importance", "importances_html"),
        ("Model details", "details_html"),
        ("Confusion matrix", "confusion_html"),
    ]:
        if output.get(key):
            sections.extend([f"<h3>{heading}</h3>", output[key]])
    sections.append("</section>")

    if tab_name == "pro_classification":
        sections.extend(
            [
                report_image("ROC curve", output.get("roc_plot"), "ROC curve"),
                report_image("Precision-recall curve", output.get("pr_plot"), "Precision-recall curve"),
                report_image("Classification tree", output.get("tree_plot"), "Classification tree"),
            ]
        )
        selected_threshold_table = csv_report_table(artifacts.get("selected_threshold_metrics"))
        if selected_threshold_table:
            sections.extend(["<section><h2>Selected threshold metrics</h2>", selected_threshold_table, "</section>"])
        threshold_table = csv_report_table(artifacts.get("threshold_analysis"))
        if threshold_table:
            sections.extend(["<section><h2>Threshold analysis</h2>", threshold_table, "</section>"])
    else:
        sections.extend(
            [
                report_image("Predicted vs actual", output.get("predicted_actual_plot"), "Predicted vs actual plot"),
                report_image("Residuals vs fitted", output.get("residuals_fitted_plot"), "Residuals vs fitted plot"),
                report_image("Residual distribution", output.get("residual_distribution_plot"), "Residual distribution plot"),
            ]
        )
        residual_table = csv_report_table(artifacts.get("residual_diagnostics"))
        if residual_table:
            sections.extend(["<section><h2>Residual diagnostics</h2>", residual_table, "</section>"])

    sections.append("</body></html>")
    return "".join(sections)


def csv_report_frame(csv_data):
    if not csv_data:
        return None
    try:
        return pd.read_csv(StringIO(csv_data))
    except Exception:
        return pd.DataFrame({"Value": [csv_data]})


def output_metrics_frame(output):
    metrics = output.get("metrics") or []
    if not metrics:
        return None
    return pd.DataFrame([{"Metric": item.get("label", ""), "Value": item.get("value", "")} for item in metrics])


def html_table_frame(html_table):
    if not html_table:
        return None
    try:
        tables = pd.read_html(StringIO(html_table))
    except (ImportError, ValueError):
        return None
    return tables[0] if tables else None


def wrap_pdf_value(value, width=28):
    text = "" if pd.isna(value) else str(value)
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or text


def add_pdf_text_page(pdf, title, lines):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.04, 0.94, title, fontsize=20, weight="bold", va="top")
    y = 0.84
    for line in lines:
        wrapped = textwrap.wrap(str(line), width=105) or [""]
        for part in wrapped:
            ax.text(0.05, y, part, fontsize=10.5, va="top")
            y -= 0.04
            if y < 0.08:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax = plt.subplots(figsize=(11, 8.5))
                ax.axis("off")
                y = 0.94
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_pdf_table_pages(pdf, title, frame, rows_per_page=24):
    if frame is None or frame.empty:
        return

    frame = display_frame(frame).copy().fillna("")
    for start in range(0, len(frame), rows_per_page):
        chunk = frame.iloc[start : start + rows_per_page].copy()
        for column in chunk.columns:
            chunk[column] = chunk[column].map(lambda value: wrap_pdf_value(value, width=24))

        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis("off")
        suffix = f" ({start + 1}-{start + len(chunk)} of {len(frame)})" if len(frame) > rows_per_page else ""
        ax.text(0.02, 0.97, f"{title}{suffix}", fontsize=16, weight="bold", va="top")
        table = ax.table(
            cellText=chunk.values,
            colLabels=list(chunk.columns),
            cellLoc="left",
            loc="center",
            bbox=[0.02, 0.05, 0.96, 0.84],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7.0 if len(chunk.columns) > 6 else 8.2)
        table.scale(1, 1.25)
        for (row, _), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#f1f5f9")
                cell.set_text_props(weight="bold")
            cell.set_edgecolor("#d8dee8")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def add_pdf_image_page(pdf, title, image_data):
    if not image_data:
        return
    try:
        image = plt.imread(BytesIO(base64.b64decode(image_data)), format="png")
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.axis("off")
    ax.text(0.02, 0.97, title, fontsize=16, weight="bold", va="top")
    ax.imshow(image, extent=[0.08, 0.92, 0.08, 0.88], aspect="auto")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def pro_report_pdf_bytes(tab_name, snapshot, dataset):
    output = snapshot.get("output") or {}
    artifacts = snapshot.get("download_artifacts") or {}
    recommendation = snapshot.get("recommendation")
    report_title = "Pro Classification Report" if tab_name == "pro_classification" else "Pro Regression Report"
    buffer = BytesIO()

    with PdfPages(buffer) as pdf:
        history_entry = snapshot.get("history_entry") or {}
        add_pdf_text_page(
            pdf,
            report_title,
            [
                "Single-run export for the selected Pro model comparison.",
                f"Dataset: {(dataset or {}).get('filename', '-')}",
                f"Target: {snapshot.get('selected_target') or '-'}",
                f"Detail model: {history_entry.get('detail_model', '-')}",
                f"Summary: {history_entry.get('summary', '-')}",
            ],
        )
        add_pdf_table_pages(pdf, "Dataset Metadata", pd.DataFrame(report_metadata_rows(tab_name, snapshot, dataset)), rows_per_page=22)
        add_pdf_table_pages(pdf, "Model Comparison", csv_report_frame(artifacts.get("model_comparison")), rows_per_page=18)
        add_pdf_table_pages(pdf, "Model Recommendation", recommendation_report_frame(recommendation), rows_per_page=20)
        add_pdf_table_pages(pdf, "Selected Detail Metrics", output_metrics_frame(output), rows_per_page=24)
        add_pdf_image_page(pdf, "Cross-Validation Fold Scores", output.get("cv_plot"))
        add_pdf_table_pages(pdf, "Cross-Validation Summary", csv_report_frame(artifacts.get("cv_summary")), rows_per_page=24)
        add_pdf_table_pages(pdf, "Cross-Validation Fold Scores", csv_report_frame(artifacts.get("cv_diagnostics")), rows_per_page=24)

        details_frame_pdf = csv_report_frame(artifacts.get("details"))
        if details_frame_pdf is None:
            details_frame_pdf = html_table_frame(output.get("details_html"))
        add_pdf_table_pages(pdf, "Model Details", details_frame_pdf, rows_per_page=24)
        add_pdf_table_pages(pdf, "Coefficients", csv_report_frame(artifacts.get("coefficients")), rows_per_page=24)
        add_pdf_table_pages(pdf, "Variable Importance", csv_report_frame(artifacts.get("variable_importance")), rows_per_page=24)
        add_pdf_table_pages(pdf, "Confusion Matrix", csv_report_frame(artifacts.get("confusion_matrix")), rows_per_page=24)

        if tab_name == "pro_classification":
            add_pdf_image_page(pdf, "ROC Curve", output.get("roc_plot"))
            add_pdf_image_page(pdf, "Precision-Recall Curve", output.get("pr_plot"))
            add_pdf_image_page(pdf, "Classification Tree", output.get("tree_plot"))
            add_pdf_table_pages(pdf, "Selected Threshold Metrics", csv_report_frame(artifacts.get("selected_threshold_metrics")), rows_per_page=24)
            add_pdf_table_pages(pdf, "Threshold Analysis", csv_report_frame(artifacts.get("threshold_analysis")), rows_per_page=24)
        else:
            add_pdf_image_page(pdf, "Predicted vs Actual", output.get("predicted_actual_plot"))
            add_pdf_image_page(pdf, "Residuals vs Fitted", output.get("residuals_fitted_plot"))
            add_pdf_image_page(pdf, "Residual Distribution", output.get("residual_distribution_plot"))
            add_pdf_table_pages(pdf, "Residual Diagnostics", csv_report_frame(artifacts.get("residual_diagnostics")), rows_per_page=28)

    buffer.seek(0)
    return buffer.getvalue()


def durable_pro_report(result_type):
    if result_type not in PRO_TAB_NAMES:
        return None
    snapshot = latest_pro_run_snapshot(result_type)
    if not snapshot:
        return None
    return pro_report_html(result_type, snapshot, current_dataset())


def durable_pro_report_pdf(result_type):
    if result_type not in PRO_TAB_NAMES:
        return None
    snapshot = latest_pro_run_snapshot(result_type)
    if not snapshot:
        return None
    return pro_report_pdf_bytes(result_type, snapshot, current_dataset())


def durable_download_artifact(result_type, artifact):
    user_id, current_id = current_pro_run_scope()
    if not user_id or not current_id or result_type not in PRO_TAB_NAMES:
        return None

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


































