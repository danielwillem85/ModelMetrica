from io import BytesIO, StringIO
import base64
import math
from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, render_template_string, request, session
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, plot_tree


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = "dev-secret-change-me"

ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
DATASETS = {}

PAGE_TEMPLATE = """
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
      .brand {
        font-size: 22px;
        font-weight: 750;
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
      @media (max-width: 760px) {
        .metric-row {
          grid-template-columns: 1fr;
        }
      }
    </style>
  </head>
  <body>
    <header>
      <div class="brand">ModelMetrica</div>
    </header>
    <main>
      <h1>Modeling workspace</h1>
      <p>Upload a CSV or Excel file, inspect the first rows, then run classification or regression analysis.</p>

      <nav class="tabs">
        <a class="tab {{ 'active' if active_tab == 'data' else '' }}" href="#data">Data</a>
        <a class="tab {{ 'active' if active_tab == 'classification' else '' }}" href="#classification">Classification</a>
        <a class="tab {{ 'active' if active_tab == 'regression' else '' }}" href="#regression">Regression</a>
      </nav>

      <section id="data" class="tab-panel {{ 'active' if active_tab == 'data' else '' }}">
        <div class="panel">
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

      <section id="classification" class="tab-panel {{ 'active' if active_tab == 'classification' else '' }}">
        <div class="panel">
          {% if not has_data %}
            <p class="error">Upload a dataset on the Data tab before running classification.</p>
          {% else %}
            <form method="post">
              <input type="hidden" name="form_name" value="classification">
              <input type="hidden" name="active_tab" value="classification">
              <div>
                <label for="classification_model">Model type</label>
                <select id="classification_model" name="classification_model" required>
                  <option value="logistic" {{ 'selected' if selected_classification_model == 'logistic' else '' }}>Logistic regression</option>
                  <option value="tree" {{ 'selected' if selected_classification_model == 'tree' else '' }}>Tree model</option>
                  <option value="random_forest" {{ 'selected' if selected_classification_model == 'random_forest' else '' }}>Random Forest</option>
                  <option value="gradient_boosting" {{ 'selected' if selected_classification_model == 'gradient_boosting' else '' }}>Gradient Boosting</option>
                  <option value="svm" {{ 'selected' if selected_classification_model == 'svm' else '' }}>Support Vector Machine</option>
                  <option value="knn" {{ 'selected' if selected_classification_model == 'knn' else '' }}>kNN</option>
                </select>
              </div>
              <div>
                <label for="target">Target column</label>
                <select id="target" name="target" required>
                  {% for column in columns %}
                    <option value="{{ column }}" {{ 'selected' if column == selected_target else '' }}>{{ column }}</option>
                  {% endfor %}
                </select>
              </div>
              <div>
                <label for="predictors">Predictor columns</label>
                <select id="predictors" name="predictors" multiple required>
                  {% for column in columns %}
                    <option value="{{ column }}" {{ 'selected' if column in selected_predictors else '' }}>{{ column }}</option>
                  {% endfor %}
                </select>
                <p>Select one or more predictors. Logistic regression requires exactly two target classes.</p>
              </div>
              <div>
                <button type="submit">Run</button>
              </div>
            </form>
          {% endif %}
          {% if classification_error %}
            <p class="error">{{ classification_error }}</p>
          {% endif %}
        </div>

        {% if model_output %}
          <div class="panel">
            <h2>{{ model_output.title }}</h2>
            <p>{{ model_output.description }}</p>
            <div class="metric-row">
              {% for metric in model_output.metrics %}
                <div class="metric"><span>{{ metric.label }}</span><strong>{{ metric.value }}</strong></div>
              {% endfor %}
            </div>
            {% if model_output.coefficients_html %}
              <h3>Coefficients</h3>
              <div class="table-wrap">
                {{ model_output.coefficients_html|safe }}
              </div>
            {% endif %}
            {% if model_output.importances_html %}
              <h3>Variable importance</h3>
              <div class="table-wrap">
                {{ model_output.importances_html|safe }}
              </div>
            {% endif %}
            {% if model_output.details_html %}
              <h3>Model details</h3>
              <div class="table-wrap">
                {{ model_output.details_html|safe }}
              </div>
            {% endif %}
            <h3>Confusion matrix</h3>
            <div class="table-wrap">
              {{ model_output.confusion_html|safe }}
            </div>
            {% if model_output.tree_plot %}
              <h3>Tree structure</h3>
              <div class="tree-plot">
                <img src="data:image/png;base64,{{ model_output.tree_plot }}" alt="Classification tree plot">
              </div>
            {% endif %}
          </div>
        {% endif %}
      </section>

      <section id="regression" class="tab-panel {{ 'active' if active_tab == 'regression' else '' }}">
        <div class="panel">
          {% if not has_data %}
            <p class="error">Upload a dataset on the Data tab before running regression.</p>
          {% else %}
            <form method="post">
              <input type="hidden" name="form_name" value="regression">
              <input type="hidden" name="active_tab" value="regression">
              <div>
                <label for="regression_model">Model type</label>
                <select id="regression_model" name="regression_model" required>
                  <option value="linear" {{ 'selected' if selected_regression_model == 'linear' else '' }}>Linear Regression</option>
                  <option value="ridge" {{ 'selected' if selected_regression_model == 'ridge' else '' }}>Ridge Regression</option>
                  <option value="lasso" {{ 'selected' if selected_regression_model == 'lasso' else '' }}>Lasso Regression</option>
                  <option value="random_forest" {{ 'selected' if selected_regression_model == 'random_forest' else '' }}>Random Forest Regression</option>
                  <option value="gradient_boosting" {{ 'selected' if selected_regression_model == 'gradient_boosting' else '' }}>Gradient Boosting Regression</option>
                  <option value="svr" {{ 'selected' if selected_regression_model == 'svr' else '' }}>Support Vector Regression</option>
                  <option value="knn" {{ 'selected' if selected_regression_model == 'knn' else '' }}>kNN Regression</option>
                </select>
              </div>
              <div>
                <label for="regression_target">Numeric target column</label>
                <select id="regression_target" name="regression_target" required>
                  {% for column in columns %}
                    <option value="{{ column }}" {{ 'selected' if column == selected_regression_target else '' }}>{{ column }}</option>
                  {% endfor %}
                </select>
              </div>
              <div>
                <label for="regression_predictors">Predictor columns</label>
                <select id="regression_predictors" name="regression_predictors" multiple required>
                  {% for column in columns %}
                    <option value="{{ column }}" {{ 'selected' if column in selected_regression_predictors else '' }}>{{ column }}</option>
                  {% endfor %}
                </select>
                <p>Select one or more predictors. Categorical predictors are automatically encoded.</p>
              </div>
              <div>
                <button type="submit">Run</button>
              </div>
            </form>
          {% endif %}
          {% if regression_error %}
            <p class="error">{{ regression_error }}</p>
          {% endif %}
        </div>

        {% if regression_output %}
          <div class="panel">
            <h2>{{ regression_output.title }}</h2>
            <p>{{ regression_output.description }}</p>
            <div class="metric-row">
              {% for metric in regression_output.metrics %}
                <div class="metric"><span>{{ metric.label }}</span><strong>{{ metric.value }}</strong></div>
              {% endfor %}
            </div>
            {% if regression_output.coefficients_html %}
              <h3>Coefficients</h3>
              <div class="table-wrap">
                {{ regression_output.coefficients_html|safe }}
              </div>
            {% endif %}
            {% if regression_output.importances_html %}
              <h3>Variable importance</h3>
              <div class="table-wrap">
                {{ regression_output.importances_html|safe }}
              </div>
            {% endif %}
            {% if regression_output.details_html %}
              <h3>Model details</h3>
              <div class="table-wrap">
                {{ regression_output.details_html|safe }}
              </div>
            {% endif %}
          </div>
        {% endif %}
      </section>
    </main>
    <script>
      const tabs = document.querySelectorAll(".tab");
      const panels = document.querySelectorAll(".tab-panel");

      function activateTab(hash) {
        const target = ["#classification", "#regression"].includes(hash) ? hash.slice(1) : "data";
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

      if (window.location.hash) {
        activateTab(window.location.hash);
      }
    </script>
  </body>
</html>
"""


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def read_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.filename).suffix.lower()
    file_bytes = uploaded_file.read()

    if suffix == ".csv":
        return pd.read_csv(StringIO(file_bytes.decode("utf-8-sig")))

    return pd.read_excel(BytesIO(file_bytes))


def current_dataset():
    dataset_id = session.get("dataset_id")
    if not dataset_id:
        return None
    return DATASETS.get(dataset_id)


def save_dataset(data, filename):
    dataset_id = str(uuid4())
    DATASETS[dataset_id] = {"data": data, "filename": filename}
    session["dataset_id"] = dataset_id


def preview_table(data):
    preview = data.head(25)
    return preview.to_html(
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


def prepare_classification_data(data, target, predictors, binary_only):
    if target in predictors:
        raise ValueError("The target column cannot also be used as a predictor.")

    selected = [target] + predictors
    model_data = data[selected].dropna()
    if len(model_data) < 5:
        raise ValueError("At least five complete rows are required.")

    target_values = model_data[target]
    classes = list(pd.unique(target_values))
    if binary_only and len(classes) != 2:
        raise ValueError("Logistic regression requires a target column with exactly two classes.")
    if len(classes) < 2:
        raise ValueError("Classification requires a target column with at least two classes.")

    x_raw = model_data[predictors]
    x_encoded = pd.get_dummies(x_raw, drop_first=True, dtype=float)
    if x_encoded.empty:
        raise ValueError("At least one usable predictor is required.")

    return model_data, target_values, classes, x_encoded


def encode_target(target_values):
    encoded_target = pd.Categorical(target_values)
    return encoded_target.codes, encoded_target.categories


def confusion_table(actual, predicted):
    return pd.crosstab(
        pd.Series(actual, name="Actual"),
        pd.Series(predicted, name="Predicted"),
        dropna=False,
    )


def details_table(details):
    return pd.DataFrame(
        [{"Setting": key, "Value": value} for key, value in details.items()]
    ).to_html(index=False, border=0, classes="model-details")


def importance_table(feature_names, importances):
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
    return importances.to_html(index=False, border=0, classes="importances", float_format="{:.4f}".format)


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


def fit_logistic_regression(data, target, predictors):
    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=True)
    y = (target_values == classes[1]).astype(float).to_numpy()
    x_matrix = np.column_stack([np.ones(len(x_encoded)), x_encoded.to_numpy(dtype=float)])
    feature_names = ["Intercept"] + list(x_encoded.columns)
    beta = np.zeros(x_matrix.shape[1])
    ridge = 1e-8

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

    predicted = np.where(probabilities >= 0.5, classes[1], classes[0])
    confusion = pd.crosstab(
        pd.Series(target_values.to_numpy(), name="Actual"),
        pd.Series(predicted, name="Predicted"),
        dropna=False,
    )
    accuracy = float(np.mean(predicted == target_values.to_numpy()))

    return {
        "title": "Logistic regression results",
        "description": f"Target: {target}. Positive class: {classes[1]}.",
        "target": target,
        "positive_class": str(classes[1]),
        "nobs": len(y),
        "metrics": [
            {"label": "Observations", "value": len(y)},
            {"label": "Accuracy", "value": f"{accuracy:.3f}"},
            {"label": "AIC", "value": f"{aic:.3f}"},
            {"label": "McFadden R2", "value": f"{pseudo_r2:.3f}"},
            {"label": "Log likelihood", "value": f"{log_likelihood:.3f}"},
            {"label": "BIC", "value": f"{bic:.3f}"},
        ],
        "coefficients_html": coefficients.to_html(index=False, border=0, classes="coefficients", float_format="{:.4f}".format),
        "importances_html": None,
        "details_html": details_table({"Model": "Logistic regression", "Decision threshold": "0.5"}),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
    }


def fit_tree_model(data, target, predictors):
    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False)
    y, class_names = encode_target(target_values)
    min_samples_leaf = max(1, int(len(y) * 0.02))

    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
    )
    tree.fit(x_encoded, y)
    predicted_codes = tree.predict(x_encoded)
    predicted_labels = class_names[predicted_codes]
    accuracy = accuracy_score(target_values, predicted_labels)

    confusion = confusion_table(target_values.to_numpy(), predicted_labels)
    tree_plot = tree_plot_image(tree, x_encoded.columns, class_names)

    return {
        "title": "Tree model results",
        "description": f"Target: {target}. Classes: {', '.join(str(value) for value in classes)}.",
        "target": target,
        "positive_class": None,
        "nobs": len(y),
        "metrics": [
            {"label": "Observations", "value": len(y)},
            {"label": "Accuracy", "value": f"{accuracy:.3f}"},
            {"label": "Tree depth", "value": tree.get_depth()},
            {"label": "Terminal nodes", "value": tree.get_n_leaves()},
        ],
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, tree.feature_importances_),
        "details_html": details_table({
            "Model": "Decision tree classifier",
            "Max depth": tree.max_depth,
            "Minimum samples per leaf": min_samples_leaf,
        }),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": tree_plot,
    }


def fit_random_forest_model(data, target, predictors):
    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False)
    y, class_names = encode_target(target_values)
    min_samples_leaf = max(1, int(len(y) * 0.01))
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_encoded, y)
    predicted_labels = class_names[model.predict(x_encoded)]
    accuracy = accuracy_score(target_values, predicted_labels)
    confusion = confusion_table(target_values.to_numpy(), predicted_labels)

    return {
        "title": "Random Forest results",
        "description": f"Target: {target}. Classes: {', '.join(str(value) for value in classes)}.",
        "target": target,
        "positive_class": None,
        "nobs": len(y),
        "metrics": [
            {"label": "Observations", "value": len(y)},
            {"label": "Accuracy", "value": f"{accuracy:.3f}"},
            {"label": "Trees", "value": model.n_estimators},
            {"label": "Classes", "value": len(class_names)},
        ],
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, model.feature_importances_),
        "details_html": details_table({
            "Model": "RandomForestClassifier",
            "Trees": model.n_estimators,
            "Minimum samples per leaf": min_samples_leaf,
            "Random seed": 42,
        }),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
    }


def fit_gradient_boosting_model(data, target, predictors):
    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False)
    y, class_names = encode_target(target_values)
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    model.fit(x_encoded, y)
    predicted_labels = class_names[model.predict(x_encoded)]
    accuracy = accuracy_score(target_values, predicted_labels)
    confusion = confusion_table(target_values.to_numpy(), predicted_labels)

    return {
        "title": "Gradient Boosting results",
        "description": f"Target: {target}. Classes: {', '.join(str(value) for value in classes)}.",
        "target": target,
        "positive_class": None,
        "nobs": len(y),
        "metrics": [
            {"label": "Observations", "value": len(y)},
            {"label": "Accuracy", "value": f"{accuracy:.3f}"},
            {"label": "Boosting stages", "value": model.n_estimators},
            {"label": "Learning rate", "value": model.learning_rate},
        ],
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, model.feature_importances_),
        "details_html": details_table({
            "Model": "GradientBoostingClassifier",
            "Boosting stages": model.n_estimators,
            "Learning rate": model.learning_rate,
            "Maximum tree depth": model.max_depth,
            "Random seed": 42,
        }),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
    }


def fit_svm_model(data, target, predictors):
    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False)
    y, class_names = encode_target(target_values)
    scaled_x = StandardScaler().fit_transform(x_encoded)
    model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    model.fit(scaled_x, y)
    predicted_labels = class_names[model.predict(scaled_x)]
    accuracy = accuracy_score(target_values, predicted_labels)
    confusion = confusion_table(target_values.to_numpy(), predicted_labels)

    return {
        "title": "Support Vector Machine results",
        "description": f"Target: {target}. Features were standardized before fitting.",
        "target": target,
        "positive_class": None,
        "nobs": len(y),
        "metrics": [
            {"label": "Observations", "value": len(y)},
            {"label": "Accuracy", "value": f"{accuracy:.3f}"},
            {"label": "Kernel", "value": model.kernel},
            {"label": "Support vectors", "value": int(np.sum(model.n_support_))},
        ],
        "coefficients_html": None,
        "importances_html": None,
        "details_html": details_table({
            "Model": "SVC",
            "Kernel": model.kernel,
            "C": model.C,
            "Gamma": model.gamma,
            "Support vectors by class": ", ".join(str(value) for value in model.n_support_),
        }),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
    }


def fit_knn_model(data, target, predictors):
    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False)
    y, class_names = encode_target(target_values)
    scaled_x = StandardScaler().fit_transform(x_encoded)
    neighbors = min(5, len(y))
    model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
    model.fit(scaled_x, y)
    predicted_labels = class_names[model.predict(scaled_x)]
    accuracy = accuracy_score(target_values, predicted_labels)
    confusion = confusion_table(target_values.to_numpy(), predicted_labels)

    return {
        "title": "kNN results",
        "description": f"Target: {target}. Features were standardized before fitting.",
        "target": target,
        "positive_class": None,
        "nobs": len(y),
        "metrics": [
            {"label": "Observations", "value": len(y)},
            {"label": "Accuracy", "value": f"{accuracy:.3f}"},
            {"label": "Neighbors", "value": neighbors},
            {"label": "Weights", "value": model.weights},
        ],
        "coefficients_html": None,
        "importances_html": None,
        "details_html": details_table({
            "Model": "KNeighborsClassifier",
            "Neighbors": neighbors,
            "Weights": model.weights,
            "Distance metric": model.metric,
        }),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
    }


def prepare_regression_data(data, target, predictors):
    if target in predictors:
        raise ValueError("The target column cannot also be used as a predictor.")

    selected = [target] + predictors
    model_data = data[selected].dropna()
    if len(model_data) < 3:
        raise ValueError("At least three complete rows are required.")

    y_series = pd.to_numeric(model_data[target], errors="coerce")
    model_data = model_data.loc[y_series.notna()].copy()
    y = y_series.loc[y_series.notna()].to_numpy(dtype=float)
    if len(y) < 3:
        raise ValueError("The regression target must contain at least three numeric values.")

    x_encoded = pd.get_dummies(model_data[predictors], drop_first=True, dtype=float)
    if x_encoded.empty:
        raise ValueError("At least one usable predictor is required.")

    return y, x_encoded


def regression_metric_list(y, predictions, parameter_count=None):
    residuals = y - predictions
    sse = float(np.sum(residuals**2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - (sse / tss) if tss > 0 else 0.0
    rmse = math.sqrt(sse / len(y))
    mae = float(np.mean(np.abs(residuals)))
    metrics = [
        {"label": "Observations", "value": len(y)},
        {"label": "R squared", "value": f"{r_squared:.3f}"},
        {"label": "RMSE", "value": f"{rmse:.3f}"},
        {"label": "MAE", "value": f"{mae:.3f}"},
    ]

    if parameter_count is not None and len(y) > parameter_count:
        residual_df = len(y) - parameter_count
        adj_r_squared = 1 - ((1 - r_squared) * (len(y) - 1) / residual_df)
        metrics.append({"label": "Adj. R squared", "value": f"{adj_r_squared:.3f}"})

    return metrics


def regression_coefficient_table(feature_names, coefficients):
    coefficients = pd.DataFrame(
        {
            "Term": feature_names,
            "Coefficient": coefficients,
        }
    )
    return coefficients.to_html(index=False, border=0, classes="coefficients", float_format="{:.4f}".format)


def fit_linear_regression(data, target, predictors):
    if target in predictors:
        raise ValueError("The target column cannot also be used as a predictor.")

    selected = [target] + predictors
    model_data = data[selected].dropna()
    if len(model_data) < 3:
        raise ValueError("At least three complete rows are required.")

    y_series = pd.to_numeric(model_data[target], errors="coerce")
    model_data = model_data.loc[y_series.notna()].copy()
    y = y_series.loc[y_series.notna()].to_numpy(dtype=float)
    if len(y) < 3:
        raise ValueError("The regression target must contain at least three numeric values.")

    x_raw = model_data[predictors]
    x_encoded = pd.get_dummies(x_raw, drop_first=True, dtype=float)
    if x_encoded.empty:
        raise ValueError("At least one usable predictor is required.")

    x_matrix = np.column_stack([np.ones(len(x_encoded)), x_encoded.to_numpy(dtype=float)])
    feature_names = ["Intercept"] + list(x_encoded.columns)
    rank = np.linalg.matrix_rank(x_matrix)
    if len(y) <= rank:
        raise ValueError("Not enough rows for the selected predictors.")

    beta = np.linalg.pinv(x_matrix) @ y
    fitted = x_matrix @ beta
    residuals = y - fitted
    sse = float(np.sum(residuals**2))
    tss = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - (sse / tss) if tss > 0 else 0.0
    nobs = len(y)
    predictor_df = max(rank - 1, 1)
    residual_df = nobs - rank
    adj_r_squared = 1 - ((1 - r_squared) * (nobs - 1) / residual_df)
    mse = sse / residual_df
    rmse = math.sqrt(sse / nobs)
    residual_se = math.sqrt(mse)
    msr = (tss - sse) / predictor_df if predictor_df else float("nan")
    f_statistic = msr / mse if mse > 0 else float("inf")

    covariance = mse * np.linalg.pinv(x_matrix.T @ x_matrix)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    t_values = np.divide(beta, standard_errors, out=np.zeros_like(beta), where=standard_errors > 0)
    p_values = [normal_two_sided_pvalue(t_value) for t_value in t_values]

    coefficients = pd.DataFrame(
        {
            "Term": feature_names,
            "Coefficient": beta,
            "Std. Error": standard_errors,
            "t value": t_values,
            "Approx. p value": p_values,
        }
    )

    return {
        "title": "Linear Regression results",
        "description": f"Target: {target}. Ordinary least squares fit.",
        "target": target,
        "metrics": [
            {"label": "Observations", "value": nobs},
            {"label": "R squared", "value": f"{r_squared:.3f}"},
            {"label": "Adj. R squared", "value": f"{adj_r_squared:.3f}"},
            {"label": "RMSE", "value": f"{rmse:.3f}"},
            {"label": "Residual SE", "value": f"{residual_se:.3f}"},
            {"label": "F statistic", "value": f"{f_statistic:.3f}"},
        ],
        "coefficients_html": coefficients.to_html(index=False, border=0, classes="coefficients", float_format="{:.4f}".format),
        "importances_html": None,
        "details_html": details_table({
            "Model": "Ordinary least squares",
            "Model df": predictor_df,
            "Residual df": residual_df,
        }),
    }


def fit_ridge_regression(data, target, predictors):
    y, x_encoded = prepare_regression_data(data, target, predictors)
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x_encoded)
    model = Ridge(alpha=1.0)
    model.fit(scaled_x, y)
    predictions = model.predict(scaled_x)

    return {
        "title": "Ridge Regression results",
        "description": f"Target: {target}. Predictors were standardized before fitting.",
        "target": target,
        "metrics": regression_metric_list(y, predictions, parameter_count=scaled_x.shape[1] + 1),
        "coefficients_html": regression_coefficient_table(["Intercept"] + list(x_encoded.columns), [model.intercept_] + list(model.coef_)),
        "importances_html": None,
        "details_html": details_table({
            "Model": "Ridge",
            "Alpha": model.alpha,
            "Feature scaling": "StandardScaler",
        }),
    }


def fit_lasso_regression(data, target, predictors):
    y, x_encoded = prepare_regression_data(data, target, predictors)
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x_encoded)
    model = Lasso(alpha=0.1, max_iter=10000, random_state=42)
    model.fit(scaled_x, y)
    predictions = model.predict(scaled_x)
    nonzero_count = int(np.sum(np.abs(model.coef_) > 1e-8))

    return {
        "title": "Lasso Regression results",
        "description": f"Target: {target}. Predictors were standardized before fitting.",
        "target": target,
        "metrics": regression_metric_list(y, predictions, parameter_count=nonzero_count + 1),
        "coefficients_html": regression_coefficient_table(["Intercept"] + list(x_encoded.columns), [model.intercept_] + list(model.coef_)),
        "importances_html": None,
        "details_html": details_table({
            "Model": "Lasso",
            "Alpha": model.alpha,
            "Non-zero coefficients": nonzero_count,
            "Feature scaling": "StandardScaler",
        }),
    }


def fit_random_forest_regression(data, target, predictors):
    y, x_encoded = prepare_regression_data(data, target, predictors)
    min_samples_leaf = max(1, int(len(y) * 0.01))
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(x_encoded, y)
    predictions = model.predict(x_encoded)

    return {
        "title": "Random Forest Regression results",
        "description": f"Target: {target}. Ensemble of regression trees.",
        "target": target,
        "metrics": regression_metric_list(y, predictions),
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, model.feature_importances_),
        "details_html": details_table({
            "Model": "RandomForestRegressor",
            "Trees": model.n_estimators,
            "Minimum samples per leaf": min_samples_leaf,
            "Random seed": 42,
        }),
    }


def fit_gradient_boosting_regression(data, target, predictors):
    y, x_encoded = prepare_regression_data(data, target, predictors)
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    model.fit(x_encoded, y)
    predictions = model.predict(x_encoded)

    return {
        "title": "Gradient Boosting Regression results",
        "description": f"Target: {target}. Sequential boosted-tree regression model.",
        "target": target,
        "metrics": regression_metric_list(y, predictions),
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, model.feature_importances_),
        "details_html": details_table({
            "Model": "GradientBoostingRegressor",
            "Boosting stages": model.n_estimators,
            "Learning rate": model.learning_rate,
            "Maximum tree depth": model.max_depth,
            "Random seed": 42,
        }),
    }


def fit_svr_regression(data, target, predictors):
    y, x_encoded = prepare_regression_data(data, target, predictors)
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x_encoded)
    model = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
    model.fit(scaled_x, y)
    predictions = model.predict(scaled_x)

    return {
        "title": "Support Vector Regression results",
        "description": f"Target: {target}. Predictors were standardized before fitting.",
        "target": target,
        "metrics": regression_metric_list(y, predictions),
        "coefficients_html": None,
        "importances_html": None,
        "details_html": details_table({
            "Model": "SVR",
            "Kernel": model.kernel,
            "C": model.C,
            "Epsilon": model.epsilon,
            "Gamma": model.gamma,
            "Support vectors": len(model.support_),
            "Feature scaling": "StandardScaler",
        }),
    }


def fit_knn_regression(data, target, predictors):
    y, x_encoded = prepare_regression_data(data, target, predictors)
    scaler = StandardScaler()
    scaled_x = scaler.fit_transform(x_encoded)
    neighbors = min(5, len(y))
    model = KNeighborsRegressor(n_neighbors=neighbors, weights="distance")
    model.fit(scaled_x, y)
    predictions = model.predict(scaled_x)

    return {
        "title": "kNN Regression results",
        "description": f"Target: {target}. Predictors were standardized before fitting.",
        "target": target,
        "metrics": regression_metric_list(y, predictions),
        "coefficients_html": None,
        "importances_html": None,
        "details_html": details_table({
            "Model": "KNeighborsRegressor",
            "Neighbors": neighbors,
            "Weights": model.weights,
            "Distance metric": model.metric,
            "Feature scaling": "StandardScaler",
        }),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    active_tab = request.form.get("active_tab", "data")
    data_error = None
    classification_error = None
    regression_error = None
    model_output = None
    regression_output = None
    selected_classification_model = "logistic"
    selected_regression_model = "linear"
    selected_target = None
    selected_predictors = []
    selected_regression_target = None
    selected_regression_predictors = []

    if request.method == "POST" and request.form.get("form_name") == "upload":
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

    dataset = current_dataset()

    if request.method == "POST" and request.form.get("form_name") == "classification":
        active_tab = "classification"
        selected_classification_model = request.form.get("classification_model", "logistic")
        selected_target = request.form.get("target")
        selected_predictors = request.form.getlist("predictors")

        if dataset is None:
            classification_error = "Upload a dataset on the Data tab before running classification."
        elif not selected_target or not selected_predictors:
            classification_error = "Select a target column and at least one predictor."
        else:
            try:
                classification_models = {
                    "logistic": fit_logistic_regression,
                    "tree": fit_tree_model,
                    "random_forest": fit_random_forest_model,
                    "gradient_boosting": fit_gradient_boosting_model,
                    "svm": fit_svm_model,
                    "knn": fit_knn_model,
                }
                fit_model = classification_models.get(selected_classification_model, fit_logistic_regression)
                model_output = fit_model(dataset["data"], selected_target, selected_predictors)
            except Exception as exc:
                classification_error = str(exc)

    if request.method == "POST" and request.form.get("form_name") == "regression":
        active_tab = "regression"
        selected_regression_model = request.form.get("regression_model", "linear")
        selected_regression_target = request.form.get("regression_target")
        selected_regression_predictors = request.form.getlist("regression_predictors")

        if dataset is None:
            regression_error = "Upload a dataset on the Data tab before running regression."
        elif not selected_regression_target or not selected_regression_predictors:
            regression_error = "Select a target column and at least one predictor."
        else:
            try:
                regression_models = {
                    "linear": fit_linear_regression,
                    "ridge": fit_ridge_regression,
                    "lasso": fit_lasso_regression,
                    "random_forest": fit_random_forest_regression,
                    "gradient_boosting": fit_gradient_boosting_regression,
                    "svr": fit_svr_regression,
                    "knn": fit_knn_regression,
                }
                fit_model = regression_models.get(selected_regression_model, fit_linear_regression)
                regression_output = fit_model(dataset["data"], selected_regression_target, selected_regression_predictors)
            except Exception as exc:
                regression_error = str(exc)

    has_data = dataset is not None
    data = dataset["data"] if has_data else None
    filename = dataset["filename"] if has_data else None
    columns = list(data.columns) if has_data else []

    if has_data and selected_target is None:
        selected_target = columns[0]
        selected_predictors = columns[1:]

    if has_data and selected_regression_target is None:
        numeric_columns = list(data.select_dtypes(include="number").columns)
        if numeric_columns:
            selected_regression_target = numeric_columns[0]
            selected_regression_predictors = [column for column in numeric_columns[1:] if column in columns]
        else:
            selected_regression_target = columns[0]
            selected_regression_predictors = columns[1:]

    table_html = preview_table(data) if has_data else None
    row_count = min(25, len(data)) if has_data else 0
    total_rows = len(data) if has_data else 0

    return render_template_string(
        PAGE_TEMPLATE,
        active_tab=active_tab,
        data_error=data_error,
        classification_error=classification_error,
        regression_error=regression_error,
        table_html=table_html,
        filename=filename,
        row_count=row_count,
        total_rows=total_rows,
        has_data=has_data,
        columns=columns,
        selected_classification_model=selected_classification_model,
        selected_target=selected_target,
        selected_predictors=selected_predictors,
        selected_regression_model=selected_regression_model,
        selected_regression_target=selected_regression_target,
        selected_regression_predictors=selected_regression_predictors,
        model_output=model_output,
        regression_output=regression_output,
    )


if __name__ == "__main__":
    app.run(debug=True)
