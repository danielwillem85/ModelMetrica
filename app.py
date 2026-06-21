from io import BytesIO, StringIO
import base64
import math
import sqlite3
from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import Flask, Response, redirect, render_template_string, request, session, url_for
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, plot_tree
from werkzeug.security import check_password_hash, generate_password_hash


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
app.secret_key = "dev-secret-change-me"

ALLOWED_EXTENSIONS = {".csv", ".xls", ".xlsx"}
DB_PATH = Path(__file__).with_name("modelmetrica_users.sqlite3")
DATASETS = {}
DOWNLOADS = {}

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
      @media (max-width: 760px) {
        .metric-row {
          grid-template-columns: 1fr;
        }
        .auth-grid {
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
            <a class="logout-link" href="{{ url_for('logout') }}">Log out</a>
          </div>
        {% endif %}
      </div>
    </header>
    {% if is_authenticated %}
    <main>
      <h1>Modeling workspace</h1>
      <p>Upload a CSV or Excel file, inspect the first rows, then run classification or regression analysis.</p>

      <nav class="tabs">
        <a class="tab {{ 'active' if active_tab == 'data' else '' }}" href="#data">Data</a>
        <a class="tab {{ 'active' if active_tab == 'classification' else '' }}" href="#classification">Classification</a>
        <a class="tab {{ 'active' if active_tab == 'regression' else '' }}" href="#regression">Regression</a>
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
                <label for="classification_test_size">Test set size</label>
                <select id="classification_test_size" name="classification_test_size" required>
                  <option value="0.2" {{ 'selected' if selected_classification_test_size == 0.2 else '' }}>20%</option>
                  <option value="0.25" {{ 'selected' if selected_classification_test_size == 0.25 else '' }}>25%</option>
                  <option value="0.3" {{ 'selected' if selected_classification_test_size == 0.3 else '' }}>30%</option>
                  <option value="0.4" {{ 'selected' if selected_classification_test_size == 0.4 else '' }}>40%</option>
                </select>
              </div>
              <div>
                <label for="classification_cv_folds">Cross-validation</label>
                <select id="classification_cv_folds" name="classification_cv_folds" required>
                  <option value="0" {{ 'selected' if selected_classification_cv_folds == 0 else '' }}>Off</option>
                  <option value="3" {{ 'selected' if selected_classification_cv_folds == 3 else '' }}>3 folds</option>
                  <option value="5" {{ 'selected' if selected_classification_cv_folds == 5 else '' }}>5 folds</option>
                  <option value="10" {{ 'selected' if selected_classification_cv_folds == 10 else '' }}>10 folds</option>
                </select>
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
            {% if model_output.downloads %}
              <div class="download-links">
                {% for download in model_output.downloads %}
                  <a href="{{ download.href }}">{{ download.label }}</a>
                {% endfor %}
              </div>
            {% endif %}
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
                <label for="regression_test_size">Test set size</label>
                <select id="regression_test_size" name="regression_test_size" required>
                  <option value="0.2" {{ 'selected' if selected_regression_test_size == 0.2 else '' }}>20%</option>
                  <option value="0.25" {{ 'selected' if selected_regression_test_size == 0.25 else '' }}>25%</option>
                  <option value="0.3" {{ 'selected' if selected_regression_test_size == 0.3 else '' }}>30%</option>
                  <option value="0.4" {{ 'selected' if selected_regression_test_size == 0.4 else '' }}>40%</option>
                </select>
              </div>
              <div>
                <label for="regression_cv_folds">Cross-validation</label>
                <select id="regression_cv_folds" name="regression_cv_folds" required>
                  <option value="0" {{ 'selected' if selected_regression_cv_folds == 0 else '' }}>Off</option>
                  <option value="3" {{ 'selected' if selected_regression_cv_folds == 3 else '' }}>3 folds</option>
                  <option value="5" {{ 'selected' if selected_regression_cv_folds == 5 else '' }}>5 folds</option>
                  <option value="10" {{ 'selected' if selected_regression_cv_folds == 10 else '' }}>10 folds</option>
                </select>
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
            {% if regression_output.downloads %}
              <div class="download-links">
                {% for download in regression_output.downloads %}
                  <a href="{{ download.href }}">{{ download.label }}</a>
                {% endfor %}
              </div>
            {% endif %}
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


def get_db_connection():
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


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


def classification_estimator(model_name):
    if model_name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    if model_name == "tree":
        return DecisionTreeClassifier(max_depth=4, random_state=42)
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    if model_name == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42))
    if model_name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights="distance"))
    raise ValueError("Unknown classification model.")


def regression_estimator(model_name):
    if model_name == "linear":
        return LinearRegression()
    if model_name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if model_name == "lasso":
        return make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=10000, random_state=42))
    if model_name == "random_forest":
        return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    if model_name == "svr":
        return make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale"))
    if model_name == "knn":
        return make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance"))
    raise ValueError("Unknown regression model.")


def append_classification_cv(metrics, model_name, x_encoded, target_values, folds):
    if folds <= 1:
        return metrics

    y_codes, _ = encode_target(target_values)
    min_class_count = int(pd.Series(y_codes).value_counts().min())
    actual_folds = min(folds, min_class_count)
    if actual_folds < 2:
        metrics.append({"label": "CV accuracy", "value": "Not enough class balance"})
        return metrics

    cv = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    scores = cross_val_score(classification_estimator(model_name), x_encoded, y_codes, cv=cv, scoring="accuracy")
    metrics.extend([
        {"label": "CV folds", "value": actual_folds},
        {"label": "CV accuracy mean", "value": f"{scores.mean():.3f}"},
        {"label": "CV accuracy SD", "value": f"{scores.std():.3f}"},
    ])
    return metrics


def append_regression_cv(metrics, model_name, x_encoded, y, folds):
    if folds <= 1:
        return metrics

    actual_folds = min(folds, len(y))
    if actual_folds < 2:
        metrics.append({"label": "CV R squared", "value": "Not enough rows"})
        return metrics

    cv = KFold(n_splits=actual_folds, shuffle=True, random_state=42)
    estimator = regression_estimator(model_name)
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


def split_classification_data(target_values, x_encoded, test_size):
    y_codes, class_names = encode_target(target_values)
    indices = np.arange(len(y_codes))
    class_counts = pd.Series(y_codes).value_counts()
    stratify = y_codes if class_counts.min() >= 2 else None

    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)

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
    return details_frame(details).to_html(index=False, border=0, classes="model-details")


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


def fit_logistic_regression(data, target, predictors, test_size, cv_folds):
    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=True)
    split = split_classification_data(target_values, x_encoded, test_size)
    y = (split["target_train"] == classes[1]).astype(float).to_numpy()
    x_matrix = np.column_stack([np.ones(len(split["x_train"])), split["x_train"].to_numpy(dtype=float)])
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
    metrics = append_classification_cv(metrics, "logistic", x_encoded, target_values, cv_folds)
    details = {
        "Model": "Logistic regression",
        "Decision threshold": "0.5",
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": 42,
    }

    return {
        "title": "Logistic regression results",
        "description": f"Target: {target}. Positive class: {classes[1]}. Metrics are computed on the held-out test set.",
        "target": target,
        "positive_class": str(classes[1]),
        "metrics": metrics,
        "coefficients_html": coefficients.to_html(index=False, border=0, classes="coefficients", float_format="{:.4f}".format),
        "importances_html": None,
        "details_html": details_table(details),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
        "download_data": {
            "coefficients": coefficients.to_csv(index=False),
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(details).to_csv(index=False),
        },
    }


def fit_tree_model(data, target, predictors, test_size, cv_folds):
    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False)
    split = split_classification_data(target_values, x_encoded, test_size)
    class_names = split["class_names"]
    min_samples_leaf = max(1, int(len(split["y_train"]) * 0.02))

    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
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
    metrics = append_classification_cv(metrics, "tree", x_encoded, target_values, cv_folds)
    details = {
        "Model": "Decision tree classifier",
        "Max depth": tree.max_depth,
        "Minimum samples per leaf": min_samples_leaf,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": 42,
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
        "details_html": details_table(details),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": tree_plot,
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(details).to_csv(index=False),
        },
    }


def fit_random_forest_model(data, target, predictors, test_size, cv_folds):
    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False)
    split = split_classification_data(target_values, x_encoded, test_size)
    class_names = split["class_names"]
    min_samples_leaf = max(1, int(len(split["y_train"]) * 0.01))
    model = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
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
    metrics = append_classification_cv(metrics, "random_forest", x_encoded, target_values, cv_folds)
    details = {
        "Model": "RandomForestClassifier",
        "Trees": model.n_estimators,
        "Minimum samples per leaf": min_samples_leaf,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": 42,
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
        "details_html": details_table(details),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(details).to_csv(index=False),
        },
    }


def fit_gradient_boosting_model(data, target, predictors, test_size, cv_folds):
    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False)
    split = split_classification_data(target_values, x_encoded, test_size)
    class_names = split["class_names"]
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
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
    metrics = append_classification_cv(metrics, "gradient_boosting", x_encoded, target_values, cv_folds)
    details = {
        "Model": "GradientBoostingClassifier",
        "Boosting stages": model.n_estimators,
        "Learning rate": model.learning_rate,
        "Maximum tree depth": model.max_depth,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": 42,
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
        "details_html": details_table(details),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(details).to_csv(index=False),
        },
    }


def fit_svm_model(data, target, predictors, test_size, cv_folds):
    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False)
    split = split_classification_data(target_values, x_encoded, test_size)
    class_names = split["class_names"]
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(split["x_train"])
    scaled_test = scaler.transform(split["x_test"])
    model = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
    model.fit(scaled_train, split["y_train"])
    predicted_labels = class_names[model.predict(scaled_test)]
    accuracy = accuracy_score(split["target_test"], predicted_labels)
    confusion = confusion_table(split["target_test"].to_numpy(), predicted_labels)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
        {"label": "Test accuracy", "value": f"{accuracy:.3f}"},
        {"label": "Kernel", "value": model.kernel},
        {"label": "Support vectors", "value": int(np.sum(model.n_support_))},
    ]
    metrics = append_classification_cv(metrics, "svm", x_encoded, target_values, cv_folds)
    details = {
        "Model": "SVC",
        "Kernel": model.kernel,
        "C": model.C,
        "Gamma": model.gamma,
        "Support vectors by class": ", ".join(str(value) for value in model.n_support_),
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
        "importances_html": None,
        "details_html": details_table(details),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
        "download_data": {
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(details).to_csv(index=False),
        },
    }


def fit_knn_model(data, target, predictors, test_size, cv_folds):
    _, target_values, classes, x_encoded = prepare_classification_data(data, target, predictors, binary_only=False)
    split = split_classification_data(target_values, x_encoded, test_size)
    class_names = split["class_names"]
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(split["x_train"])
    scaled_test = scaler.transform(split["x_test"])
    neighbors = min(5, len(split["y_train"]))
    model = KNeighborsClassifier(n_neighbors=neighbors, weights="distance")
    model.fit(scaled_train, split["y_train"])
    predicted_labels = class_names[model.predict(scaled_test)]
    accuracy = accuracy_score(split["target_test"], predicted_labels)
    confusion = confusion_table(split["target_test"].to_numpy(), predicted_labels)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
        {"label": "Test accuracy", "value": f"{accuracy:.3f}"},
        {"label": "Neighbors", "value": neighbors},
        {"label": "Weights", "value": model.weights},
    ]
    metrics = append_classification_cv(metrics, "knn", x_encoded, target_values, cv_folds)
    details = {
        "Model": "KNeighborsClassifier",
        "Neighbors": neighbors,
        "Weights": model.weights,
        "Distance metric": model.metric,
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
        "importances_html": None,
        "details_html": details_table(details),
        "confusion_html": confusion.to_html(border=0, classes="confusion"),
        "tree_plot": None,
        "download_data": {
            "confusion_matrix": confusion.to_csv(),
            "details": details_frame(details).to_csv(index=False),
        },
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


def split_regression_data(y, x_encoded, test_size):
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
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
    return coefficients.to_html(index=False, border=0, classes="coefficients", float_format="{:.4f}".format)


def fit_linear_regression(data, target, predictors, test_size, cv_folds):
    y, x_encoded = prepare_regression_data(data, target, predictors)
    split = split_regression_data(y, x_encoded, test_size)
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
    metrics = append_regression_cv(metrics, "linear", x_encoded, y, cv_folds)
    details = {
        "Model": "Ordinary least squares",
        "Train residual df": train_residual_df,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": 42,
    }

    return {
        "title": "Linear Regression results",
        "description": f"Target: {target}. Ordinary least squares fit; metrics are computed on the held-out test set.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": coefficients.to_html(index=False, border=0, classes="coefficients", float_format="{:.4f}".format),
        "importances_html": None,
        "details_html": details_table(details),
        "download_data": {
            "coefficients": coefficients.to_csv(index=False),
            "details": details_frame(details).to_csv(index=False),
        },
    }


def fit_ridge_regression(data, target, predictors, test_size, cv_folds):
    y, x_encoded = prepare_regression_data(data, target, predictors)
    split = split_regression_data(y, x_encoded, test_size)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(split["x_train"])
    scaled_test = scaler.transform(split["x_test"])
    model = Ridge(alpha=1.0)
    model.fit(scaled_train, split["y_train"])
    predictions = model.predict(scaled_test)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions, parameter_count=scaled_train.shape[1] + 1)[1:]
    metrics = append_regression_cv(metrics, "ridge", x_encoded, y, cv_folds)
    details = {
        "Model": "Ridge",
        "Alpha": model.alpha,
        "Feature scaling": "StandardScaler",
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": 42,
    }
    coefficients = pd.DataFrame({"Term": ["Intercept"] + list(x_encoded.columns), "Coefficient": [model.intercept_] + list(model.coef_)})

    return {
        "title": "Ridge Regression results",
        "description": f"Target: {target}. Predictors were standardized using the training split; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": coefficients.to_html(index=False, border=0, classes="coefficients", float_format="{:.4f}".format),
        "importances_html": None,
        "details_html": details_table(details),
        "download_data": {
            "coefficients": coefficients.to_csv(index=False),
            "details": details_frame(details).to_csv(index=False),
        },
    }


def fit_lasso_regression(data, target, predictors, test_size, cv_folds):
    y, x_encoded = prepare_regression_data(data, target, predictors)
    split = split_regression_data(y, x_encoded, test_size)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(split["x_train"])
    scaled_test = scaler.transform(split["x_test"])
    model = Lasso(alpha=0.1, max_iter=10000, random_state=42)
    model.fit(scaled_train, split["y_train"])
    predictions = model.predict(scaled_test)
    nonzero_count = int(np.sum(np.abs(model.coef_) > 1e-8))

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions, parameter_count=nonzero_count + 1)[1:]
    metrics = append_regression_cv(metrics, "lasso", x_encoded, y, cv_folds)
    details = {
        "Model": "Lasso",
        "Alpha": model.alpha,
        "Non-zero coefficients": nonzero_count,
        "Feature scaling": "StandardScaler",
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": 42,
    }
    coefficients = pd.DataFrame({"Term": ["Intercept"] + list(x_encoded.columns), "Coefficient": [model.intercept_] + list(model.coef_)})

    return {
        "title": "Lasso Regression results",
        "description": f"Target: {target}. Predictors were standardized using the training split; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": coefficients.to_html(index=False, border=0, classes="coefficients", float_format="{:.4f}".format),
        "importances_html": None,
        "details_html": details_table(details),
        "download_data": {
            "coefficients": coefficients.to_csv(index=False),
            "details": details_frame(details).to_csv(index=False),
        },
    }


def fit_random_forest_regression(data, target, predictors, test_size, cv_folds):
    y, x_encoded = prepare_regression_data(data, target, predictors)
    split = split_regression_data(y, x_encoded, test_size)
    min_samples_leaf = max(1, int(len(split["y_train"]) * 0.01))
    model = RandomForestRegressor(
        n_estimators=200,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(split["x_train"], split["y_train"])
    predictions = model.predict(split["x_test"])

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions)[1:]
    metrics = append_regression_cv(metrics, "random_forest", x_encoded, y, cv_folds)
    details = {
        "Model": "RandomForestRegressor",
        "Trees": model.n_estimators,
        "Minimum samples per leaf": min_samples_leaf,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": 42,
    }
    importances = pd.DataFrame({"Predictor": x_encoded.columns, "Importance": model.feature_importances_})

    return {
        "title": "Random Forest Regression results",
        "description": f"Target: {target}. Ensemble of regression trees; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, model.feature_importances_),
        "details_html": details_table(details),
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "details": details_frame(details).to_csv(index=False),
        },
    }


def fit_gradient_boosting_regression(data, target, predictors, test_size, cv_folds):
    y, x_encoded = prepare_regression_data(data, target, predictors)
    split = split_regression_data(y, x_encoded, test_size)
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    model.fit(split["x_train"], split["y_train"])
    predictions = model.predict(split["x_test"])

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions)[1:]
    metrics = append_regression_cv(metrics, "gradient_boosting", x_encoded, y, cv_folds)
    details = {
        "Model": "GradientBoostingRegressor",
        "Boosting stages": model.n_estimators,
        "Learning rate": model.learning_rate,
        "Maximum tree depth": model.max_depth,
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": 42,
    }
    importances = pd.DataFrame({"Predictor": x_encoded.columns, "Importance": model.feature_importances_})

    return {
        "title": "Gradient Boosting Regression results",
        "description": f"Target: {target}. Sequential boosted-tree regression model; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": importance_table(x_encoded.columns, model.feature_importances_),
        "details_html": details_table(details),
        "download_data": {
            "variable_importance": importances.to_csv(index=False),
            "details": details_frame(details).to_csv(index=False),
        },
    }


def fit_svr_regression(data, target, predictors, test_size, cv_folds):
    y, x_encoded = prepare_regression_data(data, target, predictors)
    split = split_regression_data(y, x_encoded, test_size)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(split["x_train"])
    scaled_test = scaler.transform(split["x_test"])
    model = SVR(kernel="rbf", C=1.0, epsilon=0.1, gamma="scale")
    model.fit(scaled_train, split["y_train"])
    predictions = model.predict(scaled_test)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions)[1:]
    metrics = append_regression_cv(metrics, "svr", x_encoded, y, cv_folds)
    details = {
        "Model": "SVR",
        "Kernel": model.kernel,
        "C": model.C,
        "Epsilon": model.epsilon,
        "Gamma": model.gamma,
        "Support vectors": len(model.support_),
        "Feature scaling": "StandardScaler",
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": 42,
    }

    return {
        "title": "Support Vector Regression results",
        "description": f"Target: {target}. Predictors were standardized using the training split; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": None,
        "details_html": details_table(details),
        "download_data": {
            "details": details_frame(details).to_csv(index=False),
        },
    }


def fit_knn_regression(data, target, predictors, test_size, cv_folds):
    y, x_encoded = prepare_regression_data(data, target, predictors)
    split = split_regression_data(y, x_encoded, test_size)
    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(split["x_train"])
    scaled_test = scaler.transform(split["x_test"])
    neighbors = min(5, len(split["y_train"]))
    model = KNeighborsRegressor(n_neighbors=neighbors, weights="distance")
    model.fit(scaled_train, split["y_train"])
    predictions = model.predict(scaled_test)

    metrics = [
        {"label": "Train rows", "value": len(split["x_train"])},
        {"label": "Test rows", "value": len(split["x_test"])},
    ] + regression_metric_list(split["y_test"], predictions)[1:]
    metrics = append_regression_cv(metrics, "knn", x_encoded, y, cv_folds)
    details = {
        "Model": "KNeighborsRegressor",
        "Neighbors": neighbors,
        "Weights": model.weights,
        "Distance metric": model.metric,
        "Feature scaling": "StandardScaler",
        "Test set size": f"{test_size:.0%}",
        "CV folds requested": cv_folds if cv_folds else "Off",
        "Random seed": 42,
    }

    return {
        "title": "kNN Regression results",
        "description": f"Target: {target}. Predictors were standardized using the training split; test-set metrics are shown.",
        "target": target,
        "metrics": metrics,
        "coefficients_html": None,
        "importances_html": None,
        "details_html": details_table(details),
        "download_data": {
            "details": details_frame(details).to_csv(index=False),
        },
    }


@app.route("/", methods=["GET", "POST"])
def index():
    active_tab = request.form.get("active_tab", "data")
    auth_error = None
    data_error = None
    classification_error = None
    regression_error = None
    model_output = None
    regression_output = None
    selected_classification_model = "logistic"
    selected_regression_model = "linear"
    selected_classification_test_size = 0.2
    selected_regression_test_size = 0.2
    selected_classification_cv_folds = 0
    selected_regression_cv_folds = 0
    selected_target = None
    selected_predictors = []
    selected_regression_target = None
    selected_regression_predictors = []
    form_name = request.form.get("form_name")

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

    if authenticated and request.method == "POST" and form_name == "classification":
        active_tab = "classification"
        selected_classification_model = request.form.get("classification_model", "logistic")
        selected_classification_test_size = parse_test_size(request.form.get("classification_test_size"))
        selected_classification_cv_folds = parse_cv_folds(request.form.get("classification_cv_folds"))
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
                model_output = fit_model(
                    dataset["data"],
                    selected_target,
                    selected_predictors,
                    selected_classification_test_size,
                    selected_classification_cv_folds,
                )
                model_output = register_downloads("classification", model_output)
            except Exception as exc:
                classification_error = str(exc)

    if authenticated and request.method == "POST" and form_name == "regression":
        active_tab = "regression"
        selected_regression_model = request.form.get("regression_model", "linear")
        selected_regression_test_size = parse_test_size(request.form.get("regression_test_size"))
        selected_regression_cv_folds = parse_cv_folds(request.form.get("regression_cv_folds"))
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
                regression_output = fit_model(
                    dataset["data"],
                    selected_regression_target,
                    selected_regression_predictors,
                    selected_regression_test_size,
                    selected_regression_cv_folds,
                )
                regression_output = register_downloads("regression", regression_output)
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
            selected_regression_target = "regression_target" if "regression_target" in numeric_columns else numeric_columns[0]
            selected_regression_predictors = [column for column in numeric_columns if column != selected_regression_target]
        else:
            selected_regression_target = columns[0]
            selected_regression_predictors = columns[1:]

    table_html = preview_table(data) if has_data else None
    row_count = min(25, len(data)) if has_data else 0
    total_rows = len(data) if has_data else 0

    return render_template_string(
        PAGE_TEMPLATE,
        active_tab=active_tab,
        auth_error=auth_error,
        is_authenticated=authenticated,
        current_username=session.get("username"),
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
        selected_classification_test_size=selected_classification_test_size,
        selected_classification_cv_folds=selected_classification_cv_folds,
        selected_target=selected_target,
        selected_predictors=selected_predictors,
        selected_regression_model=selected_regression_model,
        selected_regression_test_size=selected_regression_test_size,
        selected_regression_cv_folds=selected_regression_cv_folds,
        selected_regression_target=selected_regression_target,
        selected_regression_predictors=selected_regression_predictors,
        model_output=model_output,
        regression_output=regression_output,
    )


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/download/<result_type>/<artifact>")
def download_result(result_type, artifact):
    if not is_user_authenticated():
        return Response("Authentication required.", status=401, mimetype="text/plain")

    current_id = dataset_id()
    csv_data = DOWNLOADS.get(current_id, {}).get(result_type, {}).get(artifact)
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
