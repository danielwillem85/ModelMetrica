import sqlite3
from pathlib import Path


DB_PATH = Path(__file__).with_name("modelmetrica_users.sqlite3")


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
