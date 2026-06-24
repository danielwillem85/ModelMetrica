from flask import session
from werkzeug.security import check_password_hash, generate_password_hash

from db import get_db_connection
from email_notifications import send_registration_email


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
        user_id = cursor.lastrowid
    send_registration_email(user_id, username)
    return user_id


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
