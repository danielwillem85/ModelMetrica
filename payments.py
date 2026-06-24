import json
import os
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


MOLLIE_API_BASE = "https://api.mollie.com/v2"
MOLLIE_API_KEY = os.environ.get("MOLLIE_API_KEY", "")
MOLLIE_BASE_URL = os.environ.get("MOLLIE_BASE_URL", "")
SUBSCRIPTION_AMOUNT = os.environ.get("MODELMETRICA_SUBSCRIPTION_AMOUNT", "9.99")
SUBSCRIPTION_CURRENCY = os.environ.get("MODELMETRICA_SUBSCRIPTION_CURRENCY", "EUR")
SUBSCRIPTION_INTERVAL = os.environ.get("MODELMETRICA_SUBSCRIPTION_INTERVAL", "1 month")
SUBSCRIPTION_DESCRIPTION = os.environ.get("MODELMETRICA_SUBSCRIPTION_DESCRIPTION", "ModelMetrica Pro subscription")


@dataclass
class PaymentService:
    get_db_connection: object
    url_for: object
    send_subscription_success_email: object

    def api_key_valid(self):
        return MOLLIE_API_KEY.startswith(("test_", "live_"))

    def external_url_for(self, endpoint, **values):
        if MOLLIE_BASE_URL:
            return urljoin(MOLLIE_BASE_URL.rstrip("/") + "/", self.url_for(endpoint, **values).lstrip("/"))
        return self.url_for(endpoint, _external=True, **values)

    def webhook_url_valid(self):
        webhook_url = self.external_url_for("mollie_webhook")
        parsed = urlparse(webhook_url)
        return parsed.scheme in {"http", "https"} and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}

    def configured(self):
        return bool(MOLLIE_API_KEY) and self.api_key_valid() and self.webhook_url_valid()

    def request(self, method, path, payload=None, require_public_webhook=False):
        if not MOLLIE_API_KEY:
            raise RuntimeError("Mollie API key is not configured.")
        if not self.api_key_valid():
            raise RuntimeError("MOLLIE_API_KEY must be a Mollie profile API key that starts with test_ or live_.")
        if require_public_webhook and not self.webhook_url_valid():
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

    def ensure_customer(self, user):
        if user["mollie_customer_id"]:
            return user["mollie_customer_id"]
        customer = self.request(
            "POST",
            "/customers",
            {
                "name": user["username"],
                "metadata": {"user_id": user["id"]},
            },
        )
        customer_id = customer["id"]
        with self.get_db_connection() as connection:
            connection.execute("UPDATE users SET mollie_customer_id = ? WHERE id = ?", (customer_id, user["id"]))
        return customer_id

    def create_first_payment(self, user):
        customer_id = self.ensure_customer(user)
        payment = self.request(
            "POST",
            "/payments",
            {
                "amount": {"currency": SUBSCRIPTION_CURRENCY, "value": SUBSCRIPTION_AMOUNT},
                "customerId": customer_id,
                "sequenceType": "first",
                "description": SUBSCRIPTION_DESCRIPTION,
                "redirectUrl": self.external_url_for("subscription_return"),
                "webhookUrl": self.external_url_for("mollie_webhook"),
                "metadata": {"user_id": user["id"]},
            },
            require_public_webhook=True,
        )
        checkout_url = (payment.get("_links") or {}).get("checkout", {}).get("href")
        if not checkout_url:
            raise RuntimeError("Mollie did not return a checkout URL.")
        with self.get_db_connection() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO subscription_payments
                (user_id, mollie_payment_id, mollie_customer_id, status, checkout_url, updated_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (user["id"], payment["id"], customer_id, payment.get("status", "open"), checkout_url),
            )
        return checkout_url

    def create_subscription(self, user_id, customer_id):
        with self.get_db_connection() as connection:
            user = connection.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if user and user["mollie_subscription_id"]:
            return user["mollie_subscription_id"]
        subscription = self.request(
            "POST",
            f"/customers/{customer_id}/subscriptions",
            {
                "amount": {"currency": SUBSCRIPTION_CURRENCY, "value": SUBSCRIPTION_AMOUNT},
                "interval": SUBSCRIPTION_INTERVAL,
                "description": SUBSCRIPTION_DESCRIPTION,
                "webhookUrl": self.external_url_for("mollie_webhook"),
                "metadata": {"user_id": user_id},
            },
            require_public_webhook=True,
        )
        subscription_id = subscription["id"]
        with self.get_db_connection() as connection:
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
        username = user["username"] if user else str(user_id)
        self.send_subscription_success_email(user_id, username, subscription_id)
        return subscription_id

    def sync_payment(self, payment_id):
        payment = self.request("GET", f"/payments/{payment_id}")
        with self.get_db_connection() as connection:
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
                subscription_id = self.create_subscription(row["user_id"], customer_id)
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

    def cancel_subscription(self, user):
        customer_id = user["mollie_customer_id"]
        subscription_id = user["mollie_subscription_id"]
        if customer_id and subscription_id:
            self.request("DELETE", f"/customers/{customer_id}/subscriptions/{subscription_id}")
        with self.get_db_connection() as connection:
            connection.execute(
                """
                UPDATE users
                SET subscription_status = 'inactive',
                    mollie_subscription_id = NULL,
                    subscription_updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (user["id"],),
            )
