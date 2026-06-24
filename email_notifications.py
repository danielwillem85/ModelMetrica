import html
import json
import os
import ssl
from urllib.error import HTTPError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


RESEND_API_BASE = "https://api.resend.com"
RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "")
RESEND_FROM = os.environ.get("RESEND_FROM", "ModelMetrica <onboarding@resend.dev>")
RESEND_NOTIFICATION_TO = os.environ.get("RESEND_NOTIFICATION_TO", "danielvdpalm@gmail.com")


def create_api_ssl_context():
    try:
        import certifi

        ssl_context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ssl_context = ssl.create_default_context()
    if hasattr(ssl, "VERIFY_X509_STRICT"):
        ssl_context.verify_flags &= ~ssl.VERIFY_X509_STRICT
    if hasattr(ssl, "enum_certificates"):
        windows_certs = []
        for store_name in ("ROOT", "CA"):
            for certificate, encoding, trust in ssl.enum_certificates(store_name):
                if encoding == "x509_asn" and (trust is True or "1.3.6.1.5.5.7.3.1" in trust):
                    windows_certs.append(ssl.DER_cert_to_PEM_cert(certificate))
        if windows_certs:
            ssl_context.load_verify_locations(cadata="\n".join(windows_certs))
    return ssl_context


def resend_configured():
    return bool(RESEND_API_KEY and not RESEND_API_KEY.startswith("replace_"))


def send_resend_notification(subject, text_part, html_part):
    if not resend_configured():
        return None

    data = {
        "from": RESEND_FROM,
        "to": [RESEND_NOTIFICATION_TO],
        "subject": subject,
        "text": text_part,
        "html": html_part,
    }

    try:
        request_obj = Request(
            urljoin(RESEND_API_BASE, "/emails"),
            data=json.dumps(data).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": f"Bearer {RESEND_API_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "ModelMetrica/1.0",
            },
        )
        with urlopen(request_obj, timeout=20, context=create_api_ssl_context()) as response:
            body = response.read().decode("utf-8")
            return {"status_code": response.status, "body": json.loads(body) if body else {}}
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"Resend notification failed: HTTP {exc.code} {error_body}")
        return None
    except Exception as exc:
        print(f"Resend notification failed: {exc}")
        return None


def send_registration_email(user_id, username):
    escaped_username = html.escape(username)
    send_resend_notification(
        "New ModelMetrica user registration",
        f"New user registered: {username} (user id {user_id}).",
        f"<h3>New ModelMetrica user registration</h3><p>User: {escaped_username}</p><p>User ID: {user_id}</p>",
    )


def send_subscription_success_email(user_id, username, subscription_id):
    escaped_username = html.escape(username)
    escaped_subscription_id = html.escape(subscription_id or "-")
    send_resend_notification(
        "New ModelMetrica Pro subscription",
        f"User {username} (user id {user_id}) subscribed successfully. Subscription ID: {subscription_id or '-'}",
        (
            "<h3>New ModelMetrica Pro subscription</h3>"
            f"<p>User: {escaped_username}</p>"
            f"<p>User ID: {user_id}</p>"
            f"<p>Subscription ID: {escaped_subscription_id}</p>"
        ),
    )
