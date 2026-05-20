import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Optional


def hash_password(password: str, salt: Optional[str] = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120_000)
    return f"pbkdf2_sha256${salt}${base64.b64encode(digest).decode()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, salt, expected = password_hash.split("$", 2)
    except ValueError:
        return False
    if algorithm != "pbkdf2_sha256":
        return False
    actual = hash_password(password, salt).split("$", 2)[2]
    return hmac.compare_digest(actual, expected)


def create_session_token(username: str, secret: str, ttl_seconds: int = 86_400) -> str:
    payload = {"sub": username, "exp": int(time.time()) + ttl_seconds}
    raw_payload = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode()
    signature = hmac.new(secret.encode(), raw_payload.encode(), hashlib.sha256).hexdigest()
    return f"{raw_payload}.{signature}"


def verify_session_token(token: str, secret: str) -> Optional[str]:
    try:
        raw_payload, signature = token.split(".", 1)
        expected = hmac.new(secret.encode(), raw_payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            return None
        payload = json.loads(base64.urlsafe_b64decode(raw_payload.encode()).decode())
    except Exception:
        return None
    if int(payload.get("exp", 0)) < int(time.time()):
        return None
    return payload.get("sub")
