from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
import string
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PASSWORD_ALGORITHM = "pbkdf2_sha256"
PASSWORD_ITERATIONS = 260_000
SESSION_COOKIE_NAME = "catmaster_webui_session"
SESSION_TTL_SECONDS = 14 * 24 * 60 * 60
CAPTCHA_TTL_SECONDS = 10 * 60
USERNAME_CHARS = set(string.ascii_letters + string.digits + "_.-")


@dataclass(frozen=True)
class AuthIdentity:
    username: str
    authenticated: bool
    auth_enabled: bool


class AuthManager:
    def __init__(self, *, auth_root: str | Path, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.auth_root = Path(auth_root).expanduser().resolve()
        self.auth_root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.auth_root / "auth.sqlite"
        self._lock = threading.RLock()
        self._captchas: dict[str, tuple[str, float]] = {}
        self._init_db()

    @staticmethod
    def normalize_username(username: str) -> str:
        value = str(username or "").strip().lower()
        if not (3 <= len(value) <= 40):
            raise ValueError("Username must be 3-40 characters.")
        if any(ch not in USERNAME_CHARS for ch in value):
            raise ValueError("Username may only contain letters, numbers, dot, dash, and underscore.")
        if value in {".", ".."}:
            raise ValueError("Username is not allowed.")
        return value

    @staticmethod
    def validate_password(password: str) -> str:
        value = str(password or "")
        if len(value) < 8:
            raise ValueError("Password must be at least 8 characters.")
        if len(value) > 256:
            raise ValueError("Password is too long.")
        return value

    def user_root(self, username: str, *, base_project_space_root: str | Path) -> Path:
        name = self.normalize_username(username)
        return Path(base_project_space_root).expanduser().resolve() / "users" / name

    def default_identity(self) -> AuthIdentity:
        return AuthIdentity(username="admin", authenticated=True, auth_enabled=False)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_username ON sessions(username)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)")
            conn.commit()

    @staticmethod
    def _hash_password(password: str, *, salt: bytes | None = None) -> str:
        salt_bytes = salt or os.urandom(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt_bytes,
            PASSWORD_ITERATIONS,
        )
        return "$".join(
            [
                PASSWORD_ALGORITHM,
                str(PASSWORD_ITERATIONS),
                salt_bytes.hex(),
                digest.hex(),
            ]
        )

    @staticmethod
    def _verify_password(password: str, stored_hash: str) -> bool:
        try:
            algorithm, iterations_text, salt_hex, digest_hex = str(stored_hash or "").split("$", 3)
            if algorithm != PASSWORD_ALGORITHM:
                return False
            iterations = int(iterations_text)
            salt = bytes.fromhex(salt_hex)
            expected = bytes.fromhex(digest_hex)
        except Exception:
            return False
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(str(token or "").encode("utf-8")).hexdigest()

    def has_users(self) -> bool:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT 1 FROM users LIMIT 1").fetchone()
        return row is not None

    def create_captcha(self) -> dict[str, str]:
        left = secrets.randbelow(8) + 2
        right = secrets.randbelow(8) + 2
        captcha_id = secrets.token_urlsafe(18)
        answer = str(left + right)
        expires_at = time.time() + CAPTCHA_TTL_SECONDS
        with self._lock:
            self._captchas[captcha_id] = (answer, expires_at)
            self._prune_captchas_unlocked()
        return {"captcha_id": captcha_id, "question": f"{left} + {right} = ?"}

    def verify_captcha(self, captcha_id: str, answer: str) -> bool:
        key = str(captcha_id or "").strip()
        value = str(answer or "").strip()
        with self._lock:
            stored = self._captchas.pop(key, None)
            self._prune_captchas_unlocked()
        if not stored:
            return False
        expected, expires_at = stored
        if time.time() > expires_at:
            return False
        return hmac.compare_digest(value, expected)

    def _prune_captchas_unlocked(self) -> None:
        now = time.time()
        stale = [key for key, (_answer, expires_at) in self._captchas.items() if expires_at <= now]
        for key in stale:
            self._captchas.pop(key, None)

    def register_user(self, *, username: str, password: str, captcha_id: str, captcha_answer: str) -> str:
        normalized = self.normalize_username(username)
        secret = self.validate_password(password)
        if not self.verify_captcha(captcha_id, captcha_answer):
            raise ValueError("Captcha answer is incorrect or expired.")
        password_hash = self._hash_password(secret)
        created_at = time.time()
        try:
            with self._lock, self._connect() as conn:
                conn.execute(
                    "INSERT INTO users(username, password_hash, created_at) VALUES (?, ?, ?)",
                    (normalized, password_hash, created_at),
                )
                conn.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError("Username is already registered.") from exc
        return normalized

    def authenticate_user(self, *, username: str, password: str) -> str:
        normalized = self.normalize_username(username)
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT password_hash FROM users WHERE username = ?", (normalized,)).fetchone()
        if row is None or not self._verify_password(str(password or ""), str(row["password_hash"] or "")):
            raise ValueError("Invalid username or password.")
        return normalized

    def create_session(self, username: str) -> str:
        normalized = self.normalize_username(username)
        token = secrets.token_urlsafe(32)
        now = time.time()
        expires_at = now + SESSION_TTL_SECONDS
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(token_hash, username, created_at, expires_at) VALUES (?, ?, ?, ?)",
                (self._hash_token(token), normalized, now, expires_at),
            )
            conn.execute("DELETE FROM sessions WHERE expires_at <= ?", (now,))
            conn.commit()
        return token

    def identity_for_token(self, token: str) -> AuthIdentity | None:
        if not self.enabled:
            return self.default_identity()
        token_hash = self._hash_token(token)
        now = time.time()
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT username, expires_at FROM sessions WHERE token_hash = ?",
                (token_hash,),
            ).fetchone()
            if row is None:
                return None
            if float(row["expires_at"] or 0.0) <= now:
                conn.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
                conn.commit()
                return None
        return AuthIdentity(username=str(row["username"]), authenticated=True, auth_enabled=True)

    def revoke_session(self, token: str) -> None:
        if not token:
            return
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (self._hash_token(token),))
            conn.commit()

    def public_status(self, identity: AuthIdentity | None) -> dict[str, Any]:
        auth_enabled = self.enabled
        authenticated = bool(identity and identity.authenticated)
        return {
            "auth_enabled": auth_enabled,
            "authenticated": authenticated,
            "username": identity.username if identity and authenticated else "",
            "registration_enabled": auth_enabled,
            "has_users": self.has_users() if auth_enabled else True,
        }
