from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from catmaster.webui.auth import SESSION_COOKIE_NAME
from catmaster.webui.server import create_app


def _captcha_answer(question: str) -> str:
    numbers = [int(value) for value in re.findall(r"\d+", question)]
    return str(sum(numbers))


def _register(client: TestClient, username: str, password: str = "correct-password-123") -> dict:
    captcha = client.get("/api/auth/captcha")
    assert captcha.status_code == 200
    captcha_payload = captcha.json()
    response = client.post(
        "/api/auth/register",
        json={
            "username": username,
            "password": password,
            "captcha_id": captcha_payload["captcha_id"],
            "captcha_answer": _captcha_answer(captcha_payload["question"]),
        },
    )
    assert response.status_code == 200
    return response.json()


def test_login_mode_requires_authentication_for_webui_api(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)

    status = client.get("/api/auth/status")
    assert status.status_code == 200
    assert status.json()["auth_enabled"] is True
    assert status.json()["authenticated"] is False
    assert status.json()["registration_enabled"] is True

    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 401


def test_register_hashes_password_and_bootstraps_locked_user_root(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)

    payload = _register(client, "Alice_1")

    assert payload["authenticated"] is True
    assert payload["username"] == "alice_1"
    assert SESSION_COOKIE_NAME in client.cookies

    with sqlite3.connect(str(tmp_path / ".webui_auth" / "auth.sqlite")) as conn:
        row = conn.execute("SELECT password_hash FROM users WHERE username = ?", ("alice_1",)).fetchone()
    assert row is not None
    password_hash = str(row[0])
    assert password_hash.startswith("pbkdf2_sha256$")
    assert password_hash != "correct-password-123"

    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 200
    boot_payload = bootstrap.json()
    user_root = tmp_path.resolve() / "users" / "alice_1"
    assert boot_payload["workspace_root"] == str(user_root)
    assert boot_payload["workspace_root_locked"] is True
    assert boot_payload["workspace_name"] == "default"
    assert boot_payload["auth"]["username"] == "alice_1"
    assert (user_root / "default" / "files").is_dir()

    outside = tmp_path / "outside-root"
    refresh = client.post(
        f"/api/session/{boot_payload['ctx']}/workspace/refresh",
        json={"root_path": str(outside), "workspace": "default"},
    )
    assert refresh.status_code == 200
    assert refresh.json()["workspace_root"] == str(user_root)


def test_same_ctx_is_isolated_between_authenticated_users(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path))
    alice = TestClient(app)
    bob = TestClient(app)

    _register(alice, "alice")
    _register(bob, "bob")

    alice_boot = alice.get("/api/bootstrap", params={"ctx": "ctx_shared_001"})
    assert alice_boot.status_code == 200
    ctx = alice_boot.json()["ctx"]
    alice_create = alice.post(f"/api/session/{ctx}/workspace/create", json={"workspace": "private"})
    assert alice_create.status_code == 200
    assert alice_create.json()["ok"] is True

    bob_boot = bob.get("/api/bootstrap", params={"ctx": "ctx_shared_001"})
    assert bob_boot.status_code == 200
    bob_payload = bob_boot.json()

    alice_root = tmp_path.resolve() / "users" / "alice"
    bob_root = tmp_path.resolve() / "users" / "bob"
    assert alice_boot.json()["workspace_root"] == str(alice_root)
    assert bob_payload["workspace_root"] == str(bob_root)
    assert (alice_root / "private").is_dir()
    assert not (bob_root / "private").exists()
    assert "private" not in {item["value"] for item in bob_payload["workspaces"]}


def test_disable_registration_rejects_signup_but_preserves_existing_login(tmp_path: Path) -> None:
    bootstrap_client = TestClient(create_app(project_space_root=str(tmp_path)))
    _register(bootstrap_client, "existing_user")

    client = TestClient(create_app(project_space_root=str(tmp_path), disable_registration=True))
    status = client.get("/api/auth/status")
    assert status.status_code == 200
    assert status.json()["auth_enabled"] is True
    assert status.json()["authenticated"] is False
    assert status.json()["registration_enabled"] is False
    assert status.json()["has_users"] is True

    captcha = client.get("/api/auth/captcha")
    assert captcha.status_code == 403
    assert captcha.json()["detail"] == "Registration is disabled."

    register = client.post(
        "/api/auth/register",
        json={
            "username": "blocked_user",
            "password": "correct-password-123",
            "captcha_id": "",
            "captcha_answer": "",
        },
    )
    assert register.status_code == 403
    assert register.json()["detail"] == "Registration is disabled."

    login = client.post(
        "/api/auth/login",
        json={"username": "existing_user", "password": "correct-password-123"},
    )
    assert login.status_code == 200
    assert login.json()["authenticated"] is True
    assert login.json()["registration_enabled"] is False
    assert client.get("/api/bootstrap").status_code == 200


def test_no_login_uses_admin_workspace_without_auth_cookie(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)

    status = client.get("/api/auth/status")
    assert status.status_code == 200
    assert status.json()["auth_enabled"] is False
    assert status.json()["authenticated"] is True
    assert status.json()["username"] == "admin"
    assert status.json()["registration_enabled"] is False

    bootstrap = client.get("/api/bootstrap")
    assert bootstrap.status_code == 200
    payload = bootstrap.json()
    assert payload["workspace_root"] == str(tmp_path.resolve())
    assert payload["workspace_root_locked"] is True
    assert payload["workspace_name"] == "admin"
    assert (tmp_path / "admin" / "files").is_dir()
