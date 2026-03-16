from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient
from starlette.routing import Match

from catmaster.webui import server
from catmaster.webui.server import create_app
from catmaster.webui.session import WebSession


def _scope(path: str) -> dict:
    return {"type": "http", "path": path, "method": "GET", "root_path": ""}


def test_monitor_path_redirect_route_precedes_root_mount(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path))
    full_matches = [route for route in app.routes if route.matches(_scope("/monitor"))[0] == Match.FULL]
    assert full_matches
    assert getattr(full_matches[0], "path", None) == "/monitor"


def test_monitor_path_with_slash_hits_monitor_mount(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path))
    full_matches = [route for route in app.routes if route.matches(_scope("/monitor/"))[0] == Match.FULL]
    assert full_matches
    assert getattr(full_matches[0], "path", None) == "/monitor/"


def test_pages_load_react_static_bundle(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)

    home = client.get("/")
    assert home.status_code == 200
    assert '/static/app.css' in home.text
    assert '/static/app.js' in home.text

    monitor = client.get("/monitor/")
    assert monitor.status_code == 200
    assert '/static/app.css' in monitor.text
    assert '/static/app.js' in monitor.text


def test_coerce_int_treats_empty_string_as_default() -> None:
    assert server._coerce_int("", 0) == 0
    assert server._coerce_int("7", 0) == 7


def test_memory_route_returns_workspace_memory(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    (ws / "files").mkdir(parents=True)
    (ws / "metadata").mkdir(parents=True)
    db_path = ws / "metadata" / "deepagent_memory.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE store (prefix TEXT NOT NULL, key TEXT NOT NULL, value BLOB NOT NULL)")
    prefix = ".".join(("catmaster", WebSession._project_id_for_workspace(ws), "filesystem"))
    payload = {"content": ["Stable preference: prefer compact reports."]}
    conn.execute(
        "INSERT INTO store(prefix, key, value) VALUES (?, ?, ?)",
        (prefix, "/AGENTS.md", json.dumps(payload).encode("utf-8")),
    )
    conn.commit()
    conn.close()

    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo"})
    assert boot.status_code == 200
    ctx = boot.json()["ctx"]

    response = client.get(f"/api/session/{ctx}/memory")
    assert response.status_code == 200
    payload = response.json()
    assert "prefer compact reports" in payload["memory"]


def test_active_run_name_falls_back_to_run_info_when_runtime_has_no_run_name() -> None:
    class _DummyRunDir:
        name = "run_old"

    class _DummySession:
        run_info = {"run_id": "run_new"}

        @staticmethod
        def get_selected_run_dir():
            return _DummyRunDir()

    assert server._active_run_name(_DummySession(), {"run_name": ""}) == "run_new"
