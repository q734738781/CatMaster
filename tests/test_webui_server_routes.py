from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from starlette.routing import Match

from catmaster.webui import server
from catmaster.webui.server import create_app


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
