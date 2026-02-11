from __future__ import annotations

from pathlib import Path

from starlette.routing import Match

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
    # First full match should be the mounted monitor app.
    assert getattr(full_matches[0], "path", None) == "/monitor"
