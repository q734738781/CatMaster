from __future__ import annotations

from catmaster.tools.base import ensure_project_space_layout, system_root
from catmaster.webui.session import WebSession


def test_select_run_same_run_does_not_reset_event_cursor(tmp_path) -> None:
    ws = tmp_path / "workspace"
    ensure_project_space_layout(ws, create=True)
    runs_root = system_root(workspace=ws) / "runs"
    (runs_root / "run_001").mkdir(parents=True, exist_ok=True)

    session = WebSession()
    ok, _ = session.open_workspace(str(ws), create=False)
    assert ok is True

    session.select_run("run_001")
    session.last_event_seq = 77
    session.event_lines = ["event-1"]

    msg = session.select_run("run_001")

    assert "Selected run: run_001" in msg
    assert session.last_event_seq == 77
    assert session.event_lines == ["event-1"]


def test_select_run_different_run_resets_event_cursor(tmp_path) -> None:
    ws = tmp_path / "workspace"
    ensure_project_space_layout(ws, create=True)
    runs_root = system_root(workspace=ws) / "runs"
    (runs_root / "run_001").mkdir(parents=True, exist_ok=True)
    (runs_root / "run_002").mkdir(parents=True, exist_ok=True)

    session = WebSession()
    ok, _ = session.open_workspace(str(ws), create=False)
    assert ok is True

    session.select_run("run_001")
    session.last_event_seq = 88
    session.event_lines = ["event-1", "event-2"]

    msg = session.select_run("run_002")

    assert "Selected run: run_002" in msg
    assert session.last_event_seq == 0
    assert session.event_lines == []
