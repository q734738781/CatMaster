from __future__ import annotations

import json
import os
from pathlib import Path

from catmaster.specialists import RUN_STATE_FILE
from catmaster.tools.base import system_root
from catmaster.webui.session import WebSession


def _mk_resumable_run(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / RUN_STATE_FILE).write_text('{"entrypoint":"research","status":"running"}', encoding="utf-8")
    (path / "meta.json").write_text("{}", encoding="utf-8")


def test_resolve_resume_dir_prefers_selected_run(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    sys_root = system_root(workspace=ws)
    run_a = sys_root / "runs" / "run_a"
    run_b = sys_root / "runs" / "run_b"
    _mk_resumable_run(run_a)
    _mk_resumable_run(run_b)
    (sys_root / "active_runs.json").write_text(json.dumps({"research": str(run_a)}), encoding="utf-8")

    session.selected_run_dir = run_b
    picked = session._resolve_resume_dir("research", workspace=ws)
    assert picked == str(run_b.resolve())


def test_resolve_resume_dir_falls_back_to_active_then_latest(tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    session = WebSession()
    session.set_workspace_root(str(tmp_path))
    ok, _ = session.open_workspace(str(ws), create=True)
    assert ok

    sys_root = system_root(workspace=ws)
    run_old = sys_root / "runs" / "run_old"
    run_new = sys_root / "runs" / "run_new"
    _mk_resumable_run(run_old)
    _mk_resumable_run(run_new)

    (sys_root / "active_runs.json").write_text(json.dumps({"research": "runs/run_old"}), encoding="utf-8")
    picked = session._resolve_resume_dir("research", workspace=ws)
    assert picked == str(run_old.resolve())

    (sys_root / "active_runs.json").write_text(json.dumps({"research": "runs/not_found"}), encoding="utf-8")
    os.utime(run_old, (1, 1))
    os.utime(run_new, (2, 2))
    picked_latest = session._resolve_resume_dir("research", workspace=ws)
    assert picked_latest == str(run_new.resolve())
