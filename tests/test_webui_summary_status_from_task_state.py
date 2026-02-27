from __future__ import annotations

import json
from pathlib import Path

from catmaster.webui.summary_service import snapshot_summary, summarize_run


def _init_run_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(
        json.dumps({"workspace": "/tmp/ws", "model_name": "m"}),
        encoding="utf-8",
    )
    (path / "ui_events.jsonl").write_text("", encoding="utf-8")


def test_snapshot_summary_prefers_terminal_task_state_over_stale_cache(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_x"
    _init_run_dir(run_dir)
    (run_dir / "task_state.json").write_text(json.dumps({"status": "done"}), encoding="utf-8")
    (run_dir / "ui_summary.json").write_text(
        json.dumps({"status": "running", "headline": "run_x | running | m"}),
        encoding="utf-8",
    )

    summary = snapshot_summary(run_dir)

    assert summary.get("status") == "done"
    assert "done" in str(summary.get("headline") or "")


def test_summarize_run_uses_task_state_when_run_end_event_missing(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_y"
    _init_run_dir(run_dir)
    (run_dir / "task_state.json").write_text(json.dumps({"status": "failure"}), encoding="utf-8")

    summary = summarize_run(run_dir)

    assert summary.get("status") == "failure"
