from __future__ import annotations

from pathlib import Path

from catmaster.webui.session import WebSession


def test_resolve_latest_run_dir_accepts_copied_directory(tmp_path: Path) -> None:
    ws = tmp_path / "workspace"
    latest_run = ws / "reports" / "latest_run"
    (latest_run / "reports").mkdir(parents=True, exist_ok=True)
    (latest_run / "reports" / "FINAL_REPORT.md").write_text("report", encoding="utf-8")

    session = WebSession()
    resolved = session._resolve_latest_run_dir(workspace=ws)

    assert resolved is not None
    assert resolved.resolve() == latest_run.resolve()
