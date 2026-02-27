from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("gradio")
pytest.importorskip("langchain_core")

from catmaster.webui import server


def test_launch_sets_immediate_shutdown_defaults(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    monkeypatch.setattr(server, "create_app", lambda **kwargs: object())

    def _fake_run(app, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(server.uvicorn, "run", _fake_run)

    server.launch(project_space_root=str(tmp_path))

    kwargs = captured["kwargs"]
    assert kwargs["timeout_keep_alive"] == 0
    assert kwargs["timeout_graceful_shutdown"] == 0
