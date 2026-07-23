from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("langchain_core")

from catmaster.webui import server
from catmaster.webui import __main__ as webui_main


def test_launch_sets_immediate_shutdown_defaults(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def _fake_create_app(**kwargs):
        captured["app_kwargs"] = kwargs
        return object()

    monkeypatch.setattr(server, "create_app", _fake_create_app)

    def _fake_run(app, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(server.uvicorn, "run", _fake_run)

    server.launch(project_space_root=str(tmp_path), disable_registration=True)

    kwargs = captured["kwargs"]
    assert kwargs["timeout_keep_alive"] == 0
    assert kwargs["timeout_graceful_shutdown"] == 0
    assert captured["app_kwargs"]["disable_registration"] is True


def test_cli_forwards_disable_registration(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    monkeypatch.setattr(webui_main, "launch", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(
        "sys.argv",
        [
            "catmaster.webui",
            "--project-space-root",
            str(tmp_path),
            "--disable-registration",
        ],
    )

    webui_main.main()

    assert captured["project_space_root"] == str(tmp_path)
    assert captured["no_login"] is False
    assert captured["disable_registration"] is True
