from __future__ import annotations

import os
from pathlib import Path

import pytest

from catmaster.runtime.workspace_python_env import (
    ensure_workspace_python_env,
    workspace_python_env_overrides,
    workspace_python_env_ready,
    workspace_python_env_root,
)


def _venv_python_path(root: Path) -> Path:
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def test_ensure_workspace_python_env_creates_system_site_packages_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    calls: list[list[str]] = []

    def _fake_run(cmd, check: bool = False):
        assert check is True
        calls.append([str(part) for part in cmd])
        venv_root = workspace_python_env_root(workspace)
        _venv_python_path(venv_root).parent.mkdir(parents=True, exist_ok=True)
        _venv_python_path(venv_root).write_text("", encoding="utf-8")
        (venv_root / "pyvenv.cfg").write_text("include-system-site-packages = true\n", encoding="utf-8")

    monkeypatch.setattr("catmaster.runtime.workspace_python_env.subprocess.run", _fake_run)

    created = ensure_workspace_python_env(workspace, base_python="/fake/python")

    assert created == workspace_python_env_root(workspace)
    assert calls == [[
        "/fake/python",
        "-m",
        "venv",
        "--system-site-packages",
        str(workspace_python_env_root(workspace)),
    ]]
    assert workspace_python_env_ready(workspace)


def test_ensure_workspace_python_env_reuses_existing_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    venv_root = workspace_python_env_root(workspace)
    _venv_python_path(venv_root).parent.mkdir(parents=True, exist_ok=True)
    _venv_python_path(venv_root).write_text("", encoding="utf-8")
    (venv_root / "pyvenv.cfg").write_text("include-system-site-packages = true\n", encoding="utf-8")

    def _fail_run(*args, **kwargs):
        raise AssertionError("venv creation should not be retried")

    monkeypatch.setattr("catmaster.runtime.workspace_python_env.subprocess.run", _fail_run)

    reused = ensure_workspace_python_env(workspace)

    assert reused == venv_root


def test_workspace_python_env_overrides_prepend_workspace_venv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    venv_root = workspace_python_env_root(workspace)
    _venv_python_path(venv_root).parent.mkdir(parents=True, exist_ok=True)
    _venv_python_path(venv_root).write_text("", encoding="utf-8")
    (venv_root / "pyvenv.cfg").write_text("include-system-site-packages = true\n", encoding="utf-8")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")

    env = workspace_python_env_overrides(workspace)

    assert env["VIRTUAL_ENV"] == str(venv_root)
    assert env["PATH"].split(os.pathsep)[0] == str(_venv_python_path(venv_root).parent)
    assert env["PIP_DISABLE_PIP_VERSION_CHECK"] == "1"
    assert env["PIP_REQUIRE_VIRTUALENV"] == "1"
    assert env["PIP_CACHE_DIR"] == str(workspace / "metadata" / ".pip-cache")
