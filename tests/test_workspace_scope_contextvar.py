from __future__ import annotations

from pathlib import Path

from catmaster.tools.base import PROJECT_FILES_DIR_NAME, PROJECT_METADATA_DIR_NAME, system_root, workspace_root


def test_workspace_root_prefers_explicit_param(tmp_path: Path) -> None:
    ws = (tmp_path / "ws").resolve()
    ws.mkdir(parents=True, exist_ok=True)
    assert workspace_root(ws) == ws / PROJECT_FILES_DIR_NAME


def test_system_root_uses_explicit_workspace(tmp_path: Path) -> None:
    ws = (tmp_path / "proj").resolve()
    expected = ws / PROJECT_METADATA_DIR_NAME
    assert system_root(workspace=ws) == expected
