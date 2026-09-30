from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from catmaster.tools.base import ensure_system_root, system_root

_WORKSPACE_VENV_DIRNAME = ".venv"
_WORKSPACE_PIP_CACHE_DIRNAME = ".pip-cache"


def workspace_python_env_root(workspace: Path | str | None = None) -> Path:
    """Return the workspace-local Python environment root under metadata/."""
    return system_root(workspace) / _WORKSPACE_VENV_DIRNAME


def workspace_python_env_ready(workspace: Path | str | None = None) -> bool:
    root = workspace_python_env_root(workspace)
    return (root / "pyvenv.cfg").is_file() and _venv_python_path(root).is_file()


def ensure_workspace_python_env(
    workspace: Path | str | None = None,
    *,
    base_python: str | None = None,
) -> Path:
    """Create the lightweight workspace-local venv if missing and return its root."""
    metadata_root = ensure_system_root(workspace)
    venv_root = metadata_root / _WORKSPACE_VENV_DIRNAME
    if workspace_python_env_ready(workspace):
        return venv_root
    python_exe = base_python or sys.executable
    venv_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [python_exe, "-m", "venv", "--system-site-packages", str(venv_root)],
        check=True,
    )
    return venv_root


def workspace_python_env_overrides(workspace: Path | str | None = None) -> dict[str, str]:
    """Return env overrides that make the workspace-local venv the default shell Python."""
    metadata_root = ensure_system_root(workspace)
    venv_root = ensure_workspace_python_env(workspace)
    bin_dir = _venv_bin_path(venv_root)
    pip_cache_dir = metadata_root / _WORKSPACE_PIP_CACHE_DIRNAME
    pip_cache_dir.mkdir(parents=True, exist_ok=True)
    inherited_path = os.environ.get("PATH", "")
    path_value = str(bin_dir)
    if inherited_path:
        path_value = f"{bin_dir}{os.pathsep}{inherited_path}"
    return {
        "VIRTUAL_ENV": str(venv_root),
        "PATH": path_value,
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_REQUIRE_VIRTUALENV": "1",
        "PIP_CACHE_DIR": str(pip_cache_dir),
    }


def _venv_bin_path(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts"
    return venv_root / "bin"


def _venv_python_path(venv_root: Path) -> Path:
    if os.name == "nt":
        return venv_root / "Scripts" / "python.exe"
    return venv_root / "bin" / "python"
