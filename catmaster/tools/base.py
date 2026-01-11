#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base utilities for tools to create standardized outputs.
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional
import time
import os
from pathlib import Path

SYSTEM_DIR_NAME = ".catmaster"


def create_tool_output(
    tool_name: str,
    success: bool = True,
    data: Dict[str, Any] = None,
    error: str = None,
    warnings: List[str] = None,
    execution_time: float = None,
) -> Dict[str, Any]:
    """
    Create standardized tool output dictionary.
    
    Args:
        tool_name: Name of the tool
        success: Whether execution succeeded
        data: Tool-specific output data
        error: Error message if failed
        warnings: List of warning messages
        execution_time: Execution time in seconds
    Returns:
        Standardized output dictionary
    """
    return {
        "status": "success" if success else "failed",
        "tool_name": tool_name,
        "data": data or {},
        "warnings": warnings or [],
        "error": error,
        "execution_time": execution_time,
    }


def workspace_root() -> Path:
    """Resolve the workspace root from CATMASTER_WORKSPACE or cwd."""
    env = os.environ.get("CATMASTER_WORKSPACE")
    root = Path(env).expanduser().resolve() if env else Path.cwd().resolve()
    return root


def system_root() -> Path:
    """Return the system metadata root under the workspace."""
    return workspace_root() / SYSTEM_DIR_NAME


def ensure_system_root() -> Path:
    """Ensure the system root directory exists."""
    root = system_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def workspace_relpath(path: Path) -> str:
    """Return workspace-relative path string if inside workspace, else absolute."""
    root = workspace_root()
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path.resolve())


def resolve_view_path(path: str, view: str, *, must_exist: bool = False) -> Path:
    """
    Resolve a path under the requested view root.
    view='user' -> workspace root (system directory excluded)
    view='system' -> system root (.catmaster)
    """
    if view not in {"user", "system"}:
        raise ValueError(f"Invalid view: {view}")
    root = workspace_root() if view == "user" else system_root()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if not str(p).startswith(str(root)):
        raise ValueError(f"Path escapes {view} root: {p}")
    if view == "user":
        sys_root = system_root().resolve()
        if str(p).startswith(str(sys_root)):
            raise ValueError(f"Path under system root is not allowed in user view: {p}")
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    return p


def view_relpath(path: Path, view: str) -> str:
    """Return view-relative path string if inside that root, else absolute."""
    root = workspace_root() if view == "user" else system_root()
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path.resolve())


def resolve_workspace_path(path: str, *, must_exist: bool = False) -> Path:
    """
    Resolve a path under the user workspace (system root excluded).
    """
    return resolve_view_path(path, "user", must_exist=must_exist)
