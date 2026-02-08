#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base utilities for tools to create standardized outputs.
"""
from __future__ import annotations

from contextlib import contextmanager
import contextvars
from typing import Dict, List, Any, Optional
import time
from pathlib import Path

SYSTEM_DIR_NAME = ".catmaster"
_WORKSPACE_OVERRIDE: contextvars.ContextVar[Optional[Path]] = contextvars.ContextVar(
    "catmaster_workspace_override",
    default=None,
)


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


def workspace_root(workspace: Path | str | None = None) -> Path:
    """Resolve workspace root from explicit param, instance scope, or cwd."""
    if workspace is not None:
        return Path(workspace).expanduser().resolve()
    override = _WORKSPACE_OVERRIDE.get()
    if override is not None:
        return override
    return Path.cwd().resolve()


@contextmanager
def workspace_scope(path: Path | str):
    """Temporarily bind a workspace root to the current execution context."""
    resolved = Path(path).expanduser().resolve()
    token = _WORKSPACE_OVERRIDE.set(resolved)
    try:
        yield resolved
    finally:
        _WORKSPACE_OVERRIDE.reset(token)


def system_root(workspace: Path | str | None = None) -> Path:
    """Return the system metadata root under the workspace."""
    return workspace_root(workspace) / SYSTEM_DIR_NAME


def ensure_system_root(workspace: Path | str | None = None) -> Path:
    """Ensure the system root directory exists."""
    root = system_root(workspace)
    root.mkdir(parents=True, exist_ok=True)
    return root


def workspace_relpath(path: Path, workspace: Path | str | None = None) -> str:
    """Return workspace-relative path string if inside workspace, else absolute."""
    root = workspace_root(workspace)
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path.resolve())


def resolve_view_path(
    path: str,
    view: str,
    *,
    workspace: Path | str | None = None,
    must_exist: bool = False,
) -> Path:
    """
    Resolve a path under the requested view root.
    view='user' -> workspace root (system directory excluded)
    view='system' -> system root (.catmaster)
    """
    if view not in {"user", "system"}:
        raise ValueError(f"Invalid view: {view}")
    root = (workspace_root(workspace) if view == "user" else system_root(workspace)).resolve()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise ValueError(f"Path escapes {view} root: {p}")
    if view == "user":
        sys_root = system_root(workspace).resolve()
        try:
            p.relative_to(sys_root)
        except ValueError:
            pass
        else:
            raise ValueError(f"Path under system root is not allowed in user view: {p}")
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    return p


def view_relpath(path: Path, view: str, workspace: Path | str | None = None) -> str:
    """Return view-relative path string if inside that root, else absolute."""
    root = workspace_root(workspace) if view == "user" else system_root(workspace)
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path.resolve())


def resolve_workspace_path(
    path: str,
    *,
    workspace: Path | str | None = None,
    must_exist: bool = False,
) -> Path:
    """
    Resolve a path under the user workspace (system root excluded).
    """
    return resolve_view_path(path, "user", workspace=workspace, must_exist=must_exist)
