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


def create_tool_output(
    tool_name: str,
    success: bool = True,
    data: Dict[str, Any] = None,
    error: str = None,
    warnings: List[str] = None,
    execution_time: float = None
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
        "execution_time": execution_time
    }


def workspace_root() -> Path:
    """Resolve the workspace root from CATMASTER_WORKSPACE or cwd."""
    env = os.environ.get("CATMASTER_WORKSPACE")
    root = Path(env).expanduser().resolve() if env else Path.cwd().resolve()
    return root


def workspace_relpath(path: Path) -> str:
    """Return workspace-relative path string if inside workspace, else absolute."""
    root = workspace_root()
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path.resolve())


def resolve_workspace_path(path: str, *, must_exist: bool = False) -> Path:
    """
    Resolve a path under the workspace. Relative paths are rooted at workspace_root().
    Absolute paths are allowed only if they stay within workspace_root().
    """
    root = workspace_root()
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if not str(p).startswith(str(root)):
        raise ValueError(f"Path escapes workspace root: {p}")
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    return p
