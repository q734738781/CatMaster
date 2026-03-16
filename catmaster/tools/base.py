#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Base project-space and path utilities shared by CatMaster tools."""
from __future__ import annotations

from contextlib import contextmanager
import contextvars
from typing import Any, Optional
from pathlib import Path

PROJECT_FILES_DIR_NAME = "files"
PROJECT_METADATA_DIR_NAME = "metadata"
LEGACY_SYSTEM_DIR_NAME = ".catmaster"
_PROJECT_SPACE_OVERRIDE: contextvars.ContextVar[Optional[Path]] = contextvars.ContextVar(
    "catmaster_project_space_override",
    default=None,
)


def project_space_root(project_space: Path | str | None = None) -> Path:
    """Resolve project-space root from explicit param, instance scope, or cwd."""
    if project_space is not None:
        return Path(project_space).expanduser().resolve()
    override = _PROJECT_SPACE_OVERRIDE.get()
    if override is not None:
        return override
    return Path.cwd().resolve()


def ensure_project_space_layout(
    project_space: Path | str | None = None,
    *,
    create: bool = True,
) -> dict[str, Path]:
    """
    Ensure project_space uses the new two-root layout:
    - project_space/files      (LLM-visible project files)
    - project_space/metadata   (internal run metadata)
    """
    root = project_space_root(project_space)
    files = root / PROJECT_FILES_DIR_NAME
    metadata = root / PROJECT_METADATA_DIR_NAME
    legacy = root / LEGACY_SYSTEM_DIR_NAME

    # Explicitly refuse legacy-only layout (no automatic migration).
    if not metadata.exists() and legacy.exists():
        raise ValueError(
            "Legacy layout detected (.catmaster). "
            "Use project_space/{files,metadata}; automatic migration is disabled."
        )

    if create:
        root.mkdir(parents=True, exist_ok=True)
        files.mkdir(parents=True, exist_ok=True)
        metadata.mkdir(parents=True, exist_ok=True)
    else:
        if not files.is_dir() or not metadata.is_dir():
            raise ValueError(
                "Invalid project space layout. Expected directories: "
                f"{files} and {metadata}"
            )

    return {
        "project_space_root": root,
        "files_root": files,
        "metadata_root": metadata,
    }


def workspace_root(workspace: Path | str | None = None) -> Path:
    """
    Backward-compatible alias for project files root.
    NOTE: This now resolves to <project_space>/files.
    """
    root = project_space_root(workspace)
    return root / PROJECT_FILES_DIR_NAME


@contextmanager
def workspace_scope(path: Path | str):
    """Temporarily bind a project-space root to the current execution context."""
    resolved = Path(path).expanduser().resolve()
    ensure_project_space_layout(resolved, create=True)
    token = _PROJECT_SPACE_OVERRIDE.set(resolved)
    try:
        yield resolved
    finally:
        _PROJECT_SPACE_OVERRIDE.reset(token)


def system_root(workspace: Path | str | None = None) -> Path:
    """
    Backward-compatible alias for metadata root.
    NOTE: This now resolves to <project_space>/metadata.
    """
    root = project_space_root(workspace)
    return root / PROJECT_METADATA_DIR_NAME


def ensure_system_root(workspace: Path | str | None = None) -> Path:
    """Backward-compatible alias that ensures project-space layout exists."""
    return ensure_project_space_layout(workspace, create=True)["metadata_root"]


def workspace_relpath(path: Path, workspace: Path | str | None = None) -> str:
    """Return files-root-relative path string if inside files root, else absolute."""
    root = workspace_root(workspace)
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path.resolve())


def resolve_scoped_path(
    path: str,
    scope: str,
    *,
    workspace: Path | str | None = None,
    must_exist: bool = False,
) -> Path:
    """
    Resolve a path under the requested root.
    scope='files' -> project files root
    scope='metadata' -> metadata root
    """
    if scope not in {"files", "metadata"}:
        raise ValueError(f"Invalid scope: {scope}")
    root = (workspace_root(workspace) if scope == "files" else system_root(workspace)).resolve()
    raw_path = str(path or "").strip() or "."
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        resolved_absolute = p.resolve()
        if scope == "files":
            try:
                resolved_absolute.relative_to(root)
            except ValueError:
                # DeepAgent exposes the project filesystem with a virtual `/` root.
                # Treat absolute file-scope paths like `/foo/bar` as virtual paths
                # relative to <project_space>/files.
                p = (root / raw_path.lstrip("/")).resolve()
            else:
                p = resolved_absolute
        else:
            p = resolved_absolute
    else:
        p = (root / p).resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise ValueError(f"Path escapes {scope} root: {p}")
    if scope == "files":
        sys_root = system_root(workspace).resolve()
        try:
            p.relative_to(sys_root)
        except ValueError:
            pass
        else:
            raise ValueError(f"Path under metadata root is not allowed in files scope: {p}")
    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")
    return p


def resolve_view_path(
    path: str,
    view: str,
    *,
    workspace: Path | str | None = None,
    must_exist: bool = False,
) -> Path:
    """
    Backward-compatibility alias:
    view='user' -> scope='files'
    view='system' -> scope='metadata'
    """
    mapping = {"user": "files", "system": "metadata"}
    scope = mapping.get(view)
    if scope is None:
        raise ValueError(f"Invalid legacy view: {view}")
    return resolve_scoped_path(path, scope, workspace=workspace, must_exist=must_exist)


def scoped_relpath(path: Path, scope: str, workspace: Path | str | None = None) -> str:
    """Return path string relative to the chosen scope root, else absolute."""
    if scope not in {"files", "metadata"}:
        raise ValueError(f"Invalid scope: {scope}")
    root = workspace_root(workspace) if scope == "files" else system_root(workspace)
    try:
        return str(path.resolve().relative_to(root))
    except Exception:
        return str(path.resolve())


def view_relpath(path: Path, view: str, workspace: Path | str | None = None) -> str:
    """Backward-compatibility alias for scoped_relpath()."""
    mapping = {"user": "files", "system": "metadata"}
    scope = mapping.get(view)
    if scope is None:
        raise ValueError(f"Invalid legacy view: {view}")
    return scoped_relpath(path, scope, workspace=workspace)


def resolve_workspace_path(
    path: str,
    *,
    workspace: Path | str | None = None,
    must_exist: bool = False,
) -> Path:
    """
    Resolve a path under project files root (metadata excluded).
    """
    return resolve_scoped_path(path, "files", workspace=workspace, must_exist=must_exist)


def resolve_project_file_path(
    path: str,
    *,
    project_space: Path | str | None = None,
    must_exist: bool = False,
) -> Path:
    """Explicit helper resolving paths under project_space/files."""
    return resolve_workspace_path(path, workspace=project_space, must_exist=must_exist)
