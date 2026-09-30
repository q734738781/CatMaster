from __future__ import annotations

"""Deprecated archived file-manager tools; needs new implementation."""

from typing import List, Optional

from pydantic import BaseModel, Field


def _deprecated_not_implemented(tool_name: str) -> None:
    raise NotImplementedError(f"{tool_name} is deprecated and needs new implementation.")


class WorkspaceListFilesInput(BaseModel):
    """List files and directories under a project-files-relative path with bounded output."""

    path: str = Field(".", description="Relative path inside project files root")
    depth: int = Field(3, ge=0, description="Maximum depth to traverse from the root path")
    max_entries: int = Field(200, ge=1, description="Maximum number of entries to return")
    exclude_globs: List[str] = Field(default_factory=list, description="List of glob patterns to exclude")
    continuation_token: Optional[str] = Field(None, description="Opaque offset token for pagination")


def workspace_list_files(payload: dict) -> tuple[str, dict]:
    _ = WorkspaceListFilesInput(**payload)
    _deprecated_not_implemented("workspace_list_files")


class WorkspaceReadFileInput(BaseModel):
    """Read a text file from project files with a byte limit."""

    path: str = Field(..., description="File path relative to project files root")
    max_bytes: int = Field(1024, ge=1, description="Maximum bytes to read")


def workspace_read_file(payload: dict) -> tuple[str, dict]:
    _ = WorkspaceReadFileInput(**payload)
    _deprecated_not_implemented("workspace_read_file")


class WorkspaceWriteFileInput(BaseModel):
    """Write text content to a project-files-relative file path."""

    path: str = Field(..., description="File path relative to project files root")
    content: str = Field(..., description="File content to write")


def workspace_write_file(payload: dict) -> tuple[str, dict]:
    _ = WorkspaceWriteFileInput(**payload)
    _deprecated_not_implemented("workspace_write_file")


class WorkspaceMkdirInput(BaseModel):
    """Create a directory under project files root."""

    path: str = Field(..., description="Directory path relative to project files root")
    parents: bool = Field(True, description="Create parent directories if needed")
    exist_ok: bool = Field(True, description="Do not error if the directory already exists")


def workspace_mkdir(payload: dict) -> tuple[str, dict]:
    _ = WorkspaceMkdirInput(**payload)
    _deprecated_not_implemented("workspace_mkdir")


class WorkspaceCopyFilesInput(BaseModel):
    """Copy files or directories within project files root."""

    sources: List[str] = Field(..., min_length=1, description="Source files or directories (files-root-relative)")
    destination: str = Field(..., description="Destination path (files-root-relative). If multiple sources, must be a directory.")
    overwrite: bool = Field(False, description="Overwrite existing destination paths if true")
    recursive: bool = Field(True, description="Allow copying directories if true")


def workspace_copy_files(payload: dict) -> tuple[str, dict]:
    _ = WorkspaceCopyFilesInput(**payload)
    _deprecated_not_implemented("workspace_copy_files")


class WorkspaceDeleteInput(BaseModel):
    """Delete files or directories under project files root."""

    paths: List[str] = Field(..., min_length=1, description="Files or directories to delete (files-root-relative)")
    recursive: bool = Field(False, description="Allow deleting directories and their contents")
    missing_ok: bool = Field(False, description="Ignore missing paths instead of failing")


def workspace_delete(payload: dict) -> tuple[str, dict]:
    _ = WorkspaceDeleteInput(**payload)
    _deprecated_not_implemented("workspace_delete")


class WorkspaceGrepInput(BaseModel):
    """Search for a pattern in files and return matching lines."""

    path: str = Field(".", description="Base directory or file to search (files-root-relative)")
    pattern: str = Field(..., description="Search pattern (regex by default)")
    regex: bool = Field(True, description="Interpret pattern as regex if true; otherwise literal substring")
    ignore_case: bool = Field(False, description="Case-insensitive search if true")
    file_glob: Optional[str] = Field(None, description="Optional file glob filter (e.g., '*.py')")
    include_hidden: bool = Field(False, description="Include hidden files and folders if true")
    max_matches: int = Field(200, ge=1, description="Maximum number of matches to return")


def workspace_grep(payload: dict) -> tuple[str, dict]:
    _ = WorkspaceGrepInput(**payload)
    _deprecated_not_implemented("workspace_grep")


class WorkspaceHeadInput(BaseModel):
    """Return the first N lines of a text file."""

    path: str = Field(..., description="File path relative to project files root")
    lines: int = Field(20, ge=1, description="Number of lines to return from the start")
    max_bytes: int = Field(20000, ge=1, description="Maximum bytes to return")


def workspace_head(payload: dict) -> tuple[str, dict]:
    _ = WorkspaceHeadInput(**payload)
    _deprecated_not_implemented("workspace_head")


class WorkspaceTailInput(BaseModel):
    """Return the last N lines of a text file."""

    path: str = Field(..., description="File path relative to project files root")
    lines: int = Field(20, ge=1, description="Number of lines to return from the end")
    max_bytes: int = Field(20000, ge=1, description="Maximum bytes to return")


def workspace_tail(payload: dict) -> tuple[str, dict]:
    _ = WorkspaceTailInput(**payload)
    _deprecated_not_implemented("workspace_tail")


class WorkspaceMoveFilesInput(BaseModel):
    """Move or rename a file/directory within project files root."""

    src: str = Field(..., description="Source file or directory path (files-root-relative)")
    dst: str = Field(..., description="Destination path (files-root-relative). If exists, operation fails.")


def workspace_move_files(payload: dict) -> tuple[str, dict]:
    _ = WorkspaceMoveFilesInput(**payload)
    _deprecated_not_implemented("workspace_move_files")


__all__ = [
    "workspace_list_files",
    "workspace_read_file",
    "workspace_write_file",
    "workspace_mkdir",
    "workspace_copy_files",
    "workspace_delete",
    "workspace_grep",
    "workspace_head",
    "workspace_tail",
    "workspace_move_files",
    "WorkspaceListFilesInput",
    "WorkspaceReadFileInput",
    "WorkspaceWriteFileInput",
    "WorkspaceMkdirInput",
    "WorkspaceCopyFilesInput",
    "WorkspaceDeleteInput",
    "WorkspaceGrepInput",
    "WorkspaceHeadInput",
    "WorkspaceTailInput",
    "WorkspaceMoveFilesInput",
]
