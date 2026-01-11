from __future__ import annotations

"""Workspace-scoped file management helpers with user/system view separation."""

import fnmatch
import os
import re
import shutil
from collections import deque
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from catmaster.tools.base import create_tool_output, resolve_view_path, view_relpath, system_root


def _validate_view(view: str) -> str:
    if view not in {"user", "system"}:
        raise ValueError(f"Invalid view: {view}")
    return view


def _is_hidden(path: Path, base: Path) -> bool:
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = path
    for part in rel.parts:
        if part in {".", ".."}:
            continue
        if part.startswith("."):
            return True
    return False


def _should_exclude(rel_path: str, exclude_globs: List[str]) -> bool:
    for pattern in exclude_globs:
        if fnmatch.fnmatch(rel_path, pattern):
            return True
    return False


class WorkspaceListFilesInput(BaseModel):
    """List files and directories under a workspace-relative path with bounded output."""
    path: str = Field(".", description="Relative path inside the selected view")
    view: str = Field("user", description="user or system")
    depth: int = Field(3, ge=0, description="Maximum depth to traverse from the root path")
    max_entries: int = Field(200, ge=1, description="Maximum number of entries to return")
    exclude_globs: List[str] = Field(default_factory=list, description="Glob patterns to exclude")
    continuation_token: Optional[str] = Field(None, description="Opaque offset token for pagination")


def workspace_list_files(payload: dict) -> dict:
    params = WorkspaceListFilesInput(**payload)
    view = _validate_view(params.view)
    root = resolve_view_path(params.path, view, must_exist=True)
    if view == "user":
        exclude_globs = list(params.exclude_globs) + [".catmaster", ".catmaster/**"]
    else:
        exclude_globs = list(params.exclude_globs)

    offset = 0
    if params.continuation_token is not None:
        try:
            offset = int(params.continuation_token)
        except ValueError as exc:
            raise ValueError(f"Invalid continuation_token: {exc}")

    if root.is_file():
        rel = view_relpath(root, view)
        if _should_exclude(rel, exclude_globs):
            return create_tool_output(
                "workspace_list_files",
                True,
                data={"root": rel, "entries": [], "truncated": False, "next_token": None},
            )
        entry = {"path": rel, "is_dir": False}
        return create_tool_output(
            "workspace_list_files",
            True,
            data={"root": rel, "entries": [entry], "truncated": False, "next_token": None},
        )

    entries = []
    seen = 0
    truncated = False
    base_depth = len(root.parts)

    for dirpath, dirnames, filenames in os.walk(root):
        current_depth = len(Path(dirpath).parts) - base_depth
        if current_depth > params.depth:
            continue
        if current_depth >= params.depth:
            dirnames[:] = []
        dirnames.sort()
        filenames.sort()
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            rel = view_relpath(path, view)
            if _should_exclude(rel, exclude_globs):
                continue
            if seen < offset:
                seen += 1
                continue
            entries.append({"path": rel, "is_dir": path.is_dir()})
            seen += 1
            if len(entries) >= params.max_entries:
                truncated = True
                break
        if truncated:
            break

    next_token = str(offset + len(entries)) if truncated else None
    return create_tool_output(
        "workspace_list_files",
        True,
        data={
            "root": view_relpath(root, view),
            "entries": entries,
            "truncated": truncated,
            "next_token": next_token,
        },
    )


class WorkspaceReadFileInput(BaseModel):
    """Read a text file from the workspace with a byte limit."""
    path: str = Field(..., description="File path relative to the selected view")
    view: str = Field("user", description="user or system")
    max_bytes: int = Field(1024, ge=1, description="Maximum bytes to read")


def workspace_read_file(payload: dict) -> dict:
    params = WorkspaceReadFileInput(**payload)
    view = _validate_view(params.view)
    path = resolve_view_path(params.path, view, must_exist=True)
    data = path.read_bytes()[: params.max_bytes]
    rel = view_relpath(path, view)
    return create_tool_output(
        "workspace_read_file",
        True,
        data={"path": rel, "content": data.decode(errors="replace")},
    )


class WorkspaceWriteFileInput(BaseModel):
    """Write text content to a workspace-relative file path."""
    path: str = Field(..., description="File path relative to the selected view")
    view: str = Field("user", description="user or system")
    content: str = Field(..., description="File content to write")


def workspace_write_file(payload: dict) -> dict:
    params = WorkspaceWriteFileInput(**payload)
    view = _validate_view(params.view)
    path = resolve_view_path(params.path, view)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(params.content, encoding="utf-8")
    rel = view_relpath(path, view)
    return create_tool_output(
        "workspace_write_file",
        True,
        data={"path": rel},
    )


class WorkspaceMkdirInput(BaseModel):
    """Create a directory under the workspace view."""
    path: str = Field(..., description="Directory path relative to the selected view")
    view: str = Field("user", description="user or system")
    parents: bool = Field(True, description="Create parent directories if needed")
    exist_ok: bool = Field(True, description="Do not error if the directory already exists")


def workspace_mkdir(payload: dict) -> dict:
    params = WorkspaceMkdirInput(**payload)
    view = _validate_view(params.view)
    path = resolve_view_path(params.path, view)
    existed = path.exists()
    try:
        path.mkdir(parents=params.parents, exist_ok=params.exist_ok)
    except FileExistsError:
        return create_tool_output("workspace_mkdir", False, error=f"Path exists and is not a directory: {path}")
    rel = view_relpath(path, view)
    return create_tool_output(
        "workspace_mkdir",
        True,
        data={"path": rel, "existed": existed, "created": not existed},
    )


class WorkspaceCopyFilesInput(BaseModel):
    """Copy files or directories within the same view."""
    sources: List[str] = Field(..., min_length=1, description="Source files or directories (view-relative)")
    destination: str = Field(..., description="Destination path (view-relative). If multiple sources, must be a directory.")
    view: str = Field("user", description="user or system")
    overwrite: bool = Field(False, description="Overwrite existing destination paths if true")
    recursive: bool = Field(True, description="Allow copying directories if true")


def _copy_single(src: Path, dst: Path, *, overwrite: bool, recursive: bool) -> None:
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"Destination exists: {dst}")
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_dir():
        if not recursive:
            raise IsADirectoryError(f"Source is a directory but recursive is false: {src}")
        shutil.copytree(src, dst, symlinks=False)
    else:
        shutil.copy2(src, dst, follow_symlinks=True)


def workspace_copy_files(payload: dict) -> dict:
    params = WorkspaceCopyFilesInput(**payload)
    view = _validate_view(params.view)
    sources = [resolve_view_path(p, view, must_exist=True) for p in params.sources]
    dest_root = resolve_view_path(params.destination, view)

    results: List[dict] = []
    if len(sources) > 1:
        if dest_root.exists() and not dest_root.is_dir():
            return create_tool_output("workspace_copy_files", False, error="Destination must be a directory when copying multiple sources")
        if not dest_root.exists():
            dest_root.mkdir(parents=True, exist_ok=True)
        for src in sources:
            target = dest_root / src.name
            if src.resolve() == target.resolve():
                return create_tool_output("workspace_copy_files", False, error=f"Source and destination are the same: {src}")
            _copy_single(src, target, overwrite=params.overwrite, recursive=params.recursive)
            results.append({"src_rel": view_relpath(src, view), "dst_rel": view_relpath(target, view)})
    else:
        src = sources[0]
        if dest_root.exists() and dest_root.is_dir():
            target = dest_root / src.name
        else:
            target = dest_root
        if src.resolve() == target.resolve():
            return create_tool_output("workspace_copy_files", False, error=f"Source and destination are the same: {src}")
        if not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        _copy_single(src, target, overwrite=params.overwrite, recursive=params.recursive)
        results.append({"src_rel": view_relpath(src, view), "dst_rel": view_relpath(target, view)})

    return create_tool_output(
        "workspace_copy_files",
        True,
        data={"copied": results, "count": len(results)},
    )


class WorkspaceDeleteInput(BaseModel):
    """Delete files or directories under the workspace view."""
    paths: List[str] = Field(..., min_length=1, description="Files or directories to delete (view-relative)")
    view: str = Field("user", description="user or system")
    recursive: bool = Field(False, description="Allow deleting directories and their contents")
    missing_ok: bool = Field(False, description="Ignore missing paths instead of failing")


def workspace_delete(payload: dict) -> dict:
    params = WorkspaceDeleteInput(**payload)
    view = _validate_view(params.view)
    deleted: List[str] = []
    skipped: List[str] = []

    for raw in params.paths:
        path = resolve_view_path(raw, view)
        if view == "user" and path.resolve() == system_root().resolve():
            return create_tool_output("workspace_delete", False, error="Refusing to delete system root")
        if not path.exists():
            if params.missing_ok:
                skipped.append(str(Path(raw)))
                continue
            return create_tool_output("workspace_delete", False, error=f"Path does not exist: {path}")
        if path.is_dir():
            if not params.recursive:
                return create_tool_output("workspace_delete", False, error=f"Path is a directory (recursive=false): {path}")
            shutil.rmtree(path)
        else:
            path.unlink()
        deleted.append(view_relpath(path, view))

    return create_tool_output(
        "workspace_delete",
        True,
        data={"deleted": deleted, "skipped": skipped},
    )


class WorkspaceGrepInput(BaseModel):
    """Search for a pattern in files and return matching lines."""
    path: str = Field(".", description="Base directory or file to search (view-relative)")
    view: str = Field("user", description="user or system")
    pattern: str = Field(..., description="Search pattern (regex by default)")
    regex: bool = Field(True, description="Interpret pattern as regex if true; otherwise literal substring")
    ignore_case: bool = Field(False, description="Case-insensitive search if true")
    file_glob: Optional[str] = Field(None, description="Optional file glob filter (e.g., '*.py')")
    include_hidden: bool = Field(False, description="Include hidden files and folders if true")
    max_matches: int = Field(200, ge=1, description="Maximum number of matches to return")


def workspace_grep(payload: dict) -> dict:
    params = WorkspaceGrepInput(**payload)
    view = _validate_view(params.view)
    base = resolve_view_path(params.path, view, must_exist=True)

    flags = re.IGNORECASE if params.ignore_case else 0
    if params.regex:
        try:
            pattern_re = re.compile(params.pattern, flags=flags)
        except re.error as exc:
            return create_tool_output("workspace_grep", False, error=f"Invalid regex: {exc}")
    else:
        pattern_re = None
        pattern_literal = params.pattern.lower() if params.ignore_case else params.pattern

    matches: List[dict] = []
    files_scanned = 0
    files_skipped = 0
    matched_files = set()

    def _iter_files():
        if base.is_file():
            yield base
            return
        iterator = base.rglob(params.file_glob) if params.file_glob else base.rglob("*")
        for p in iterator:
            if p.is_file():
                yield p

    for file in _iter_files():
        if not params.include_hidden and _is_hidden(file, base):
            files_skipped += 1
            continue
        files_scanned += 1
        try:
            with file.open("r", encoding="utf-8", errors="ignore") as fh:
                for line_no, line in enumerate(fh, start=1):
                    hit = False
                    if pattern_re:
                        if pattern_re.search(line):
                            hit = True
                    else:
                        hay = line.lower() if params.ignore_case else line
                        if pattern_literal in hay:
                            hit = True
                    if hit:
                        snippet = line.rstrip("\n")
                        if len(snippet) > 400:
                            snippet = snippet[:400] + "..."
                        matches.append(
                            {"file": view_relpath(file, view), "line_number": line_no, "line": snippet}
                        )
                        matched_files.add(view_relpath(file, view))
                        if len(matches) >= params.max_matches:
                            return create_tool_output(
                                "workspace_grep",
                                True,
                                data={
                                    "matches": matches,
                                    "truncated": True,
                                    "files_scanned": files_scanned,
                                    "files_skipped": files_skipped,
                                },
                            )
        except OSError:
            files_skipped += 1
            continue

    return create_tool_output(
        "workspace_grep",
        True,
        data={
            "matches": matches,
            "truncated": False,
            "files_scanned": files_scanned,
            "files_skipped": files_skipped,
        },
    )


class WorkspaceHeadInput(BaseModel):
    """Return the first N lines of a text file."""
    path: str = Field(..., description="File path relative to the selected view")
    view: str = Field("user", description="user or system")
    lines: int = Field(20, ge=1, description="Number of lines to return from the start")
    max_bytes: int = Field(20000, ge=1, description="Maximum bytes to return")


def workspace_head(payload: dict) -> dict:
    params = WorkspaceHeadInput(**payload)
    view = _validate_view(params.view)
    path = resolve_view_path(params.path, view, must_exist=True)
    if path.is_dir():
        return create_tool_output("workspace_head", False, error=f"Path is a directory: {path}")

    lines_out: List[str] = []
    total_bytes = 0
    truncated = False
    has_more = False
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if len(lines_out) >= params.lines:
                has_more = True
                break
            total_bytes += len(line.encode("utf-8", errors="ignore"))
            if total_bytes > params.max_bytes:
                truncated = True
                break
            lines_out.append(line)

    rel = view_relpath(path, view)
    return create_tool_output(
        "workspace_head",
        True,
        data={
            "path": rel,
            "lines": len(lines_out),
            "truncated": truncated,
            "has_more": has_more or truncated,
            "content": "".join(lines_out),
        },
    )


class WorkspaceTailInput(BaseModel):
    """Return the last N lines of a text file."""
    path: str = Field(..., description="File path relative to the selected view")
    view: str = Field("user", description="user or system")
    lines: int = Field(20, ge=1, description="Number of lines to return from the end")
    max_bytes: int = Field(20000, ge=1, description="Maximum bytes to return")


def workspace_tail(payload: dict) -> dict:
    params = WorkspaceTailInput(**payload)
    view = _validate_view(params.view)
    path = resolve_view_path(params.path, view, must_exist=True)
    if path.is_dir():
        return create_tool_output("workspace_tail", False, error=f"Path is a directory: {path}")

    buf: deque[str] = deque(maxlen=params.lines)
    total_lines = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            total_lines += 1
            buf.append(line)

    content = "".join(buf)
    truncated = False
    if len(content.encode("utf-8", errors="ignore")) > params.max_bytes:
        truncated = True
        content = content.encode("utf-8", errors="ignore")[-params.max_bytes :].decode("utf-8", errors="ignore")

    rel = view_relpath(path, view)
    return create_tool_output(
        "workspace_tail",
        True,
        data={
            "path": rel,
            "lines": len(buf),
            "truncated": truncated,
            "has_more": total_lines > params.lines,
            "content": content,
        },
    )


class WorkspaceMoveFilesInput(BaseModel):
    """Move or rename a file/directory within the workspace view."""
    src: str = Field(..., description="Source file or directory path (view-relative)")
    dst: str = Field(..., description="Destination path (view-relative). If exists, operation fails.")
    view: str = Field("user", description="user or system")


def workspace_move_files(payload: dict) -> dict:
    params = WorkspaceMoveFilesInput(**payload)
    view = _validate_view(params.view)
    src = resolve_view_path(params.src, view, must_exist=True)
    dst = resolve_view_path(params.dst, view)
    if dst.exists():
        return create_tool_output("workspace_move_files", False, error=f"Destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    src_rel = view_relpath(src, view)
    dst_rel = view_relpath(dst, view)
    return create_tool_output(
        "workspace_move_files",
        True,
        data={"src_rel": src_rel, "dst_rel": dst_rel},
    )


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
