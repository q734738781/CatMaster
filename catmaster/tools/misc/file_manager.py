from __future__ import annotations

"""Workspace-scoped file management helpers for LLM agents."""

import os
import re
import shutil
from collections import deque
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from catmaster.tools.base import create_tool_output


def _workspace_root() -> Path:
    env = os.environ.get("CATMASTER_WORKSPACE")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()


def _resolve(path: str) -> Path:
    root = _workspace_root()
    p = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    if not str(p).startswith(str(root)):
        raise ValueError("Path escapes workspace root")
    return p


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


class ListFilesInput(BaseModel):
    """List files and directories under a workspace-relative path."""
    path: str = Field(".", description="Relative path inside workspace")


def list_files(payload: dict) -> dict:
    params = ListFilesInput(**payload)
    root = _resolve(params.path)
    if not root.exists():
        return create_tool_output("list_files", False, error=f"Path does not exist: {root}")
    files = []
    for item in sorted(root.glob("**/*")):
        files.append({"path": str(item.relative_to(_workspace_root())), "is_dir": item.is_dir()})
    return create_tool_output("list_files", True, data={"files": files})


class ReadFileInput(BaseModel):
    """Read a text file from the workspace with a byte limit. Should be used for small files. Use other tools for large files."""
    path: str = Field(..., description="File path relative to workspace")
    max_bytes: int = Field(1024, description="Maximum bytes to read")


def read_file(payload: dict) -> dict:
    params = ReadFileInput(**payload)
    path = _resolve(params.path)
    data = path.read_bytes()[: params.max_bytes]
    return create_tool_output("read_file", True, data={"path": str(path.relative_to(_workspace_root())), "content": data.decode(errors="replace")})


class WriteFileInput(BaseModel):
    """Write text content to a workspace-relative file path."""
    path: str = Field(..., description="File path relative to workspace")
    content: str = Field(..., description="File content to write")


def write_file(payload: dict) -> dict:
    params = WriteFileInput(**payload)
    path = _resolve(params.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(params.content)
    return create_tool_output("write_file", True, data={"path": str(path.relative_to(_workspace_root()))})


class MkdirInput(BaseModel):
    """Create a directory under the workspace."""
    path: str = Field(..., description="Directory path relative to workspace")
    parents: bool = Field(True, description="Create parent directories if needed")
    exist_ok: bool = Field(True, description="Do not error if the directory already exists")


class CopyFilesInput(BaseModel):
    """Copy files or directories within the workspace (preserve metadata, follow symlinks, create destination dirs)."""
    sources: List[str] = Field(..., min_length=1, description="Source files or directories (workspace-relative)")
    destination: str = Field(..., description="Destination path (workspace-relative). If multiple sources, must be a directory.")
    overwrite: bool = Field(False, description="Overwrite existing destination paths if true")
    recursive: bool = Field(True, description="Allow copying directories if true")


class DeleteInput(BaseModel):
    """Delete files or directories under the workspace."""
    paths: List[str] = Field(..., min_length=1, description="Files or directories to delete (workspace-relative)")
    recursive: bool = Field(False, description="Allow deleting directories and their contents")
    missing_ok: bool = Field(False, description="Ignore missing paths instead of failing")


class GrepToolInput(BaseModel):
    """Search for a pattern in files and return matching lines (truncates lines to 400 chars)."""
    path: str = Field(".", description="Base directory or file to search (workspace-relative)")
    pattern: str = Field(..., description="Search pattern (regex by default)")
    regex: bool = Field(True, description="Interpret pattern as regex if true; otherwise literal substring")
    ignore_case: bool = Field(False, description="Case-insensitive search if true")
    file_glob: Optional[str] = Field(None, description="Optional file glob filter (e.g., '*.py')")
    include_hidden: bool = Field(False, description="Include hidden files and folders if true")
    max_matches: int = Field(200, ge=1, description="Maximum number of matches to return")


class HeadInput(BaseModel):
    """Return the first N lines of a text file."""
    path: str = Field(..., description="File path relative to workspace")
    lines: int = Field(20, ge=1, description="Number of lines to return from the start")
    max_bytes: int = Field(20000, ge=1, description="Maximum bytes to return")


class TailInput(BaseModel):
    """Return the last N lines of a text file."""
    path: str = Field(..., description="File path relative to workspace")
    lines: int = Field(20, ge=1, description="Number of lines to return from the end")
    max_bytes: int = Field(20000, ge=1, description="Maximum bytes to return")

class MoveFilesInput(BaseModel):
    """Move or rename a file/directory within the workspace."""
    src: str = Field(..., description="Source file or directory path (workspace-relative)")
    dst: str = Field(..., description="Destination path (workspace-relative). If exists, operation fails.")


def mkdir(payload: dict) -> dict:
    """Create a directory under the workspace and report whether it already existed."""
    params = MkdirInput(**payload)
    path = _resolve(params.path)
    existed = path.exists()
    try:
        path.mkdir(parents=params.parents, exist_ok=params.exist_ok)
    except FileExistsError:
        return create_tool_output("mkdir", False, error=f"Path exists and is not a directory: {path}")
    return create_tool_output(
        "mkdir",
        True,
        data={
            "path": str(path.relative_to(_workspace_root())),
            "existed": existed,
            "created": not existed,
        },
    )


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


def copy_files(payload: dict) -> dict:
    """Copy files or directories within the workspace (preserves metadata and follows symlinks)."""
    params = CopyFilesInput(**payload)
    sources = [ _resolve(p) for p in params.sources ]
    dest_root = _resolve(params.destination)

    for src in sources:
        if not src.exists():
            return create_tool_output("copy_files", False, error=f"Source does not exist: {src}")

    results: List[dict] = []

    if len(sources) > 1:
        if dest_root.exists() and not dest_root.is_dir():
            return create_tool_output("copy_files", False, error="Destination must be a directory when copying multiple sources")
        if not dest_root.exists():
            dest_root.mkdir(parents=True, exist_ok=True)

        for src in sources:
            target = dest_root / src.name
            if src.resolve() == target.resolve():
                return create_tool_output("copy_files", False, error=f"Source and destination are the same: {src}")
            try:
                _copy_single(
                    src,
                    target,
                    overwrite=params.overwrite,
                    recursive=params.recursive,
                )
            except Exception as exc:
                return create_tool_output("copy_files", False, error=str(exc))
            results.append({"src_rel": str(src.relative_to(_workspace_root())), "dst_rel": str(target.relative_to(_workspace_root()))})
    else:
        src = sources[0]
        if dest_root.exists() and dest_root.is_dir():
            target = dest_root / src.name
        else:
            target = dest_root
        if src.resolve() == target.resolve():
            return create_tool_output("copy_files", False, error=f"Source and destination are the same: {src}")
        if not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        try:
            _copy_single(
                src,
                target,
                overwrite=params.overwrite,
                recursive=params.recursive,
            )
        except Exception as exc:
            return create_tool_output("copy_files", False, error=str(exc))
        results.append({"src_rel": str(src.relative_to(_workspace_root())), "dst_rel": str(target.relative_to(_workspace_root()))})

    return create_tool_output(
        "copy_files",
        True,
        data={
            "copied": results,
            "count": len(results),
        },
    )


def delete(payload: dict) -> dict:
    """Delete files or directories within the workspace."""
    params = DeleteInput(**payload)
    root = _workspace_root()
    deleted: List[str] = []
    skipped: List[str] = []

    for raw in params.paths:
        path = _resolve(raw)
        if path == root:
            return create_tool_output("delete", False, error="Refusing to delete workspace root")
        if not path.exists():
            if params.missing_ok:
                skipped.append(str(Path(raw)))
                continue
            return create_tool_output("delete", False, error=f"Path does not exist: {path}")
        if path.is_dir():
            if not params.recursive:
                return create_tool_output("delete", False, error=f"Path is a directory (recursive=false): {path}")
            shutil.rmtree(path)
        else:
            path.unlink()
        deleted.append(str(path.relative_to(root)))

    return create_tool_output(
        "delete",
        True,
        data={
            "deleted": deleted,
            "skipped": skipped,
        },
    )


def grep_tool(payload: dict) -> dict:
    """Search files for a pattern and return matching lines with file and line number."""
    params = GrepToolInput(**payload)
    base = _resolve(params.path)
    if not base.exists():
        return create_tool_output("grep_tool", False, error=f"Path does not exist: {base}")

    flags = re.IGNORECASE if params.ignore_case else 0
    if params.regex:
        try:
            pattern_re = re.compile(params.pattern, flags=flags)
        except re.error as exc:
            return create_tool_output("grep_tool", False, error=f"Invalid regex: {exc}")
    else:
        pattern_re = None
        pattern_literal = params.pattern.lower() if params.ignore_case else params.pattern

    matches: List[dict] = []
    files_scanned = 0
    files_skipped = 0
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
                            {
                                "file": str(file.relative_to(_workspace_root())),
                                "line_number": line_no,
                                "line": snippet,
                            }
                        )
                        if len(matches) >= params.max_matches:
                            return create_tool_output(
                                "grep_tool",
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
        "grep_tool",
        True,
        data={
            "matches": matches,
            "truncated": False,
            "files_scanned": files_scanned,
            "files_skipped": files_skipped,
        },
    )


def head(payload: dict) -> dict:
    """Return the first N lines of a text file."""
    params = HeadInput(**payload)
    path = _resolve(params.path)
    if not path.exists():
        return create_tool_output("head", False, error=f"Path does not exist: {path}")
    if path.is_dir():
        return create_tool_output("head", False, error=f"Path is a directory: {path}")

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

    return create_tool_output(
        "head",
        True,
        data={
            "path": str(path.relative_to(_workspace_root())),
            "lines": len(lines_out),
            "truncated": truncated,
            "has_more": has_more or truncated,
            "content": "".join(lines_out),
        },
    )


def tail(payload: dict) -> dict:
    """Return the last N lines of a text file."""
    params = TailInput(**payload)
    path = _resolve(params.path)
    if not path.exists():
        return create_tool_output("tail", False, error=f"Path does not exist: {path}")
    if path.is_dir():
        return create_tool_output("tail", False, error=f"Path is a directory: {path}")

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

    return create_tool_output(
        "tail",
        True,
        data={
            "path": str(path.relative_to(_workspace_root())),
            "lines": len(buf),
            "truncated": truncated,
            "has_more": total_lines > params.lines,
            "content": content,
        },
    )


def move_files(payload: dict) -> dict:
    params = MoveFilesInput(**payload)
    src = _resolve(params.src)
    dst = _resolve(params.dst)
    if not src.exists():
        return create_tool_output("move_files", False, error=f"Source does not exist: {src}")
    if dst.exists():
        return create_tool_output("move_files", False, error=f"Destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    return create_tool_output(
        "move_files",
        True,
        data={
            "src_rel": str(src.relative_to(_workspace_root())),
            "dst_rel": str(dst.relative_to(_workspace_root())),
        },
    )


__all__ = [
    "list_files",
    "read_file",
    "write_file",
    "mkdir",
    "copy_files",
    "delete",
    "grep_tool",
    "head",
    "tail",
    "move_files",
    "ListFilesInput",
    "ReadFileInput",
    "WriteFileInput",
    "MkdirInput",
    "CopyFilesInput",
    "DeleteInput",
    "GrepToolInput",
    "HeadInput",
    "TailInput",
    "MoveFilesInput",
]
