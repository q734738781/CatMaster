from __future__ import annotations

"""Workspace-scoped file management helpers for LLM agents."""

import os
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


class ListFilesInput(BaseModel):
    path: str = Field(".", description="Relative path inside workspace")


def list_files(payload: dict) -> dict:
    params = ListFilesInput(**payload)
    root = _resolve(params.path)
    files = []
    for item in sorted(root.glob("**/*")):
        files.append({"path": str(item.relative_to(_workspace_root())), "is_dir": item.is_dir()})
    return create_tool_output("list_files", True, data={"files": files})


class ReadFileInput(BaseModel):
    path: str = Field(..., description="File path relative to workspace")
    max_bytes: int = Field(20000, description="Maximum bytes to read")


def read_file(payload: dict) -> dict:
    params = ReadFileInput(**payload)
    path = _resolve(params.path)
    data = path.read_bytes()[: params.max_bytes]
    return create_tool_output("read_file", True, data={"path": str(path.relative_to(_workspace_root())), "content": data.decode(errors="replace")})


class WriteFileInput(BaseModel):
    path: str = Field(..., description="File path relative to workspace")
    content: str = Field(..., description="File content to write")


def write_file(payload: dict) -> dict:
    params = WriteFileInput(**payload)
    path = _resolve(params.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(params.content)
    return create_tool_output("write_file", True, data={"path": str(path.relative_to(_workspace_root()))})


class FindTextInput(BaseModel):
    path: str = Field(".", description="Directory to search relative to workspace")
    pattern: str = Field(..., description="Substring to look for")
    max_matches: int = Field(50)


def find_text(payload: dict) -> dict:
    params = FindTextInput(**payload)
    base = _resolve(params.path)
    matches: List[dict] = []
    for file in base.rglob("*"):
        if file.is_dir():
            continue
        text = file.read_text(errors="ignore")
        if params.pattern in text:
            matches.append({"file": str(file.relative_to(_workspace_root()))})
        if len(matches) >= params.max_matches:
            break
    return create_tool_output("find_text", True, data={"matches": matches})


__all__ = [
    "list_files",
    "read_file",
    "write_file",
    "find_text",
    "ListFilesInput",
    "ReadFileInput",
    "WriteFileInput",
    "FindTextInput",
]

