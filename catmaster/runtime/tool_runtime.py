from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_TOOLCALL_KEY: ContextVar[str] = ContextVar("_catmaster_toolcall_key", default="")
_RUN_DIR: ContextVar[str] = ContextVar("_catmaster_run_dir", default="")
_TOOL_AUDIENCE: ContextVar[str] = ContextVar("_catmaster_tool_audience", default="")


def current_toolcall_key() -> str:
    return _TOOLCALL_KEY.get("") or ""


def current_run_dir() -> str:
    return _RUN_DIR.get("") or ""


def current_tool_audience() -> str:
    return _TOOL_AUDIENCE.get("") or ""


@contextmanager
def toolcall_context(toolcall_key: str, *, run_dir: str = "", audience: str = "") -> Iterator[None]:
    tool_token = _TOOLCALL_KEY.set(toolcall_key or "")
    run_token = _RUN_DIR.set(run_dir or "")
    audience_token = _TOOL_AUDIENCE.set(audience or "")
    try:
        yield
    finally:
        _TOOL_AUDIENCE.reset(audience_token)
        _RUN_DIR.reset(run_token)
        _TOOLCALL_KEY.reset(tool_token)


__all__ = ["toolcall_context", "current_toolcall_key", "current_run_dir", "current_tool_audience"]
