from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Mapping

_TOOLCALL_KEY: ContextVar[str] = ContextVar("_catmaster_toolcall_key", default="")
_RUN_DIR: ContextVar[str] = ContextVar("_catmaster_run_dir", default="")
_TOOL_AUDIENCE: ContextVar[str] = ContextVar("_catmaster_tool_audience", default="")
_TOOL_CONTEXT: ContextVar[dict[str, Any]] = ContextVar(
    "_catmaster_tool_context",
    default={},
)


def current_toolcall_key() -> str:
    return _TOOLCALL_KEY.get("") or ""


def current_run_dir() -> str:
    return _RUN_DIR.get("") or ""


def current_tool_audience() -> str:
    return _TOOL_AUDIENCE.get("") or ""


def current_tool_context() -> dict[str, Any]:
    """Return trusted runtime values injected by the tool host, never by tool args."""

    return dict(_TOOL_CONTEXT.get({}) or {})


@contextmanager
def toolcall_context(
    toolcall_key: str,
    *,
    run_dir: str = "",
    audience: str = "",
    context: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    tool_token = _TOOLCALL_KEY.set(toolcall_key or "")
    run_token = _RUN_DIR.set(run_dir or "")
    audience_token = _TOOL_AUDIENCE.set(audience or "")
    context_token = _TOOL_CONTEXT.set(dict(context or {}))
    try:
        yield
    finally:
        _TOOL_CONTEXT.reset(context_token)
        _TOOL_AUDIENCE.reset(audience_token)
        _RUN_DIR.reset(run_token)
        _TOOLCALL_KEY.reset(tool_token)


__all__ = [
    "current_run_dir",
    "current_tool_audience",
    "current_tool_context",
    "current_toolcall_key",
    "toolcall_context",
]
