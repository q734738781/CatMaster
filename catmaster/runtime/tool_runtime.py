from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

_TOOLCALL_KEY: ContextVar[str] = ContextVar("_catmaster_toolcall_key", default="")


def current_toolcall_key() -> str:
    return _TOOLCALL_KEY.get("") or ""


@contextmanager
def toolcall_context(toolcall_key: str) -> Iterator[None]:
    token = _TOOLCALL_KEY.set(toolcall_key or "")
    try:
        yield
    finally:
        _TOOLCALL_KEY.reset(token)


__all__ = ["toolcall_context", "current_toolcall_key"]
