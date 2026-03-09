from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")
from langchain_core.messages import ToolMessage

from catmaster.agents.graph import _make_tool_call_budget_middleware


def _invoke_budget_middleware(middleware_obj, request, handler):
    if hasattr(middleware_obj, "awrap_tool_call"):
        return asyncio.run(middleware_obj.awrap_tool_call(request, handler))
    return middleware_obj.wrap_tool_call(request, handler)


def test_budget_counts_failed_tool_calls() -> None:
    reset_mw, budget_mw = _make_tool_call_budget_middleware(role="task_runner", max_tool_calls=2)
    reset_mw.before_agent({}, None)

    request = SimpleNamespace(tool_call={"name": "dummy_tool", "id": "call-1"})
    call_counter = {"count": 0}

    def _ok(_request):
        call_counter["count"] += 1
        return "ok"

    def _bad(_request):
        call_counter["count"] += 1
        raise ValueError("validation failed")

    assert _invoke_budget_middleware(budget_mw, request, _ok) == "ok"

    failed = _invoke_budget_middleware(budget_mw, request, _bad)
    assert isinstance(failed, ToolMessage)
    assert failed.status == "error"

    blocked = _invoke_budget_middleware(budget_mw, request, _ok)
    assert isinstance(blocked, ToolMessage)
    assert blocked.status == "error"
    assert "tool-call budget exceeded" in str(blocked.content)
    assert call_counter["count"] == 2


def test_budget_maps_generic_exception_to_tool_message() -> None:
    reset_mw, budget_mw = _make_tool_call_budget_middleware(role="task_runner", max_tool_calls=2)
    reset_mw.before_agent({}, None)
    request = SimpleNamespace(tool_call={"name": "dummy_tool", "id": "call-2"})

    def _boom(_request):
        raise RuntimeError("unexpected")

    out = _invoke_budget_middleware(budget_mw, request, _boom)
    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert "RuntimeError" in str(out.content)


def test_budget_supports_async_handler_for_mcp_tools() -> None:
    reset_mw, budget_mw = _make_tool_call_budget_middleware(role="task_runner", max_tool_calls=2)
    reset_mw.before_agent({}, None)
    request = SimpleNamespace(tool_call={"name": "read_text_file", "id": "call-async-1"})

    async def _async_ok(_request):
        return "ok-async"

    out = _invoke_budget_middleware(budget_mw, request, _async_ok)
    assert out == "ok-async"


def test_sync_wrap_returns_error_when_handler_is_awaitable() -> None:
    reset_mw, budget_mw = _make_tool_call_budget_middleware(role="task_runner", max_tool_calls=2)
    reset_mw.before_agent({}, None)
    request = SimpleNamespace(tool_call={"name": "read_text_file", "id": "call-sync-async-1"})

    async def _async_ok(_request):
        return "ok-async"

    out = budget_mw.wrap_tool_call(request, _async_ok)
    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert "ainvoke/arun" in str(out.content)
