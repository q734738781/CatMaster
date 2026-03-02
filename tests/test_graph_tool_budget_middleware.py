from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")
from langchain_core.messages import ToolMessage

from catmaster.agents.graph import _make_tool_call_budget_middleware


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

    assert budget_mw.wrap_tool_call(request, _ok) == "ok"

    failed = budget_mw.wrap_tool_call(request, _bad)
    assert isinstance(failed, ToolMessage)
    assert failed.status == "error"

    blocked = budget_mw.wrap_tool_call(request, _ok)
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

    out = budget_mw.wrap_tool_call(request, _boom)
    assert isinstance(out, ToolMessage)
    assert out.status == "error"
    assert "RuntimeError" in str(out.content)
