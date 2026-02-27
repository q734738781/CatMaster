from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, Field

pytest.importorskip("langchain_core")

from catmaster.tools.registry import _make_langchain_tool


class _DummyInput(BaseModel):
    text: str = Field(..., description="dummy")


def test_wrapper_marks_missing_status_with_error_as_failed() -> None:
    def _tool(payload: dict) -> dict:
        return {
            "tool_name": "dummy_tool",
            "data": {},
            "error": "validation failed",
        }

    tool = _make_langchain_tool("dummy_tool", _tool, _DummyInput)
    raw = tool.invoke({"text": "x"})
    parsed = json.loads(raw)
    assert parsed["status"] == "failed"
    assert parsed["tool_name"] == "dummy_tool"


def test_wrapper_marks_non_dict_output_as_failed() -> None:
    def _tool(payload: dict) -> int:
        return 123

    tool = _make_langchain_tool("dummy_tool", _tool, _DummyInput)
    raw = tool.invoke({"text": "x"})
    parsed = json.loads(raw)
    assert parsed["status"] == "failed"
    assert parsed["tool_name"] == "dummy_tool"
