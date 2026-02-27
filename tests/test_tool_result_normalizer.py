from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import ToolMessage

from catmaster.runtime.tool_result_normalizer import (
    normalize_tool_result,
    to_tool_message_status,
)


def test_normalize_dict_without_status_but_with_error_to_failed() -> None:
    out = normalize_tool_result(
        {"tool_name": "demo_tool", "data": {}, "error": "boom"},
        tool_name="demo_tool",
        is_control_tool=False,
    )
    assert out["status"] == "failed"
    assert out["tool_name"] == "demo_tool"
    assert to_tool_message_status(out) == "error"


def test_normalize_dict_without_status_with_data_to_success() -> None:
    out = normalize_tool_result(
        {"tool_name": "demo_tool", "data": {"x": 1}, "error": None},
        tool_name="demo_tool",
        is_control_tool=False,
    )
    assert out["status"] == "success"
    assert out["data"] == {"x": 1}
    assert to_tool_message_status(out) == "success"


def test_normalize_non_json_string_defaults_to_failed() -> None:
    out = normalize_tool_result(
        "ValidationError: invalid payload",
        tool_name="demo_tool",
        is_control_tool=False,
    )
    assert out["status"] == "failed"
    assert "validationerror" in str(out.get("error", "")).lower()
    assert to_tool_message_status(out) == "error"


def test_normalize_tool_message_with_success_payload_beats_raw_error_status() -> None:
    message = ToolMessage(
        content=json.dumps(
            {
                "status": "success",
                "tool_name": "demo_tool",
                "data": {"ok": True},
                "error": "",
            },
            ensure_ascii=False,
        ),
        tool_call_id="call_1",
        name="demo_tool",
        status="error",
    )
    out = normalize_tool_result(message, tool_name="demo_tool", is_control_tool=False)
    assert out["status"] == "success"
    assert to_tool_message_status(out) == "success"


def test_normalize_control_tool_keeps_control_status() -> None:
    out = normalize_tool_result(
        {
            "status": "control",
            "tool_name": "task_finish",
            "payload": {"summary": "done"},
        },
        tool_name="task_finish",
        is_control_tool=True,
    )
    assert out["status"] == "control"
    assert out["payload"]["summary"] == "done"
    assert to_tool_message_status(out) == "success"

