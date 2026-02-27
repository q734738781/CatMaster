from __future__ import annotations

import json

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from langchain_core.messages import ToolMessage

from catmaster.agents.graph import _compact_tool_message_for_llm


def test_compact_tool_message_prefers_payload_success_over_raw_error() -> None:
    message = ToolMessage(
        content=json.dumps({
            "status": "success",
            "tool_name": "vasp_relax_prepare",
            "data": {"output_root": "o2_box/01_relax"},
            "error": "",
        }, ensure_ascii=False),
        tool_call_id="call_1",
        name="vasp_relax_prepare",
        status="error",
    )

    compacted = _compact_tool_message_for_llm(message)

    assert compacted.status == "success"
    parsed = json.loads(str(compacted.content))
    assert parsed.get("status") == "success"


def test_compact_tool_message_maps_payload_failed_to_error_status() -> None:
    message = ToolMessage(
        content=json.dumps({
            "status": "failed",
            "tool_name": "vasp_relax_prepare",
            "data": {},
            "error": "missing POTCAR",
        }, ensure_ascii=False),
        tool_call_id="call_2",
        name="vasp_relax_prepare",
        status="success",
    )

    compacted = _compact_tool_message_for_llm(message)
    assert compacted.status == "error"


def test_compact_tool_message_infers_error_status_from_payload_error_text() -> None:
    message = ToolMessage(
        content=json.dumps({
            "tool_name": "vasp_relax_prepare",
            "data": {},
            "error": "runtime exception",
        }, ensure_ascii=False),
        tool_call_id="call_3",
        name="vasp_relax_prepare",
        status="success",
    )

    compacted = _compact_tool_message_for_llm(message)
    assert compacted.status == "error"


def test_compact_tool_message_non_json_error_text_is_not_misclassified_as_success() -> None:
    message = ToolMessage(
        content="ValidationError: user_incar_settings.2.value MAGMOM invalid",
        tool_call_id="call_4",
        name="vasp_relax_prepare",
        status="error",
    )

    compacted = _compact_tool_message_for_llm(message)
    assert compacted.status == "error"
    parsed = json.loads(str(compacted.content))
    assert parsed.get("status") == "failed"
    assert "validationerror" in str(parsed.get("error", "")).lower()
