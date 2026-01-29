from __future__ import annotations

import json

from pydantic import BaseModel, Field

from catmaster.runtime import ArtifactStore, TraceStore, ToolExecutor
from catmaster.runtime.local_tool_backend import LocalToolBackend
from catmaster.tools.registry import ToolRegistry


class DummyInput(BaseModel):
    """Dummy tool input."""

    text: str = Field(..., description="Text to echo")


def dummy_tool(payload: dict) -> dict:
    return {
        "status": "success",
        "tool_name": "dummy_tool",
        "data": {"text": payload.get("text")},
    }


def test_local_tool_backend_call(tmp_path) -> None:
    registry = ToolRegistry(register_all_tools=False)
    registry.register_tool("dummy_tool", dummy_tool, DummyInput)

    tool_executor = ToolExecutor(registry)
    artifact_store = ArtifactStore(tmp_path)
    trace_store = TraceStore(tmp_path)
    backend = LocalToolBackend(
        registry=registry,
        tool_executor=tool_executor,
        artifact_store=artifact_store,
        trace_store=trace_store,
    )

    output = backend.call(
        "dummy_tool",
        json.dumps({"text": "hello"}),
        toolcall_key="call-1",
        call_id="call-1",
    )

    assert output["status"] == "success"
    input_path = tmp_path / "toolcalls" / "call-1" / "input.json"
    output_path = tmp_path / "toolcalls" / "call-1" / "output.json"
    assert input_path.exists()
    assert output_path.exists()
