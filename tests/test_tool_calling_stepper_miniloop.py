from __future__ import annotations

import json

from pydantic import BaseModel, Field

from catmaster.agents.tool_calling_stepper import ToolCallingTaskStepper
from catmaster.llm.fake_driver import FakeDriver
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


def test_tool_calling_stepper_miniloop(tmp_path) -> None:
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

    driver = FakeDriver([
        [
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "dummy_tool",
                "arguments": json.dumps({"text": "hello"}),
            }
        ],
        [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "done"}],
            }
        ],
    ])

    stepper = ToolCallingTaskStepper(
        driver=driver,
        backend=backend,
    )

    result = stepper.run(
        task_id="task-1",
        task_goal="Say hello",
        context_pack={},
        function_tools=backend.list_function_tools(),
        builtin_tools=[],
    )

    assert result["status"] == "done"
    assert result["finish_reason"] == "model_text"

    toolcalls_dir = tmp_path / "toolcalls"
    toolcall_dirs = [p for p in toolcalls_dir.iterdir() if p.is_dir()]
    assert len(toolcall_dirs) == 1
    input_path = toolcall_dirs[0] / "input.json"
    output_path = toolcall_dirs[0] / "output.json"
    assert input_path.exists()
    assert output_path.exists()
    input_payload = json.loads(input_path.read_text())
    assert input_payload.get("call_id") == "call-1"

    trace_path = tmp_path / "tool_trace.jsonl"
    assert trace_path.exists()
    records = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    assert records
    assert records[0]["tool_name"] == "dummy_tool"
    assert records[0]["call_id"] == "call-1"
