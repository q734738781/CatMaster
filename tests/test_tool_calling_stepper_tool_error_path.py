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


def test_tool_calling_stepper_tool_error_path(tmp_path) -> None:
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
                "call_id": "call-err",
                "name": "dummy_tool",
                "arguments": json.dumps({}),
            }
        ],
        [
            {
                "type": "function_call",
                "call_id": "fail-1",
                "name": "task_fail",
                "arguments": json.dumps({"error": "validation failed"}),
            }
        ],
    ])

    stepper = ToolCallingTaskStepper(
        driver=driver,
        backend=backend,
    )

    result = stepper.run(
        task_id="task-err",
        task_goal="Trigger validation error",
        context_pack={},
        function_tools=backend.list_function_tools(),
        builtin_tools=[],
    )

    toolcalls_dir = tmp_path / "toolcalls"
    toolcall_dirs = [p for p in toolcalls_dir.iterdir() if p.is_dir()]
    assert toolcall_dirs
    input_payload = json.loads((toolcall_dirs[0] / "input.json").read_text())
    output_payload = json.loads((toolcall_dirs[0] / "output.json").read_text())
    assert input_payload["status"] == "validation_failed"
    assert output_payload.get("status") == "validation_failed"
    assert result["finish_reason"] != "task_finish"
