from __future__ import annotations

import json

from pydantic import BaseModel, Field

from catmaster.agents.tool_calling_stepper import ToolCallingTaskStepper
from catmaster.llm.fake_driver import FakeDriver
from catmaster.runtime import ArtifactStore, TraceStore, ToolExecutor
from catmaster.runtime.local_tool_backend import LocalToolBackend
from catmaster.tools.registry import ToolRegistry


class BashLikeInput(BaseModel):
    script: str = Field(..., description="Script content")


def _bash_like_tool(_payload: dict) -> dict:
    return {
        "status": "success",
        "tool_name": "bash",
        "data": {
            "stdout": "x" * 6000,
            "stderr": "y" * 4000,
            "exit_code": 0,
            "timed_out": False,
            "cwd": ".",
            "stdout_path": ".logs/bash/fake.stdout.txt",
            "stderr_path": ".logs/bash/fake.stderr.txt",
        },
        "warnings": [],
        "error": "",
    }


class CaptureInputDriver(FakeDriver):
    def __init__(self, script):
        super().__init__(script)
        self.turn_inputs: list[list[dict]] = []

    def create_turn(self, *, input_items, tools=None, **kwargs):
        self.turn_inputs.append(list(input_items))
        return super().create_turn(input_items=input_items, tools=tools, **kwargs)


def test_stepper_preserves_bash_exec_streams_for_next_turn(tmp_path) -> None:
    registry = ToolRegistry(register_all_tools=False)
    registry.register_tool("bash", _bash_like_tool, BashLikeInput)

    tool_executor = ToolExecutor(registry)
    artifact_store = ArtifactStore(tmp_path)
    trace_store = TraceStore(tmp_path)
    backend = LocalToolBackend(
        registry=registry,
        tool_executor=tool_executor,
        artifact_store=artifact_store,
        trace_store=trace_store,
    )

    driver = CaptureInputDriver(
        [
            [
                {
                    "type": "function_call",
                    "call_id": "call-bash",
                    "name": "bash",
                    "arguments": json.dumps({"script": "echo test"}),
                }
            ],
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done"}],
                }
            ],
        ]
    )
    stepper = ToolCallingTaskStepper(driver=driver, backend=backend)
    result = stepper.run(
        task_id="task-1",
        task_goal="run bash",
        context_pack={},
        function_tools=backend.list_function_tools(),
        builtin_tools=[],
    )
    assert result["finish_reason"] == "model_text"
    assert len(driver.turn_inputs) >= 2

    second_turn_items = driver.turn_inputs[1]
    fco_items = [item for item in second_turn_items if item.get("type") == "function_call_output"]
    assert fco_items
    output_payload = json.loads(str(fco_items[-1].get("output") or "{}"))
    data = output_payload.get("data") or {}
    assert data.get("stdout") == "x" * 6000
    assert data.get("stderr") == "y" * 4000
    assert "stdout_path" not in data
    assert "stderr_path" not in data
