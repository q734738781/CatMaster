from __future__ import annotations

import json

from catmaster.agents.tool_calling_stepper import ToolCallingTaskStepper
from catmaster.llm.fake_driver import FakeDriver
from catmaster.runtime import ArtifactStore, TraceStore, ToolExecutor
from catmaster.runtime.local_tool_backend import LocalToolBackend
from catmaster.tools.registry import ToolRegistry


def test_tool_calling_stepper_finish_tool(tmp_path) -> None:
    registry = ToolRegistry(register_all_tools=False)
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
                "call_id": "finish-1",
                "name": "task_finish",
                "arguments": json.dumps({"summary": "done"}),
            }
        ]
    ])

    stepper = ToolCallingTaskStepper(
        driver=driver,
        backend=backend,
    )

    result = stepper.run(
        task_id="task-1",
        task_goal="Finish",
        context_pack={},
        function_tools=[],
        builtin_tools=[],
    )

    assert result["status"] == "done"
    assert result["finish_reason"] == "task_finish"
    assert result["control_payload"]["summary"] == "done"
