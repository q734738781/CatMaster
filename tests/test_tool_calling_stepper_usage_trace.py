from __future__ import annotations

import json

from catmaster.agents.tool_calling_stepper import ToolCallingTaskStepper
from catmaster.llm.fake_driver import FakeDriver
from catmaster.llm.types import LLMTokenUsage, TurnResult
from catmaster.runtime import ArtifactStore, TraceStore, ToolExecutor
from catmaster.runtime.local_tool_backend import LocalToolBackend
from catmaster.tools.registry import ToolRegistry


def test_tool_calling_stepper_records_usage_event(tmp_path) -> None:
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

    turn = TurnResult(
        output_text="done",
        tool_calls=[],
        output_items_raw=[{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "done"}],
        }],
        usage=LLMTokenUsage(
            input_tokens=10,
            input_cached_tokens=4,
            output_tokens=6,
            total_tokens=16,
            source="provider",
            raw={"input_tokens": 10},
        ),
    )
    driver = FakeDriver([turn])
    stepper = ToolCallingTaskStepper(
        driver=driver,
        backend=backend,
        trace_store=trace_store,
    )

    result = stepper.run(
        task_id="task-1",
        task_goal="Return text",
        context_pack={},
        function_tools=[],
        builtin_tools=[],
    )

    assert result["finish_reason"] == "model_text"
    records = [
        json.loads(line)
        for line in (tmp_path / "event_trace.jsonl").read_text().splitlines()
        if line.strip()
    ]
    usage_records = [r for r in records if r.get("event") == "LLM_USAGE"]
    assert usage_records
    usage = usage_records[0]["payload"]["usage"]
    assert usage["input_tokens"] == 10
    assert usage["input_cached_tokens"] == 4
    assert usage["output_tokens"] == 6
    assert usage["total_tokens"] == 16
