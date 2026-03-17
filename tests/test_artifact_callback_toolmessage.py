from __future__ import annotations

import json
import uuid

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from catmaster.runtime.artifact_callback import ArtifactPersistenceHandler, UIEventHandler
from catmaster.runtime.artifact_store import ArtifactStore
from catmaster.runtime.trace_store import TraceStore
from catmaster.ui.reporters import Reporter


class _CollectReporter(Reporter):
    def __init__(self) -> None:
        self.events = []

    def emit(self, event):
        self.events.append(event)


def test_artifact_persistence_parses_tool_message_output(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    trace = TraceStore(tmp_path)
    handler = ArtifactPersistenceHandler(store, trace, run_id="run_x")

    rid = uuid.uuid4()
    handler.on_tool_start(
        serialized={"name": "bash"},
        input_str=json.dumps({"cmd": "echo ok"}),
        run_id=rid,
    )

    handler.on_tool_end(
        ToolMessage(
            content=json.dumps({
                "status": "success",
                "tool_name": "bash",
                "data": {"stdout": "ok"},
                "error": None,
            }, ensure_ascii=False),
            tool_call_id="call_1",
            name="bash",
        ),
        run_id=rid,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "tool_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records
    assert records[0]["status"] == "success"
    assert records[0]["tool_name"] == "bash"


def test_ui_event_handler_emits_tool_status_for_tool_message() -> None:
    reporter = _CollectReporter()
    handler = UIEventHandler(reporter, run_id="run_x")

    rid = uuid.uuid4()
    handler.on_tool_start(
        serialized={"name": "demo_tool"},
        input_str='{"alpha": 1}',
        run_id=rid,
        metadata={"lc_agent_name": "literature_agent"},
    )

    handler.on_tool_end(
        ToolMessage(
            content=json.dumps({
                "status": "success",
                "tool_name": "demo_tool",
                "data": {"ok": True},
            }, ensure_ascii=False),
            tool_call_id="call_2",
            name="demo_tool",
        ),
        run_id=rid,
    )

    end_events = [e for e in reporter.events if e.name == "TOOL_CALL_END"]
    assert end_events
    start_events = [e for e in reporter.events if e.name == "TOOL_CALL_START"]
    assert start_events
    start_payload = start_events[-1].payload
    assert start_payload.get("tool") == "demo_tool"
    assert start_payload.get("agent_name") == "literature_agent"
    assert start_payload.get("toolcall_id") == str(rid)
    assert "alpha" in str(start_payload.get("params_compact") or "")
    payload = end_events[-1].payload
    assert payload.get("tool") == "demo_tool"
    assert payload.get("status") == "success"
    assert payload.get("toolcall_id") == str(rid)
    assert payload.get("agent_name") == "literature_agent"
    assert "alpha" in str(payload.get("params_compact") or "")


def test_artifact_persistence_non_json_tool_message_is_failed(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    trace = TraceStore(tmp_path)
    handler = ArtifactPersistenceHandler(store, trace, run_id="run_x")

    rid = uuid.uuid4()
    handler.on_tool_start(
        serialized={"name": "vasp_relax_prepare"},
        input_str=json.dumps({"input_path": "x.vasp"}),
        run_id=rid,
    )

    handler.on_tool_end(
        ToolMessage(
            content="ValidationError: user_incar_settings invalid",
            tool_call_id="call_3",
            name="vasp_relax_prepare",
            status="error",
        ),
        run_id=rid,
    )

    records = [
        json.loads(line)
        for line in (tmp_path / "tool_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert records
    assert records[0]["tool_name"] == "vasp_relax_prepare"
    assert records[0]["status"] == "error"


def test_artifact_persistence_tool_start_json_safes_non_serializable_inputs(tmp_path) -> None:
    store = ArtifactStore(tmp_path)
    trace = TraceStore(tmp_path)
    handler = ArtifactPersistenceHandler(store, trace, run_id="run_x")

    class _RuntimeLike:
        def __repr__(self) -> str:
            return "<ToolRuntime stub>"

    rid = uuid.uuid4()
    handler.on_tool_start(
        serialized={"name": "read_text_file"},
        input_str="{}",
        run_id=rid,
        inputs={"path": "x.txt", "runtime": _RuntimeLike()},
    )

    input_files = list((tmp_path / "toolcalls").glob("*/input.json"))
    assert input_files
    payload = json.loads(input_files[0].read_text(encoding="utf-8"))
    assert payload["raw_params"]["path"] == "x.txt"
    assert payload["raw_params"]["runtime"] == "<ToolRuntime stub>"


def test_ui_event_handler_emits_llm_preview_and_tool_plan() -> None:
    reporter = _CollectReporter()
    handler = UIEventHandler(reporter, run_id="run_x")

    rid = uuid.uuid4()
    handler.on_llm_start(
        serialized={"kwargs": {"model_name": "gpt-5"}},
        prompts=["prompt"],
        run_id=rid,
        metadata={"lc_agent_name": "experiment_specialist"},
    )
    handler.on_llm_end(
        LLMResult(
            generations=[[
                ChatGeneration(
                    message=AIMessage(
                        content="Progress: built O2 and preparing relax.",
                        tool_calls=[{
                            "id": "call_1",
                            "type": "tool_call",
                            "name": "mace_relax_batch",
                            "args": {"input": "o2/O2.vasp"},
                        }],
                    )
                )
            ]],
            llm_output={},
        ),
        run_id=rid,
    )

    end_events = [e for e in reporter.events if e.name == "LLM_CALL_END"]
    assert end_events
    payload = end_events[-1].payload
    assert payload.get("text_preview") == "Progress: built O2 and preparing relax."
    assert payload.get("tool_calls") == ["mace_relax_batch"]
    assert payload.get("agent_name") == "experiment_specialist"


def test_ui_event_handler_reasoning_delta_emits_only_new_suffix() -> None:
    reporter = _CollectReporter()
    handler = UIEventHandler(reporter, run_id="run_x")

    rid = uuid.uuid4()
    handler.on_llm_start(
        serialized={"kwargs": {"model_name": "gpt-5"}},
        prompts=["prompt"],
        run_id=rid,
    )
    handler.on_llm_new_token(
        "",
        run_id=rid,
        chunk={"reasoning_details": [{"type": "reasoning.summary", "summary": "Running batch process"}]},
    )
    handler.on_llm_new_token(
        "",
        run_id=rid,
        chunk={"reasoning_details": [{"type": "reasoning.summary", "summary": "Running batch process on boxed O2"}]},
    )

    reasoning_events = [e for e in reporter.events if e.name == "LLM_REASONING_DELTA"]
    assert len(reasoning_events) == 2
    assert reasoning_events[0].payload.get("text") == "Running batch process"
    assert reasoning_events[1].payload.get("text") == " on boxed O2"
