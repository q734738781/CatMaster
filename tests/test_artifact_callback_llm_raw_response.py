from __future__ import annotations

import json
import uuid

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from catmaster.runtime.artifact_callback import LLMTracingHandler
from catmaster.runtime.trace_store import TraceStore


def test_llm_tracing_handler_persists_raw_response_and_toolcall_arguments(tmp_path) -> None:
    trace = TraceStore(tmp_path)
    handler = LLMTracingHandler(trace, run_id="run_x")
    rid = uuid.uuid4()

    args_raw = '{"MAGMOM":"1 1","NUPDOWN":2}'
    message = AIMessage(
        content="Preparing relax input.",
        additional_kwargs={
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "vasp_relax_prepare",
                    "arguments": args_raw,
                },
            }]
        },
        tool_calls=[{
            "id": "call_1",
            "type": "tool_call",
            "name": "vasp_relax_prepare",
            "args": {"MAGMOM": "1 1", "NUPDOWN": 2},
        }],
    )
    result = LLMResult(
        generations=[[ChatGeneration(message=message)]],
        llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    )

    handler.on_llm_start(
        serialized={"kwargs": {"model_name": "openai/gpt-5.2:online"}},
        prompts=["prompt text"],
        run_id=rid,
    )
    handler.on_llm_end(result, run_id=rid)

    records = [
        json.loads(line)
        for line in (tmp_path / "event_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    raw_records = [r for r in records if r.get("event") == "LLM_RAW_RESPONSE"]
    assert raw_records

    payload = raw_records[-1]["payload"]
    generation = payload["generations"][0]
    assert generation["response_text"] == "Preparing relax input."
    assert generation["raw_tool_calls"][0]["arguments_raw"] == args_raw
    assert generation["parsed_tool_calls"][0]["args_json"] == json.dumps(
        {"MAGMOM": "1 1", "NUPDOWN": 2},
        ensure_ascii=False,
    )


def test_llm_tracing_handler_extracts_reasoning_summary_and_usage_details(tmp_path) -> None:
    trace = TraceStore(tmp_path)
    handler = LLMTracingHandler(trace, run_id="run_reasoning")
    rid = uuid.uuid4()

    message = AIMessage(
        content=[
            {
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": "Checking whether spin needs to be constrained."}
                ],
            },
            {"type": "text", "text": "Triplet is more stable."},
        ],
        usage_metadata={
            "input_tokens": 10,
            "output_tokens": 8,
            "total_tokens": 18,
            "input_token_details": {"cache_read": 2},
            "output_token_details": {"reasoning": 3},
        },
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]], llm_output={})

    handler.on_llm_start(
        serialized={"kwargs": {"model_name": "gpt-5"}},
        prompts=["prompt"],
        run_id=rid,
    )
    handler.on_llm_end(result, run_id=rid)

    records = [
        json.loads(line)
        for line in (tmp_path / "event_trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    usage_record = next(r for r in records if r.get("event") == "LLM_USAGE")
    raw_record = next(r for r in records if r.get("event") == "LLM_RAW_RESPONSE")

    assert usage_record["payload"]["usage"]["input_token_details"]["cache_read"] == 2
    assert usage_record["payload"]["usage"]["output_token_details"]["reasoning"] == 3
    assert "spin needs to be constrained" in usage_record["payload"]["reasoning_text"]
    assert "spin needs to be constrained" in raw_record["payload"]["generations"][0]["reasoning_text"]
