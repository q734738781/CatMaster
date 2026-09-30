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
                    "name": "vasp_prepare",
                    "arguments": args_raw,
                },
            }]
        },
        tool_calls=[{
            "id": "call_1",
            "type": "tool_call",
            "name": "vasp_prepare",
            "args": {"MAGMOM": "1 1", "NUPDOWN": 2},
        }],
    )
    result = LLMResult(
        generations=[[ChatGeneration(message=message)]],
        llm_output={"token_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}},
    )

    handler.on_llm_start(
        serialized={"kwargs": {"model_name": "openai/gpt-5.2"}},
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
        additional_kwargs={
            "reasoning_content": "Need a compact explanation before tool use.",
            "reasoning_details": [
                {
                    "type": "reasoning.summary",
                    "summary": "Use write_todos first, then inspect the current workspace.",
                }
            ],
        },
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
    raw_record = next(r for r in records if r.get("event") == "LLM_RAW_RESPONSE")

    assert not any(r.get("event") == "LLM_USAGE" for r in records)
    generation = raw_record["payload"]["generations"][0]
    assert generation["usage_metadata"]["input_token_details"]["cache_read"] == 2
    assert generation["usage_metadata"]["output_token_details"]["reasoning"] == 3
    assert "spin needs to be constrained" in raw_record["payload"]["generations"][0]["reasoning_text"]
    assert "compact explanation before tool use" in raw_record["payload"]["generations"][0]["reasoning_text"]
    assert "Use write_todos first" in raw_record["payload"]["generations"][0]["reasoning_text"]


def test_llm_tracing_handler_merges_fragmented_reasoning_tokens(tmp_path) -> None:
    trace = TraceStore(tmp_path)
    handler = LLMTracingHandler(trace, run_id="run_fragments")
    rid = uuid.uuid4()

    message = AIMessage(
        content=[],
        additional_kwargs={
            "reasoning_details": [
                {"type": "reasoning.summary", "summary": "Evalu"},
                {"type": "reasoning.summary", "summary": "ating"},
                {"type": "reasoning.summary", "summary": "M"},
                {"type": "reasoning.summary", "summary": "ACE"},
                {"type": "reasoning.summary", "summary": "setup"},
            ],
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
    raw_record = next(r for r in records if r.get("event") == "LLM_RAW_RESPONSE")
    reasoning_text = raw_record["payload"]["generations"][0]["reasoning_text"]
    assert "\n" not in reasoning_text
    assert reasoning_text == "EvaluatingMACEsetup"


def test_llm_tracing_handler_prefers_complete_reasoning_summary_over_fragments(tmp_path) -> None:
    trace = TraceStore(tmp_path)
    handler = LLMTracingHandler(trace, run_id="run_reasoning_mix")
    rid = uuid.uuid4()

    message = AIMessage(
        content=[],
        additional_kwargs={
            "reasoning_details": [
                {"type": "reasoning.summary", "summary": "Running batch process"},
                {"type": "reasoning.summary", "summary": "Running"},
                {"type": "reasoning.summary", "summary": "batch"},
                {"type": "reasoning.summary", "summary": "process"},
                {"type": "reasoning.summary", "summary": "on boxed O2"},
                {"type": "reasoning.summary", "summary": "on"},
                {"type": "reasoning.summary", "summary": "boxed"},
                {"type": "reasoning.summary", "summary": "O2"},
            ],
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
    raw_record = next(r for r in records if r.get("event") == "LLM_RAW_RESPONSE")
    reasoning_text = raw_record["payload"]["generations"][0]["reasoning_text"]
    assert reasoning_text == "Running batch process on boxed O2"
