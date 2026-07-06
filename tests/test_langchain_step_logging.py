from __future__ import annotations

import logging
import uuid

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from catmaster.runtime.artifact_callback import LangChainStepLogger, LLMTracingHandler, ObservabilityCallbackHandler, build_callbacks
from catmaster.runtime.artifact_store import ArtifactStore
from catmaster.runtime.observability_store import OBSERVABILITY_DB_NAME, ObservabilityStore
from catmaster.runtime.trace_store import TraceStore
from catmaster.ui.reporters import NullReporter


def test_langchain_step_logger_avoids_prompt_context_dump(caplog) -> None:
    logger = LangChainStepLogger(run_id="run_step")
    rid = uuid.uuid4()
    secret_prompt = "SECRET_PROMPT_SHOULD_NOT_BE_LOGGED"

    caplog.set_level(logging.INFO, logger="catmaster.langchain")
    logger.on_chat_model_start(
        serialized={"kwargs": {"model_name": "gpt-test"}},
        messages=[[AIMessage(content=secret_prompt)]],
        run_id=rid,
    )
    logger.on_tool_start(
        serialized={"name": "bash"},
        input_str='{"secret":"tool args should stay out of logs"}',
        run_id=rid,
    )
    logger.on_tool_end(
        {"status": "success", "tool_name": "bash", "summary": "command completed"},
        run_id=rid,
    )
    logger.on_llm_end(
        LLMResult(
            generations=[[ChatGeneration(message=AIMessage(content="finished"))]],
            llm_output={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}},
        ),
        run_id=rid,
    )

    text = caplog.text
    assert "phase=llm.start" in text
    assert "phase=tool.start" in text
    assert "phase=tool.end" in text
    assert "phase=llm.end" in text
    assert secret_prompt not in text
    assert "tool args should stay out of logs" not in text


def test_build_callbacks_can_enable_step_logs(tmp_path) -> None:
    callbacks = build_callbacks(
        artifact_store=ArtifactStore(tmp_path),
        trace_store=TraceStore(tmp_path),
        reporter=NullReporter(),
        run_id="run_x",
        enable_step_logs=True,
    )

    assert any(isinstance(callback, LangChainStepLogger) for callback in callbacks)
    assert any(isinstance(callback, ObservabilityCallbackHandler) for callback in callbacks)
    assert not any(isinstance(callback, LLMTracingHandler) for callback in callbacks)


def test_build_callbacks_default_runtime_writes_only_observability_db(tmp_path) -> None:
    callbacks = build_callbacks(
        artifact_store=ArtifactStore(tmp_path),
        reporter=NullReporter(),
        run_id="run_x",
    )
    callback_run_id = uuid.uuid4()
    for callback in callbacks:
        try:
            callback.on_chat_model_start(
                serialized={"kwargs": {"model_name": "model-a"}},
                messages=[[AIMessage(content="hello")]],
                run_id=callback_run_id,
            )
        except NotImplementedError:
            pass
        try:
            callback.on_llm_end(
                LLMResult(generations=[[ChatGeneration(message=AIMessage(content="done"))]], llm_output={}),
                run_id=callback_run_id,
            )
        except NotImplementedError:
            pass
    tool_run_id = uuid.uuid4()
    for callback in callbacks:
        try:
            callback.on_tool_start(serialized={"name": "bash"}, input_str='{"cmd":"echo ok"}', run_id=tool_run_id)
        except NotImplementedError:
            pass
        try:
            callback.on_tool_end({"status": "success", "tool_name": "bash"}, run_id=tool_run_id)
        except NotImplementedError:
            pass

    assert (tmp_path / OBSERVABILITY_DB_NAME).exists()
    assert not (tmp_path / "event_trace.jsonl").exists()
    assert not (tmp_path / "tool_trace.jsonl").exists()
    assert ObservabilityStore(tmp_path).read_metrics()["tool_calls"] == 1
