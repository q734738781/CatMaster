from __future__ import annotations

import logging
import uuid

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from catmaster.runtime.artifact_callback import LangChainStepLogger, build_callbacks
from catmaster.runtime.artifact_store import ArtifactStore
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
