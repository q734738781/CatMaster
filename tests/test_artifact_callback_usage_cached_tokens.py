from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from catmaster.runtime.artifact_callback import _extract_usage_from_llm_result


def test_extract_usage_reads_cached_tokens_from_prompt_details() -> None:
    result = LLMResult(
        generations=[[ChatGeneration(message=AIMessage(content="ok"))]],
        llm_output={
            "token_usage": {
                "prompt_tokens": 120,
                "completion_tokens": 16,
                "total_tokens": 136,
                "prompt_tokens_details": {
                    "cached_tokens": 88,
                },
            }
        },
    )

    usage = _extract_usage_from_llm_result(result)

    assert usage["input_tokens"] == 120
    assert usage["output_tokens"] == 16
    assert usage["total_tokens"] == 136
    assert usage["input_cached_tokens"] == 88


def test_extract_usage_reads_cached_tokens_from_usage_metadata() -> None:
    result = LLMResult(
        generations=[[
            ChatGeneration(
                message=AIMessage(
                    content="ok",
                    usage_metadata={
                        "input_tokens": 44,
                        "output_tokens": 7,
                        "total_tokens": 51,
                        "input_token_details": {
                            "flex_cache_read": 33,
                        },
                    },
                )
            )
        ]],
        llm_output={},
    )

    usage = _extract_usage_from_llm_result(result)

    assert usage["input_tokens"] == 44
    assert usage["output_tokens"] == 7
    assert usage["total_tokens"] == 51
    assert usage["input_cached_tokens"] == 33

