from __future__ import annotations

import json

from catmaster.runtime.usage_stats import (
    summarize_usage_from_metadata,
    write_usage_summary_from_metadata,
)


def test_usage_summary_from_langchain_metadata_aggregates_tokens_and_calls(tmp_path) -> None:
    summary = summarize_usage_from_metadata(
        {
            "openai/gpt-5.4-20260305": {
                "input_tokens": 1200,
                "output_tokens": 140,
                "total_tokens": 1340,
                "input_token_details": {"cache_read": 900, "cache_creation": 50},
                "output_token_details": {"reasoning": 20},
            }
        },
        run_dir=tmp_path,
        call_counts_by_model={"openai/gpt-5.4-20260305": 3},
        usage_metadata_by_role={
            "writing_specialist": {
                "openai/gpt-5.4-20260305": {
                    "input_tokens": 400,
                    "output_tokens": 50,
                    "total_tokens": 450,
                    "input_token_details": {"cache_read": 200},
                    "output_token_details": {"reasoning": 10},
                }
            }
        },
        call_counts_by_role={"writing_specialist": 2},
    )

    assert summary["source"] == "langchain_usage_metadata"
    assert summary["calls"] == 3
    assert summary["input_tokens"] == 1200
    assert summary["input_uncached_tokens"] == 250
    assert summary["input_cached_tokens"] == 900
    assert summary["input_cache_read_tokens"] == 900
    assert summary["input_cache_write_tokens"] == 50
    assert summary["output_tokens"] == 140
    assert summary["reasoning_tokens"] == 20
    assert summary["total_tokens"] == 1340
    assert summary["by_model"][0]["name"] == "openai/gpt-5.4-20260305"
    assert summary["by_model"][0]["calls"] == 3
    assert summary["by_model"][0]["input_uncached_tokens"] == 250
    assert summary["by_role"][0]["name"] == "writing_specialist"
    assert summary["by_role"][0]["calls"] == 2
    assert summary["by_role"][0]["input_tokens"] == 400
    assert summary["by_role"][0]["input_uncached_tokens"] == 200
    assert summary["by_role"][0]["output_tokens"] == 50


def test_write_usage_summary_from_metadata_appends_existing_totals(tmp_path) -> None:
    first = write_usage_summary_from_metadata(
        tmp_path,
        usage_metadata={
            "model-a": {
                "input_tokens": 100,
                "output_tokens": 20,
                "total_tokens": 120,
                "input_token_details": {"cache_read": 40},
            }
        },
        call_counts_by_model={"model-a": 1},
        append=True,
    )
    second = write_usage_summary_from_metadata(
        tmp_path,
        usage_metadata={
            "model-a": {
                "input_tokens": 50,
                "output_tokens": 10,
                "total_tokens": 60,
                "input_token_details": {"cache_read": 5},
                "output_token_details": {"reasoning": 2},
            }
        },
        call_counts_by_model={"model-a": 2},
        usage_metadata_by_role={
            "literature_agent": {
                "model-a": {
                    "input_tokens": 25,
                    "output_tokens": 4,
                    "total_tokens": 29,
                }
            }
        },
        call_counts_by_role={"literature_agent": 1},
        append=True,
    )

    assert first["input_tokens"] == 100
    assert second["input_tokens"] == 150
    assert second["input_uncached_tokens"] == 105
    assert second["output_tokens"] == 30
    assert second["input_cached_tokens"] == 45
    assert second["reasoning_tokens"] == 2
    assert second["calls"] == 3
    assert second["by_role"][0]["name"] == "literature_agent"
    assert second["by_role"][0]["calls"] == 1


def test_write_usage_summary_from_metadata_writes_file(tmp_path) -> None:
    summary = write_usage_summary_from_metadata(
        tmp_path,
        usage_metadata={},
        call_counts_by_model={},
        usage_metadata_by_role={},
        call_counts_by_role={},
        append=True,
    )

    assert summary["calls"] == 0
    payload = json.loads((tmp_path / "usage_summary.json").read_text(encoding="utf-8"))
    assert payload["calls"] == 0
