from __future__ import annotations

import json
from pathlib import Path

from catmaster.runtime import usage_stats
from catmaster.webui.view_utils import render_cost_card_markdown


def _write_event_trace(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "event_trace.jsonl"
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_usage_summary_aggregates_exact_costs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(usage_stats, "resolve_model_pricing", lambda _model: (None, {}))
    run_dir = tmp_path / "run_a"
    _write_event_trace(
        run_dir,
        [
            {
                "event": "LLM_USAGE",
                "payload": {
                    "role": "task_runner",
                    "model": "openai/gpt-5.4:online",
                    "usage": {
                        "input_tokens": 1000,
                        "input_cached_tokens": 400,
                        "output_tokens": 200,
                        "total_tokens": 1200,
                        "cost": 0.0125,
                        "cost_details": {
                            "upstream_inference_prompt_cost": 0.0075,
                            "upstream_inference_completions_cost": 0.005,
                        },
                    },
                },
            }
        ],
    )

    summary = usage_stats.summarize_usage_from_event_trace(run_dir)

    assert summary["cost_source"] == "exact"
    assert summary["cost_usd"] == 0.0125
    assert summary["exact_cost_usd"] == 0.0125
    assert summary["estimated_cost_usd"] == 0.0
    assert summary["input_tokens"] == 1000
    assert summary["input_cached_tokens"] == 400
    assert summary["output_tokens"] == 200
    assert summary["by_role"][0]["name"] == "task_runner"


def test_usage_summary_estimates_cost_from_pricing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        usage_stats,
        "resolve_model_pricing",
        lambda _model: (
            "openai/gpt-5.4",
            {
                "prompt": 0.000002,
                "input_cache_read": 0.0000005,
                "input_cache_write": 0.000001,
                "completion": 0.000008,
                "internal_reasoning": 0.0000015,
            },
        ),
    )
    run_dir = tmp_path / "run_b"
    _write_event_trace(
        run_dir,
        [
            {
                "event": "LLM_USAGE",
                "payload": {
                    "role": "tool_selector",
                    "model": "openai/gpt-5.4:online-minimum",
                    "usage": {
                        "input_tokens": 1000,
                        "input_cached_tokens": 300,
                        "output_tokens": 120,
                        "total_tokens": 1120,
                        "input_token_details": {"cache_write": 100},
                        "output_token_details": {"reasoning": 20},
                    },
                },
            }
        ],
    )

    summary = usage_stats.summarize_usage_from_event_trace(run_dir)

    expected = round(
        600 * 0.000002 + 300 * 0.0000005 + 100 * 0.000001 + 100 * 0.000008 + 20 * 0.0000015,
        8,
    )
    assert summary["cost_source"] == "estimated"
    assert summary["cost_usd"] == expected
    assert summary["estimated_cost_usd"] == expected
    assert summary["breakdown_usd"]["prompt_uncached"] == round(600 * 0.000002, 8)
    assert summary["breakdown_usd"]["cache_read"] == round(300 * 0.0000005, 8)
    assert summary["breakdown_usd"]["cache_write"] == round(100 * 0.000001, 8)
    assert summary["breakdown_usd"]["completion"] == round(100 * 0.000008, 8)
    assert summary["breakdown_usd"]["internal_reasoning"] == round(20 * 0.0000015, 8)


def test_render_cost_card_markdown_shows_key_fields() -> None:
    text = render_cost_card_markdown(
        {
            "cost_usd": 0.1234,
            "exact_cost_usd": 0.1,
            "estimated_cost_usd": 0.0234,
            "cost_source": "mixed",
            "calls": 9,
            "input_tokens": 12000,
            "input_cached_tokens": 4000,
            "input_cache_write_tokens": 500,
            "output_tokens": 3200,
            "reasoning_tokens": 800,
            "missing_cost_calls": 0,
            "breakdown_usd": {
                "prompt_uncached": 0.04,
                "cache_read": 0.01,
                "cache_write": 0.002,
                "completion": 0.06,
                "internal_reasoning": 0.0114,
            },
            "by_role": [
                {"name": "task_runner", "cost_usd": 0.09},
                {"name": "director", "cost_usd": 0.02},
            ],
        }
    )

    assert "### Cost" in text
    assert "`$0.1234`" in text
    assert "Source: `mixed`" in text
    assert "prompt uncached" in text
    assert "`task_runner`: `$0.0900`" in text
