from __future__ import annotations

import json

from catmaster.runtime.usage_stats import summarize_usage_from_event_trace, write_usage_summary


def test_usage_summary_aggregates_llm_usage_events(tmp_path) -> None:
    event_path = tmp_path / "event_trace.jsonl"
    events = [
        {
            "event": "LLM_USAGE",
            "payload": {
                "task_id": "task_01",
                "usage": {
                    "input_tokens": 100,
                    "input_cached_tokens": 20,
                    "output_tokens": 30,
                    "total_tokens": 130,
                    "source": "provider",
                },
            },
        },
        {
            "event": "LLM_USAGE",
            "payload": {
                "task_id": "task_02",
                "usage": {
                    "input_tokens": None,
                    "input_cached_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "source": "missing",
                },
            },
        },
        {
            "event": "LLM_USAGE",
            "payload": {
                "task_id": "task_03",
                "usage": {
                    "input_tokens": 10,
                    "input_cached_tokens": 2,
                    "output_tokens": 5,
                    "total_tokens": None,
                    "source": "provider",
                },
            },
        },
    ]
    event_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in events),
        encoding="utf-8",
    )

    summary = summarize_usage_from_event_trace(tmp_path)

    assert summary["calls"] == 3
    assert summary["missing_usage_calls"] == 1
    assert summary["input_tokens"] == 110
    assert summary["input_cached_tokens"] == 22
    assert summary["output_tokens"] == 35
    # third entry total falls back to input+output
    assert summary["total_tokens"] == 145


def test_write_usage_summary_writes_file(tmp_path) -> None:
    (tmp_path / "event_trace.jsonl").write_text("", encoding="utf-8")

    summary = write_usage_summary(tmp_path)

    assert summary["calls"] == 0
    payload = json.loads((tmp_path / "usage_summary.json").read_text(encoding="utf-8"))
    assert payload["calls"] == 0
