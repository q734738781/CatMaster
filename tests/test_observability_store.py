from __future__ import annotations

from pathlib import Path

from catmaster.runtime.observability_store import OBSERVABILITY_DB_NAME, ObservabilityStore


def test_observability_store_records_metrics_trace_decisions_and_state(tmp_path: Path) -> None:
    store = ObservabilityStore(tmp_path)

    store.record_ui_event(
        {
            "seq": 1,
            "ts": 10.0,
            "name": "LLM_CALL_START",
            "category": "llm",
            "run_id": "run_demo",
            "payload": {
                "model": "gpt-test",
                "agent_name": "worker",
                "callback_run_id": "llm_1",
            },
        }
    )
    store.record_ui_event(
        {
            "seq": 2,
            "ts": 11.0,
            "name": "TOOL_CALL_START",
            "category": "tool",
            "run_id": "run_demo",
            "payload": {
                "tool": "write_todos",
                "agent_name": "worker",
                "callback_run_id": "tool_1",
                "parent_callback_run_id": "llm_1",
                "params_full": {"todos": [{"content": "Prepare O2", "status": "in_progress"}]},
            },
        }
    )
    store.record_ui_event(
        {
            "seq": 3,
            "ts": 12.0,
            "name": "TOOL_CALL_END",
            "category": "tool",
            "run_id": "run_demo",
            "payload": {
                "tool": "write_todos",
                "status": "success",
                "agent_name": "worker",
                "callback_run_id": "tool_1",
                "parent_callback_run_id": "llm_1",
            },
        }
    )
    store.record_ui_event(
        {
            "seq": 4,
            "ts": 13.0,
            "name": "LLM_CALL_END",
            "category": "llm",
            "run_id": "run_demo",
            "payload": {
                "model": "gpt-test",
                "agent_name": "worker",
                "callback_run_id": "llm_1",
                "elapsed_ms": 3000,
                "reasoning_text": "Need the staged O2 input before remote submission.",
                "text_preview": "I will prepare the O2 stage.",
                "tool_calls": ["write_todos"],
                "usage": {"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 2},
            },
        }
    )
    store.record_run_state({"status": "running", "phase": "executing", "text_preview": "Prepare O2"}, reason="test")

    snapshot = store.read_snapshot()

    assert (tmp_path / OBSERVABILITY_DB_NAME).exists()
    assert snapshot["metrics"]["total_events"] >= 5
    assert snapshot["metrics"]["llm_calls"] == 1
    assert snapshot["metrics"]["tool_calls"] == 1
    assert snapshot["metrics"]["input_tokens"] == 10
    assert snapshot["metrics"]["error_rate"] == 0.0
    assert snapshot["decisions"][0]["reason"] == "Need the staged O2 input before remote submission."
    assert snapshot["task_state"]["plan_revision_count"] == 1
    assert snapshot["task_state"]["todos"][0]["content"] == "Prepare O2"
    tool_node = [node for node in snapshot["trace_tree"]["nodes"] if node["id"] == "tool_1"][0]
    assert tool_node["parent_id"] == "llm_1"
    page = store.read_ui_events_page(limit=2)
    assert page is not None
    assert [event["seq"] for event in page["events"]] == [3, 4]
    assert page["has_more"] is True
    assert store.last_ui_event_seq() == 4


def test_observability_store_backfills_existing_ui_events(tmp_path: Path) -> None:
    (tmp_path / "ui_events.jsonl").write_text(
        '{"seq": 1, "ts": 1.0, "name": "RUN_START", "category": "run", "payload": {"status": "running"}}\n',
        encoding="utf-8",
    )

    snapshot = ObservabilityStore(tmp_path).read_snapshot()

    assert snapshot["metrics"]["event_counts"]["RUN_START"] == 1
