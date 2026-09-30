from __future__ import annotations

import sqlite3
from pathlib import Path

from catmaster.runtime.observability_store import OBSERVABILITY_DB_NAME, OBSERVABILITY_SCHEMA_VERSION, ObservabilityStore
from catmaster.__main__ import main as catmaster_main


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


def test_observability_store_deduplicates_llm_metrics_by_callback_id(tmp_path: Path) -> None:
    store = ObservabilityStore(tmp_path)
    usage = {"input_tokens": 10, "output_tokens": 4, "total_tokens": 14}
    for source, seq in (("langchain_callback", None), ("ui_event", 1)):
        store.record_event(
            source=source,
            channel="callback" if source == "langchain_callback" else "ui",
            name="LLM_CALL_END",
            category="llm",
            ts=1.0,
            seq=seq,
            run_id="run_demo",
            task_id="",
            step_id=None,
            payload={
                "model": "gpt-test",
                "agent_name": "worker",
                "callback_run_id": "llm_shared",
                "usage": usage,
            },
        )

    snapshot = store.read_snapshot()

    assert snapshot["metrics"]["llm_calls"] == 1
    assert snapshot["metrics"]["input_tokens"] == 10
    assert snapshot["metrics"]["output_tokens"] == 4


def test_observability_store_queries_core_observation_surfaces(tmp_path: Path) -> None:
    store = ObservabilityStore(tmp_path)
    store.record_event(
        source="langchain_callback",
        channel="callback",
        name="LLM_CALL_END",
        category="llm",
        ts=1.0,
        seq=None,
        run_id="run_demo",
        task_id="",
        step_id=None,
        payload={"model": "gpt-test", "callback_run_id": "llm_1", "usage": {"input_tokens": 3, "output_tokens": 2}},
    )
    store.record_event(
        source="langchain_callback",
        channel="callback",
        name="TOOL_CALL_END",
        category="tool",
        ts=2.0,
        seq=None,
        run_id="run_demo",
        task_id="",
        step_id=None,
        payload={"tool": "mace_relax_batch", "callback_run_id": "tool_1", "status": "success"},
    )
    for name, category, channel, payload in (
        ("MACHINE_TIME_RECORD", "machine_time", "machine", {"core_hours": 1.5}),
        ("task_receipt.updated", "thread", "thread", {"receipt": {"run_id": "remote_1", "status": "done"}}),
        ("subagent.status", "thread", "thread", {"agent_name": "materials_worker", "status": "done"}),
        ("artifact.created", "thread", "thread", {"artifact_id": "art_1", "path": "files/result.xyz"}),
        ("interrupt.created", "thread", "thread", {"interrupt_id": "hitl_1", "status": "pending"}),
    ):
        store.record_event(
            source="test",
            channel=channel,
            name=name,
            category=category,
            ts=3.0,
            seq=None,
            run_id="run_demo",
            task_id="",
            step_id=None,
            thread_id="thread_1",
            message_id="msg_1",
            payload=payload,
        )

    assert store.list_tool_names(event_names=["TOOL_CALL_END"]) == ["mace_relax_batch"]
    assert store.read_metrics()["llm_calls"] == 1
    page = store.read_events_page(limit=20)
    names = {event["name"] for event in page["events"]}
    assert {
        "LLM_CALL_END",
        "TOOL_CALL_END",
        "MACHINE_TIME_RECORD",
        "task_receipt.updated",
        "subagent.status",
        "artifact.created",
        "interrupt.created",
    }.issubset(names)


def test_observability_store_imports_existing_ui_events_explicitly(tmp_path: Path) -> None:
    (tmp_path / "ui_events.jsonl").write_text(
        '{"seq": 1, "ts": 1.0, "name": "RUN_START", "category": "run", "payload": {"status": "running"}}\n',
        encoding="utf-8",
    )

    store = ObservabilityStore(tmp_path)
    assert store.read_snapshot()["metrics"]["total_events"] == 0
    assert store.import_legacy_jsonl(include_ui_events=True) == 1
    snapshot = store.read_snapshot()

    assert snapshot["metrics"]["event_counts"]["RUN_START"] == 1


def test_catmaster_migrate_observability_cli_imports_legacy_jsonl(tmp_path: Path, capsys) -> None:
    (tmp_path / "ui_events.jsonl").write_text(
        '{"seq": 1, "ts": 1.0, "name": "RUN_START", "category": "run", "payload": {"status": "running"}}\n',
        encoding="utf-8",
    )

    assert catmaster_main(["migrate-observability", str(tmp_path), "--no-trace-records"]) == 0

    assert "Imported 1 legacy observation records." in capsys.readouterr().out
    assert ObservabilityStore(tmp_path).read_snapshot()["metrics"]["event_counts"]["RUN_START"] == 1


def test_observability_store_migrates_pre_thread_schema(tmp_path: Path) -> None:
    db_path = tmp_path / OBSERVABILITY_DB_NAME
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE observation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                seq INTEGER,
                source TEXT NOT NULL,
                category TEXT NOT NULL,
                name TEXT NOT NULL,
                run_id TEXT,
                task_id TEXT,
                step_id INTEGER,
                agent_name TEXT,
                callback_run_id TEXT,
                parent_callback_run_id TEXT,
                node TEXT,
                model TEXT,
                tool TEXT,
                status TEXT,
                duration_ms INTEGER,
                payload_json TEXT NOT NULL
            )
            """
        )

    store = ObservabilityStore(tmp_path)
    store.record_ui_event(
        {
            "seq": 1,
            "name": "message.delta",
            "thread_id": "thread_old",
            "message_id": "msg_old",
            "part_id": "part_old",
            "payload": {"text": "ok"},
        }
    )

    with sqlite3.connect(str(db_path)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(observation_events)").fetchall()}
        schema_version = conn.execute("PRAGMA user_version").fetchone()[0]
    assert {"channel", "thread_id", "message_id", "part_id"} <= columns
    assert schema_version == OBSERVABILITY_SCHEMA_VERSION

    event = store.read_snapshot()["events"][0]
    assert event["channel"] == "ui"
    assert event["thread_id"] == "thread_old"
    assert event["message_id"] == "msg_old"
    assert event["part_id"] == "part_old"


def test_observability_store_query_wrappers_filter_events_and_metrics(tmp_path: Path) -> None:
    store = ObservabilityStore(tmp_path)
    store.record_ui_event({"seq": 1, "name": "TOOL_CALL_END", "category": "tool", "payload": {"tool": "bash", "status": "success"}})
    store.record_ui_event({"seq": 2, "name": "LLM_CALL_END", "category": "llm", "payload": {"model": "model-a", "usage": {"input_tokens": 1}}})

    page = store.read_events_page(category="tool")

    assert [event["name"] for event in page["events"]] == ["TOOL_CALL_END"]
    assert store.read_metrics()["tool_calls"] == 1
    assert store.read_run_snapshot(limit=1)["metrics"]["llm_calls"] == 1


def test_observability_snapshot_hides_legacy_trace_records_by_default(tmp_path: Path) -> None:
    store = ObservabilityStore(tmp_path)
    store.record_ui_event(
        {
            "seq": 1,
            "ts": 1.0,
            "name": "LLM_CALL_END",
            "category": "llm",
            "payload": {
                "model": "stream-model",
                "usage": {"input_tokens": 2, "output_tokens": 3},
            },
        }
    )
    store.record_trace_record(
        "event_trace.jsonl",
        {
            "event": "LLM_RAW_RESPONSE",
            "payload": {
                "model": "legacy-trace-model",
                "generations": [{"response_text": "legacy raw response"}],
            },
        },
    )
    store.record_trace_record(
        "tool_trace.jsonl",
        {"tool_name": "legacy_tool", "status": "success"},
    )

    snapshot = store.read_snapshot()

    assert snapshot["metrics"]["total_events"] == 1
    assert [event["name"] for event in snapshot["events"]] == ["LLM_CALL_END"]
    assert snapshot["raw_logs"]["total_events"] == 1

    legacy_snapshot = store.read_snapshot(include_legacy_trace_records=True)
    assert legacy_snapshot["metrics"]["total_events"] == 3
    assert "LLM_RAW_RESPONSE" in [event["name"] for event in legacy_snapshot["events"]]
    assert "TOOL_TRACE" in [event["name"] for event in legacy_snapshot["events"]]
