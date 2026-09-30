from __future__ import annotations

import json
from pathlib import Path

from catmaster.runtime.machine_time_stats import (
    append_machine_time_record,
    build_machine_time_record,
    load_machine_time_summary,
    machine_time_records_path,
    machine_time_summary_path,
    summarize_machine_time_from_observability,
)
from catmaster.runtime.observability_store import ObservabilityStore


def test_append_machine_time_record_summarizes_cpu_core_and_node_hours(tmp_path: Path) -> None:
    append_machine_time_record(
        tmp_path,
        {
            "status": "success",
            "tool_name": "remote_submission",
            "task_name": "vasp_execute",
            "resources": "vasp_cpu",
            "machine": "cpu_server",
            "task_count": 1,
            "number_node": 2,
            "cpu_per_node": 32,
            "gpu_per_node": 0,
            "elapsed_seconds": 1800,
            "core_hours": 32.0,
            "node_hours": 1.0,
        },
    )

    summary = load_machine_time_summary(tmp_path)

    assert summary["requests"] == 1
    assert summary["successful_requests"] == 1
    assert summary["core_hours"] == 32.0
    assert summary["node_hours"] == 1.0
    assert summary["gpu_node_hours"] == 0.0
    assert summary["by_resource"][0]["name"] == "vasp_cpu"
    assert summary["source"] == "observability_store"
    assert machine_time_records_path(tmp_path).is_file()
    assert machine_time_summary_path(tmp_path).is_file()


def test_build_machine_time_record_uses_status_elapsed_for_gpu_node_hours() -> None:
    class _Result:
        duration_s = 999.0
        work_base = "remote_submission_stage"
        remote_context_id = "dp_ctx"
        submission_hash = "abc"
        receipt_rel = ".deepagents/dpdispatcher/receipts/dp_ctx.json"
        jobs = [{"job_hash": "jobhash", "job_id": "123", "status": "finished"}]
        status_records = [
            {"task_index": 0, "task_work_path": ".", "returncode": 0, "elapsed_seconds": 3600.0},
            {"task_index": 1, "task_work_path": ".", "returncode": 0, "elapsed_seconds": 1800.0},
        ]

    record = build_machine_time_record(
        status="success",
        tool_name="remote_submission_batch",
        task_name="mace_relax_dir",
        work_dir_rel="mace_batch",
        work_base="remote_submission_stage",
        resources_key="mace_gpu",
        resource_cfg={
            "machine": "gpu_server",
            "queue_name": "main",
            "number_node": 1,
            "cpu_per_node": 16,
            "gpu_per_node": 1,
            "group_size": 1,
        },
        task_count=2,
        result=_Result(),
    )

    assert record["elapsed_source"] == "remote_status"
    assert record["elapsed_seconds"] == 5400.0
    assert record["core_hours"] == 24.0
    assert record["node_hours"] == 1.5
    assert record["gpu_node_hours"] == 1.5
    assert record["gpu_hours"] == 1.5
    assert record["usage_basis"] == "node_hours"
    assert record["remote_context_id"] == "dp_ctx"
    assert record["jobs"][0]["job_id"] == "123"


def test_load_machine_time_summary_returns_empty_for_legacy_run(tmp_path: Path) -> None:
    summary = load_machine_time_summary(tmp_path)

    assert summary["requests"] == 0
    assert summary["core_hours"] == 0.0
    assert summary["records"] == []
    assert not machine_time_records_path(tmp_path).exists()


def test_append_machine_time_record_writes_jsonl(tmp_path: Path) -> None:
    append_machine_time_record(
        tmp_path,
        {
            "status": "failed",
            "tool_name": "remote_submission",
            "resources": "general_cpu",
            "machine": "cpu_server",
        },
    )

    line = machine_time_records_path(tmp_path).read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["status"] == "failed"
    assert payload["core_hours"] == 0.0


def test_machine_time_summary_pages_all_events_and_reports_coverage(tmp_path: Path) -> None:
    store = ObservabilityStore(tmp_path)
    rows = []
    for index in range(5_001):
        payload = {
            "record_id": f"record-{index}",
            "status": "success",
            "tool_name": "remote_submission",
            "resources": "cpu",
            "machine": "worker",
            "task_count": 1,
            "core_hours": 0.5,
            "node_hours": 0.25,
        }
        rows.append(
            (
                float(index), None, "machine_time", "machine", "machine",
                "MACHINE_TIME_RECORD", "run-many-machine-events", f"task-{index}",
                None, "", "", "", "", "", "", "", "", "remote_submission",
                "success", None, json.dumps(payload, sort_keys=True),
            )
        )
    with store._connect() as conn:
        conn.executemany(
            """
            INSERT INTO observation_events (
                ts, seq, source, channel, category, name,
                run_id, task_id, step_id, thread_id, message_id, part_id,
                agent_name, callback_run_id, parent_callback_run_id,
                node, model, tool, status, duration_ms, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    summary = summarize_machine_time_from_observability(tmp_path)

    assert summary["requests"] == 5_001
    assert summary["successful_requests"] == 5_001
    assert summary["task_count"] == 5_001
    assert summary["core_hours"] == 2_500.5
    assert summary["node_hours"] == 1_250.25
    assert summary["source_event_count"] == 5_001
    assert summary["source_min_event_id"] == 1
    assert summary["source_max_event_id"] == 5_001
    assert summary["partial"] is False
