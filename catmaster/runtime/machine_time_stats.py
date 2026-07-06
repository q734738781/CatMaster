#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-level remote machine-time accounting summaries."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List
import json
import uuid

from catmaster.runtime import observation_events as obs_events
from catmaster.runtime.observability_store import ObservabilityStore

MACHINE_TIME_RECORDS_NAME = "machine_time_records.jsonl"
MACHINE_TIME_SUMMARY_NAME = "machine_time_summary.json"


def machine_time_records_path(run_dir: Path) -> Path:
    return Path(run_dir).expanduser().resolve() / MACHINE_TIME_RECORDS_NAME


def machine_time_summary_path(run_dir: Path) -> Path:
    return Path(run_dir).expanduser().resolve() / MACHINE_TIME_SUMMARY_NAME


def load_machine_time_summary(run_dir: Path | None) -> Dict[str, Any]:
    if run_dir is None:
        return empty_machine_time_summary()
    run_path = Path(run_dir).expanduser().resolve()
    observed = summarize_machine_time_from_observability(run_path)
    if observed.get("requests"):
        return observed
    records_path = machine_time_records_path(run_path)
    if records_path.exists():
        summary = summarize_machine_time_records(_iter_record_file(records_path), run_dir=run_path)
        _write_summary(run_path, summary)
        return summary
    path = machine_time_summary_path(run_path)
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return empty_machine_time_summary(run_path)
        return payload if isinstance(payload, dict) else empty_machine_time_summary(run_path)
    return empty_machine_time_summary(run_path)


def append_machine_time_record(run_dir: Path | str | None, record: Dict[str, Any]) -> Dict[str, Any]:
    if run_dir is None:
        return empty_machine_time_summary()
    run_path = Path(run_dir).expanduser().resolve()
    normalized = normalize_machine_time_record(record)
    try:
        ObservabilityStore(run_path).record_event(
            source="machine_time",
            channel="machine",
            name=obs_events.MACHINE_TIME_RECORD,
            category="machine",
            ts=_to_timestamp(normalized.get("recorded_at")),
            seq=None,
            run_id=run_path.name,
            task_id=str(normalized.get("task_name") or ""),
            step_id=None,
            payload=normalized,
        )
    except Exception:
        pass
    path = machine_time_records_path(run_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(normalized, ensure_ascii=False, sort_keys=True) + "\n")
    summary = summarize_machine_time_records(_iter_record_file(path), run_dir=run_path)
    _write_summary(run_path, summary)
    return summary


def summarize_machine_time_from_observability(run_dir: Path | str | None) -> Dict[str, Any]:
    if run_dir is None:
        return empty_machine_time_summary()
    run_path = Path(run_dir).expanduser().resolve()
    try:
        page = ObservabilityStore(run_path).read_events_page(names=[obs_events.MACHINE_TIME_RECORD], limit=5000)
    except Exception:
        return empty_machine_time_summary(run_path)
    records: list[Dict[str, Any]] = []
    for event in page.get("events") if isinstance(page, dict) else []:
        payload = event.get("payload") if isinstance(event, dict) and isinstance(event.get("payload"), dict) else {}
        if payload:
            records.append(payload)
    summary = summarize_machine_time_records(records, run_dir=run_path)
    summary["source"] = "observability_store"
    return summary


def normalize_machine_time_record(record: Dict[str, Any]) -> Dict[str, Any]:
    payload = _json_safe(record if isinstance(record, dict) else {})
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("record_id", uuid.uuid4().hex)
    payload.setdefault("recorded_at", _now_iso())
    payload["status"] = str(payload.get("status") or "unknown")
    payload["tool_name"] = str(payload.get("tool_name") or "")
    payload["task_name"] = str(payload.get("task_name") or "")
    payload["resources"] = str(payload.get("resources") or "")
    payload["machine"] = str(payload.get("machine") or "")
    payload["queue_name"] = str(payload.get("queue_name") or "")
    payload["task_count"] = max(0, _to_int(payload.get("task_count"), 0))
    payload["number_node"] = max(0.0, _to_float(payload.get("number_node"), 0.0))
    payload["cpu_per_node"] = max(0.0, _to_float(payload.get("cpu_per_node"), 0.0))
    payload["gpu_per_node"] = max(0.0, _to_float(payload.get("gpu_per_node"), 0.0))
    payload["elapsed_seconds"] = max(0.0, _to_float(payload.get("elapsed_seconds"), 0.0))
    payload["dispatch_duration_seconds"] = max(0.0, _to_float(payload.get("dispatch_duration_seconds"), 0.0))
    payload["core_hours"] = _round_hours(_to_float(payload.get("core_hours"), 0.0))
    payload["node_hours"] = _round_hours(_to_float(payload.get("node_hours"), 0.0))
    payload["gpu_node_hours"] = _round_hours(_to_float(payload.get("gpu_node_hours"), 0.0))
    payload["gpu_hours"] = _round_hours(_to_float(payload.get("gpu_hours"), 0.0))
    if not payload.get("usage_basis"):
        payload["usage_basis"] = "node_hours" if payload["gpu_per_node"] > 0 else "core_hours"
    return payload


def build_machine_time_record(
    *,
    status: str,
    tool_name: str,
    task_name: str,
    work_dir_rel: str,
    work_base: str,
    resources_key: str,
    resource_cfg: Dict[str, Any],
    task_count: int,
    toolcall_id: str = "",
    result: Any = None,
    remote_context: Dict[str, Any] | None = None,
    error: str = "",
) -> Dict[str, Any]:
    status_records = _status_records_from_result(result)
    elapsed_seconds = _status_elapsed_seconds(status_records)
    elapsed_source = "remote_status"
    if elapsed_seconds <= 0:
        duration = _to_float(getattr(result, "duration_s", 0.0), 0.0)
        if status == "success" and duration > 0:
            elapsed_seconds = duration
            elapsed_source = "dispatch_duration"
        else:
            elapsed_source = "unavailable"

    number_node = max(1.0, _to_float(resource_cfg.get("number_node"), 1.0))
    cpu_per_node = max(0.0, _to_float(resource_cfg.get("cpu_per_node"), 0.0))
    gpu_per_node = max(0.0, _to_float(resource_cfg.get("gpu_per_node"), 0.0))
    elapsed_hours = elapsed_seconds / 3600.0
    node_hours = elapsed_hours * number_node
    core_hours = node_hours * cpu_per_node
    gpu_hours = node_hours * gpu_per_node
    gpu_node_hours = node_hours if gpu_per_node > 0 else 0.0
    context = dict(remote_context or {})
    if not context and result is not None:
        context = {
            key: getattr(result, key, "")
            for key in ("remote_context_id", "submission_hash", "receipt_rel")
            if getattr(result, key, "") not in (None, "", [], {})
        }
    jobs = list(getattr(result, "jobs", []) or context.get("jobs") or [])
    record = {
        "source": "dpdispatcher_remote_submission",
        "status": str(status or "unknown"),
        "tool_name": str(tool_name or ""),
        "task_name": str(task_name or ""),
        "toolcall_id": str(toolcall_id or ""),
        "work_dir_rel": str(work_dir_rel or ""),
        "work_base": str(work_base or getattr(result, "work_base", "") or ""),
        "resources": str(resources_key or ""),
        "machine": str(resource_cfg.get("machine") or ""),
        "queue_name": str(resource_cfg.get("queue_name") or ""),
        "task_count": int(task_count or 0),
        "number_node": number_node,
        "cpu_per_node": cpu_per_node,
        "gpu_per_node": gpu_per_node,
        "group_size": max(1, _to_int(resource_cfg.get("group_size"), 1)),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_source": elapsed_source,
        "dispatch_duration_seconds": _to_float(getattr(result, "duration_s", 0.0), 0.0),
        "core_hours": core_hours,
        "node_hours": node_hours,
        "gpu_node_hours": gpu_node_hours,
        "gpu_hours": gpu_hours,
        "usage_basis": "node_hours" if gpu_per_node > 0 else "core_hours",
        "remote_context_id": str(context.get("remote_context_id") or context.get("context_id") or ""),
        "submission_hash": str(context.get("submission_hash") or ""),
        "receipt_rel": str(context.get("receipt_rel") or ""),
        "jobs": _compact_jobs(jobs),
        "status_records": _compact_status_records(status_records),
    }
    if error:
        record["error"] = str(error)
    return normalize_machine_time_record(record)


def summarize_machine_time_records(records: Iterable[Dict[str, Any]], *, run_dir: Path | None = None) -> Dict[str, Any]:
    normalized = [normalize_machine_time_record(record) for record in records if isinstance(record, dict)]
    requests = len(normalized)
    success = sum(1 for record in normalized if str(record.get("status") or "") == "success")
    failed = sum(1 for record in normalized if str(record.get("status") or "") not in {"", "success"})
    total_core = sum(_to_float(record.get("core_hours"), 0.0) for record in normalized)
    total_node = sum(_to_float(record.get("node_hours"), 0.0) for record in normalized)
    total_gpu_node = sum(_to_float(record.get("gpu_node_hours"), 0.0) for record in normalized)
    total_gpu = sum(_to_float(record.get("gpu_hours"), 0.0) for record in normalized)
    task_count = sum(_to_int(record.get("task_count"), 0) for record in normalized)
    summary = {
        "generated_at": _now_iso(),
        "source": "machine_time_records_jsonl",
        "run_dir": str(Path(run_dir).expanduser().resolve()) if run_dir is not None else "",
        "requests": requests,
        "successful_requests": success,
        "failed_requests": failed,
        "task_count": task_count,
        "core_hours": _round_hours(total_core),
        "node_hours": _round_hours(total_node),
        "gpu_node_hours": _round_hours(total_gpu_node),
        "gpu_hours": _round_hours(total_gpu),
        "usage_basis_note": "CPU资源通常看core_hours；GPU资源通常看node_hours或gpu_node_hours。",
        "by_resource": _bucket_summary(normalized, "resources"),
        "by_machine": _bucket_summary(normalized, "machine"),
        "records": normalized[-100:],
    }
    return summary


def empty_machine_time_summary(run_dir: Path | None = None) -> Dict[str, Any]:
    return {
        "generated_at": _now_iso(),
        "source": "machine_time_records_jsonl",
        "run_dir": str(Path(run_dir).expanduser().resolve()) if run_dir is not None else "",
        "requests": 0,
        "successful_requests": 0,
        "failed_requests": 0,
        "task_count": 0,
        "core_hours": 0.0,
        "node_hours": 0.0,
        "gpu_node_hours": 0.0,
        "gpu_hours": 0.0,
        "usage_basis_note": "CPU资源通常看core_hours；GPU资源通常看node_hours或gpu_node_hours。",
        "by_resource": [],
        "by_machine": [],
        "records": [],
    }


def _write_summary(run_dir: Path, summary: Dict[str, Any]) -> None:
    try:
        path = machine_time_summary_path(run_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return


def _iter_record_file(path: Path) -> Iterable[Dict[str, Any]]:
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                text = line.strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    yield payload
    except Exception:
        return


def _status_records_from_result(result: Any) -> List[Dict[str, Any]]:
    if result is None:
        return []
    records = getattr(result, "status_records", None)
    if isinstance(records, list):
        return [record for record in records if isinstance(record, dict)]
    return []


def _status_elapsed_seconds(records: List[Dict[str, Any]]) -> float:
    total = 0.0
    for record in records:
        elapsed = _to_float(record.get("elapsed_seconds"), 0.0)
        if elapsed <= 0:
            start = _to_float(record.get("t_start"), 0.0)
            end = _to_float(record.get("t_end"), 0.0)
            if end > start:
                elapsed = end - start
        total += max(0.0, elapsed)
    return total


def _compact_status_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for record in records[:200]:
        out.append(
            {
                "task_index": _to_int(record.get("task_index"), 0),
                "task_work_path": str(record.get("task_work_path") or ""),
                "returncode": record.get("returncode"),
                "elapsed_seconds": max(0.0, _to_float(record.get("elapsed_seconds"), 0.0)),
                "status_missing_or_invalid": bool(record.get("status_missing_or_invalid")),
            }
        )
    return out


def _compact_jobs(jobs: List[Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for job in jobs[:200]:
        if not isinstance(job, dict):
            continue
        out.append(
            {
                "job_hash": str(job.get("job_hash") or ""),
                "job_id": str(job.get("job_id") or ""),
                "status": str(job.get("status") or ""),
            }
        )
    return out


def _bucket_summary(records: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    buckets: Dict[str, Dict[str, Any]] = {}
    for record in records:
        name = str(record.get(key) or "unknown")
        bucket = buckets.setdefault(
            name,
            {
                "name": name,
                "requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "task_count": 0,
                "core_hours": 0.0,
                "node_hours": 0.0,
                "gpu_node_hours": 0.0,
                "gpu_hours": 0.0,
            },
        )
        bucket["requests"] += 1
        if str(record.get("status") or "") == "success":
            bucket["successful_requests"] += 1
        else:
            bucket["failed_requests"] += 1
        bucket["task_count"] += _to_int(record.get("task_count"), 0)
        bucket["core_hours"] += _to_float(record.get("core_hours"), 0.0)
        bucket["node_hours"] += _to_float(record.get("node_hours"), 0.0)
        bucket["gpu_node_hours"] += _to_float(record.get("gpu_node_hours"), 0.0)
        bucket["gpu_hours"] += _to_float(record.get("gpu_hours"), 0.0)
    rows = []
    for bucket in buckets.values():
        row = dict(bucket)
        for field in ("core_hours", "node_hours", "gpu_node_hours", "gpu_hours"):
            row[field] = _round_hours(row[field])
        rows.append(row)
    return sorted(rows, key=lambda item: (-float(item.get("core_hours") or 0.0), str(item.get("name") or "")))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_timestamp(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc).timestamp()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return datetime.now(timezone.utc).timestamp()


def _round_hours(value: Any) -> float:
    return round(float(value or 0.0), 6)


def _to_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return float(default)


def _to_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(float(str(value)))
    except Exception:
        return int(default)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump(mode="json"))
        except Exception:
            pass
    return str(value)


__all__ = [
    "MACHINE_TIME_RECORDS_NAME",
    "MACHINE_TIME_SUMMARY_NAME",
    "append_machine_time_record",
    "build_machine_time_record",
    "empty_machine_time_summary",
    "load_machine_time_summary",
    "machine_time_records_path",
    "machine_time_summary_path",
    "normalize_machine_time_record",
    "summarize_machine_time_records",
]
