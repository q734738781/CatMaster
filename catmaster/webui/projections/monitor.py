from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import (
    decode_public_cursor,
    display_path,
    encode_public_cursor,
    humanize_identifier,
    redact_internal_text,
    safe_scalar_fields,
)
from .models import TruncationInfo


def _record(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        return [{"name": key, "count": item} for key, item in value.items()]
    return []


def _nonnegative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _model_usage_rows(value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in _rows(value):
        label = str(item.get("name") or "").strip()
        if not label:
            continue
        input_tokens = _nonnegative_int(item.get("input_tokens"))
        cached_tokens = _nonnegative_int(
            item.get("input_cache_read_tokens", item.get("input_cached_tokens"))
        )
        cache_write_tokens = _nonnegative_int(item.get("input_cache_write_tokens"))
        if item.get("input_uncached_tokens") is None:
            uncached_tokens = max(
                0,
                input_tokens - cached_tokens - cache_write_tokens,
            )
        else:
            uncached_tokens = _nonnegative_int(item.get("input_uncached_tokens"))
        output_tokens = _nonnegative_int(item.get("output_tokens"))
        total_tokens = _nonnegative_int(item.get("total_tokens"))
        if not total_tokens:
            total_tokens = input_tokens + output_tokens
        rows.append(
            {
                "model_label": label[:240],
                "calls": _nonnegative_int(item.get("calls")),
                "input_uncached_tokens": uncached_tokens,
                "input_cached_tokens": cached_tokens,
                "input_cache_write_tokens": cache_write_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        )
    return rows


def _collection_page(
    *,
    shown_count: int,
    total_count: int,
    full_content_ref: str = "",
) -> dict[str, Any]:
    truncated = total_count > shown_count
    return TruncationInfo(
        shown_count=shown_count,
        total_count=total_count,
        total_unknown=False,
        truncated=truncated,
        full_content_ref=full_content_ref if truncated else "",
        unit="items",
        range_start=0,
        range_end=shown_count,
    ).model_dump(mode="json")


def _monitor_event(event: dict[str, Any], *, workspace: Path | None) -> dict[str, Any] | None:
    name = str(event.get("name") or event.get("event") or "")
    payload = _record(event.get("payload") or event.get("data"))
    status = str(payload.get("status") or event.get("status") or "").strip().lower()
    title = ""
    summary = ""
    kind = "progress"
    fields: list[dict[str, Any]] = []

    if name in {"RUN_START", "RUN_INIT_DONE"}:
        title, status = "Task started", "running"
    elif name in {"RUN_END"}:
        failed = status in {"error", "failed", "failure"}
        title, status = ("Task failed", "failed") if failed else ("Task completed", "completed")
        summary = redact_internal_text(
            payload.get("summary") or payload.get("error"),
            workspace=workspace,
            limit=360,
        )
    elif name in {"RUN_PAUSED", "RUN_WAITING_INPUT", "PROPOSAL_REVIEW_WAIT_INPUT"}:
        title, status = "Waiting for your decision", "waiting"
    elif name in {"RUN_INPUT_RECEIVED"}:
        title, status = "Decision received", "running"
    elif name in {"TASK_START"}:
        title, status = "Research step started", "running"
        summary = redact_internal_text(payload.get("goal"), workspace=workspace, limit=360)
    elif name in {"TASK_SUMMARY", "TASK_END"}:
        failed = str(payload.get("outcome") or "").lower() in {"failure", "failed", "error"}
        title, status = ("Research step failed", "failed") if failed else ("Research step completed", "completed")
        summary = redact_internal_text(
            payload.get("summary_snippet") or payload.get("summary"),
            workspace=workspace,
            limit=420,
        )
    elif name in {"TOOL_CALL_START", "TOOL_CALL_END", "TOOL_VALIDATE_FAILED", "TOOL_CALL_INTERRUPTED"}:
        tool = str(payload.get("tool") or payload.get("tool_name") or event.get("tool") or "")
        verb = "Running" if name == "TOOL_CALL_START" else "Completed"
        if name in {"TOOL_VALIDATE_FAILED"}:
            verb, status = "Could not start", "failed"
        elif name in {"TOOL_CALL_INTERRUPTED"}:
            verb, status = "Paused", "waiting"
        elif status in {"error", "failed", "failure"}:
            verb, status = "Failed", "failed"
        elif name == "TOOL_CALL_END":
            status = "completed"
        else:
            status = "running"
        title = f"{verb}: {humanize_identifier(tool, fallback='operation')}"
        summary = redact_internal_text(
            payload.get("highlights") or payload.get("status_message") or payload.get("error"),
            workspace=workspace,
            limit=420,
        )
        fields = [
            field.model_dump(mode="json")
            for field in safe_scalar_fields(
                payload,
                workspace=workspace,
                preferred_keys=("job_id", "scheduler_job_id", "energy", "converged", "count"),
                limit=5,
                include_remaining=False,
            )
        ]
        kind = "tool"
    elif name in {"USAGE_UPDATE", "usage.updated"}:
        title, status, kind = "Usage updated", "updated", "usage"
    elif name in {"MACHINE_TIME_RECORD"}:
        title, status, kind = "Compute usage recorded", status or "completed", "compute"
        fields = [
            field.model_dump(mode="json")
            for field in safe_scalar_fields(
                payload,
                preferred_keys=("machine", "core_hours", "node_hours", "task_count"),
                limit=4,
                include_remaining=False,
            )
        ]
    elif name in {"INTERRUPT_REQUESTED", "INTERRUPT_ACKED", "interrupt.created"}:
        title, status, kind = "Waiting for your decision", "waiting", "review"
    elif name in {"LLM_ERROR"}:
        title, status, kind = "Model request failed", "failed", "error"
        summary = redact_internal_text(payload.get("error"), workspace=workspace, limit=420)
    elif name in {
        "LLM_CALL_START",
        "LLM_CALL_END",
        "LLM_RAW_REQUEST",
        "LLM_RAW_RESPONSE",
        "TOOL_RAW_INPUT",
        "TOOL_RAW_OUTPUT",
        "RUN_STATE_CHANGE",
    }:
        return None
    else:
        readable = redact_internal_text(
            payload.get("summary") or payload.get("text_preview"),
            workspace=workspace,
            limit=360,
        )
        if not readable:
            return None
        title = "Execution update"
        summary = readable
        status = status or "updated"

    return {
        "id": str(event.get("id") or event.get("seq") or ""),
        "kind": kind,
        "title": title,
        "summary": summary,
        "status": status or "updated",
        "timestamp": float(event.get("ts") or event.get("created_at") or 0.0),
        "fields": fields,
        "diagnostics_ref": f"run-event:{event.get('id') or event.get('seq') or ''}",
    }


def project_monitor_snapshot(
    payload: dict[str, Any],
    *,
    workspace: Path | None,
    diagnostics_available: bool,
    timeline_cursor: str = "",
    timeline_limit: int = 200,
    timeline_identity: str = "",
    timeline_ref: str = "",
    details_ref: str = "",
) -> dict[str, Any]:
    metrics = _record(payload.get("metrics"))
    usage = _record(payload.get("usage_summary"))
    machine = _record(payload.get("machine_time_summary"))
    live = _record(payload.get("live_state"))
    llm = _record(live.get("llm"))
    progress = _record(live.get("progress"))
    usage_by_model = _model_usage_rows(usage.get("by_model"))

    timeline: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for event in list(payload.get("events") or []):
        if not isinstance(event, dict):
            continue
        item = _monitor_event(event, workspace=workspace)
        if item is None:
            continue
        fingerprint = (item["title"], item["status"], item["summary"])
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        timeline.append(item)

    all_todo_rows = []
    source_todos = live.get("todo_rows") if isinstance(live.get("todo_rows"), list) else payload.get("todo_items")
    for item in list(source_todos or []):
        if isinstance(item, dict):
            label = redact_internal_text(
                item.get("content") or item.get("task"),
                workspace=workspace,
                limit=300,
            )
            status = str(item.get("status") or "pending")
        else:
            label = redact_internal_text(item, workspace=workspace, limit=300)
            status = "pending"
        if label:
            all_todo_rows.append({"label": label, "status": status})
    todo_rows = all_todo_rows[:200]

    all_tools = []
    for row in list(live.get("recent_toolcalls") or []):
        if not isinstance(row, dict):
            continue
        all_tools.append(
            {
                "title": humanize_identifier(row.get("tool"), fallback="Operation"),
                "status": str(row.get("status") or "updated"),
                "summary": redact_internal_text(
                    row.get("highlights") or row.get("status_message"),
                    workspace=workspace,
                    limit=300,
                ),
            }
        )
    tools = all_tools[-30:]

    all_agents = []
    for row in _rows(live.get("agents")):
        all_agents.append(
            {
                "title": humanize_identifier(row.get("name"), fallback="Specialist"),
                "status": str(row.get("status") or "updated"),
                "summary": redact_internal_text(
                    row.get("current_task") or row.get("summary"),
                    workspace=workspace,
                    limit=300,
                ),
            }
        )
    agents = all_agents[:30]

    all_artifact_rows = []
    for item in list(payload.get("_artifacts") or []):
        if not isinstance(item, dict):
            continue
        path = display_path(item.get("path"), workspace=workspace)
        all_artifact_rows.append(
            {
                "title": redact_internal_text(
                    item.get("title") or Path(path).name or "Artifact",
                    workspace=workspace,
                    limit=240,
                ),
                "summary": redact_internal_text(item.get("summary"), workspace=workspace, limit=300),
                "path": path,
                "renderer": str(item.get("renderer") or item.get("kind") or "file"),
            }
        )
    artifact_rows = all_artifact_rows[:100]

    source_events = list(payload.get("events") or [])
    raw_total = int(_record(payload.get("raw_logs")).get("total_events") or len(source_events))
    cursor_identity = timeline_identity or str(payload.get("selected_run") or "monitor")
    timeline_end = len(timeline)
    if timeline_cursor:
        position = decode_public_cursor(
            timeline_cursor,
            kind="monitor_timeline",
            identity=cursor_identity,
        )
        if not isinstance(position, int) or position < 0 or position > len(timeline):
            raise ValueError("Monitor timeline cursor is invalid or stale.")
        timeline_end = position
    capped_timeline_limit = min(500, max(1, int(timeline_limit or 200)))
    timeline_start = max(0, timeline_end - capped_timeline_limit)
    visible_timeline = timeline[timeline_start:timeline_end]
    source_truncated = raw_total > len(source_events)
    timeline_has_more = timeline_start > 0
    timeline_truncated = timeline_has_more or source_truncated
    return {
        "has_run": bool(payload.get("selected_run")),
        "overview": {
            "status": str(payload.get("run_status") or live.get("status") or "idle"),
            "status_text": redact_internal_text(
                payload.get("run_status_text"),
                workspace=workspace,
                limit=360,
            ),
            "phase": humanize_identifier(live.get("current_phase"), fallback="Ready"),
            "current_task": redact_internal_text(
                live.get("current_task_goal"),
                workspace=workspace,
                limit=360,
            ),
            "model": str(llm.get("model") or ""),
            "duration_sec": float(metrics.get("duration_sec") or 0.0),
            "llm_calls": int(metrics.get("llm_calls") or usage.get("calls") or 0),
            "tool_calls": int(metrics.get("tool_calls") or 0),
            "tool_failures": int(metrics.get("tool_failures") or 0),
            "total_tokens": int(usage.get("total_tokens") or 0),
            "input_tokens": int(usage.get("input_tokens") or 0),
            "input_uncached_tokens": int(usage.get("input_uncached_tokens") or 0),
            "input_cached_tokens": int(usage.get("input_cached_tokens") or 0),
            "output_tokens": int(usage.get("output_tokens") or 0),
            "cost_usd": float(usage.get("cost_usd") or 0.0),
            "core_hours": float(machine.get("core_hours") or 0.0),
            "node_hours": float(machine.get("node_hours") or 0.0),
        },
        "usage": {
            "by_model": usage_by_model,
        },
        "live": {
            "progress": {
                "completed": int(progress.get("completed") or 0),
                "pending": int(progress.get("pending") or 0),
                "failed": int(progress.get("failed") or 0),
                "total": int(progress.get("total") or 0),
            },
            "todos": todo_rows,
            "todos_page": _collection_page(
                shown_count=len(todo_rows),
                total_count=len(all_todo_rows),
                full_content_ref=details_ref,
            ),
            "tools": tools,
            "tools_page": _collection_page(
                shown_count=len(tools),
                total_count=len(all_tools),
                full_content_ref=details_ref,
            ),
            "agents": agents,
            "agents_page": _collection_page(
                shown_count=len(agents),
                total_count=len(all_agents),
                full_content_ref=details_ref,
            ),
        },
        "artifacts": artifact_rows,
        "artifacts_page": _collection_page(
            shown_count=len(artifact_rows),
            total_count=len(all_artifact_rows),
            full_content_ref=details_ref,
        ),
        "timeline": visible_timeline,
        "page": TruncationInfo(
            shown_count=len(visible_timeline),
            total_count=len(timeline),
            total_unknown=source_truncated,
            truncated=timeline_truncated,
            next_cursor=(
                encode_public_cursor(
                    "monitor_timeline",
                    cursor_identity,
                    timeline_start,
                )
                if timeline_has_more
                else ""
            ),
            full_content_ref=timeline_ref if timeline_truncated else "",
            unit="items",
            range_start=timeline_start,
            range_end=timeline_end,
        ).model_dump(mode="json"),
        "developer_diagnostics_available": bool(diagnostics_available),
    }


__all__ = ["project_monitor_snapshot"]
