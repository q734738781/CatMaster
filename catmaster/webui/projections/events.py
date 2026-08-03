from __future__ import annotations

from pathlib import Path
from typing import Any

from catmaster.webui.thread_models import ThreadEventEnvelope

from .common import redact_internal_text, safe_scalar_fields
from .errors import project_error_part
from .messages import project_current_todo_parts, project_message, project_part
from .models import PublicEvent, PublicEventData, PublicField, PublicPart


def _event_diagnostics_ref(event: ThreadEventEnvelope) -> str:
    return f"thread-event:{event.thread_id}:{event.seq}"


def project_event(
    event: ThreadEventEnvelope,
    *,
    workspace: Path | None = None,
) -> PublicEvent:
    data = dict(event.data or {})
    name = str(event.event or "")
    public_name = name
    projected = PublicEventData(
        message_id=str(data.get("message_id") or event.message_id or ""),
        part_id=str(data.get("part_id") or ""),
        status=str(data.get("status") or event.status or ""),
        diagnostics_ref=_event_diagnostics_ref(event),
    )

    if name == "message.created":
        raw_message = data.get("message")
        if isinstance(raw_message, dict):
            projected.message = project_message(raw_message, workspace=workspace)
    elif name in {"message.delta", "reasoning.delta", "subagent.delta"}:
        projected.delta = str(data.get("delta") or "")
    elif name in {"message.part.created", "subagent.started"}:
        raw_part = data.get("part")
        if isinstance(raw_part, dict):
            projected.part = project_part(
                raw_part,
                workspace=workspace,
                thread_id=event.thread_id,
                message_id=str(event.message_id or data.get("message_id") or ""),
            )
    elif name == "message.completed":
        projected.status = "completed"
        raw_message = data.get("message")
        if isinstance(raw_message, dict):
            projected.message = project_message(raw_message, workspace=workspace)
            projected.todo_parts = project_current_todo_parts(
                [raw_message],
                workspace=workspace,
            )
        public_name = "message.completed"
    elif name in {"message.failed", "error"}:
        public_name = "run.failed"
        error_part = project_error_part(
            part_id=str(data.get("part_id") or f"part_error_{event.seq}"),
            summary=data.get("error") or data.get("summary"),
            error_code=data.get("error_code") or data.get("code"),
            retry_safe=bool(data.get("retry_safe")),
            diagnostics_ref=_event_diagnostics_ref(event),
            workspace=workspace,
        )
        projected.status = "failed"
        projected.title = error_part.title
        projected.summary = error_part.summary
        projected.fields = error_part.fields
        projected.actions = error_part.actions
    elif name in {"tool_call.started", "tool_call.delta", "tool_call.completed", "tool_call.failed"}:
        status = {
            "tool_call.started": "running",
            "tool_call.delta": "running",
            "tool_call.completed": "completed",
            "tool_call.failed": "failed",
        }[name]
        raw_part = {
            "id": str(data.get("part_id") or f"part_tool_{event.seq}"),
            "type": "tool-call",
            "status": status,
            "meta": {
                "tool": data.get("tool"),
                "input": data.get("input") if isinstance(data.get("input"), dict) else {},
                "output": data.get("output"),
                "agent_name": data.get("agent_name"),
                "subagent_source": data.get("subagent_source"),
                "agent_run_id": data.get("agent_run_id"),
                "stream_namespace": data.get("stream_namespace"),
            },
        }
        projected.part = project_part(
            raw_part,
            workspace=workspace,
            thread_id=event.thread_id,
            message_id=str(event.message_id or data.get("message_id") or ""),
        )
        public_name = "activity.updated"
    elif name in {"artifact.created", "artifact.updated"}:
        raw_part = {
            **data,
            "id": str(data.get("part_id") or f"part_{data.get('artifact_id') or event.seq}"),
            "type": "artifact",
            "status": event.status or "completed",
        }
        projected.part = project_part(
            raw_part,
            workspace=workspace,
            thread_id=event.thread_id,
            message_id=str(event.message_id or data.get("message_id") or ""),
        )
        public_name = "activity.updated"
    elif name == "task_receipt.updated":
        receipt = data.get("receipt") if isinstance(data.get("receipt"), dict) else {}
        raw_part = {
            "id": str(data.get("part_id") or f"part_receipt_{event.seq}"),
            "type": "receipt",
            "status": event.status or receipt.get("status") or "updated",
            "meta": receipt,
        }
        projected.part = project_part(
            raw_part,
            workspace=workspace,
            thread_id=event.thread_id,
            message_id=str(event.message_id or data.get("message_id") or ""),
        )
        public_name = "activity.updated"
    elif name == "interrupt.created":
        raw_part = {
            "id": str(data.get("part_id") or f"part_interrupt_{event.seq}"),
            "type": "interrupt",
            "status": "pending",
            "text": data.get("body"),
            "meta": data,
        }
        projected.part = project_part(
            raw_part,
            workspace=workspace,
            thread_id=event.thread_id,
            message_id=str(event.message_id or data.get("message_id") or ""),
        )
        public_name = "activity.updated"
    elif name == "interrupt.resolved":
        projected.status = "resolved"
    elif name == "usage.updated":
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else data
        projected.usage = safe_scalar_fields(
            usage,
            preferred_keys=("total_tokens", "input_tokens", "output_tokens", "cost_usd"),
            limit=6,
        )
    elif name in {"thread.status", "thread.updated", "thread.created"}:
        projected.status = str(data.get("status") or event.status or "")
        thread_payload = data.get("thread") if isinstance(data.get("thread"), dict) else {}
        if not projected.message_id:
            projected.message_id = str(thread_payload.get("active_message_id") or "")
        if projected.status in {"error", "failed", "failure"}:
            public_name = "run.failed"
            error_part = project_error_part(
                part_id=f"part_error_{event.seq}",
                summary=data.get("error") or thread_payload.get("error"),
                error_code=data.get("error_code"),
                retry_safe=bool(data.get("retry_safe")),
                diagnostics_ref=_event_diagnostics_ref(event),
                workspace=workspace,
            )
            projected.status = "failed"
            projected.title = error_part.title
            projected.summary = error_part.summary
            projected.fields = error_part.fields
            projected.actions = error_part.actions
        else:
            projected.title = {
                "running": "Task started",
                "interrupted": "Waiting for your decision",
                "stopping": "Stopping task",
                "stopped": "Task stopped",
                "completed": "Task completed",
                "idle": "Task ready",
            }.get(projected.status, "Task status updated")
            projected.summary = projected.title
    elif name in {"multimodal.prepared"}:
        public_name = "activity.updated"
        projected.title = "Attachments prepared"
        projected.summary = "The attached scientific material is ready for the agent."
        projected.fields = safe_scalar_fields(data, preferred_keys=("image_count", "file_count"), limit=4)
    else:
        public_name = "activity.updated"
        projected.status = str(event.status or "updated")
        projected.title = "Execution update"
        projected.summary = (
            redact_internal_text(
                data.get("summary") or data.get("text"),
                workspace=workspace,
                limit=400,
            )
            if isinstance(data, dict)
            else ""
        ) or "A runtime update was recorded. Raw data remains available only in developer diagnostics."

    return PublicEvent(
        seq=int(event.seq),
        event=public_name,
        thread_id=event.thread_id,
        message_id=projected.message_id or event.message_id,
        status=projected.status or event.status,
        created_at=float(event.created_at),
        data=projected,
    )
