from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.parse import quote

from .common import (
    display_path,
    encode_public_cursor,
    humanize_agent_name,
    humanize_identifier,
    public_activity_identity,
    redact_internal_text,
    safe_scalar_fields,
    truncate_text,
)
from .models import (
    PublicAction,
    PublicField,
    PublicFormField,
    PublicItem,
    PublicMessage,
    PublicPart,
    TruncationInfo,
)
from .errors import project_error_part
from .tools import project_receipt_part, project_todo_items, project_tool_part


_TEXT_LIMIT = 24_000
_PROGRESS_LIMIT = 12_000
_INLINE_EDIT_FIELD_LIMIT = 12
_INLINE_EDIT_VALUE_LIMIT = 12_000
_INLINE_ITEM_LIMIT = 200
_INLINE_CITATION_LIMIT = 100


def _part_content_ref(thread_id: str, message_id: str, part_id: str) -> str:
    return f"/api/threads/{thread_id}/messages/{message_id}/parts/{part_id}/content"


def _part_items_ref(thread_id: str, message_id: str, part_id: str) -> str:
    return f"/api/threads/{thread_id}/messages/{message_id}/parts/{part_id}/items"


def _interrupt_rows(raw: dict[str, Any]) -> list[tuple[dict[str, Any], set[str]]]:
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    payload = meta.get("payload") if isinstance(meta.get("payload"), dict) else {}
    interrupts = payload.get("interrupts")
    if isinstance(interrupts, list):
        rows = interrupts
    elif interrupts is not None:
        rows = [interrupts]
    elif payload.get("action_requests") or payload.get("actionRequests"):
        # Compatibility for projections written before the Agent Protocol
        # interrupt envelope was preserved in full.
        rows = [{"value": payload}]
    else:
        rows = []
    out: list[tuple[dict[str, Any], set[str]]] = []
    for row in rows:
        value = row.get("value") if isinstance(row, dict) and isinstance(row.get("value"), dict) else row
        if not isinstance(value, dict):
            continue
        requests = value.get("action_requests") or value.get("actionRequests") or []
        configs = value.get("review_configs") or value.get("reviewConfigs") or []
        allowed_by_name: dict[str, set[str]] = {}
        if isinstance(configs, list):
            for config in configs:
                if not isinstance(config, dict):
                    continue
                action_name = str(config.get("action_name") or config.get("actionName") or "")
                decisions = config.get("allowed_decisions") or config.get("allowedDecisions") or []
                if action_name:
                    allowed_by_name[action_name] = {
                        str(item).strip().lower() for item in decisions if str(item).strip()
                    }
        if not isinstance(requests, list):
            continue
        for request in requests:
            if not isinstance(request, dict):
                continue
            name = str(request.get("name") or "")
            allowed = allowed_by_name.get(name) or {"approve", "reject", "respond"}
            out.append((request, allowed))
    return out


def _form_fields(args: Any) -> tuple[list[PublicFormField], int]:
    if not isinstance(args, dict):
        return [], 0
    fields: list[PublicFormField] = []
    editable_count = 0
    for key, value in args.items():
        name = str(key)
        if name.startswith("_") or any(token in name.lower() for token in ("password", "secret", "token", "api_key")):
            continue
        if isinstance(value, bool):
            input_type = "boolean"
            text = "true" if value else "false"
        elif isinstance(value, (int, float)):
            input_type = "number"
            text = str(value)
        elif isinstance(value, str):
            input_type = "textarea" if len(value) > 160 or "\n" in value or name in {"content", "prompt"} else "text"
            text = value
        else:
            continue
        editable_count += 1
        # A partially projected default is unsafe: submitting an unchanged edit
        # would replace the complete original string with the displayed prefix.
        # Omitted fields remain byte-for-byte unchanged because the resume layer
        # merges submitted fields into the persisted action arguments.
        if len(fields) >= _INLINE_EDIT_FIELD_LIMIT or len(text) > _INLINE_EDIT_VALUE_LIMIT:
            continue
        fields.append(
            PublicFormField(
                name=name,
                label=humanize_identifier(name),
                value=text,
                input_type=input_type,
                required=False,
            )
        )
    return fields, editable_count


def project_citation_items(
    message: Any,
    *,
    workspace: Path | None = None,
) -> list[PublicItem]:
    raw = message.model_dump(mode="json") if hasattr(message, "model_dump") else dict(message or {})
    sidecar = raw.get("structured_sidecar") if isinstance(raw.get("structured_sidecar"), dict) else {}
    citations = sidecar.get("citations")
    if not isinstance(citations, list):
        return []
    items: list[PublicItem] = []
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        url = str(citation.get("url") or "").strip()
        if not url.startswith(("http://", "https://")):
            continue
        items.append(
            PublicItem(
                label=redact_internal_text(
                    citation.get("title") or url,
                    workspace=workspace,
                    limit=500,
                ),
                href=url,
            )
        )
    return items


def _project_interrupt(
    raw: dict[str, Any],
    *,
    thread_id: str,
    message_id: str,
    workspace: Path | None,
) -> PublicPart:
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    part_id = str(raw.get("id") or "")
    status = str(raw.get("status") or meta.get("status") or "pending")
    actions: list[PublicAction] = []
    action_items: list[PublicItem] = []
    interrupt_rows = _interrupt_rows(raw)
    shown_edit_fields = 0
    total_edit_fields = 0
    for index, (request, allowed) in enumerate(interrupt_rows):
        name = str(request.get("name") or "")
        args = request.get("args") if isinstance(request.get("args"), dict) else {}
        action_id = str(index)
        form_fields, editable_count = _form_fields(args)
        shown_edit_fields += len(form_fields)
        total_edit_fields += editable_count
        action_items.append(
            PublicItem(
                label=humanize_identifier(name, fallback=f"Action {index + 1}"),
                status="Waiting for review",
                summary="; ".join(
                    f"{field.label}: {field.value}"
                    for field in safe_scalar_fields(
                        args,
                        preferred_keys=("path", "command", "task"),
                        limit=3,
                        include_remaining=False,
                    )
                ),
            )
        )
        if "approve" in allowed:
            actions.append(
                PublicAction(
                    id=action_id,
                    label=f"Approve {index + 1}" if len(interrupt_rows) > 1 else "Approve",
                    kind="primary",
                    decision="approve",
                    confirmation="The agent will continue with this action.",
                )
            )
        if "edit" in allowed and form_fields:
            actions.append(
                PublicAction(
                    id=action_id,
                    label=f"Edit {index + 1}" if len(interrupt_rows) > 1 else "Edit and approve",
                    kind="secondary",
                    decision="edit",
                    confirmation="The edited values will replace the corresponding action fields.",
                    fields=form_fields,
                )
            )
        if "respond" in allowed:
            actions.append(
                PublicAction(
                    id=action_id,
                    label=f"Respond {index + 1}" if len(interrupt_rows) > 1 else "Respond",
                    decision="respond",
                    requires_reason=True,
                    fields=[
                        PublicFormField(
                            name="reason",
                            label="Response",
                            input_type="textarea",
                            required=True,
                        )
                    ],
                )
            )
        if "reject" in allowed:
            actions.append(
                PublicAction(
                    id=action_id,
                    label=f"Reject {index + 1}" if len(interrupt_rows) > 1 else "Reject",
                    kind="danger",
                    decision="reject",
                    requires_reason=True,
                    confirmation="This action will not run.",
                    fields=[
                        PublicFormField(
                            name="reason",
                            label="Reason",
                            input_type="textarea",
                            required=True,
                        )
                    ],
                )
            )
    if not action_items:
        action_items.append(PublicItem(label="Agent action", status="Waiting for review"))
    omitted_edit_fields = max(0, total_edit_fields - shown_edit_fields)
    summary = redact_internal_text(
        meta.get("body") or raw.get("text") or "The task is paused until you choose an action.",
        workspace=workspace,
        limit=700,
    )
    if omitted_edit_fields:
        summary = (
            f"{summary} Inline editing shows {shown_edit_fields} of "
            f"{total_edit_fields} editable fields; omitted fields remain unchanged."
        ).strip()
        action_items.append(
            PublicItem(
                label="Inline editing",
                status="Limited",
                summary=(
                    f"{omitted_edit_fields} field"
                    f"{'s are' if omitted_edit_fields != 1 else ' is'} too large "
                    "or numerous for the inline form and will remain unchanged."
                ),
            )
        )
    full_ref = (
        f"/api/diagnostics/threads/{thread_id}/messages/{message_id}/parts/{part_id}"
        if omitted_edit_fields
        else ""
    )
    return PublicPart(
        id=part_id,
        type="interrupt",
        status=status,
        title=redact_internal_text(
            meta.get("title") or "Your decision is required",
            workspace=workspace,
            limit=240,
        ),
        summary=summary,
        actions=actions,
        items=action_items,
        diagnostics_ref=f"thread-message-part:{message_id}:{part_id}",
        truncation=TruncationInfo(
            shown_count=shown_edit_fields,
            total_count=total_edit_fields,
            total_unknown=False,
            truncated=bool(omitted_edit_fields),
            full_content_ref=full_ref,
            unit="items",
            range_start=0,
            range_end=shown_edit_fields,
        ),
    )


def project_part(
    raw_part: Any,
    *,
    workspace: Path | None,
    thread_id: str,
    message_id: str,
    text_limit: int = _TEXT_LIMIT,
    progress_limit: int = _PROGRESS_LIMIT,
) -> PublicPart:
    if hasattr(raw_part, "model_dump"):
        raw = raw_part.model_dump(mode="json")
    elif isinstance(raw_part, dict):
        raw = dict(raw_part)
    else:
        raw = {}
    part_id = str(raw.get("id") or "part_unknown")
    part_type = str(raw.get("type") or "").strip().lower()
    status = str(raw.get("status") or "")
    diagnostics_ref = f"thread-message-part:{message_id}:{part_id}"

    if part_type == "text":
        text, truncation = truncate_text(
            raw.get("text"),
            limit=max(0, int(text_limit)),
            full_content_ref=_part_content_ref(thread_id, message_id, part_id),
        )
        if truncation.truncated:
            truncation.next_cursor = encode_public_cursor("part_content", part_id, len(text))
        return PublicPart(
            id=part_id,
            type="text",
            status=status,
            text=text,
            diagnostics_ref=diagnostics_ref,
            truncation=truncation,
        )
    if part_type == "reasoning":
        meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
        activity_group_id, activity_group_title = public_activity_identity(
            meta,
            fallback_title="CatMaster",
        )
        text, truncation = truncate_text(
            raw.get("text"),
            limit=max(0, int(progress_limit)),
            full_content_ref=_part_content_ref(thread_id, message_id, part_id),
        )
        if truncation.truncated:
            truncation.next_cursor = encode_public_cursor("part_content", part_id, len(text))
        return PublicPart(
            id=part_id,
            type="reasoning",
            status=status,
            title="Progress",
            activity_group_id=activity_group_id,
            activity_group_title=activity_group_title,
            text=text,
            diagnostics_ref=diagnostics_ref,
            truncation=truncation,
        )
    if part_type == "subagent":
        meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
        activity_group_id, activity_group_title = public_activity_identity(meta)
        task_id = str(meta.get("task_id") or meta.get("thread_id") or "").strip()
        native_status = str(meta.get("native_status") or status).strip().lower()
        endpoint = (f"/api/threads/{quote(thread_id, safe='')}/async-subagents/"
                    f"{quote(task_id, safe='')}") if task_id else ""
        actions: list[PublicAction] = [PublicAction(id="open_async_subagent", label="查看过程",
            kind="primary", endpoint=endpoint)] if endpoint else []
        if task_id and native_status in {"created", "pending", "running", "streaming"}:
            actions += [
                PublicAction(
                    id="update_async_subagent",
                    label="Steer",
                    kind="secondary",
                    endpoint=f"{endpoint}/update",
                ),
                PublicAction(
                    id="stop_async_subagent",
                    label="Stop",
                    kind="danger",
                    endpoint=f"{endpoint}/stop",
                    confirmation="Stop this async specialist run and keep its saved progress?",
                ),
            ]
        text, truncation = truncate_text(
            raw.get("text"),
            limit=max(0, int(progress_limit)),
            full_content_ref=_part_content_ref(thread_id, message_id, part_id),
        )
        if truncation.truncated:
            truncation.next_cursor = encode_public_cursor("part_content", part_id, len(text))
        return PublicPart(
            id=part_id,
            type="subagent" if task_id else "progress",
            status=status,
            on_completion=str(meta.get("on_completion") or "resume_parent"),
            task_description=str(meta.get("task_description") or ""),
            task_cost=str(meta.get("task_cost") or ""),
            capacity_state=str(meta.get("capacity_state") or ""),
            task_followup=str(meta.get("task_followup") or ""),
            task_followup_status=str(meta.get("task_followup_status") or ""),
            progress_summary=str((meta.get("research_progress") or {}).get("summary") or ""),
            progress_next_step=str((meta.get("research_progress") or {}).get("next_step") or ""),
            title=humanize_agent_name(meta.get("source"), fallback="Specialist progress"),
            activity_group_id=activity_group_id,
            activity_group_title=activity_group_title,
            text=text,
            actions=actions,
            detail_ref=endpoint,
            fields=[PublicField(label="当前动作", value=str(meta["activity_tool"]))] if meta.get("activity_tool") else [],
            diagnostics_ref=diagnostics_ref,
            truncation=truncation,
        )
    if part_type == "tool-call":
        projected = project_tool_part(
            raw,
            workspace=workspace,
            thread_id=thread_id,
            message_id=message_id,
        )
        todo_items = project_todo_items(raw, workspace=workspace)
        if todo_items:
            shown_items = todo_items[:_INLINE_ITEM_LIMIT]
            total_items = len(todo_items)
            shown_complete = sum(
                1
                for item in shown_items
                if item.status in {"done", "completed", "complete"}
            )
            item_ref = _part_items_ref(thread_id, message_id, part_id)
            projected.items = shown_items
            projected.summary = (
                f"{shown_complete} of {len(shown_items)} shown items complete."
                if total_items <= len(shown_items)
                else (
                    f"{shown_complete} of {len(shown_items)} shown items complete; "
                    f"{total_items} items total."
                )
            )
            projected.detail_ref = item_ref if total_items > len(shown_items) else projected.detail_ref
            projected.truncation = TruncationInfo(
                shown_count=len(shown_items),
                total_count=total_items,
                total_unknown=False,
                truncated=total_items > len(shown_items),
                next_cursor=(
                    encode_public_cursor("part_items", part_id, len(shown_items))
                    if total_items > len(shown_items)
                    else ""
                ),
                full_content_ref=item_ref if total_items > len(shown_items) else "",
                unit="items",
                range_start=0,
                range_end=len(shown_items),
            )
        return projected
    if part_type == "artifact":
        path = display_path(raw.get("path"), workspace=workspace)
        return PublicPart(
            id=part_id,
            type="artifact",
            status=status,
            title=redact_internal_text(
                raw.get("title") or Path(path).name or "Artifact",
                workspace=workspace,
                limit=240,
            ),
            summary=redact_internal_text(raw.get("summary"), workspace=workspace, limit=360),
            artifact_id=str(raw.get("artifact_id") or ""),
            renderer=str(raw.get("renderer") or "file"),
            path=path,
            actions=[
                PublicAction(
                    id="open_artifact",
                    label="Open",
                    kind="primary",
                )
            ],
            diagnostics_ref=diagnostics_ref,
        )
    if part_type == "receipt":
        return project_receipt_part(raw, workspace=workspace, message_id=message_id)
    if part_type == "interrupt":
        return _project_interrupt(
            raw,
            thread_id=thread_id,
            message_id=message_id,
            workspace=workspace,
        )
    if part_type == "error":
        meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
        return project_error_part(
            part_id=part_id,
            summary=(
                raw.get("text")
                or raw.get("summary")
                or meta.get("error")
                or meta.get("summary")
            ),
            error_code=meta.get("error_code") or meta.get("code"),
            retry_safe=bool(meta.get("retry_safe")),
            checkpoint_resumable=bool(
                meta.get("checkpoint_resume_available")
            ),
            diagnostics_ref=diagnostics_ref,
            workspace=workspace,
        )
    if part_type == "trace":
        meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
        if meta.get("kind") == "context_compaction":
            return PublicPart(
                id=part_id, type="progress", status=status, title="上下文压缩",
                text=str(raw.get("text") or ""), diagnostics_ref=diagnostics_ref,
            )
        return PublicPart(
            id=part_id,
            type="progress",
            status=status,
            title="Execution update",
            summary=redact_internal_text(raw.get("text"), workspace=workspace, limit=500)
            or "An internal execution update was recorded.",
            diagnostics_ref=diagnostics_ref,
        )
    return PublicPart(
        id=part_id,
        type="unknown",
        status=status or "unsupported",
        title="This activity cannot be displayed yet",
        summary="CatMaster kept the underlying record for developer diagnostics without exposing the raw payload here.",
        diagnostics_ref=diagnostics_ref,
    )


def project_message(
    message: Any,
    *,
    workspace: Path | None = None,
    max_parts: int = 20,
    text_budget: int = 64_000,
) -> PublicMessage:
    raw = message.model_dump(mode="json") if hasattr(message, "model_dump") else dict(message or {})
    thread_id = str(raw.get("thread_id") or "")
    message_id = str(raw.get("id") or "")
    raw_parts = list(raw.get("parts") or [])
    visible_raw_parts = raw_parts[: max(1, int(max_parts))]
    remaining_text_budget = max(0, int(text_budget))
    parts: list[PublicPart] = []
    for raw_part in visible_raw_parts:
        raw_part_payload = (
            raw_part.model_dump(mode="json")
            if hasattr(raw_part, "model_dump")
            else raw_part if isinstance(raw_part, dict) else {}
        )
        part_type = str(raw_part_payload.get("type") or "")
        requested_limit = (
            _TEXT_LIMIT
            if part_type == "text"
            else _PROGRESS_LIMIT
            if part_type in {"reasoning", "subagent"}
            else 0
        )
        part_budget = min(requested_limit, remaining_text_budget) if requested_limit else 0
        projected_part = project_part(
            raw_part,
            workspace=workspace,
            thread_id=thread_id,
            message_id=message_id,
            text_limit=part_budget if part_type == "text" else _TEXT_LIMIT,
            progress_limit=part_budget if part_type in {"reasoning", "subagent"} else _PROGRESS_LIMIT,
        )
        parts.append(projected_part)
        if requested_limit:
            remaining_text_budget = max(0, remaining_text_budget - len(projected_part.text))
    message_status = str(raw.get("status") or "completed")
    if message_status == "failed" and not any(part.type == "error" for part in parts):
        meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
        parts.append(
            project_error_part(
                part_id=f"part_error_{message_id}",
                summary=meta.get("error") or meta.get("failure"),
                error_code=meta.get("error_code"),
                retry_safe=bool(meta.get("retry_safe")),
                checkpoint_resumable=bool(
                    meta.get("checkpoint_resume_available")
                ),
                diagnostics_ref=f"thread-message:{message_id}",
                workspace=workspace,
            )
        )
    if message_status == "interrupted" and not any(part.type == "error" for part in parts):
        meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
        interruption_reason = redact_internal_text(
            meta.get("interruption_reason"),
            workspace=workspace,
            limit=700,
        )
        if interruption_reason:
            parts.append(
                PublicPart(
                    id=f"part_interrupted_{message_id}",
                    type="error",
                    status="interrupted",
                    title="Task interrupted",
                    summary=interruption_reason,
                    fields=[PublicField(label="Task state", value="Interrupted")],
                    diagnostics_ref=f"thread-message:{message_id}",
                )
            )
    citation_items = project_citation_items(raw, workspace=workspace)
    if citation_items:
        shown_citations = citation_items[:_INLINE_CITATION_LIMIT]
        citation_part_id = f"part_citations_{message_id}"
        citation_ref = _part_items_ref(thread_id, message_id, citation_part_id)
        citation_total = len(citation_items)
        parts.append(
            PublicPart(
                id=citation_part_id,
                type="citations",
                status="completed",
                title="Sources",
                summary=(
                    f"{citation_total} source{'s' if citation_total != 1 else ''}"
                    if citation_total <= len(shown_citations)
                    else f"Showing {len(shown_citations)} of {citation_total} sources"
                ),
                items=shown_citations,
                detail_ref=citation_ref if citation_total > len(shown_citations) else "",
                diagnostics_ref=f"thread-message:{message_id}",
                truncation=TruncationInfo(
                    shown_count=len(shown_citations),
                    total_count=citation_total,
                    total_unknown=False,
                    truncated=citation_total > len(shown_citations),
                    next_cursor=(
                        encode_public_cursor(
                            "part_items",
                            citation_part_id,
                            len(shown_citations),
                        )
                        if citation_total > len(shown_citations)
                        else ""
                    ),
                    full_content_ref=(
                        citation_ref
                        if citation_total > len(shown_citations)
                        else ""
                    ),
                    unit="items",
                    range_start=0,
                    range_end=len(shown_citations),
                ),
            )
        )
    role = str(raw.get("role") or "assistant").lower()
    if role not in {"user", "assistant", "system", "tool"}:
        role = "assistant"
    return PublicMessage(
        id=message_id,
        origin=str((raw.get("meta") or {}).get("origin") or ""),
        notification_title=str((raw.get("meta") or {}).get("notification_title") or ""),
        source=str((raw.get("meta") or {}).get("source") or "").removesuffix("_agent").replace("_", " ").title(),
        role=role,
        status=message_status,
        created_at=float(raw.get("created_at") or 0.0),
        updated_at=float(raw.get("updated_at") or 0.0),
        parts=parts,
        parts_page=TruncationInfo(
            shown_count=len(visible_raw_parts),
            total_count=len(raw_parts),
            total_unknown=False,
            truncated=len(raw_parts) > len(visible_raw_parts),
            next_cursor=(
                encode_public_cursor(
                    "message_parts",
                    message_id,
                    str(raw_part_payload.get("id") or ""),
                )
                if len(raw_parts) > len(visible_raw_parts) and visible_raw_parts
                else ""
            ),
            full_content_ref=(
                f"/api/threads/{thread_id}/messages/{message_id}/parts"
                if len(raw_parts) > len(visible_raw_parts)
                else ""
            ),
            unit="items",
        ),
    )


def project_messages(messages: list[Any], *, workspace: Path | None = None) -> list[PublicMessage]:
    count = max(1, len(messages))
    per_message_budget = min(64_000, max(4_000, 1_200_000 // count))
    return [
        project_message(
            message,
            workspace=workspace,
            text_budget=per_message_budget,
        )
        for message in messages
    ]


def project_current_todo_parts(
    messages: list[Any],
    *,
    workspace: Path | None = None,
) -> list[PublicPart]:
    """Project the latest persisted plan for each agent in the current turn."""

    rows = [
        message.model_dump(mode="json")
        if hasattr(message, "model_dump")
        else message if isinstance(message, dict) else {}
        for message in messages
    ]
    latest_user = max(
        (
            index
            for index, message in enumerate(rows)
            if str(message.get("role") or "").lower() == "user"
        ),
        default=-1,
    )
    latest_by_source: dict[str, tuple[int, PublicPart]] = {}
    order = 0
    for message in rows[latest_user + 1 :]:
        message_id = str(message.get("id") or "")
        thread_id = str(message.get("thread_id") or "")
        for raw_part in list(message.get("parts") or []):
            raw = (
                raw_part.model_dump(mode="json")
                if hasattr(raw_part, "model_dump")
                else raw_part if isinstance(raw_part, dict) else {}
            )
            if not project_todo_items(raw, workspace=workspace):
                continue
            projected = project_tool_part(
                raw,
                workspace=workspace,
                thread_id=thread_id,
                message_id=message_id,
            )
            source_key = str(projected.title or "Research plan").casefold()
            latest_by_source[source_key] = (order, projected)
            order += 1
    projected_parts = [
        projected
        for _order, projected in sorted(
            latest_by_source.values(),
            key=lambda item: item[0],
            reverse=True,
        )
    ]
    # A reply ending does not complete its plan or independently running child.
    return projected_parts


def project_current_active_parts(
    messages: list[Any],
    *,
    workspace: Path | None = None,
) -> list[PublicPart]:
    """Project current-turn tools; async activity comes from LocalThreadService."""

    rows = [
        message.model_dump(mode="json")
        if hasattr(message, "model_dump")
        else message if isinstance(message, dict) else {}
        for message in messages
    ]
    latest_user = max(
        (
            index
            for index, message in enumerate(rows)
            if str(message.get("role") or "").lower() == "user"
        ),
        default=-1,
    )
    active_statuses = {"created", "streaming", "running", "queued", "pending", "started"}
    projected_parts: list[tuple[int, PublicPart]] = []
    order = 0
    for index, message in enumerate(rows):
        message_id = str(message.get("id") or "")
        thread_id = str(message.get("thread_id") or "")
        for raw_part in list(message.get("parts") or []):
            raw = (
                raw_part.model_dump(mode="json")
                if hasattr(raw_part, "model_dump")
                else raw_part if isinstance(raw_part, dict) else {}
            )
            raw_type = str(raw.get("type") or "").lower()
            if index <= latest_user:
                continue
            if raw_type != "tool-call":
                continue
            if str(raw.get("status") or "").lower() not in active_statuses:
                continue
            projected = project_tool_part(
                raw,
                workspace=workspace,
                thread_id=thread_id,
                message_id=message_id,
            )
            if projected.type not in {"tool", "progress", "subagent"}:
                continue
            projected_parts.append((order, projected))
            order += 1
    return [
        part
        for _order, part in sorted(
            projected_parts,
            key=lambda item: (
                item[1].started_at <= 0,
                item[1].started_at if item[1].started_at > 0 else 0,
                item[0],
            ),
        )
    ]
