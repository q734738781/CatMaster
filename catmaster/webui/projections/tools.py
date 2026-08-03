from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Any

from .common import (
    humanize_agent_name,
    humanize_identifier,
    parse_json_object,
    redact_internal_text,
    safe_scalar_fields,
)
from .models import PublicField, PublicItem, PublicPart


_TOOL_TITLES = {
    "write_todos": "Research plan",
    "task": "Specialist task",
    "read_file": "Read file",
    "write_file": "Write file",
    "edit_file": "Edit file",
    "list_files": "List files",
    "glob": "Find files",
    "grep": "Search files",
    "web_search": "Search the web",
    "literature_search": "Search literature",
}
_PREFERRED_INPUT_KEYS = (
    "task",
    "query",
    "path",
    "file_path",
    "filename",
    "formula",
    "smiles",
    "method",
    "machine",
    "job_id",
    "temperature",
    "pressure",
    "steps",
    "count",
)
_PREFERRED_OUTPUT_KEYS = (
    "status",
    "summary",
    "message",
    "result",
    "job_id",
    "scheduler_job_id",
    "energy",
    "converged",
    "count",
)
_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_MARKDOWN_CITATION_RE = re.compile(r"\ue200cite.*?\ue201")
_CONCLUSION_HEADING_RE = re.compile(
    r"^(?:(?:provisional|practical|overall|main)\s+)?"
    r"(?:conclusion|judg(?:e)?ment|synthesis|findings?|assessment|verdict|summary|discriminating takeaway)\b",
    re.IGNORECASE,
)


def _tool_title(name: str) -> str:
    normalized = str(name or "").strip()
    if normalized in _TOOL_TITLES:
        return _TOOL_TITLES[normalized]
    title = humanize_identifier(normalized, fallback="Tool activity")
    replacements = {
        "Dpdispatcher": "Remote",
        "Vasp execute": "VASP",
        "Mace": "MACE",
        "Xtb": "xTB",
    }
    for source, target in replacements.items():
        if title.startswith(source):
            title = target + title[len(source):]
            break
    return title


def _raw_tool_payload(raw_part: Any) -> tuple[dict[str, Any], str, Any]:
    if hasattr(raw_part, "model_dump"):
        raw = raw_part.model_dump(mode="json")
    elif isinstance(raw_part, dict):
        raw = dict(raw_part)
    else:
        raw = {}
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    name = str(raw.get("tool") or meta.get("tool") or "").strip()
    input_payload = raw.get("input") if isinstance(raw.get("input"), dict) else meta.get("input")
    return raw, name, input_payload


def _structured_result_summary(payload: Any) -> str:
    if not isinstance(payload, dict):
        return "Structured results are available in details."
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    count = payload.get("count") if isinstance(payload.get("count"), (int, float)) else meta.get("count")
    rows = next(
        (
            value
            for key in ("results", "items", "papers", "records", "data")
            if isinstance((value := payload.get(key)), list)
        ),
        None,
    )
    if isinstance(count, (int, float)) and not isinstance(count, bool):
        shown = len(rows) if rows is not None else 0
        suffix = f"; {shown:,} shown" if shown else ""
        return f"Found {int(count):,} structured records{suffix}."
    if rows is not None:
        return f"Returned {len(rows):,} structured items."
    return "Structured results are available in details."


def _command_message_content(payload: Any) -> str:
    """Extract the final ToolMessage content from a serialized LangGraph Command."""

    if isinstance(payload, dict):
        update = payload.get("update") if isinstance(payload.get("update"), dict) else payload
        messages = update.get("messages") if isinstance(update, dict) else None
        if isinstance(messages, list):
            for message in reversed(messages):
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    content = str(message.get("content") or "").strip()
                    if content:
                        return content
        return ""
    if not isinstance(payload, str) or "ToolMessage(content=" not in payload:
        return ""

    # Compatibility for persisted runs created before Command acquired a
    # structured JSON projection. tokenize isolates the Python string literal;
    # literal_eval decodes it without evaluating the surrounding repr.
    marker = "ToolMessage(content="
    snippet = payload[payload.rfind(marker) + len(marker) :]
    try:
        tokens = tokenize.generate_tokens(io.StringIO(snippet).readline)
        token = next(
            candidate
            for candidate in tokens
            if candidate.type not in {tokenize.ENCODING, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}
        )
        if token.type == tokenize.STRING:
            content = ast.literal_eval(token.string)
            return content.strip() if isinstance(content, str) else ""
    except (StopIteration, SyntaxError, ValueError, tokenize.TokenError):
        return ""
    return ""


def _plain_markdown(
    value: Any,
    *,
    workspace: Path | None,
    limit: int,
) -> str:
    text = _MARKDOWN_CITATION_RE.sub("", str(value or ""))
    text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[`*_~]+", "", text)
    text = re.sub(r"(^|\s)>\s+", r"\1", text)
    text = re.sub(r"^\s*(?:[-+*]|\d+[.)])\s+", "", text)
    return redact_internal_text(text, workspace=workspace, limit=limit)


def _section_excerpt(
    lines: list[str],
    *,
    start: int,
    end: int,
    workspace: Path | None,
    limit: int,
) -> str:
    paragraph: list[str] = []
    for raw_line in lines[start:end]:
        line = raw_line.strip()
        if not line:
            if paragraph:
                if paragraph[-1].endswith(":"):
                    continue
                break
            continue
        if line.startswith("```") or _MARKDOWN_HEADING_RE.match(line):
            continue
        paragraph.append(line)
        if re.match(r"^(?:[-+*]|\d+[.)])\s+", line):
            break
    return _plain_markdown(" ".join(paragraph), workspace=workspace, limit=limit)


def _task_result_presentation(
    content: str,
    *,
    workspace: Path | None,
) -> tuple[str, str, list[PublicItem]]:
    lines = str(content or "").splitlines()
    headings: list[tuple[int, int, str]] = []
    for index, line in enumerate(lines):
        match = _MARKDOWN_HEADING_RE.match(line)
        if match:
            title = _plain_markdown(match.group(2), workspace=workspace, limit=120)
            if title:
                headings.append((index, len(match.group(1)), title))

    title = headings[0][2] if headings else "Specialist result"
    sections: list[tuple[str, str]] = []
    for heading_index, (line_index, _level, heading) in enumerate(headings[1:], start=1):
        next_index = headings[heading_index + 1][0] if heading_index + 1 < len(headings) else len(lines)
        excerpt = _section_excerpt(
            lines,
            start=line_index + 1,
            end=next_index,
            workspace=workspace,
            limit=240,
        )
        sections.append((heading, excerpt))

    summary = next(
        (excerpt for heading, excerpt in sections if excerpt and _CONCLUSION_HEADING_RE.search(heading)),
        "",
    )
    if not summary:
        start = headings[0][0] + 1 if headings else 0
        end = headings[1][0] if len(headings) > 1 else len(lines)
        summary = _section_excerpt(
            lines,
            start=start,
            end=end,
            workspace=workspace,
            limit=360,
        )
    if not summary:
        summary = _plain_markdown(content, workspace=workspace, limit=360)

    items = [
        PublicItem(label=heading, summary=excerpt)
        for heading, excerpt in sections[:5]
    ]
    if not items:
        for line in lines:
            if not re.match(r"^\s*(?:[-+*]|\d+[.)])\s+", line):
                continue
            label = _plain_markdown(line, workspace=workspace, limit=180)
            if label:
                items.append(PublicItem(label=label))
            if len(items) >= 5:
                break
    return title, summary, items


def project_todo_items(
    raw_part: Any,
    *,
    workspace: Path | None = None,
) -> list[PublicItem]:
    _raw, name, input_payload = _raw_tool_payload(raw_part)
    if name != "write_todos" or not isinstance(input_payload, dict):
        return []
    rows = input_payload.get("todos")
    if not isinstance(rows, list):
        return []
    items: list[PublicItem] = []
    for row in rows:
        if isinstance(row, str):
            label = redact_internal_text(row, workspace=workspace, limit=None)
            status = "pending"
        elif isinstance(row, dict):
            label = redact_internal_text(
                row.get("content") or row.get("task") or row.get("title"),
                workspace=workspace,
                limit=None,
            )
            status = str(row.get("status") or "pending").strip().lower()
        else:
            continue
        if label:
            items.append(PublicItem(label=label, status=status))
    return items


def project_tool_part(
    raw: dict[str, Any],
    *,
    workspace: Path | None,
    thread_id: str,
    message_id: str,
) -> PublicPart:
    raw, name, input_payload = _raw_tool_payload(raw)
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    output_payload = raw.get("output") if "output" in raw else meta.get("output")
    status = str(raw.get("status") or "updated").strip().lower()
    part_id = str(raw.get("id") or raw.get("part_id") or "")
    diagnostics_ref = f"thread-message-part:{message_id}:{part_id}"

    todos = project_todo_items(raw, workspace=workspace)
    if todos:
        complete = sum(1 for item in todos if item.status in {"done", "completed", "complete"})
        source = str(meta.get("subagent_source") or meta.get("agent_name") or "").strip()
        return PublicPart(
            id=part_id,
            type="progress",
            status=status,
            title=f"{humanize_agent_name(source)} plan" if source else "Research plan",
            summary=f"{complete} of {len(todos)} items complete.",
            items=todos,
            diagnostics_ref=diagnostics_ref,
        )

    output_object = parse_json_object(output_payload)
    fields = safe_scalar_fields(
        input_payload,
        workspace=workspace,
        preferred_keys=_PREFERRED_INPUT_KEYS,
        limit=5,
        include_remaining=False,
    )
    result_fields = safe_scalar_fields(
        output_object if output_object else output_payload,
        workspace=workspace,
        preferred_keys=_PREFERRED_OUTPUT_KEYS,
        limit=5,
    )
    existing_labels = {field.label for field in fields}
    fields.extend(field for field in result_fields if field.label not in existing_labels)
    fields = fields[:8]

    task_content = _command_message_content(output_payload) if name == "task" else ""
    if task_content and status not in {"failed", "error", "incomplete"}:
        result_title, summary, items = _task_result_presentation(
            task_content,
            workspace=workspace,
        )
        source = str(
            meta.get("subagent_source")
            or meta.get("agent_name")
            or (input_payload or {}).get("subagent_type")
            or ""
        ).strip()
        title = result_title
        if source:
            title = f"{humanize_agent_name(source)} · {result_title}"
        return PublicPart(
            id=part_id,
            type="tool",
            status=status,
            title=title,
            summary=summary,
            fields=[field for field in fields if field.label != "Type"],
            items=items,
            detail_ref=f"/api/threads/{thread_id}/messages/{message_id}/parts/{part_id}",
            diagnostics_ref=diagnostics_ref,
        )

    if status in {"running", "streaming", "started"}:
        summary = "Work is in progress."
    elif status in {"failed", "error", "incomplete"}:
        summary = "The operation did not complete. Review the error card or diagnostics reference."
    elif status in {"queued", "pending"}:
        summary = "The operation is waiting to run."
    else:
        summary = "The operation completed."
    detail = ""
    structured_text = isinstance(output_payload, str) and output_payload.lstrip().startswith(("{", "["))
    if output_object or structured_text:
        summary = _structured_result_summary(output_object)
    elif isinstance(output_payload, str):
        detail = redact_internal_text(output_payload, workspace=workspace, limit=1200)
        if detail:
            summary = detail
    source = str(meta.get("subagent_source") or meta.get("agent_name") or "").strip()
    title = _tool_title(name)
    if source:
        title = f"{humanize_agent_name(source)} · {title}"
    return PublicPart(
        id=part_id,
        type="tool",
        status=status,
        title=title,
        summary=summary,
        fields=fields,
        detail_ref=f"/api/threads/{thread_id}/messages/{message_id}/parts/{part_id}",
        diagnostics_ref=diagnostics_ref,
    )


def project_receipt_part(
    raw: dict[str, Any],
    *,
    workspace: Path | None,
    message_id: str,
) -> PublicPart:
    meta = raw.get("meta") if isinstance(raw.get("meta"), dict) else {}
    status = str(raw.get("status") or meta.get("status") or "updated").strip().lower()
    part_id = str(raw.get("id") or raw.get("part_id") or "")
    job_id = (
        meta.get("scheduler_job_id")
        or meta.get("job_id")
        or meta.get("slurm_job_id")
        or meta.get("pbs_job_id")
        or ""
    )
    fields = safe_scalar_fields(
        meta,
        workspace=workspace,
        preferred_keys=("task", "task_type", "machine", "host", "scheduler", "job_id", "scheduler_job_id", "status"),
        limit=6,
        include_remaining=False,
    )
    if job_id and not any(field.label == "Job id" for field in fields):
        fields.append(PublicField(label="Job", value=str(job_id), copy_value=str(job_id)))
    if status in {"queued", "pending", "submitted"}:
        summary = f"Job {job_id} is waiting in the scheduler." if job_id else "The remote task was submitted and is waiting."
    elif status in {"running"}:
        summary = f"Job {job_id} is running." if job_id else "The remote task is running."
    elif status in {"failed", "error"}:
        summary = f"Job {job_id} failed." if job_id else "The remote task failed."
    elif status in {"completed", "done", "success"}:
        summary = f"Job {job_id} completed." if job_id else "The remote task completed."
    else:
        summary = "Remote task status was updated."
    receipt_path = str(meta.get("receipt_rel") or "").strip()
    actions = []
    if receipt_path and not Path(receipt_path).is_absolute():
        from .models import PublicAction

        actions.append(
            PublicAction(
                id="open_receipt",
                label="Open receipt",
                kind="link",
                href=receipt_path.replace("\\", "/"),
            )
        )
    return PublicPart(
        id=part_id,
        type="receipt",
        status=status,
        title="Remote calculation",
        summary=summary,
        fields=fields,
        actions=actions,
        diagnostics_ref=f"thread-message-part:{message_id}:{part_id}",
    )
