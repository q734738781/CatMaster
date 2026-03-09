from __future__ import annotations

"""Lightweight note tool for keeping run-local scratch notes."""

from datetime import datetime, timezone
from typing import Dict

from pydantic import BaseModel, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

class MemoryNoteInput(BaseModel):
    """Write a short reusable note to workspace notes for later steps."""

    note: str = Field(..., description="Freeform note to remember.")
    tags: list[str] = Field(default_factory=list, description="Optional short tags.")


def write_note(payload: Dict[str, object]) -> tuple[str, dict[str, object]]:
    params = MemoryNoteInput(**payload)
    note = str(params.note or "").strip()
    tags = [str(tag).strip() for tag in params.tags if str(tag).strip()]

    if not note:
        raise CatMasterToolExecutionError(
            tool_name="write_note",
            public_message="note is empty",
            artifact={"tool_name": "write_note", "data": {"error_code": "empty_note"}},
            error_code="empty_note",
        )

    notes_path = resolve_workspace_path("notes/agent_notes.md", must_exist=False)
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    tags_line = ", ".join(tags) if tags else "-"

    block_lines = [
        f"## {timestamp}",
        f"tags: {tags_line}",
        "note:",
        note,
        "",
    ]
    block = "\n".join(block_lines)

    try:
        with notes_path.open("a", encoding="utf-8") as f:
            f.write(block)
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name="write_note",
            public_message=f"failed to persist note: {type(exc).__name__}: {exc}",
            artifact={"tool_name": "write_note", "data": {"error_code": "write_failed"}},
            error_code="write_failed",
        ) from exc

    note_rel = workspace_relpath(notes_path)
    preview = note if len(note) <= 200 else note[:197] + "..."
    content_lines = [
        "write_note completed.",
        f"note_file_rel={note_rel}",
        f"note_chars={len(note)}",
    ]
    if tags:
        content_lines.append("tags=" + ", ".join(tags))
    content_lines.append(f"note_preview={preview}")

    artifact = {
        "tool_name": "write_note",
        "data": {
            "note_file_rel": note_rel,
            "timestamp": timestamp,
            "note_chars": len(note),
            "tags": tags,
            "note_preview": preview,
        },
    }
    return "\n".join(content_lines), artifact


__all__ = ["write_note", "MemoryNoteInput"]
