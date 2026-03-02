from __future__ import annotations

"""Deprecated memory-note tool stubs. Needs new implementation."""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryNoteInput(BaseModel):
    """Attach a short note to the agent's observation log."""

    note: str = Field(..., description="Freeform note to remember.")
    tags: Optional[List[str]] = Field(None, description="Optional tags for filtering later.")


def write_note(payload: Dict[str, object]) -> tuple[str, dict[str, object]]:
    _ = MemoryNoteInput(**payload)
    raise NotImplementedError("write_note is deprecated and needs new implementation.")


__all__ = ["write_note", "MemoryNoteInput"]
