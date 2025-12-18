from __future__ import annotations

"""
Lightweight tools that let the LLM persist notes into the observations stream.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from catmaster.tools.base import create_tool_output


class MemoryNoteInput(BaseModel):
    """Attach a short note to the agent's observation log."""

    note: str = Field(..., description="Freeform note to remember.")
    tags: Optional[List[str]] = Field(None, description="Optional tags for filtering later.")


def write_note(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Store a textual note so the agent can recall decisions or extracted values.

    Returns a standardized tool output; the orchestrator will add it to observations.
    """
    params = MemoryNoteInput(**payload)
    return create_tool_output(
        tool_name="write_note",
        success=True,
        data={"note": params.note, "tags": params.tags or []},
    )


__all__ = ["write_note", "MemoryNoteInput"]
