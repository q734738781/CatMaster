from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class NotifyProgressInput(BaseModel):
    """[workflow/progress] Report one user-visible scientific phase change before major delegation, long managed computation, a meaningful plan revision, or a real wait/blocker; do not use for routine reads, searches, tool calls, or heartbeat updates."""

    model_config = ConfigDict(extra="forbid")

    summary: str = Field(
        ...,
        min_length=1,
        description=(
            "Concise scientific stage update that tells the user what is changing or why the next major action matters."
        ),
    )
    next_step: str = Field(
        "",
        description=(
            "The next major scientific action when it is already known; otherwise leave empty."
        ),
    )


def notify_progress(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Persist a semantic progress update through the ordinary tool lifecycle."""

    params = NotifyProgressInput.model_validate(payload)
    data = {
        "summary": params.summary.strip(),
        "next_step": params.next_step.strip(),
    }
    return "Progress update recorded.", {
        "tool_name": "notify_progress",
        "data": data,
    }


__all__ = ["NotifyProgressInput", "notify_progress"]
