from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SpecialistEntrypoint = Literal["research", "experiment", "writing"]


class ProposalCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_md: str = Field(..., description="Compact executable proposal in markdown.")
    todo_items: list[str] = Field(default_factory=list, description="Short flat execution checklist.")
    questions_for_human: list[str] = Field(
        default_factory=list,
        description="Only blocking clarification questions that need a human answer.",
    )


__all__ = [
    "ProposalCheckpoint",
    "SpecialistEntrypoint",
]
