from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


SpecialistEntrypoint = Literal["research", "experiment", "writing", "peer_review"]


class ProposalCheckpoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_md: str = Field(..., description="Compact executable proposal in markdown.")
    todo_items: list[str] = Field(default_factory=list, description="Short flat execution checklist.")
    questions_for_human: list[str] = Field(
        default_factory=list,
        description="Only blocking clarification questions that need a human answer.",
    )


class ResearchRunCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field("", description="Subagent or source that produced this card.")
    summary: str = Field("", description="Short execution summary.")
    facts: list[str] = Field(default_factory=list, description="Minimal decision-relevant facts.")
    artifacts: list[str] = Field(default_factory=list, description="Relevant artifact paths.")


class ResearchKernel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field("", description="Current research question.")
    hypotheses: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Only the currently active 3-5 hypotheses; keep this list short.",
    )
    run_cards: list[ResearchRunCard] = Field(
        default_factory=list,
        description="Compact cards from subagent executions.",
    )
    frontier: list[str] = Field(
        default_factory=list,
        description="Next unresolved questions or actions to validate.",
    )
    conclusion_draft: str = Field("", description="Tentative current conclusion draft.")


__all__ = [
    "ProposalCheckpoint",
    "ResearchKernel",
    "ResearchRunCard",
    "SpecialistEntrypoint",
]
