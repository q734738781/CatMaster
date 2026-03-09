from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from catmaster.runtime.writing.models import SectionDraftModel, SectionReviewModel, WritingPlanModel


class WritingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str = Field(...)
    source_campaign_id: str | None = Field(None)


class WritingPlanOutput(WritingPlanModel):
    model_config = ConfigDict(extra="forbid")


class SectionDraftOutput(SectionDraftModel):
    model_config = ConfigDict(extra="forbid")


class SectionReviewOutput(SectionReviewModel):
    model_config = ConfigDict(extra="forbid")


class WritingFinalizeOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(...)
    compile_notes: list[str] = Field(default_factory=list)
    final_latex_path: str | None = Field(None)


__all__ = [
    "WritingFinalizeOutput",
    "SectionDraftOutput",
    "SectionReviewOutput",
    "WritingPlanOutput",
    "WritingRequest",
]
