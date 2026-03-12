from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from catmaster.runtime.writing.models import SectionDraftModel, SectionReviewModel, WritingPlanModel


class WritingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request: str = Field(...)
    session_context_text: str = Field("")
    chat_session_id: str | None = Field(None)
    entry_context_tokens_estimate: int = Field(0, ge=0)
    source_campaign_id: str | None = Field(None)
    writing_mode: Literal["internal_report", "paper_outline", "section_draft", "full_draft"] = Field("internal_report")
    output_format: Literal["md", "tex"] = Field("tex")
    target_section: str | None = Field(None)


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
    final_output_path: str | None = Field(None)
    final_latex_path: str | None = Field(None)


__all__ = [
    "WritingFinalizeOutput",
    "SectionDraftOutput",
    "SectionReviewOutput",
    "WritingPlanOutput",
    "WritingRequest",
]
