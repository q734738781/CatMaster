from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FigureRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    figure_id: str = Field(...)
    title: str = Field(...)
    purpose: str = Field(...)
    source_refs: list[str] = Field(default_factory=list)


class WritingSectionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(...)
    heading: str = Field(...)
    purpose: str = Field(...)
    required_evidence: list[str] = Field(default_factory=list)
    required_figures: list[str] = Field(default_factory=list)
    required_citations: list[str] = Field(default_factory=list)


class WritingPlanModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(...)
    writing_mode: Literal["internal_report", "paper_outline", "section_draft", "full_draft"] = Field("internal_report")
    target_audience: str = Field(...)
    abstract_md: str = Field("")
    outline_md: str = Field("")
    section_specs: list[WritingSectionSpec] = Field(default_factory=list)
    figure_requests: list[FigureRequest] = Field(default_factory=list)
    citation_needs: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    preferred_output_format: Literal["tex"] = Field("tex")


class ClaimEvidenceMapRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim: str = Field(...)
    evidence_refs: list[str] = Field(default_factory=list)


class SectionDraftModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(...)
    heading: str = Field(...)
    status: Literal["drafted", "revised"] = Field(...)
    title: str = Field("")
    section_tex: str = Field("")
    citations: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    planned_figure_ids: list[str] = Field(default_factory=list)
    realized_figure_refs: list[str] = Field(default_factory=list)
    figure_refs: list[str] = Field(default_factory=list)
    latex_artifact_refs: list[str] = Field(default_factory=list)
    claim_evidence_map: list[ClaimEvidenceMapRow] = Field(default_factory=list)
    unresolved_gaps: list[str] = Field(default_factory=list)


class SectionReviewModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str = Field(...)
    status: Literal["approved", "needs_revision"] = Field(...)
    revision_notes: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    missing_citations: list[str] = Field(default_factory=list)


class ManuscriptBundleModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_campaign_id: str | None = Field(None)
    writing_mode: str = Field(...)
    title: str = Field(...)
    ordered_sections: list[str] = Field(default_factory=list)
    bibliography_shortlist: list[str] = Field(default_factory=list)
    figure_manifest: list[dict] = Field(default_factory=list)
    final_manuscript_path: str = Field(...)
    final_latex_path: str | None = Field(None)


class WritingBoard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(...)
    source_campaign_id: str | None = Field(None)
    writing_mode: str = Field("auto")
    status: Literal["planning", "drafting", "reviewing", "finalizing", "done", "failed"] = Field("planning")
    title: str = Field("")
    current_section_index: int = Field(0, ge=0)
    revision_counts: dict[str, int] = Field(default_factory=dict)
    latest_manuscript_ref: str | None = Field(None)
    latest_bundle_ref: str | None = Field(None)


__all__ = [
    "ClaimEvidenceMapRow",
    "FigureRequest",
    "ManuscriptBundleModel",
    "SectionDraftModel",
    "SectionReviewModel",
    "WritingBoard",
    "WritingPlanModel",
    "WritingSectionSpec",
]
