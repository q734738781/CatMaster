from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PaperRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paper_id: str = Field("", description="Semantic Scholar paper id when available.")
    title: str = Field(..., description="Paper title.")
    year: int | None = Field(None, description="Publication year if available.")
    venue: str | None = Field(None, description="Venue/journal/conference.")
    url: str | None = Field(None, description="Canonical paper or landing-page URL.")
    doi: str | None = Field(None, description="DOI when available.")
    abstract: str | None = Field(None, description="Abstract snippet or summary.")
    authors: list[str] = Field(default_factory=list, description="Short author list.")
    citation_count: int | None = Field(None, description="Citation count when available.")
    influential_citation_count: int | None = Field(None, description="Influential citation count when available.")
    landing_page_url: str | None = Field(None, description="Canonical landing page URL for the work.")
    open_access_pdf_url: str | None = Field(None, description="Direct open-access PDF URL when available.")
    is_open_access: bool | None = Field(None, description="Whether an open-access location is available.")
    has_abstract: bool | None = Field(None, description="Whether abstract text was available in normalized metadata.")
    has_fulltext: bool | None = Field(None, description="Whether the source reports accessible fulltext availability.")
    source: str = Field("semantic_scholar", description="Primary provenance source.")
    snippet: str | None = Field(None, description="Short task-relevant snippet.")


class PaperSearchHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., description="Query that produced this hit.")
    rank: int = Field(..., ge=1, description="1-based rank in the current retrieval pass.")
    score_hint: float | None = Field(
        None,
        exclude=True,
        description="Internal retrieval-order hint; never part of scientific evidence output.",
    )
    paper: PaperRecord = Field(..., description="Normalized paper record.")


class PublicWebHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., description="Result title.")
    url: str | None = Field(None, description="Public URL.")
    snippet: str = Field(..., description="Short normalized snippet.")
    source: str = Field("public_web", description="Source bucket.")


class PublicWebSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[PublicWebHit] = Field(default_factory=list, description="Public web search hits.")


class PublicPageSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_url: str = Field(..., description="Original requested URL.")
    final_url: str = Field(..., description="Final URL after redirects.")
    status_code: int = Field(..., description="HTTP status code for the fetched page.")
    content_type: str | None = Field(None, description="Response content type when available.")
    title: str | None = Field(None, description="Page title when detected.")
    description: str | None = Field(None, description="Short description or abstract-like meta text when detected.")
    text: str = Field(..., description="Normalized visible page text, truncated to the requested maximum.")


class InPageMatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pattern: str = Field(..., description="Query pattern that was searched.")
    start_char: int = Field(..., ge=0, description="Start character offset in normalized page text.")
    end_char: int = Field(..., ge=0, description="End character offset in normalized page text.")
    snippet: str = Field(..., description="Context snippet around the match.")


class FindInPageResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requested_url: str = Field(..., description="Original requested URL.")
    final_url: str = Field(..., description="Final URL after redirects.")
    pattern: str = Field(..., description="Pattern that was searched.")
    total_matches: int = Field(..., ge=0, description="Total number of matches found in normalized text.")
    matches: list[InPageMatch] = Field(default_factory=list, description="First page-text matches with context snippets.")


__all__ = [
    "PaperRecord",
    "PaperSearchHit",
    "PublicWebHit",
    "PublicWebSearchResult",
    "PublicPageSnapshot",
    "InPageMatch",
    "FindInPageResult",
]
