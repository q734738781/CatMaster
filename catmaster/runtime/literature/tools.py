from __future__ import annotations

import json
from typing import Any

import httpx
from pydantic import BaseModel, Field, ValidationError

from catmaster.llm.config import LLMProfile
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError

from .subagent import LiteratureSubagent
from .models import PaperRecord
from .online_search_adapter import OnlineSearchAdapter
from .openalex_client import OpenAlexClient
from .semanticscholar_client import SemanticScholarClient, SemanticScholarRateLimitError


def _literature_components() -> tuple[LLMProfile, OpenAlexClient, SemanticScholarClient, OnlineSearchAdapter]:
    profile = LLMProfile.from_env_or_file()
    return (
        profile,
        OpenAlexClient(),
        SemanticScholarClient(
            retry_429_attempts=profile.literature.semantic_scholar_retry_429_attempts,
            retry_429_wait_seconds=profile.literature.semantic_scholar_retry_429_wait_seconds,
        ),
        OnlineSearchAdapter(),
    )


def _paper_payload(paper: PaperRecord) -> dict[str, Any]:
    return paper.model_dump()


def _json_tool_result(*, data: dict[str, Any], tool_name: str) -> tuple[str, dict[str, Any]]:
    content = json.dumps(data, ensure_ascii=False)
    return content, {
        "tool_name": tool_name,
        "data": data,
        "suppress_content_offload_ref": True,
    }


def _semantic_scholar_rate_limited_result(
    *,
    tool_name: str,
    query: str = "",
    paper_id_or_doi: str = "",
    seed_paper_ids: list[str] | None = None,
    exc: SemanticScholarRateLimitError,
) -> tuple[str, dict[str, Any]]:
    data: dict[str, Any] = {
        "status": "rate_limited",
        "source": "semantic_scholar",
        "message": str(exc),
        "retry_exhausted": True,
        "attempts": int(exc.attempts),
        "wait_seconds": float(exc.wait_seconds),
    }
    if query:
        data["query"] = query
    if paper_id_or_doi:
        data["paper_id_or_doi"] = paper_id_or_doi
    if seed_paper_ids:
        data["seed_paper_ids"] = [str(item).strip() for item in seed_paper_ids if str(item).strip()]
    return _json_tool_result(data=data, tool_name=tool_name)


def _external_tool_soft_error_result(
    *,
    tool_name: str,
    source: str,
    message: str,
    status: str,
    extra: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    data: dict[str, Any] = {
        "status": status,
        "source": source,
        "message": str(message or "").strip(),
    }
    if isinstance(extra, dict):
        data.update({str(key): value for key, value in extra.items()})
    return _json_tool_result(data=data, tool_name=tool_name)


def _soft_external_error(tool_name: str, source: str, exc: Exception, *, extra: dict[str, Any] | None = None) -> tuple[str, dict[str, Any]] | None:
    if isinstance(exc, SemanticScholarRateLimitError):
        return _external_tool_soft_error_result(
            tool_name=tool_name,
            source=source,
            status="rate_limited",
            message=str(exc),
            extra=extra,
        )
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = int(exc.response.status_code)
        if status_code == 404:
            return _external_tool_soft_error_result(
                tool_name=tool_name,
                source=source,
                status="not_found",
                message=f"{source} record not found.",
                extra={**(extra or {}), "http_status": status_code},
            )
        if status_code == 429:
            return _external_tool_soft_error_result(
                tool_name=tool_name,
                source=source,
                status="rate_limited",
                message=f"{source} rate limited.",
                extra={**(extra or {}), "http_status": status_code},
            )
        if 500 <= status_code < 600:
            return _external_tool_soft_error_result(
                tool_name=tool_name,
                source=source,
                status="upstream_error",
                message=f"{source} upstream error ({status_code}).",
                extra={**(extra or {}), "http_status": status_code},
            )
    if isinstance(exc, httpx.RequestError):
        return _external_tool_soft_error_result(
            tool_name=tool_name,
            source=source,
            status="network_error",
            message=f"{source} request failed: {exc}",
            extra=extra,
        )
    if isinstance(exc, ValidationError):
        return _external_tool_soft_error_result(
            tool_name=tool_name,
            source=source,
            status="invalid_request",
            message=f"{tool_name} received invalid arguments.",
            extra={**(extra or {}), "validation_error": str(exc)},
        )
    if isinstance(exc, ValueError) and "Unexpected" in str(exc):
        return _external_tool_soft_error_result(
            tool_name=tool_name,
            source=source,
            status="bad_payload",
            message=f"{source} returned an unexpected payload.",
            extra=extra,
        )
    if isinstance(exc, ValueError):
        return _external_tool_soft_error_result(
            tool_name=tool_name,
            source=source,
            status="invalid_request",
            message=str(exc),
            extra=extra,
        )
    if isinstance(exc, RuntimeError):
        message = str(exc).strip()
        if "TAVILY_API_KEY is required" in message:
            return _external_tool_soft_error_result(
                tool_name=tool_name,
                source=source,
                status="unavailable",
                message=message,
                extra=extra,
            )
        if "request failed without response" in message:
            return _external_tool_soft_error_result(
                tool_name=tool_name,
                source=source,
                status="upstream_error",
                message=message,
                extra=extra,
            )
        return _external_tool_soft_error_result(
            tool_name=tool_name,
            source=source,
            status="runtime_error",
            message=message or f"{tool_name} runtime error.",
            extra=extra,
        )
    return _external_tool_soft_error_result(
        tool_name=tool_name,
        source=source,
        status="internal_error",
        message=f"{tool_name} failed: {exc}",
        extra=extra,
    )


class SearchOpenAlexInput(BaseModel):
    """[literature/metadata] Search OpenAlex for exact scholarly metadata, not broad background search."""

    query: str = Field(..., description="Short paper-lookup query for OpenAlex.")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of records to return.")


class SearchSemanticScholarInput(BaseModel):
    """[literature/metadata] Search Semantic Scholar for exact paper metadata, abstracts, or seed papers."""

    query: str = Field(..., description="Short paper-lookup query for Semantic Scholar.")
    limit: int = Field(10, ge=1, le=50, description="Maximum number of records to return.")
    year_from: int | None = Field(None, description="Optional lower year bound.")
    year_to: int | None = Field(None, description="Optional upper year bound.")


class GetOpenAlexRecordInput(BaseModel):
    """[literature/metadata] Fetch one OpenAlex work by OpenAlex id or DOI."""

    work_id_or_doi: str = Field(..., description="OpenAlex work id or DOI.")


class GetSemanticScholarRecordInput(BaseModel):
    """[literature/metadata] Fetch one Semantic Scholar paper by paper id or DOI."""

    paper_id_or_doi: str = Field(..., description="Semantic Scholar paper id or DOI.")


class RecommendSemanticScholarInput(BaseModel):
    """[literature/metadata] Expand a small seed set with Semantic Scholar recommendations."""

    seed_paper_ids: list[str] = Field(..., description="Seed Semantic Scholar paper ids.")
    limit: int = Field(6, ge=1, le=50, description="Maximum number of recommendations to return.")
    positive_ids: list[str] | None = Field(None, description="Optional explicit positive paper ids.")
    negative_ids: list[str] | None = Field(None, description="Optional negative paper ids to avoid.")


class SearchPublicWebInput(BaseModel):
    """[web/search] Search the public web for broad scientific background or landing-page summaries."""

    query: str = Field(..., description="Public-web query.")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results to return.")


class OpenPublicPageInput(BaseModel):
    """[web/read] Open a public page and extract normalized visible text."""

    url: str = Field(..., description="Public http(s) URL.")
    max_chars: int = Field(12000, ge=500, le=50000, description="Maximum number of normalized text characters to retain.")


class FindInPageInput(BaseModel):
    """[web/find] Search normalized text within a previously identified public page."""

    url: str = Field(..., description="Public http(s) URL.")
    pattern: str = Field(..., description="Pattern to search for.")
    max_matches: int = Field(5, ge=1, le=50, description="Maximum number of match snippets to return.")
    context_chars: int = Field(240, ge=40, le=2000, description="Context characters around each match.")
    page_max_chars: int = Field(20000, ge=500, le=80000, description="Maximum page text to inspect.")


class RunLiteratureResearchInput(BaseModel):
    """[literature/review] Run a precise literature-grounding workflow and return a compact context pack."""

    query: str = Field(..., description="Literature question or retrieval topic to investigate.")
    depth: str = Field(
        "auto",
        description="Research depth: auto | none | quick | standard | focused | deep_report.",
    )
    topic: str | None = Field(
        None,
        description="Optional short topic/search phrase. If provided, scholarly databases use this instead of the full natural-language request.",
    )
    seed_papers: list[str] | None = Field(
        None,
        description="Optional Semantic Scholar ids / DOI-like seeds to anchor recommendations.",
    )
    role: str | None = Field(
        None,
        description="Internal caller role used for depth-policy gating. Usually injected automatically.",
    )


def _pack_summary_text(data: dict[str, Any]) -> str:
    summary_limit = max(1, LLMProfile.from_env_or_file().literature.summary_key_paper_count)
    lines = [
        f"Literature research depth: {data.get('depth', 'unknown')}",
        f"Summary: {data.get('summary', '').strip()}",
        "Key papers:",
    ]
    key_papers = data.get("key_papers") or []
    if key_papers:
        for idx, paper in enumerate(key_papers[:summary_limit], start=1):
            if not isinstance(paper, dict):
                continue
            title = str(paper.get("title") or "Untitled paper").strip()
            year = str(paper.get("year") or "?").strip()
            venue = str(paper.get("venue") or "").strip()
            doi = str(paper.get("doi") or paper.get("url") or "").strip()
            oa_pdf = str(paper.get("open_access_pdf_url") or "").strip()
            abstract_flag = paper.get("has_abstract")
            suffix = f" ({venue})" if venue else ""
            cite = f" - {doi}" if doi else ""
            extra_parts = []
            if abstract_flag is True:
                extra_parts.append("abstract")
            if oa_pdf:
                extra_parts.append("oa_pdf")
            extra = f" [{', '.join(extra_parts)}]" if extra_parts else ""
            lines.append(f"- [{idx}] {title} [{year}]{suffix}{cite}{extra}")
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def run_literature_research(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[literature/review] Run the literature grounding workflow and return a compact context pack."""
    tool_name = "run_literature_research"
    try:
        params = RunLiteratureResearchInput(**payload)
        subagent = LiteratureSubagent.create_default()
        pack = subagent.run(
            query=params.query,
            requested_depth=params.depth,
            topic=params.topic,
            seed_papers=params.seed_papers,
            role=params.role,
        )
        data = pack.model_dump()
        return _pack_summary_text(data), {
            "tool_name": tool_name,
            "data": data,
            # Keep representative citations visible in tool content even when the
            # full artifact is offloaded to disk.
            "suppress_content_offload_ref": True,
        }
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} failed: {exc}",
            artifact={"tool_name": tool_name, "data": {"query": payload.get("query")}},
            error_code="run_literature_research_failed",
        ) from exc


def search_openalex(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[literature/metadata] Search OpenAlex for paper metadata candidates."""
    tool_name = "search_openalex"
    try:
        params = SearchOpenAlexInput(**payload)
        _profile, client, _scholar, _web = _literature_components()
        hits = client.search_works(params.query, limit=params.limit)
        data = {
            "status": "ok",
            "source": "openalex",
            "query": params.query,
            "count": len(hits),
            "papers": [_paper_payload(hit.paper) for hit in hits],
        }
        return _json_tool_result(data=data, tool_name=tool_name)
    except Exception as exc:
        soft = _soft_external_error(
            tool_name,
            "openalex",
            exc,
            extra={"query": payload.get("query")},
        )
        return soft


def search_semantic_scholar(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[literature/metadata] Search Semantic Scholar for paper metadata candidates."""
    tool_name = "search_semantic_scholar"
    try:
        params = SearchSemanticScholarInput(**payload)
        _profile, _openalex, client, _web = _literature_components()
        hits = client.search_papers(
            params.query,
            limit=params.limit,
            year_from=params.year_from,
            year_to=params.year_to,
        )
        data = {
            "status": "ok",
            "source": "semantic_scholar",
            "query": params.query,
            "count": len(hits),
            "papers": [_paper_payload(hit.paper) for hit in hits],
        }
        return _json_tool_result(data=data, tool_name=tool_name)
    except SemanticScholarRateLimitError as exc:
        return _semantic_scholar_rate_limited_result(
            tool_name=tool_name,
            query=params.query if "params" in locals() else str(payload.get("query") or ""),
            exc=exc,
        )
    except Exception as exc:
        soft = _soft_external_error(
            tool_name,
            "semantic_scholar",
            exc,
            extra={"query": payload.get("query")},
        )
        return soft


def get_openalex_record(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[literature/metadata] Fetch one OpenAlex record by DOI or OpenAlex id."""
    tool_name = "get_openalex_record"
    try:
        params = GetOpenAlexRecordInput(**payload)
        _profile, client, _scholar, _web = _literature_components()
        paper = client.get_work(params.work_id_or_doi)
        data = {
            "status": "ok",
            "source": "openalex",
            "paper": _paper_payload(paper),
        }
        return _json_tool_result(data=data, tool_name=tool_name)
    except Exception as exc:
        soft = _soft_external_error(
            tool_name,
            "openalex",
            exc,
            extra={"work_id_or_doi": payload.get("work_id_or_doi")},
        )
        return soft


def get_semantic_scholar_record(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[literature/metadata] Fetch one Semantic Scholar record by DOI or paper id."""
    tool_name = "get_semantic_scholar_record"
    try:
        params = GetSemanticScholarRecordInput(**payload)
        _profile, _openalex, client, _web = _literature_components()
        paper = client.get_paper(params.paper_id_or_doi)
        data = {
            "status": "ok",
            "source": "semantic_scholar",
            "paper": _paper_payload(paper),
        }
        return _json_tool_result(data=data, tool_name=tool_name)
    except SemanticScholarRateLimitError as exc:
        return _semantic_scholar_rate_limited_result(
            tool_name=tool_name,
            paper_id_or_doi=params.paper_id_or_doi if "params" in locals() else str(payload.get("paper_id_or_doi") or ""),
            exc=exc,
        )
    except Exception as exc:
        soft = _soft_external_error(
            tool_name,
            "semantic_scholar",
            exc,
            extra={"paper_id_or_doi": payload.get("paper_id_or_doi")},
        )
        return soft


def recommend_semantic_scholar(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[literature/metadata] Expand a seed set with Semantic Scholar recommendations."""
    tool_name = "recommend_semantic_scholar"
    try:
        params = RecommendSemanticScholarInput(**payload)
        _profile, _openalex, client, _web = _literature_components()
        hits = client.get_recommendations(
            seed_paper_ids=params.seed_paper_ids,
            positive_ids=params.positive_ids,
            negative_ids=params.negative_ids,
            limit=params.limit,
        )
        data = {
            "status": "ok",
            "source": "semantic_scholar_recommendation",
            "seed_paper_ids": list(params.seed_paper_ids),
            "count": len(hits),
            "papers": [_paper_payload(hit.paper) for hit in hits],
        }
        return _json_tool_result(data=data, tool_name=tool_name)
    except SemanticScholarRateLimitError as exc:
        return _semantic_scholar_rate_limited_result(
            tool_name=tool_name,
            seed_paper_ids=params.seed_paper_ids if "params" in locals() else list(payload.get("seed_paper_ids") or []),
            exc=exc,
        )
    except Exception as exc:
        soft = _soft_external_error(
            tool_name,
            "semantic_scholar",
            exc,
            extra={"seed_paper_ids": list(payload.get("seed_paper_ids") or [])},
        )
        return soft


def search_public_web(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[web/search] Search the public web for broad scientific context."""
    tool_name = "search_public_web"
    try:
        params = SearchPublicWebInput(**payload)
        _profile, _openalex, _scholar, web = _literature_components()
        result = web.search_public_web(params.query, max_results=params.max_results)
        data = {
            "status": "ok",
            "source": "public_web",
            "query": params.query,
            "count": len(result.results),
            "hits": [item.model_dump() for item in result.results],
        }
        return _json_tool_result(data=data, tool_name=tool_name)
    except Exception as exc:
        soft = _soft_external_error(
            tool_name,
            "public_web",
            exc,
            extra={"query": payload.get("query")},
        )
        return soft


def open_public_page(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[web/read] Open a public page and return normalized visible text."""
    tool_name = "open_public_page"
    try:
        params = OpenPublicPageInput(**payload)
        _profile, _openalex, _scholar, web = _literature_components()
        page = web.open_public_page(params.url, max_chars=params.max_chars)
        data = {
            "status": "ok",
            "source": "public_web_page",
            "page": page.model_dump(),
        }
        return _json_tool_result(data=data, tool_name=tool_name)
    except Exception as exc:
        soft = _soft_external_error(
            tool_name,
            "public_web",
            exc,
            extra={"url": payload.get("url")},
        )
        return soft


def find_in_page(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[web/find] Search a public page for pattern matches and snippets."""
    tool_name = "find_in_page"
    try:
        params = FindInPageInput(**payload)
        _profile, _openalex, _scholar, web = _literature_components()
        result = web.find_in_page(
            params.url,
            params.pattern,
            max_matches=params.max_matches,
            context_chars=params.context_chars,
            page_max_chars=params.page_max_chars,
        )
        data = {
            "status": "ok",
            "source": "public_web_page",
            "result": result.model_dump(),
        }
        return _json_tool_result(data=data, tool_name=tool_name)
    except Exception as exc:
        soft = _soft_external_error(
            tool_name,
            "public_web",
            exc,
            extra={"url": payload.get("url"), "pattern": payload.get("pattern")},
        )
        return soft


__all__ = [
    "RunLiteratureResearchInput",
    "run_literature_research",
    "SearchOpenAlexInput",
    "search_openalex",
    "SearchSemanticScholarInput",
    "search_semantic_scholar",
    "GetOpenAlexRecordInput",
    "get_openalex_record",
    "GetSemanticScholarRecordInput",
    "get_semantic_scholar_record",
    "RecommendSemanticScholarInput",
    "recommend_semantic_scholar",
    "SearchPublicWebInput",
    "search_public_web",
    "OpenPublicPageInput",
    "open_public_page",
    "FindInPageInput",
    "find_in_page",
]
