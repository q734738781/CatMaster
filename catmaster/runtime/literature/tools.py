from __future__ import annotations

import json
import tempfile
import re
import threading
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from catmaster.llm.config import LLMProfile
from catmaster.runtime.tool_runtime import current_run_dir, current_tool_context
from catmaster.tools.base import resolve_workspace_path, workspace_relpath, workspace_root
from .models import PaperRecord, PublicPageSnapshot
from .online_search_adapter import OnlineSearchAdapter, classify_tavily_failure
from .openalex_client import OpenAlexClient
from .semanticscholar_client import SemanticScholarClient, SemanticScholarRateLimitError


_PUBLIC_WEB_CIRCUIT_LOCK = threading.Lock()
_PUBLIC_WEB_CIRCUITS: dict[str, str] = {}
_PUBLIC_WEB_CIRCUIT_LIMIT = 512
_PUBLIC_PAGE_SOURCE_LOCK = threading.RLock()
_PUBLIC_PAGE_SOURCE_DIR = Path("literature/public_pages")
_PUBLIC_PAGE_SOURCE_FORMAT = "catmaster-public-page-source-v1"
_PUBLIC_PAGE_SOURCE_SEPARATOR = "\n\n--- CATMASTER PUBLIC PAGE TEXT ---\n\n"


def _public_web_circuit_scope() -> str:
    context = current_tool_context()
    return str(
        context.get("search_scope")
        or context.get("run_id")
        or current_run_dir()
        or ""
    ).strip()


def _public_web_circuit_failure(scope: str) -> str:
    if not scope:
        return ""
    with _PUBLIC_WEB_CIRCUIT_LOCK:
        return str(_PUBLIC_WEB_CIRCUITS.get(scope) or "")


def _trip_public_web_circuit(scope: str, category: str) -> None:
    if not scope:
        return
    with _PUBLIC_WEB_CIRCUIT_LOCK:
        _PUBLIC_WEB_CIRCUITS[scope] = str(category or "upstream_error")
        while len(_PUBLIC_WEB_CIRCUITS) > _PUBLIC_WEB_CIRCUIT_LIMIT:
            _PUBLIC_WEB_CIRCUITS.pop(next(iter(_PUBLIC_WEB_CIRCUITS)))


def _reset_public_web_circuits_for_tests() -> None:
    with _PUBLIC_WEB_CIRCUIT_LOCK:
        _PUBLIC_WEB_CIRCUITS.clear()


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
    limit: int = Field(10, ge=1, le=100, description="Provider page size, up to 100 records.")
    cursor: str = Field(
        "",
        description="Opaque OpenAlex cursor returned by the preceding page; leave empty for the first page.",
    )


class SearchSemanticScholarInput(BaseModel):
    """[literature/metadata] Search Semantic Scholar for exact paper metadata, abstracts, or seed papers."""

    query: str = Field(..., description="Short paper-lookup query for Semantic Scholar.")
    limit: int = Field(10, ge=1, le=100, description="Provider page size, up to 100 records.")
    offset: int = Field(0, ge=0, description="Provider offset returned as next_offset; use 0 for the first page.")
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
    limit: int = Field(10, ge=1, le=500, description="Maximum number of provider recommendations to return, up to 500.")
    positive_ids: list[str] = Field(default_factory=list, description="Explicit positive paper ids; leave empty to use seed_paper_ids.")
    negative_ids: list[str] = Field(default_factory=list, description="Negative paper ids to avoid; leave empty when none are needed.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_null_lists(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        for field_name in ("positive_ids", "negative_ids"):
            if normalized.get(field_name) is None:
                normalized[field_name] = []
        return normalized


class WebSearchInput(BaseModel):
    """[web/search] Search for scientific background; may return scholarly-index results when public-web search is unavailable."""

    query: str = Field(..., description="Public-web query.")
    max_results: int = Field(5, ge=1, le=20, description="Maximum number of results to return.")


SearchPublicWebInput = WebSearchInput


class OpenPublicPageInput(BaseModel):
    """[web/read] Fetch one public URL once, then page its stable workspace source."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(
        "",
        description=(
            "Public http(s) URL for the first read. The complete normalized page is saved once "
            "and the result returns source_path; leave url empty on continuation calls."
        ),
    )
    source_path: str = Field(
        "",
        description=(
            "Stable workspace source_path returned by the first read. Use it for every "
            "continuation so the page is read locally without another network request."
        ),
    )
    offset: int = Field(
        0,
        ge=0,
        description=(
            "Character offset in the saved normalized page text. Leave 0 when fetching url; "
            "use next_offset together with source_path to continue."
        ),
    )
    max_chars: int = Field(
        12000,
        ge=500,
        le=50000,
        description="Maximum normalized text characters in this response page.",
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_nulls(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        for key in ("url", "source_path"):
            if data.get(key) is None:
                data[key] = ""
        return data

    @model_validator(mode="after")
    def _validate_source(self) -> "OpenPublicPageInput":
        has_url = bool(self.url.strip())
        has_source = bool(self.source_path.strip())
        if has_url == has_source:
            raise ValueError("provide exactly one of url or source_path")
        if has_url and self.offset:
            raise ValueError("offset continuation requires source_path returned by the first read")
        return self


class FindInPageInput(BaseModel):
    """[web/find] Search one stable saved public-page source."""

    model_config = ConfigDict(extra="forbid")

    url: str = Field(
        "",
        description=(
            "Public http(s) URL when no saved source exists yet. It is fetched and saved once; "
            "use the returned source_path for later match pages."
        ),
    )
    source_path: str = Field(
        "",
        description=(
            "Stable workspace source_path returned by open_public_page or an earlier find. "
            "Prefer this path so searching never refetches the URL."
        ),
    )
    pattern: str = Field(..., description="Pattern to search for.")
    match_offset: int = Field(
        0,
        ge=0,
        description="Match index to start from; use next_match_offset to continue.",
    )
    max_matches: int = Field(5, ge=1, le=50, description="Maximum number of match snippets to return.")
    context_chars: int = Field(240, ge=40, le=2000, description="Context characters around each match.")

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_nulls(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        data = dict(value)
        for key in ("url", "source_path"):
            if data.get(key) is None:
                data[key] = ""
        return data

    @model_validator(mode="after")
    def _validate_source(self) -> "FindInPageInput":
        has_url = bool(self.url.strip())
        has_source = bool(self.source_path.strip())
        if has_url == has_source:
            raise ValueError("provide exactly one of url or source_path")
        if has_url and self.match_offset:
            raise ValueError(
                "match_offset continuation requires source_path returned by the first search"
            )
        return self


def _compact_search_text(value: Any, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _format_web_search_content(
    data: dict[str, Any],
    *,
    max_results: int = 5,
) -> str:
    status = str(data.get("status") or "").strip().lower()
    if status == "error":
        query = _compact_search_text(data.get("query") or "", max_chars=160)
        message = _compact_search_text(data.get("message") or "unknown error", max_chars=280)
        source = _compact_search_text(data.get("source") or "search backend", max_chars=40)
        return f"web_search failed for query={query!r} via {source}: {message}"

    query = _compact_search_text(data.get("query") or "", max_chars=200)
    hits_raw = data.get("hits") or data.get("results") or []
    lines = [f"Query: {query}", "Top results:"]
    for idx, item in enumerate(hits_raw[: max(1, int(max_results or 1))], start=1):
        if not isinstance(item, dict):
            continue
        title = _compact_search_text(item.get("title") or "Untitled result", max_chars=120)
        url = _compact_search_text(item.get("url") or "", max_chars=220)
        snippet = _compact_search_text(item.get("snippet") or item.get("content") or "", max_chars=800)
        if not snippet:
            snippet = "(no summary provided)"
        lines.append(f"- [{idx}] {title}")
        if url:
            lines.append(f"  URL: {url}")
        lines.append(f"  Snippet: {snippet}")
    if len(lines) == 2:
        lines.append("- (no results)")
    return "\n".join(lines)


def search_openalex(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[literature/metadata] Search OpenAlex for paper metadata candidates."""
    tool_name = "search_openalex"
    try:
        params = SearchOpenAlexInput(**payload)
        _profile, client, _scholar, _web = _literature_components()
        if hasattr(client, "search_works_page"):
            hits, total, next_cursor = client.search_works_page(
                params.query,
                limit=params.limit,
                cursor=params.cursor,
            )
        else:
            hits = client.search_works(params.query, limit=params.limit)
            total, next_cursor = len(hits), ""
        data = {
            "status": "ok",
            "source": "openalex",
            "query": params.query,
            "count": len(hits),
            "total": total,
            "next_cursor": next_cursor,
            "content_complete": not bool(next_cursor),
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
        if hasattr(client, "search_papers_page"):
            hits, total, next_offset = client.search_papers_page(
                params.query,
                limit=params.limit,
                offset=params.offset,
                year_from=params.year_from,
                year_to=params.year_to,
            )
        else:
            hits = client.search_papers(
                params.query,
                limit=params.limit,
                year_from=params.year_from,
                year_to=params.year_to,
            )
            total, next_offset = len(hits), None
        data = {
            "status": "ok",
            "source": "semantic_scholar",
            "query": params.query,
            "count": len(hits),
            "total": total,
            "next_offset": next_offset,
            "content_complete": next_offset is None,
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


def _paper_search_hit(paper: PaperRecord) -> dict[str, Any]:
    url = str(
        paper.landing_page_url
        or paper.url
        or paper.open_access_pdf_url
        or ""
    ).strip()
    if not url and paper.doi:
        url = f"https://doi.org/{quote(str(paper.doi).strip(), safe='/():;')}"
    snippet = str(paper.abstract or paper.snippet or "").strip()
    if not snippet:
        context = [str(paper.venue or "").strip()]
        if paper.year is not None:
            context.append(str(paper.year))
        snippet = ", ".join(item for item in context if item) or "Scholarly metadata record."
    return {
        "title": str(paper.title or "Untitled paper").strip(),
        "url": url,
        "snippet": snippet,
        "source": str(paper.source or "scholarly_index").strip(),
    }


def _deduplicate_papers(papers: list[PaperRecord], *, limit: int) -> list[PaperRecord]:
    kept: list[PaperRecord] = []
    seen: set[str] = set()
    for paper in papers:
        key = str(paper.doi or paper.paper_id or paper.title or "").strip().casefold()
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(paper)
        if len(kept) >= limit:
            break
    return kept


def _fallback_semantic_scholar_client(client: Any) -> Any:
    if type(client) is not SemanticScholarClient:
        return client
    return SemanticScholarClient(
        api_key=client.api_key,
        base_url=client.base_url,
        timeout_s=client.timeout_s,
        retry_429_attempts=0,
        retry_429_wait_seconds=0,
    )


def _scholarly_search_fallback(
    *,
    query: str,
    max_results: int,
    openalex: Any,
    scholar: Any,
) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Return a bounded scholarly discovery fallback without Tavily."""

    limit = max(1, min(int(max_results or 1), 20))
    errors: list[str] = []
    sources: list[tuple[str, Any]] = []
    if getattr(openalex, "api_key", None):
        sources.append(("openalex", openalex))
    sources.append(("semantic_scholar", _fallback_semantic_scholar_client(scholar)))
    if not getattr(openalex, "api_key", None):
        sources.append(("openalex", openalex))

    for source, client in sources:
        try:
            if source == "openalex":
                raw = client.search_works(query, limit=limit)
            else:
                raw = client.search_papers(query, limit=limit)
            papers = [item.paper for item in list(raw or []) if isinstance(getattr(item, "paper", None), PaperRecord)]
            papers = _deduplicate_papers(papers, limit=limit)
            if papers:
                return [_paper_search_hit(paper) for paper in papers], source, errors
        except SemanticScholarRateLimitError:
            errors.append(f"{source}:rate_limited")
        except httpx.HTTPStatusError as exc:
            errors.append(f"{source}:http_{int(exc.response.status_code)}")
        except httpx.RequestError:
            errors.append(f"{source}:network_error")
        except Exception:
            errors.append(f"{source}:unavailable")
    return [], "scholarly_index_unavailable", errors


def _web_search_result(data: dict[str, Any], *, max_results: int) -> tuple[str, dict[str, Any]]:
    parent = workspace_root() / "notes" / "web_search"
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="search-", dir=parent, delete=False, encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        source = str(Path(handle.name).relative_to(workspace_root()))
    data["source_path"] = source
    content = _format_web_search_content(data, max_results=max_results) + f"\nFull search results: {source}"
    return content, {"tool_name": "web_search", "data": data, "suppress_content_offload_ref": True}


def _degraded_web_search_result(
    *,
    query: str,
    max_results: int,
    category: str,
    openalex: Any,
    scholar: Any,
    circuit_open: bool,
) -> tuple[str, dict[str, Any]]:
    hits, backend, fallback_errors = _scholarly_search_fallback(
        query=query,
        max_results=max_results,
        openalex=openalex,
        scholar=scholar,
    )
    data: dict[str, Any] = {
        "status": "degraded" if hits else "error",
        "source": backend,
        "backend": backend,
        "degraded_from": "tavily",
        "failure_category": category,
        "retryable": False,
        "circuit_open": bool(circuit_open),
        "query": query,
        "count": len(hits),
        "hits": hits,
    }
    if fallback_errors:
        data["fallback_errors"] = fallback_errors
    if not hits:
        data["message"] = (
            "Public-web search is unavailable and the scholarly-index fallback "
            "did not return usable results."
        )
    return _web_search_result(data, max_results=max_results)


def web_search(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[web/search] Search the public web, with scholarly discovery fallback when configured."""
    tool_name = "web_search"
    try:
        params = WebSearchInput(**payload)
        profile, openalex, scholar, web = _literature_components()
        scope = _public_web_circuit_scope()
        circuit_category = _public_web_circuit_failure(scope)
        if circuit_category:
            return _degraded_web_search_result(
                query=params.query,
                max_results=params.max_results,
                category=circuit_category,
                openalex=openalex,
                scholar=scholar,
                circuit_open=True,
            )
        result = web.search_public_web(params.query, max_results=params.max_results)
        data = {
            "status": "ok",
            "source": "tavily",
            "backend": "tavily",
            "query": params.query,
            "count": len(result.results),
            "hits": [item.model_dump() for item in result.results],
        }
        return _web_search_result(data, max_results=params.max_results)
    except Exception as exc:
        if "params" in locals() and "profile" in locals():
            failure = classify_tavily_failure(exc)
            category = str(failure["category"])
            if bool(failure["disable_for_scope"]):
                _trip_public_web_circuit(scope if "scope" in locals() else "", category)
            if profile.literature.public_web_on_search_failure:
                return _degraded_web_search_result(
                    query=params.query,
                    max_results=params.max_results,
                    category=category,
                    openalex=openalex,
                    scholar=scholar,
                    circuit_open=bool(failure["disable_for_scope"]),
                )
            return _external_tool_soft_error_result(
                tool_name=tool_name,
                source="tavily",
                status=category,
                message=f"Public-web search is unavailable ({category}).",
                extra={
                    "query": params.query,
                    "backend": "tavily",
                    "retryable": bool(failure["retryable"]),
                    "circuit_open": bool(failure["disable_for_scope"]),
                },
            )
        soft = _soft_external_error(
            tool_name,
            "public_web",
            exc,
            extra={"query": payload.get("query")},
        )
        return soft


def search_public_web(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Deprecated alias for `web_search`."""
    return web_search(payload)


def _public_page_source_stem(url: str) -> str:
    parsed = urlsplit(str(url or "").strip())
    host = re.sub(r"[^A-Za-z0-9.-]+", "_", parsed.netloc).strip("._-")
    route = parsed.path.strip("/")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", route).strip("._-")
    return "_".join(item for item in (host or "public-page", slug or "index") if item)[:180]


def _write_public_page_source(page: PublicPageSnapshot) -> str:
    source_dir = workspace_root() / _PUBLIC_PAGE_SOURCE_DIR
    source_dir.mkdir(parents=True, exist_ok=True)
    metadata = page.model_dump(mode="json", exclude={"text"})
    metadata["format"] = _PUBLIC_PAGE_SOURCE_FORMAT
    body = (
        json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        + _PUBLIC_PAGE_SOURCE_SEPARATOR
        + str(page.text or "")
        + "\n"
    )
    stem = _public_page_source_stem(page.requested_url)
    with _PUBLIC_PAGE_SOURCE_LOCK:
        index = 1
        while True:
            suffix = "" if index == 1 else f"-{index}"
            target = source_dir / f"{stem}{suffix}.txt"
            if not target.exists():
                target.write_text(body, encoding="utf-8")
                return "/" + workspace_relpath(target).replace("\\", "/")
            index += 1


def _read_public_page_source(source_path: str) -> tuple[PublicPageSnapshot, str]:
    target = resolve_workspace_path(source_path, must_exist=True)
    source_root = (workspace_root() / _PUBLIC_PAGE_SOURCE_DIR).resolve()
    try:
        target.resolve().relative_to(source_root)
    except ValueError as exc:
        raise ValueError(
            "source_path must be a public-page source returned by this tool"
        ) from exc
    raw = target.read_text(encoding="utf-8")
    header, separator, text = raw.partition(_PUBLIC_PAGE_SOURCE_SEPARATOR)
    if not separator:
        raise ValueError("source_path is not a CatMaster public-page source")
    metadata = json.loads(header)
    if not isinstance(metadata, dict) or metadata.pop("format", "") != _PUBLIC_PAGE_SOURCE_FORMAT:
        raise ValueError("source_path uses an unsupported public-page source format")
    metadata["text"] = text[:-1] if text.endswith("\n") else text
    page = PublicPageSnapshot.model_validate(metadata)
    normalized = "/" + workspace_relpath(target).replace("\\", "/")
    return page, normalized


def _load_or_fetch_public_page(
    *,
    url: str,
    source_path: str,
    web: Any,
) -> tuple[PublicPageSnapshot, str]:
    if str(source_path or "").strip():
        return _read_public_page_source(source_path)
    page = web.open_public_page(url)
    if not isinstance(page, PublicPageSnapshot):
        payload = page.model_dump(mode="json") if hasattr(page, "model_dump") else vars(page)
        page = PublicPageSnapshot.model_validate(payload)
    return page, _write_public_page_source(page)


def open_public_page(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[web/read] Fetch one public URL once, then page its stable workspace source."""
    tool_name = "open_public_page"
    try:
        params = OpenPublicPageInput(**payload)
        _profile, _openalex, _scholar, web = _literature_components()
        page, source_path = _load_or_fetch_public_page(
            url=params.url,
            source_path=params.source_path,
            web=web,
        )
        snapshot = page.model_dump(mode="json")
        text = str(snapshot.get("text") or "")
        if params.offset > len(text):
            raise ValueError("offset is outside the normalized page text")
        end = min(len(text), params.offset + params.max_chars)
        next_offset = end if end < len(text) else 0
        page_payload = {key: value for key, value in snapshot.items() if key != "text"}
        page_payload.update(
            {
                "text": text[params.offset:end],
                "offset": params.offset,
                "total_chars": len(text),
                "content_complete": next_offset == 0,
                "next_offset": next_offset,
                "source_path": source_path,
            }
        )
        data = {
            "status": "ok",
            "source": "public_web_page",
            "page": page_payload,
        }
        return _json_tool_result(data=data, tool_name=tool_name)
    except Exception as exc:
        soft = _soft_external_error(
            tool_name,
            "public_web",
            exc,
            extra={
                "url": payload.get("url"),
                "source_path": payload.get("source_path"),
            },
        )
        return soft


def find_in_page(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[web/find] Search complete normalized text from one stable saved source."""
    tool_name = "find_in_page"
    try:
        params = FindInPageInput(**payload)
        _profile, _openalex, _scholar, web = _literature_components()
        page, source_path = _load_or_fetch_public_page(
            url=params.url,
            source_path=params.source_path,
            web=web,
        )
        snapshot = page.model_dump(mode="json")
        match_offset = params.match_offset
        text = str(snapshot.get("text") or "")
        needle = " ".join(params.pattern.split()).strip()
        if not needle:
            raise ValueError("pattern is required")
        folded = [character.casefold() for character in text]
        original_offsets = [index for index, part in enumerate(folded) for _ in part]
        lower_text = "".join(folded)
        lower_needle = needle.casefold()
        positions: list[tuple[int, int]] = []
        start = 0
        while True:
            index = lower_text.find(lower_needle, start)
            if index < 0:
                break
            end = index + len(lower_needle)
            span = (original_offsets[index], original_offsets[end - 1] + 1)
            if not positions or positions[-1] != span:
                positions.append(span)
            start = end
        if match_offset > len(positions):
            raise ValueError("match_offset is outside the result set")
        selected = positions[match_offset : match_offset + params.max_matches]
        matches = [
            {
                "pattern": needle,
                "start_char": start_char,
                "end_char": end_char,
                "snippet": text[
                    max(0, start_char - params.context_chars) :
                    min(len(text), end_char + params.context_chars)
                ].strip(),
            }
            for start_char, end_char in selected
        ]
        next_offset = match_offset + len(matches)
        next_match_offset = next_offset if next_offset < len(positions) else 0
        result = {
            "requested_url": str(snapshot.get("requested_url") or ""),
            "final_url": str(snapshot.get("final_url") or ""),
            "source_path": source_path,
            "pattern": needle,
            "total_matches": len(positions),
            "match_offset": match_offset,
            "matches": matches,
            "next_match_offset": next_match_offset,
            "content_complete": next_match_offset == 0,
            "source_completeness": str(
                snapshot.get("source_completeness") or "complete"
            ),
        }
        data = {
            "status": "ok",
            "source": "public_web_page",
            "result": result,
        }
        return _json_tool_result(data=data, tool_name=tool_name)
    except Exception as exc:
        soft = _soft_external_error(
            tool_name,
            "public_web",
            exc,
            extra={
                "url": payload.get("url"),
                "source_path": payload.get("source_path"),
                "pattern": payload.get("pattern"),
            },
        )
        return soft


__all__ = [
    "WebSearchInput",
    "web_search",
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
