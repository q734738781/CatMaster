from __future__ import annotations

import os
import time
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from .models import PaperRecord, PaperSearchHit

_FIELDS = ",".join(
    [
        "paperId",
        "title",
        "year",
        "abstract",
        "venue",
        "url",
        "authors",
        "citationCount",
        "influentialCitationCount",
        "externalIds",
        "openAccessPdf",
        "fieldsOfStudy",
        "tldr",
    ]
)


class SemanticScholarRateLimitError(RuntimeError):
    def __init__(
        self,
        *,
        attempts: int,
        wait_seconds: float,
        response: httpx.Response | None = None,
    ) -> None:
        self.attempts = max(1, int(attempts))
        self.wait_seconds = max(0.0, float(wait_seconds))
        self.response = response
        super().__init__(
            "Semantic Scholar rate limited after "
            f"{self.attempts} attempt(s); waited about {self.wait_seconds:.1f}s between retries."
        )


class SemanticScholarClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.semanticscholar.org",
        timeout_s: float = 30.0,
        retry_429_attempts: int = 3,
        retry_429_wait_seconds: float = 15.0,
    ) -> None:
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip() or None
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.retry_429_attempts = max(0, int(retry_429_attempts))
        self.retry_429_wait_seconds = max(0.0, float(retry_429_wait_seconds))

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "CatMaster/1.0 literature"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _client(self) -> httpx.Client:
        return httpx.Client(timeout=self.timeout_s, headers=self._headers())

    @staticmethod
    def _retry_after_seconds(response: httpx.Response) -> float | None:
        raw = str(response.headers.get("Retry-After") or "").strip()
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    def _request_with_retry(self, method: str, path: str, **kwargs) -> httpx.Response:
        attempts = self.retry_429_attempts + 1
        last_exc: httpx.HTTPStatusError | None = None
        for attempt in range(attempts):
            with self._client() as client:
                response = client.request(method, f"{self.base_url}{path}", **kwargs)
            try:
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code != 429:
                    raise
                wait_s = self._retry_after_seconds(exc.response)
                if wait_s is None:
                    wait_s = self.retry_429_wait_seconds
                if attempt >= attempts - 1:
                    raise SemanticScholarRateLimitError(
                        attempts=attempts,
                        wait_seconds=wait_s,
                        response=exc.response,
                    ) from exc
                if wait_s > 0:
                    time.sleep(wait_s)
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Semantic Scholar request failed without response")

    @staticmethod
    def _normalize_doi(raw: str) -> str | None:
        text = str(raw or "").strip()
        if not text:
            return None
        lower = text.lower()
        if lower.startswith("https://doi.org/") or lower.startswith("http://doi.org/"):
            parsed = urlparse(text)
            doi = parsed.path.lstrip("/")
            return doi or None
        if text.startswith("10."):
            return text
        return None

    @classmethod
    def _paper_identifier_candidates(cls, ident: str) -> list[str]:
        normalized = str(ident or "").strip()
        if not normalized:
            return []
        doi = cls._normalize_doi(normalized)
        if doi:
            return [f"DOI:{doi}", doi]
        return [normalized]

    @staticmethod
    def _paper_from_payload(payload: dict[str, Any], *, source: str = "semantic_scholar") -> PaperRecord:
        external_ids = payload.get("externalIds") if isinstance(payload.get("externalIds"), dict) else {}
        authors_raw = payload.get("authors") if isinstance(payload.get("authors"), list) else []
        authors = [str(item.get("name") or "").strip() for item in authors_raw if isinstance(item, dict)]
        authors = [item for item in authors if item]
        tldr_raw = payload.get("tldr") if isinstance(payload.get("tldr"), dict) else {}
        snippet = str(tldr_raw.get("text") or "").strip() or None
        if snippet is None:
            abstract = str(payload.get("abstract") or "").strip()
            snippet = abstract[:400].strip() or None
        open_access_raw = payload.get("openAccessPdf") if isinstance(payload.get("openAccessPdf"), dict) else {}
        landing_page_url = str(payload.get("url") or "").strip() or None
        open_access_pdf_url = str(open_access_raw.get("url") or "").strip() or None
        url = landing_page_url or open_access_pdf_url
        abstract = str(payload.get("abstract") or "").strip() or None
        doi = str(external_ids.get("DOI") or external_ids.get("ArXiv") or "").strip() or None
        return PaperRecord(
            paper_id=str(payload.get("paperId") or "").strip(),
            title=str(payload.get("title") or "").strip() or "Untitled paper",
            year=int(payload["year"]) if payload.get("year") is not None else None,
            venue=str(payload.get("venue") or "").strip() or None,
            url=url,
            landing_page_url=landing_page_url,
            open_access_pdf_url=open_access_pdf_url,
            is_open_access=bool(open_access_pdf_url),
            has_abstract=bool(abstract),
            has_fulltext=bool(open_access_pdf_url),
            doi=doi,
            abstract=abstract,
            authors=authors[:8],
            citation_count=int(payload["citationCount"]) if payload.get("citationCount") is not None else None,
            influential_citation_count=(
                int(payload["influentialCitationCount"]) if payload.get("influentialCitationCount") is not None else None
            ),
            source=source,
            snippet=snippet,
        )

    def search_papers(
        self,
        query: str,
        limit: int,
        year_from: int | None = None,
        year_to: int | None = None,
    ) -> list[PaperSearchHit]:
        params = {
            "query": str(query).strip(),
            "limit": max(1, min(int(limit), 100)),
            "fields": _FIELDS,
        }
        response = self._request_with_retry("GET", "/graph/v1/paper/search", params=params)
        payload = response.json()
        items = payload.get("data") if isinstance(payload, dict) else []
        hits: list[PaperSearchHit] = []
        for idx, raw in enumerate(items if isinstance(items, list) else [], start=1):
            if not isinstance(raw, dict):
                continue
            paper = self._paper_from_payload(raw)
            year = paper.year
            if year_from is not None and year is not None and year < int(year_from):
                continue
            if year_to is not None and year is not None and year > int(year_to):
                continue
            hits.append(PaperSearchHit(query=str(query).strip(), rank=idx, score_hint=None, paper=paper))
        return hits

    def get_paper(self, paper_id_or_doi: str) -> PaperRecord:
        ident = str(paper_id_or_doi or "").strip()
        if not ident:
            raise ValueError("paper_id_or_doi is required")
        last_not_found: httpx.HTTPStatusError | None = None
        for candidate in self._paper_identifier_candidates(ident):
            try:
                response = self._request_with_retry(
                    "GET",
                    f"/graph/v1/paper/{quote(candidate, safe='')}",
                    params={"fields": _FIELDS},
                )
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 404:
                    last_not_found = exc
                    continue
                raise
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Unexpected Semantic Scholar response payload")
            return self._paper_from_payload(payload)
        if last_not_found is not None:
            raise last_not_found
        raise RuntimeError("Semantic Scholar paper lookup failed without a usable identifier")

    def get_recommendations(
        self,
        seed_paper_ids: list[str],
        positive_ids: list[str] | None = None,
        negative_ids: list[str] | None = None,
        limit: int = 8,
    ) -> list[PaperSearchHit]:
        positives = [str(item).strip() for item in (positive_ids or seed_paper_ids) if str(item).strip()]
        negatives = [str(item).strip() for item in (negative_ids or []) if str(item).strip()]
        if not positives:
            return []
        response = self._request_with_retry(
            "POST",
            "/recommendations/v1/papers/",
            params={"fields": _FIELDS, "limit": max(1, min(int(limit), 100))},
            json={
                "positivePaperIds": positives,
                "negativePaperIds": negatives,
            },
        )
        payload = response.json()
        items = payload.get("recommendedPapers") if isinstance(payload, dict) else []
        hits: list[PaperSearchHit] = []
        for idx, raw in enumerate(items if isinstance(items, list) else [], start=1):
            if not isinstance(raw, dict):
                continue
            hits.append(
                PaperSearchHit(
                    query="recommendations",
                    rank=idx,
                    score_hint=None,
                    paper=self._paper_from_payload(raw, source="semantic_scholar_recommendation"),
                )
            )
        return hits


__all__ = ["SemanticScholarClient", "SemanticScholarRateLimitError"]
