from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import httpx

from .models import PaperRecord, PaperSearchHit

_SELECT_FIELDS = ",".join(
    [
        "id",
        "doi",
        "title",
        "display_name",
        "publication_year",
        "primary_location",
        "best_oa_location",
        "open_access",
        "authorships",
        "abstract_inverted_index",
        "cited_by_count",
        "is_paratext",
        "has_fulltext",
        "type_crossref",
        "primary_topic",
    ]
)


class OpenAlexClient:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        mailto: str | None = None,
        base_url: str = "https://api.openalex.org",
        timeout_s: float = 30.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENALEX_API_KEY", "").strip() or None
        self.mailto = mailto or os.getenv("OPENALEX_MAILTO", "").strip() or None
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _client(self) -> httpx.Client:
        headers = {"Accept": "application/json", "User-Agent": "CatMaster/1.0 literature"}
        return httpx.Client(timeout=self.timeout_s, headers=headers)

    def _params(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if self.api_key:
            params["api_key"] = self.api_key
        if self.mailto:
            params["mailto"] = self.mailto
        if extra:
            params.update(extra)
        return params

    @staticmethod
    def _reconstruct_abstract(value: Any) -> str | None:
        if not isinstance(value, dict) or not value:
            return None
        size = -1
        for positions in value.values():
            if isinstance(positions, list):
                for pos in positions:
                    try:
                        size = max(size, int(pos))
                    except Exception:
                        continue
        if size < 0:
            return None
        words = [""] * (size + 1)
        for token, positions in value.items():
            if not isinstance(token, str) or not isinstance(positions, list):
                continue
            for pos in positions:
                try:
                    idx = int(pos)
                except Exception:
                    continue
                if 0 <= idx < len(words):
                    words[idx] = token
        text = " ".join(part for part in words if part)
        return text.strip() or None

    @staticmethod
    def _authors(payload: dict[str, Any]) -> list[str]:
        authorships = payload.get("authorships")
        if not isinstance(authorships, list):
            return []
        out: list[str] = []
        for item in authorships:
            if not isinstance(item, dict):
                continue
            author = item.get("author") if isinstance(item.get("author"), dict) else {}
            name = str(author.get("display_name") or "").strip()
            if name:
                out.append(name)
        return out[:8]

    @staticmethod
    def _location_url(location: Any, key: str) -> str | None:
        if not isinstance(location, dict):
            return None
        value = str(location.get(key) or "").strip()
        return value or None

    @staticmethod
    def _venue(payload: dict[str, Any]) -> str | None:
        primary_location = payload.get("primary_location") if isinstance(payload.get("primary_location"), dict) else {}
        source = primary_location.get("source") if isinstance(primary_location.get("source"), dict) else {}
        venue = str(source.get("display_name") or "").strip()
        return venue or None

    @classmethod
    def _paper_from_payload(cls, payload: dict[str, Any]) -> PaperRecord:
        title = str(payload.get("display_name") or payload.get("title") or "").strip() or "Untitled paper"
        abstract = cls._reconstruct_abstract(payload.get("abstract_inverted_index"))
        open_access = payload.get("open_access") if isinstance(payload.get("open_access"), dict) else {}
        best_oa_location = payload.get("best_oa_location") if isinstance(payload.get("best_oa_location"), dict) else {}
        primary_location = payload.get("primary_location") if isinstance(payload.get("primary_location"), dict) else {}

        landing_page_url = (
            cls._location_url(best_oa_location, "landing_page_url")
            or cls._location_url(primary_location, "landing_page_url")
            or str(payload.get("id") or "").strip()
            or None
        )
        open_access_pdf_url = (
            cls._location_url(best_oa_location, "pdf_url")
            or cls._location_url(primary_location, "pdf_url")
            or None
        )
        doi = str(payload.get("doi") or "").strip() or None
        source_hint = (
            payload.get("primary_topic") if isinstance(payload.get("primary_topic"), dict) else {}
        )
        snippet = None
        if isinstance(source_hint, dict):
            snippet = str(source_hint.get("display_name") or "").strip() or None

        return PaperRecord(
            paper_id=str(payload.get("id") or "").strip(),
            title=title,
            year=int(payload["publication_year"]) if payload.get("publication_year") is not None else None,
            venue=cls._venue(payload),
            url=landing_page_url or open_access_pdf_url,
            doi=doi,
            abstract=abstract,
            authors=cls._authors(payload),
            citation_count=int(payload["cited_by_count"]) if payload.get("cited_by_count") is not None else None,
            influential_citation_count=None,
            landing_page_url=landing_page_url,
            open_access_pdf_url=open_access_pdf_url,
            is_open_access=bool(open_access.get("is_oa")) if open_access else bool(open_access_pdf_url),
            has_abstract=bool(abstract),
            has_fulltext=bool(payload.get("has_fulltext")) if payload.get("has_fulltext") is not None else None,
            source="openalex",
            snippet=snippet,
        )

    def search_works(self, query: str, limit: int) -> list[PaperSearchHit]:
        params = self._params(
            {
                "search": str(query).strip(),
                "per-page": max(1, min(int(limit), 200)),
                "select": _SELECT_FIELDS,
            }
        )
        with self._client() as client:
            response = client.get(f"{self.base_url}/works", params=params)
            response.raise_for_status()
            payload = response.json()
        items = payload.get("results") if isinstance(payload, dict) else []
        hits: list[PaperSearchHit] = []
        for idx, raw in enumerate(items if isinstance(items, list) else [], start=1):
            if not isinstance(raw, dict):
                continue
            hits.append(
                PaperSearchHit(
                    query=str(query).strip(),
                    rank=idx,
                    score_hint=None,
                    paper=self._paper_from_payload(raw),
                )
            )
        return hits

    def get_work(self, work_id_or_doi: str) -> PaperRecord:
        ident = str(work_id_or_doi or "").strip()
        if not ident:
            raise ValueError("work_id_or_doi is required")
        path_ident = ident if ident.startswith("https://openalex.org/") else quote(ident, safe="")
        params = self._params({"select": _SELECT_FIELDS})
        with self._client() as client:
            response = client.get(f"{self.base_url}/works/{path_ident}", params=params)
            response.raise_for_status()
            payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected OpenAlex response payload")
        return self._paper_from_payload(payload)


__all__ = ["OpenAlexClient"]
