from __future__ import annotations

import os
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from tavily import TavilyClient

from .models import FindInPageResult, InPageMatch, PublicPageSnapshot, PublicWebHit, PublicWebSearchResult


class OnlineSearchAdapter:
    def __init__(
        self,
        *,
        tavily_api_key: str | None = None,
        search_depth: str = "advanced",
        topic: str = "general",
    ) -> None:
        api_key = str(tavily_api_key if tavily_api_key is not None else os.environ.get("TAVILY_API_KEY", "")).strip()
        self._tavily_client = TavilyClient(api_key=api_key) if api_key else None
        self.search_depth = str(search_depth or "advanced").strip().lower() or "advanced"
        self.topic = str(topic or "general").strip().lower() or "general"

    @staticmethod
    def _normalize_public_url(url: str) -> str:
        text = str(url or "").strip()
        parsed = urlparse(text)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("Only public http(s) URLs are supported")
        return text

    @staticmethod
    def _http_client() -> httpx.Client:
        return httpx.Client(
            timeout=30.0,
            follow_redirects=True,
            headers={
                "User-Agent": "CatMaster/1.0 literature-page-fetch",
                "Accept": "text/html, text/plain, application/xhtml+xml;q=0.9, */*;q=0.1",
            },
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    def public_search_enabled(self) -> bool:
        return self._tavily_client is not None

    def _require_tavily_client(self) -> TavilyClient:
        if self._tavily_client is None:
            raise RuntimeError("TAVILY_API_KEY is required for public web search.")
        return self._tavily_client

    @classmethod
    def _normalize_tavily_hit(cls, payload: Any) -> PublicWebHit:
        data = payload if isinstance(payload, dict) else {}
        title = cls._clean_text(data.get("title") or "")
        url = cls._clean_text(data.get("url") or "") or None
        snippet = cls._clean_text(data.get("content") or data.get("raw_content") or "")
        if not title:
            title = url or "Untitled result"
        if not snippet:
            snippet = title
        return PublicWebHit(title=title, url=url, snippet=snippet)

    @classmethod
    def _extract_page_text(cls, html_text: str) -> tuple[str | None, str | None, str]:
        soup = BeautifulSoup(html_text, "html.parser")
        title = cls._clean_text(soup.title.get_text(" ", strip=True)) if soup.title else None

        description = None
        for attrs in (
            {"name": "description"},
            {"property": "og:description"},
            {"name": "dc.description"},
            {"name": "citation_abstract"},
        ):
            tag = soup.find("meta", attrs=attrs)
            if tag is None:
                continue
            content = cls._clean_text(tag.get("content") or "")
            if content:
                description = content
                break

        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        body = soup.body or soup
        text = cls._clean_text(body.get_text("\n", strip=True))
        return title, description, text

    def search_public_web(self, query: str, max_results: int = 5) -> PublicWebSearchResult:
        normalized_query = self._clean_text(query)
        if not normalized_query:
            raise ValueError("query is required")
        response = self._require_tavily_client().search(
            normalized_query,
            max_results=max(1, int(max_results or 1)),
            topic=self.topic,
            search_depth=self.search_depth,  # type: ignore[arg-type]
            include_raw_content=False,
            include_answer=False,
            include_images=False,
            include_usage=False,
            timeout=30.0,
        )
        raw_results = response.get("results") if isinstance(response, dict) else []
        return PublicWebSearchResult(
            results=[self._normalize_tavily_hit(item) for item in raw_results]
        )

    def open_public_page(self, url: str, max_chars: int = 12000) -> PublicPageSnapshot:
        normalized_url = self._normalize_public_url(url)
        with self._http_client() as client:
            response = client.get(normalized_url)
        response.raise_for_status()
        content_type = str(response.headers.get("content-type") or "").strip() or None
        text: str
        title: str | None = None
        description: str | None = None
        if "html" in (content_type or "").lower():
            title, description, text = self._extract_page_text(response.text)
        else:
            text = self._clean_text(response.text)
        limit = max(500, int(max_chars or 0))
        return PublicPageSnapshot(
            requested_url=normalized_url,
            final_url=str(response.url),
            status_code=int(response.status_code),
            content_type=content_type,
            title=title,
            description=description,
            text=text[:limit],
        )

    def find_in_page(
        self,
        url: str,
        pattern: str,
        *,
        max_matches: int = 5,
        context_chars: int = 240,
        page_max_chars: int = 20000,
    ) -> FindInPageResult:
        needle = self._clean_text(pattern)
        if not needle:
            raise ValueError("pattern is required")
        page = self.open_public_page(url, max_chars=page_max_chars)
        haystack = page.text
        lower_haystack = haystack.lower()
        lower_needle = needle.lower()
        matches: list[InPageMatch] = []
        start = 0
        total = 0
        max_items = max(1, int(max_matches or 1))
        context = max(40, int(context_chars or 40))
        while True:
            idx = lower_haystack.find(lower_needle, start)
            if idx < 0:
                break
            total += 1
            end = idx + len(needle)
            if len(matches) < max_items:
                snippet_start = max(0, idx - context)
                snippet_end = min(len(haystack), end + context)
                matches.append(
                    InPageMatch(
                        pattern=needle,
                        start_char=idx,
                        end_char=end,
                        snippet=haystack[snippet_start:snippet_end].strip(),
                    )
                )
            start = end
        return FindInPageResult(
            requested_url=page.requested_url,
            final_url=page.final_url,
            pattern=needle,
            total_matches=total,
            matches=matches,
        )


__all__ = ["OnlineSearchAdapter"]
