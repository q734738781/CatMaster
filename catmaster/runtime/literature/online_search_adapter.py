from __future__ import annotations

from dataclasses import replace
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.tool_output_adapter import content_to_text

from .models import FindInPageResult, InPageMatch, PublicPageSnapshot, PublicWebHit, PublicWebSearchResult


class _WebSearchHitModel(BaseModel):
    title: str = Field(...)
    url: str | None = Field(None)
    snippet: str = Field(...)


class _WebSearchResponseModel(BaseModel):
    results: list[_WebSearchHitModel] = Field(default_factory=list)


class OnlineSearchAdapter:
    def __init__(self, *, model_override: str | None = None) -> None:
        self.model_override = str(model_override or "").strip() or None

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

    def _resolve_model_config(self):
        profile = LLMProfile.from_env_or_file()
        base_cfg = profile.config_for_role("literature_web_search")
        if self.model_override:
            if self.model_override in profile.models:
                return profile.models[self.model_override]
            return replace(base_cfg, model=self.model_override)
        model_name = str(base_cfg.model or "")
        if ":online" not in model_name and "deep-research" not in model_name:
            return replace(base_cfg, model=f"{model_name}:online")
        return base_cfg

    def search_public_web(self, query: str, max_results: int = 5) -> PublicWebSearchResult:
        cfg = self._resolve_model_config()
        model = build_chat_model(cfg).with_structured_output(_WebSearchResponseModel)
        prompt = (
            "Find a few public web results that help answer the literature query. "
            "Prefer scholarly landing pages, lab/project pages, or public summaries that add context beyond paper metadata. "
            f"Return at most {max(1, int(max_results))} results."
        )
        response = model.invoke(
            [
                SystemMessage(content=prompt),
                HumanMessage(content=f"Query: {str(query).strip()}"),
            ]
        )
        parsed = response if isinstance(response, _WebSearchResponseModel) else _WebSearchResponseModel.model_validate(response)
        return PublicWebSearchResult(
            results=[
                PublicWebHit(
                    title=str(item.title).strip() or "Untitled result",
                    url=str(item.url).strip() or None if item.url is not None else None,
                    snippet=str(item.snippet).strip() or content_to_text(item),
                )
                for item in parsed.results
            ]
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
