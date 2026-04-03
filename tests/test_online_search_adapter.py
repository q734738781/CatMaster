from __future__ import annotations

import httpx
import pytest

from catmaster.runtime.literature import OnlineSearchAdapter


class _FakeResponse:
    def __init__(self, *, url: str, text: str, content_type: str = "text/html; charset=utf-8", status_code: int = 200):
        self.url = httpx.URL(url)
        self.text = text
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.request = httpx.Request("GET", url)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code} error",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request, headers=self.headers),
            )


class _FakeClient:
    def __init__(self, response: _FakeResponse):
        self._response = response

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url: str):
        _ = url
        return self._response


class _FakeTavilyClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.extract_calls: list[dict[str, object]] = []

    def search(self, query: str, **kwargs):
        self.calls.append({"query": query, **kwargs})
        return {
            "results": [
                {
                    "title": "CO adsorption on Fe surfaces",
                    "url": "https://example.org/fe-co",
                    "content": "Representative background summary from a public landing page.",
                },
                {
                    "title": "",
                    "url": "https://example.org/untitled",
                    "raw_content": "Fallback raw content should still produce a snippet.",
                },
            ]
        }

    def extract(self, urls, **kwargs):
        self.extract_calls.append({"urls": urls, **kwargs})
        return {
            "results": [
                {
                    "url": "https://example.org/paper",
                    "title": "Extracted paper page",
                    "description": "Extracted abstract-like summary.",
                    "raw_content": "Pt(111) hydrogen adsorption benchmark text from extracted page.",
                }
            ]
        }


def test_search_public_web_uses_tavily_results() -> None:
    adapter = OnlineSearchAdapter(tavily_api_key="test-key")
    fake_client = _FakeTavilyClient()
    adapter._tavily_client = fake_client

    result = adapter.search_public_web("CO adsorption Fe surfaces", max_results=4)

    assert fake_client.calls == [
        {
            "query": "CO adsorption Fe surfaces",
            "max_results": 4,
            "topic": "general",
            "search_depth": "advanced",
            "include_raw_content": False,
            "include_answer": False,
            "include_images": False,
            "include_usage": False,
            "timeout": 30.0,
        }
    ]
    assert [hit.title for hit in result.results] == [
        "CO adsorption on Fe surfaces",
        "https://example.org/untitled",
    ]
    assert result.results[0].url == "https://example.org/fe-co"
    assert result.results[1].snippet == "Fallback raw content should still produce a snippet."


def test_search_public_web_requires_tavily_key() -> None:
    adapter = OnlineSearchAdapter(tavily_api_key="")

    assert adapter.public_search_enabled() is False
    with pytest.raises(RuntimeError, match="TAVILY_API_KEY"):
        adapter.search_public_web("CO adsorption Fe surfaces")


def test_open_public_page_prefers_tavily_extract() -> None:
    adapter = OnlineSearchAdapter(tavily_api_key="test-key")
    fake_client = _FakeTavilyClient()
    adapter._tavily_client = fake_client

    page = adapter.open_public_page("https://example.org/paper", max_chars=2000)

    assert fake_client.extract_calls == [
        {
            "urls": "https://example.org/paper",
            "extract_depth": "advanced",
            "format": "text",
            "include_images": False,
            "include_usage": False,
            "timeout": 30.0,
        }
    ]
    assert page.title == "Extracted paper page"
    assert page.description == "Extracted abstract-like summary."
    assert "Pt(111) hydrogen adsorption benchmark text" in page.text


def test_open_public_page_falls_back_to_http_extracts_title_description_and_text(monkeypatch) -> None:
    html = """
    <html>
      <head>
        <title>NIH Example Abstract</title>
        <meta name="description" content="Short abstract-like description." />
      </head>
      <body>
        <main>
          <h1>Fe(110) CO study</h1>
          <p>This page summarizes CO adsorption on Fe(110).</p>
        </main>
      </body>
    </html>
    """
    adapter = OnlineSearchAdapter()
    monkeypatch.setattr(
        adapter,
        "_http_client",
        lambda: _FakeClient(_FakeResponse(url="https://example.org/paper", text=html)),
    )

    page = adapter.open_public_page("https://example.org/paper", max_chars=2000)

    assert page.title == "NIH Example Abstract"
    assert page.description == "Short abstract-like description."
    assert "Fe(110) CO study" in page.text
    assert "CO adsorption on Fe(110)" in page.text


def test_open_public_page_falls_back_when_tavily_extract_returns_empty(monkeypatch) -> None:
    html = """
    <html><head><title>Fallback title</title></head><body><p>Fallback body text.</p></body></html>
    """
    adapter = OnlineSearchAdapter(tavily_api_key="test-key")
    fake_client = _FakeTavilyClient()
    fake_client.extract = lambda urls, **kwargs: {"results": [{"url": str(urls), "title": "Empty extract", "raw_content": ""}]}
    adapter._tavily_client = fake_client
    monkeypatch.setattr(
        adapter,
        "_http_client",
        lambda: _FakeClient(_FakeResponse(url="https://example.org/paper", text=html)),
    )

    page = adapter.open_public_page("https://example.org/paper", max_chars=2000)

    assert page.title == "Fallback title"
    assert "Fallback body text" in page.text


def test_find_in_page_returns_context_snippets(monkeypatch) -> None:
    adapter = OnlineSearchAdapter()
    monkeypatch.setattr(
        adapter,
        "open_public_page",
        lambda url, max_chars=12000: type(
            "_Page",
            (),
            {
                "requested_url": url,
                "final_url": url,
                "text": "CO adsorption on Fe(110) is discussed here. Dispersion is enabled explicitly.",
            },
        )(),
    )

    result = adapter.find_in_page("https://example.org/paper", "dispersion", max_matches=3, context_chars=30)

    assert result.total_matches == 1
    assert len(result.matches) == 1
    assert "Dispersion is enabled explicitly" in result.matches[0].snippet
