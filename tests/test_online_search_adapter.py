from __future__ import annotations

import httpx

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


def test_open_public_page_extracts_title_description_and_text(monkeypatch) -> None:
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
