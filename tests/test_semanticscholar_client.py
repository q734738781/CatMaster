from __future__ import annotations

import httpx

from catmaster.runtime.literature import OpenAlexClient, SemanticScholarClient


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict, headers: dict[str, str] | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.request = httpx.Request("GET", "https://api.semanticscholar.org/test")

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"{self.status_code} error",
                request=self.request,
                response=httpx.Response(self.status_code, request=self.request, headers=self.headers),
            )

    def json(self):
        return self._payload


def test_semanticscholar_client_retries_429_then_succeeds(monkeypatch) -> None:
    responses = [
        _FakeResponse(429, {}, headers={"Retry-After": "0"}),
        _FakeResponse(429, {}, headers={"Retry-After": "0"}),
        _FakeResponse(
            200,
            {
                "data": [
                    {
                        "paperId": "p1",
                        "title": "CO adsorption on Fe(110)",
                        "year": 2024,
                        "authors": [],
                    }
                ]
            },
        ),
    ]
    observed = {"calls": 0, "sleeps": 0}

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, method: str, url: str, **kwargs):
            _ = method, url, kwargs
            observed["calls"] += 1
            return responses.pop(0)

    monkeypatch.setattr("catmaster.runtime.literature.semanticscholar_client.time.sleep", lambda _: observed.__setitem__("sleeps", observed["sleeps"] + 1))

    client = SemanticScholarClient(retry_429_attempts=3, retry_429_wait_seconds=0.0)
    monkeypatch.setattr(client, "_client", lambda: _FakeClient())

    hits = client.search_papers("CO adsorption Fe(110)", limit=3)

    assert len(hits) == 1
    assert hits[0].paper.title == "CO adsorption on Fe(110)"
    assert observed["calls"] == 3
    assert observed["sleeps"] == 0


def test_semanticscholar_client_raises_after_retry_budget_exhausted(monkeypatch) -> None:
    responses = [
        _FakeResponse(429, {}, headers={"Retry-After": "0"}),
        _FakeResponse(429, {}, headers={"Retry-After": "0"}),
    ]
    observed = {"calls": 0}

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, method: str, url: str, **kwargs):
            _ = method, url, kwargs
            observed["calls"] += 1
            return responses.pop(0)

    client = SemanticScholarClient(retry_429_attempts=1, retry_429_wait_seconds=0.0)
    monkeypatch.setattr(client, "_client", lambda: _FakeClient())

    try:
        client.search_papers("CO adsorption Fe(110)", limit=3)
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 429
    else:
        raise AssertionError("Expected HTTPStatusError for exhausted retry budget")

    assert observed["calls"] == 2


def test_semanticscholar_client_preserves_open_access_and_abstract_fields() -> None:
    paper = SemanticScholarClient._paper_from_payload(
        {
            "paperId": "p1",
            "title": "CO adsorption on Fe(110)",
            "year": 2024,
            "abstract": "Short abstract text.",
            "url": "https://example.org/landing",
            "openAccessPdf": {"url": "https://example.org/paper.pdf"},
            "authors": [],
        }
    )

    assert paper.landing_page_url == "https://example.org/landing"
    assert paper.open_access_pdf_url == "https://example.org/paper.pdf"
    assert paper.is_open_access is True
    assert paper.has_abstract is True
    assert paper.has_fulltext is True


def test_openalex_client_reconstructs_abstract_and_open_access_fields() -> None:
    paper = OpenAlexClient._paper_from_payload(
        {
            "id": "https://openalex.org/W123",
            "display_name": "OpenAlex paper",
            "publication_year": 2025,
            "doi": "https://doi.org/10.1234/example",
            "cited_by_count": 7,
            "has_fulltext": True,
            "authorships": [{"author": {"display_name": "A. Author"}}],
            "abstract_inverted_index": {"CO": [0], "adsorption": [1], "study": [2]},
            "open_access": {"is_oa": True},
            "best_oa_location": {
                "landing_page_url": "https://example.org/landing",
                "pdf_url": "https://example.org/paper.pdf",
            },
            "primary_location": {
                "source": {"display_name": "J. Catal."},
            },
            "primary_topic": {"display_name": "Catalysis"},
        }
    )

    assert paper.paper_id == "https://openalex.org/W123"
    assert paper.abstract == "CO adsorption study"
    assert paper.venue == "J. Catal."
    assert paper.landing_page_url == "https://example.org/landing"
    assert paper.open_access_pdf_url == "https://example.org/paper.pdf"
    assert paper.is_open_access is True
    assert paper.has_abstract is True
    assert paper.has_fulltext is True
