from __future__ import annotations

import httpx

from catmaster.runtime.literature import OpenAlexClient, SemanticScholarClient, SemanticScholarRateLimitError


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


def test_semanticscholar_client_raises_rate_limit_after_retry_budget_exhausted(monkeypatch) -> None:
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
    except SemanticScholarRateLimitError as exc:
        assert exc.attempts == 2
        assert exc.wait_seconds == 0.0
    else:
        raise AssertionError("Expected SemanticScholarRateLimitError for exhausted retry budget")

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


def test_openalex_client_normalizes_doi_identifier() -> None:
    observed = {}

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def get(self, url: str, params=None):
            observed["url"] = url
            observed["params"] = params or {}
            return _FakeResponse(
                200,
                {
                    "id": "https://openalex.org/W1",
                    "display_name": "Normalized DOI result",
                    "publication_year": 2024,
                    "authorships": [],
                },
            )

    client = OpenAlexClient()
    client._client = lambda: _FakeClient()  # type: ignore[method-assign]

    paper = client.get_work("10.1016/j.susc.2017.09.002")

    assert "https%3A%2F%2Fdoi.org%2F10.1016%2Fj.susc.2017.09.002" in observed["url"]
    assert paper.title == "Normalized DOI result"


def test_semanticscholar_client_tries_doi_prefix_before_plain_doi(monkeypatch) -> None:
    observed: list[str] = []

    class _FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def request(self, method: str, url: str, **kwargs):
            _ = method, kwargs
            observed.append(url)
            if "DOI%3A10.1016%2Fj.susc.2017.09.002" in url:
                return _FakeResponse(
                    200,
                    {
                        "paperId": "p1",
                        "title": "Semantic Scholar DOI result",
                        "year": 2024,
                        "authors": [],
                    },
                )
            return _FakeResponse(404, {})

    client = SemanticScholarClient(retry_429_attempts=0, retry_429_wait_seconds=0.0)
    monkeypatch.setattr(client, "_client", lambda: _FakeClient())

    paper = client.get_paper("10.1016/j.susc.2017.09.002")

    assert "DOI%3A10.1016%2Fj.susc.2017.09.002" in observed[0]
    assert paper.title == "Semantic Scholar DOI result"
