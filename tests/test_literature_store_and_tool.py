from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from tavily.errors import InvalidAPIKeyError, UsageLimitExceededError

from catmaster.runtime.literature import PaperRecord, SemanticScholarRateLimitError
from catmaster.runtime.literature.citations import finalize_citations
from catmaster.runtime.literature.corpus import ingest_literature_files, query_literature_corpus
from catmaster.runtime.literature.tools import (
    _reset_public_web_circuits_for_tests,
    get_openalex_record,
    open_public_page,
    search_openalex,
    search_public_web,
    search_semantic_scholar,
    web_search,
)
from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.tools.base import ensure_project_space_layout, workspace_scope
from catmaster.tools.registry import get_tool_registry


def test_literature_corpus_ingest_query_and_cache(tmp_path: Path) -> None:
    project = tmp_path / "project"
    layout = ensure_project_space_layout(project)
    source = layout["files_root"] / "papers" / "her.txt"
    source.parent.mkdir(parents=True)
    source.write_text(
        "Hydrogen evolution catalysts include platinum, transition-metal sulfides, "
        "phosphides, carbides, and nitrides. Platinum has near-thermoneutral hydrogen adsorption.",
        encoding="utf-8",
    )

    with workspace_scope(project):
        first_content, first_artifact = ingest_literature_files(
            {"paths": ["papers/her.txt"], "doi_by_path": {}}
        )
        second_content, _ = ingest_literature_files({"paths": ["papers/her.txt"]})
        query_content, query_artifact = query_literature_corpus(
            {"query": "platinum hydrogen adsorption", "top_k": 3}
        )

    assert "ingested: papers/her.txt" in first_content
    assert "cached: papers/her.txt" in second_content
    assert "p.1" in query_content
    assert "Platinum" in query_content
    assert first_artifact["data"]["manifest_path"] == "notes/literature/acquisition_manifest.json"
    assert query_artifact["data"]["evidence"][0]["source_path"] == "papers/her.txt"
    assert (layout["metadata_root"] / "literature" / "corpus.sqlite").is_file()


def test_literature_corpus_accepts_jats_and_keeps_successes_when_one_file_fails(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    layout = ensure_project_space_layout(project)
    papers = layout["files_root"] / "papers"
    papers.mkdir(parents=True)
    (papers / "operando.xml").write_text(
        """
        <article>
          <front><article-meta><title-group>
            <article-title>Operando reconstruction of catalyst A</article-title>
          </title-group></article-meta></front>
          <body><sec><title>Results</title><p>
            Operando XAS reveals a reversible coordination change under reaction conditions.
          </p></sec></body>
        </article>
        """,
        encoding="utf-8",
    )
    (papers / "binary.dat").write_bytes(b"\x00\x01not-readable-full-text")

    with workspace_scope(project):
        content, artifact = ingest_literature_files(
            {"paths": ["papers/operando.xml", "papers/binary.dat"]}
        )
        query_content, query_artifact = query_literature_corpus(
            {"query": "reversible coordination change", "top_k": 3}
        )

    assert artifact["data"]["status"] == "partial"
    assert [item["path"] for item in artifact["data"]["documents"]] == [
        "papers/operando.xml"
    ]
    assert artifact["data"]["documents"][0]["title"] == (
        "Operando reconstruction of catalyst A"
    )
    assert [item["path"] for item in artifact["data"]["errors"]] == [
        "papers/binary.dat"
    ]
    assert "1 document(s) processed, 1 skipped" in content
    assert "coordination change" in query_content
    assert query_artifact["data"]["evidence"][0]["source_path"] == (
        "papers/operando.xml"
    )


def test_finalize_citations_deduplicates_and_writes_batch_artifacts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    layout = ensure_project_space_layout(project)

    def _fake_resolve(doi: str):
        return (
            {
                "title": "A catalyst paper",
                "authors": ["A. Author"],
                "venue": "Journal of Catalysis",
                "year": 2025,
                "doi": doi,
                "url": f"https://doi.org/{doi}",
                "metadata_source": "crossref",
            },
            "",
        )

    monkeypatch.setattr("catmaster.runtime.literature.citations._resolve", _fake_resolve)
    with workspace_scope(project):
        content, artifact = finalize_citations(
            {
                "items": [
                    "10.1234/example.1",
                    "https://doi.org/10.1234/example.1",
                    "not-a-doi",
                ],
                "output_stem": "her-review",
            }
        )

    assert "Finalized 1 unique citation" in content
    assert artifact["data"]["resolved_count"] == 1
    assert artifact["data"]["unresolved_count"] == 1
    assert artifact["data"]["deduplicated_input_count"] == 1
    assert (layout["files_root"] / "notes" / "literature" / "her-review.bib").is_file()
    assert (layout["files_root"] / "notes" / "literature" / "her-review.json").is_file()


def test_crossref_finalizer_retries_rate_limit_and_cleans_title(monkeypatch) -> None:
    import httpx

    request = httpx.Request("GET", "https://api.crossref.org/works/10.1234/example")
    responses = [
        httpx.Response(429, headers={"Retry-After": "0.5"}, request=request),
        httpx.Response(
            200,
            request=request,
            json={
                "message": {
                    "title": ["Hydrogen <sub>2</sub> evolution"],
                    "container-title": ["Catalysis Journal"],
                    "published-online": {"date-parts": [[2025, 1, 2]]},
                    "author": [{"given": "A.", "family": "Author"}],
                    "DOI": "10.1234/example",
                }
            },
        ),
    ]
    sleeps = []

    class _Client:
        def __init__(self, *args, **kwargs):
            _ = (args, kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            _ = args

        def get(self, *args, **kwargs):
            _ = (args, kwargs)
            return responses.pop(0)

    monkeypatch.setattr("catmaster.runtime.literature.citations.httpx.Client", _Client)
    monkeypatch.setattr("catmaster.runtime.literature.citations.time.sleep", sleeps.append)

    from catmaster.runtime.literature.citations import _crossref_record

    record = _crossref_record("10.1234/example")

    assert record["title"] == "Hydrogen 2 evolution"
    assert record["year"] == 2025
    assert sleeps == [0.5]


def test_literature_agent_visible_schemas_are_non_nullable() -> None:
    registry = get_tool_registry()
    tools = registry.as_openai_tools(
        allowlist=["web_search", "ingest_literature_files", "query_literature_corpus", "finalize_citations"]
    )
    serialized = json.dumps(tools)

    assert '"type": "null"' not in serialized
    assert '"default": null' not in serialized
    assert {tool["name"] for tool in tools} == {
        "web_search",
        "ingest_literature_files",
        "query_literature_corpus",
        "finalize_citations",
    }


def test_direct_search_openalex_tool_returns_normalized_json(monkeypatch) -> None:
    class _FakeOpenAlex:
        def search_works(self, query: str, limit: int):
            assert query == "CO adsorption Fe(110)"
            assert limit == 3
            return [
                type(
                    "_Hit",
                    (),
                    {
                        "paper": PaperRecord(
                            paper_id="https://openalex.org/W1",
                            title="OpenAlex result",
                            year=2024,
                            source="openalex",
                        )
                    },
                )()
            ]

    monkeypatch.setattr(
        "catmaster.runtime.literature.tools._literature_components",
        lambda: (object(), _FakeOpenAlex(), object(), object()),
    )
    content, artifact = search_openalex({"query": "CO adsorption Fe(110)", "limit": 3})
    payload = json.loads(content)

    assert payload["count"] == 1
    assert payload["papers"][0]["title"] == "OpenAlex result"
    assert artifact["tool_name"] == "search_openalex"


def test_direct_web_search_tool_returns_compact_hits(monkeypatch) -> None:
    class _FakeWeb:
        def search_public_web(self, query: str, max_results: int = 5):
            assert query == "CO adsorption Fe surfaces"
            assert max_results == 2
            return type(
                "_Result",
                (),
                {
                    "results": [
                        type(
                            "_Hit",
                            (),
                            {
                                "model_dump": lambda self: {
                                    "title": "Result",
                                    "url": "https://example.org",
                                    "snippet": "A" * 600,
                                    "source": "public_web",
                                }
                            },
                        )()
                    ]
                },
            )()

    monkeypatch.setattr(
        "catmaster.runtime.literature.tools._literature_components",
        lambda: (object(), object(), object(), _FakeWeb()),
    )
    content, artifact = web_search({"query": "CO adsorption Fe surfaces", "max_results": 2})

    assert "Top results:" in content
    assert "https://example.org" in content
    assert "A" * 600 in content
    assert artifact["data"]["count"] == 1


def test_web_search_falls_back_and_skips_tavily_after_quota_failure_in_same_run(
    monkeypatch,
) -> None:
    class _QuotaWeb:
        def __init__(self) -> None:
            self.calls = 0

        def search_public_web(self, query: str, max_results: int = 5):
            _ = (query, max_results)
            self.calls += 1
            raise UsageLimitExceededError("monthly usage limit exceeded")

    class _OpenAlex:
        api_key = "configured"

        def search_works(self, query: str, limit: int):
            assert query == "Pt CeO2 CO oxidation"
            assert limit == 3
            return [
                type(
                    "_Hit",
                    (),
                    {
                        "paper": PaperRecord(
                            paper_id="https://openalex.org/W1",
                            title="Dynamic Pt sites on ceria",
                            year=2025,
                            url="https://example.org/pt-ceria",
                            abstract="Operando evidence for dynamic Pt sites.",
                            source="openalex",
                        )
                    },
                )()
            ]

    web = _QuotaWeb()
    profile = SimpleNamespace(
        literature=SimpleNamespace(public_web_on_search_failure=True)
    )
    monkeypatch.setattr(
        "catmaster.runtime.literature.tools._literature_components",
        lambda: (profile, _OpenAlex(), object(), web),
    )
    _reset_public_web_circuits_for_tests()

    with toolcall_context(
        "search-1",
        context={"run_id": "run-quota", "search_scope": "run-quota"},
    ):
        first_content, first_artifact = web_search(
            {"query": "Pt CeO2 CO oxidation", "max_results": 3}
        )
    with toolcall_context(
        "search-2",
        context={"run_id": "run-quota", "search_scope": "run-quota"},
    ):
        second_content, second_artifact = web_search(
            {"query": "Pt CeO2 CO oxidation", "max_results": 3}
        )

    assert web.calls == 1
    assert "Dynamic Pt sites on ceria" in first_content
    assert "Dynamic Pt sites on ceria" in second_content
    for artifact in (first_artifact, second_artifact):
        data = artifact["data"]
        assert data["status"] == "degraded"
        assert data["backend"] == "openalex"
        assert data["degraded_from"] == "tavily"
        assert data["failure_category"] == "quota_exhausted"
        assert data["retryable"] is False
        assert data["circuit_open"] is True


def test_web_search_classifies_auth_failure_without_exposing_error_text_when_fallback_disabled(
    monkeypatch,
) -> None:
    class _AuthWeb:
        def search_public_web(self, query: str, max_results: int = 5):
            _ = (query, max_results)
            raise InvalidAPIKeyError("sensitive provider detail")

    profile = SimpleNamespace(
        literature=SimpleNamespace(public_web_on_search_failure=False)
    )
    monkeypatch.setattr(
        "catmaster.runtime.literature.tools._literature_components",
        lambda: (profile, object(), object(), _AuthWeb()),
    )
    _reset_public_web_circuits_for_tests()

    with toolcall_context(
        "search-auth",
        context={"run_id": "run-auth", "search_scope": "run-auth"},
    ):
        content, artifact = web_search({"query": "test query"})
    data = json.loads(content)

    assert data["status"] == "authentication_failed"
    assert data["backend"] == "tavily"
    assert data["retryable"] is False
    assert data["circuit_open"] is True
    assert "sensitive provider detail" not in content
    assert artifact["data"] == data


def test_search_semantic_scholar_tool_soft_fails_on_rate_limit(monkeypatch) -> None:
    class _RateLimitedClient:
        def search_papers(self, query: str, limit: int = 10, year_from=None, year_to=None):
            _ = (query, limit, year_from, year_to)
            raise SemanticScholarRateLimitError(attempts=5, wait_seconds=15.0)

    monkeypatch.setattr(
        "catmaster.runtime.literature.tools._literature_components",
        lambda: (object(), object(), _RateLimitedClient(), object()),
    )
    content, artifact = search_semantic_scholar({"query": "CO adsorption Fe(110)"})
    payload = json.loads(content)

    assert payload["status"] == "rate_limited"
    assert payload["attempts"] == 5
    assert artifact["tool_name"] == "search_semantic_scholar"


def test_metadata_and_page_tools_keep_soft_failure_contract(monkeypatch) -> None:
    class _MissingOpenAlexClient:
        def get_work(self, ident: str):
            import httpx

            request = httpx.Request("GET", "https://api.openalex.org/works/missing")
            response = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError("404 error", request=request, response=response)

    class _FakeWeb:
        def open_public_page(self, url: str, max_chars: int = 12000):
            _ = (url, max_chars)
            raise ValueError("Only public http(s) URLs are supported")

    monkeypatch.setattr(
        "catmaster.runtime.literature.tools._literature_components",
        lambda: (object(), _MissingOpenAlexClient(), object(), _FakeWeb()),
    )
    record_content, _ = get_openalex_record({"work_id_or_doi": "10.1234/missing"})
    page_content, _ = open_public_page({"url": "file:///tmp/secret.txt"})

    assert json.loads(record_content)["status"] == "not_found"
    assert json.loads(page_content)["status"] == "invalid_request"


def test_search_public_web_alias_uses_web_search(monkeypatch) -> None:
    monkeypatch.setattr(
        "catmaster.runtime.literature.tools.web_search",
        lambda payload: ("alias ok", {"tool_name": "web_search", "data": payload}),
    )
    content, artifact = search_public_web({"query": "alias"})

    assert content == "alias ok"
    assert artifact["tool_name"] == "web_search"
