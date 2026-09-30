from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage
from tavily.errors import InvalidAPIKeyError, UsageLimitExceededError

from catmaster.runtime.literature import (
    PaperRecord,
    PublicPageSnapshot,
    SemanticScholarRateLimitError,
)
from catmaster.runtime.literature.citations import finalize_citations
from catmaster.runtime.literature.corpus import ingest_literature_files, query_literature_corpus
from catmaster.runtime.literature.tools import (
    _reset_public_web_circuits_for_tests,
    find_in_page,
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


def test_literature_corpus_ingest_query_and_explicit_reingest(tmp_path: Path) -> None:
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
            {"query": "platinum hydrogen adsorption", "page_size": 3}
        )

    assert "ingested: papers/her.txt" in first_content
    assert "ingested: papers/her.txt" in second_content
    assert "p.1" in query_content
    assert "Platinum" in query_content
    assert first_artifact["data"]["manifest_path"] == "notes/literature/acquisition_manifest.json"
    assert query_artifact["data"]["evidence"][0]["source_path"] == "papers/her.txt"
    manifest = json.loads(
        (layout["files_root"] / "notes/literature/acquisition_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "file_hash" not in manifest[0]
    assert "document_id" not in first_artifact["data"]["documents"][0]
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
            {"query": "reversible coordination change", "page_size": 3}
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


def test_literature_corpus_migrates_legacy_hash_schema_without_losing_text(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    layout = ensure_project_space_layout(project)
    database = layout["metadata_root"] / "literature" / "corpus.sqlite"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE documents (document_id TEXT PRIMARY KEY, source_path TEXT NOT NULL, "
            "file_hash TEXT NOT NULL UNIQUE, doi TEXT NOT NULL DEFAULT '', title TEXT NOT NULL DEFAULT '', "
            "page_count INTEGER NOT NULL DEFAULT 0, ingested_at TEXT NOT NULL)"
        )
        connection.execute(
            "CREATE VIRTUAL TABLE chunks USING fts5(document_id UNINDEXED, source_path UNINDEXED, "
            "doi UNINDEXED, page UNINDEXED, section UNINDEXED, text)"
        )
        connection.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "doc-legacy",
                "papers/legacy.txt",
                "legacy-digest",
                "",
                "Legacy source",
                1,
                "2026-01-01T00:00:00+00:00",
            ),
        )
        connection.execute(
            "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
            (
                "doc-legacy",
                "papers/legacy.txt",
                "",
                1,
                "page 1",
                "legacy operando evidence",
            ),
        )

    with workspace_scope(project):
        _content, artifact = query_literature_corpus(
            {"query": "legacy operando", "page_size": 5}
        )

    assert artifact["data"]["evidence"][0]["source_path"] == "papers/legacy.txt"
    with sqlite3.connect(database) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(documents)")
        }
        chunk_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(chunks)")
        }
    assert "file_hash" not in columns
    assert "document_id" not in columns
    assert "document_id" not in chunk_columns


def test_literature_corpus_offset_reaches_every_partial_locator_once(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    layout = ensure_project_space_layout(project)
    source = layout["files_root"] / "papers" / "long-operando.txt"
    source.parent.mkdir(parents=True)
    source.write_text(
        "\n".join(
            f"operando-cursor-marker section {index} " + ("measurement detail " * 80)
            for index in range(12)
        ),
        encoding="utf-8",
    )

    with workspace_scope(project):
        ingest_literature_files({"paths": ["papers/long-operando.txt"]})
        offset = 0
        locators: list[tuple[str, int, str, str]] = []
        total = 0
        while True:
            _content, artifact = query_literature_corpus(
                {
                    "query": "operando cursor marker",
                    "page_size": 1,
                    "offset": offset,
                }
            )
            data = artifact["data"]
            assert set(data) == {
                "query",
                "partial",
                "evidence",
                "total_count",
                "next_offset",
            }
            total = data["total_count"]
            assert data["partial"] is True
            assert "score" not in json.dumps(data["evidence"])
            assert all("chunk_rowid" not in item for item in data["evidence"])
            locators.extend(
                (
                    str(item["source_path"]),
                    int(item["page"]),
                    str(item["section"]),
                    str(item["snippet"]),
                )
                for item in data["evidence"]
            )
            offset = data["next_offset"]
            if not offset:
                break

    assert total > 1
    assert len(locators) == total
    assert len(set(locators)) == total


def test_literature_corpus_query_does_not_discard_later_terms(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    layout = ensure_project_space_layout(project)
    source = layout["files_root"] / "papers" / "late-query-term.txt"
    source.parent.mkdir(parents=True)
    source.write_text(
        "A measurement reports the seventeenthmarker under controlled conditions.",
        encoding="utf-8",
    )
    earlier_terms = " ".join(f"missingterm{index}" for index in range(16))

    with workspace_scope(project):
        ingest_literature_files({"paths": ["papers/late-query-term.txt"]})
        _content, artifact = query_literature_corpus(
            {
                "query": f"{earlier_terms} seventeenthmarker",
                "page_size": 1,
            }
        )

    assert artifact["data"]["total_count"] == 1
    assert artifact["data"]["evidence"][0]["source_path"] == (
        "papers/late-query-term.txt"
    )


def test_finalize_citations_deduplicates_and_defaults_to_one_bib(
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
    output_root = layout["files_root"] / "notes" / "literature"
    assert [path.name for path in output_root.iterdir()] == ["her-review.bib"]
    assert artifact["data"]["files"] == ["notes/literature/her-review.bib"]
    assert "not-a-doi" in content
    assert "No DOI found" in content


@pytest.mark.parametrize("output_format", ["bib", "json", "md"])
def test_final_citation_tool_writes_only_selected_format(tmp_path, monkeypatch, output_format) -> None:
    project = tmp_path / "project"
    layout = ensure_project_space_layout(project)
    monkeypatch.setattr("catmaster.runtime.literature.citations._resolve", lambda doi: ({
        "title": "Selected paper", "authors": [f"Researcher {i}" for i in range(14)],
        "venue": "Journal", "year": 2026, "doi": doi, "url": f"https://doi.org/{doi}",
    }, ""))
    tool = get_tool_registry().as_langchain_tools(allowlist=["finalize_citations"])[0]
    schema = tool.args_schema["properties"]["output_format"]
    assert schema["type"] == "string"
    assert schema["default"] == "bib"
    assert set(schema["enum"]) == {"bib", "json", "md"}
    provider_schema = get_tool_registry().as_openai_tools(allowlist=["finalize_citations"])[0]
    assert provider_schema["parameters"]["properties"]["output_format"] == schema
    with workspace_scope(project):
        result = tool.invoke({
            "name": "finalize_citations", "type": "tool_call", "id": "finalize-one",
            "args": {"items": ["10.1234/example"], "output_format": output_format},
        })
    assert isinstance(result, ToolMessage)
    assert f"notes/literature/references.{output_format}" in result.content
    output_root = layout["files_root"] / "notes/literature"
    assert [path.name for path in output_root.iterdir()] == [f"references.{output_format}"]
    assert "Researcher 13" in (output_root / f"references.{output_format}").read_text()


def test_default_bib_export_preserves_existing_other_formats(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project"
    root = ensure_project_space_layout(project)["files_root"] / "notes/literature"
    root.mkdir(parents=True, exist_ok=True)
    (root / "references.json").write_text('"user-owned JSON"', encoding="utf-8")
    (root / "references.md").write_text("User-owned notes", encoding="utf-8")
    monkeypatch.setattr("catmaster.runtime.literature.citations._resolve", lambda doi: ({
        "title": "Paper", "authors": ["Author"], "doi": doi,
    }, ""))
    with workspace_scope(project):
        _, artifact = finalize_citations({"items": ["10.1234/example"]})
    assert artifact["data"]["files"] == ["notes/literature/references.bib"]
    assert (root / "references.json").read_text() == '"user-owned JSON"'
    assert (root / "references.md").read_text() == "User-owned notes"


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
        allowlist=[
            "web_search",
            "search_openalex",
            "search_semantic_scholar",
            "open_public_page",
            "find_in_page",
            "ingest_literature_files",
            "query_literature_corpus",
            "finalize_citations",
        ]
    )
    serialized = json.dumps(tools)

    assert '"type": "null"' not in serialized
    assert '"default": null' not in serialized
    assert {tool["name"] for tool in tools} == {
        "web_search",
        "search_openalex",
        "search_semantic_scholar",
        "open_public_page",
        "find_in_page",
        "ingest_literature_files",
        "query_literature_corpus",
        "finalize_citations",
    }
    query_schema = next(
        tool["parameters"]
        for tool in tools
        if tool["name"] == "query_literature_corpus"
    )
    assert set(query_schema["properties"]) == {"query", "page_size", "offset", "query_mode", "source_paths"}
    assert "top_k" not in query_schema["properties"]
    by_name = {tool["name"]: tool["parameters"] for tool in tools}
    assert by_name["search_openalex"]["properties"]["cursor"]["type"] == "string"
    assert by_name["search_openalex"]["properties"]["cursor"]["default"] == ""
    assert by_name["search_semantic_scholar"]["properties"]["offset"]["type"] == "integer"
    assert by_name["search_semantic_scholar"]["properties"]["offset"]["default"] == 0
    assert set(by_name["open_public_page"]["properties"]) == {
        "url",
        "source_path",
        "offset",
        "max_chars",
    }
    assert set(by_name["find_in_page"]["properties"]) == {
        "url",
        "source_path",
        "pattern",
        "match_offset",
        "max_matches",
        "context_chars",
    }
    assert by_name["open_public_page"]["properties"]["offset"]["default"] == 0
    assert by_name["find_in_page"]["properties"]["match_offset"]["default"] == 0
    assert set(by_name["web_search"]["properties"]) == {"query", "max_results"}
    assert by_name["web_search"]["properties"]["max_results"]["maximum"] == 20


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


def test_registered_metadata_tools_follow_two_provider_native_pages(monkeypatch) -> None:
    class _FakeOpenAlex:
        def __init__(self) -> None:
            self.calls: list[tuple[str, int]] = []

        def search_works_page(self, query: str, limit: int, *, cursor: str = ""):
            assert query == "catalysis"
            self.calls.append((cursor, limit))
            if not cursor:
                return [
                    SimpleNamespace(
                        paper=PaperRecord(
                            paper_id="https://openalex.org/W1",
                            title="OpenAlex first page",
                            source="openalex",
                        )
                    )
                ], 2, "cursor-page-2"
            assert cursor == "cursor-page-2"
            return [
                SimpleNamespace(
                    paper=PaperRecord(
                        paper_id="https://openalex.org/W2",
                        title="OpenAlex second page",
                        source="openalex",
                    )
                )
            ], 2, ""

    class _FakeSemanticScholar:
        def __init__(self) -> None:
            self.calls: list[tuple[int, int]] = []

        def search_papers_page(
            self,
            query: str,
            limit: int,
            *,
            offset: int = 0,
            year_from=None,
            year_to=None,
        ):
            assert query == "catalysis"
            assert year_from is None and year_to is None
            self.calls.append((offset, limit))
            if offset == 0:
                return [
                    SimpleNamespace(
                        paper=PaperRecord(
                            paper_id="S1",
                            title="Semantic Scholar first page",
                            source="semantic_scholar",
                        )
                    )
                ], 2, 1
            assert offset == 1
            return [
                SimpleNamespace(
                    paper=PaperRecord(
                        paper_id="S2",
                        title="Semantic Scholar second page",
                        source="semantic_scholar",
                    )
                )
            ], 2, None

    openalex = _FakeOpenAlex()
    scholar = _FakeSemanticScholar()
    monkeypatch.setattr(
        "catmaster.runtime.literature.tools._literature_components",
        lambda: (object(), openalex, scholar, object()),
    )
    registered = {
        tool.name: tool
        for tool in get_tool_registry().as_langchain_tools(
            allowlist=["search_openalex", "search_semantic_scholar"]
        )
    }

    openalex_first = registered["search_openalex"].invoke(
        {
            "name": "search_openalex",
            "args": {"query": "catalysis", "limit": 1},
            "id": "openalex-page-1",
            "type": "tool_call",
        }
    )
    assert isinstance(openalex_first, ToolMessage)
    openalex_first_payload = json.loads(str(openalex_first.content))
    openalex_second = registered["search_openalex"].invoke(
        {
            "name": "search_openalex",
            "args": {
                "query": "catalysis",
                "limit": 1,
                "cursor": openalex_first_payload["next_cursor"],
            },
            "id": "openalex-page-2",
            "type": "tool_call",
        }
    )
    openalex_second_payload = json.loads(str(openalex_second.content))
    assert openalex_first_payload["total"] == 2
    assert openalex_first_payload["papers"][0]["paper_id"].endswith("W1")
    assert openalex_second_payload["papers"][0]["paper_id"].endswith("W2")
    assert openalex_second_payload["next_cursor"] == ""
    assert openalex.calls == [("", 1), ("cursor-page-2", 1)]

    scholar_first = registered["search_semantic_scholar"].invoke(
        {
            "name": "search_semantic_scholar",
            "args": {"query": "catalysis", "limit": 1},
            "id": "scholar-page-1",
            "type": "tool_call",
        }
    )
    assert isinstance(scholar_first, ToolMessage)
    scholar_first_payload = json.loads(str(scholar_first.content))
    scholar_second = registered["search_semantic_scholar"].invoke(
        {
            "name": "search_semantic_scholar",
            "args": {
                "query": "catalysis",
                "limit": 1,
                "offset": scholar_first_payload["next_offset"],
            },
            "id": "scholar-page-2",
            "type": "tool_call",
        }
    )
    scholar_second_payload = json.loads(str(scholar_second.content))
    assert scholar_first_payload["total"] == 2
    assert scholar_first_payload["papers"][0]["paper_id"] == "S1"
    assert scholar_second_payload["papers"][0]["paper_id"] == "S2"
    assert scholar_second_payload["next_offset"] is None
    assert scholar.calls == [(0, 1), (1, 1)]


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


def test_public_page_offsets_are_complete_continuable_and_searchable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    ensure_project_space_layout(project)
    decisive = "DECISIVE-BEYOND-OLD-PREFIX"
    full_text = ("prefix material " * 60) + decisive + (" suffix material" * 50)

    class _CompleteWeb:
        def __init__(self) -> None:
            self.calls = 0

        def open_public_page(self, url: str, max_chars: int = 12000):
            _ = max_chars
            self.calls += 1
            return PublicPageSnapshot(
                requested_url=url,
                final_url=url,
                status_code=200,
                content_type="text/plain",
                text=full_text,
                source_completeness="complete",
            )

    web = _CompleteWeb()
    monkeypatch.setattr(
        "catmaster.runtime.literature.tools._literature_components",
        lambda: (object(), object(), object(), web),
    )
    with workspace_scope(project):
        first_content, _ = open_public_page(
            {"url": "https://example.org/long", "max_chars": 500}
        )
        first = json.loads(first_content)["page"]
        source_path = first["source_path"]
        chunks = [first["text"]]
        offset = first["next_offset"]
        while offset:
            content, _ = open_public_page(
                {
                    "source_path": source_path,
                    "offset": offset,
                    "max_chars": 500,
                }
            )
            page = json.loads(content)["page"]
            chunks.append(page["text"])
            offset = page["next_offset"]

        find_content, _ = find_in_page(
            {
                "source_path": source_path,
                "pattern": decisive,
                "max_matches": 1,
            }
        )
        found = json.loads(find_content)["result"]

    assert "".join(chunks) == full_text
    assert first["total_chars"] == len(full_text)
    assert "snapshot_ref" not in first
    assert "full_content_ref" not in first
    assert source_path.startswith("/literature/public_pages/example.org_long")
    assert (project / "files" / source_path.lstrip("/")).is_file()
    assert web.calls == 1
    assert found["total_matches"] == 1
    assert found["source_path"] == source_path
    assert found["matches"][0]["start_char"] > 500
    assert decisive in found["matches"][0]["snippet"]


def test_public_page_read_creates_visible_reusable_workspace_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    ensure_project_space_layout(project)

    class _CompleteWeb:
        def open_public_page(self, url: str, max_chars: int = 12000):
            _ = max_chars
            return PublicPageSnapshot(
                requested_url=url,
                final_url=url,
                status_code=200,
                text="complete source text",
            )

    monkeypatch.setattr(
        "catmaster.runtime.literature.tools._literature_components",
        lambda: (object(), object(), object(), _CompleteWeb()),
    )
    with workspace_scope(project):
        content, _ = open_public_page({"url": "https://example.org/long"})
    payload = json.loads(content)

    assert payload["status"] == "ok"
    assert payload["page"]["text"] == "complete source text"
    source_path = payload["page"]["source_path"]
    assert source_path.startswith("/literature/public_pages/")
    saved = project / "files" / source_path.lstrip("/")
    assert saved.is_file()
    assert "complete source text" in saved.read_text(encoding="utf-8")
    assert not (project / "files/literature/public_page_snapshots").exists()


def test_public_page_continuation_rejects_url_refetch(tmp_path: Path) -> None:
    project = tmp_path / "project"
    ensure_project_space_layout(project)

    with workspace_scope(project):
        content, _ = open_public_page(
            {
                "url": "https://example.org/long",
                "offset": 500,
                "max_chars": 500,
            }
        )

    payload = json.loads(content)
    assert payload["status"] == "invalid_request"
    assert not (project / "files/literature/public_pages").exists()


def test_search_public_web_alias_uses_web_search(monkeypatch) -> None:
    monkeypatch.setattr(
        "catmaster.runtime.literature.tools.web_search",
        lambda payload: ("alias ok", {"tool_name": "web_search", "data": payload}),
    )
    content, artifact = search_public_web({"query": "alias"})

    assert content == "alias ok"
    assert artifact["tool_name"] == "web_search"
