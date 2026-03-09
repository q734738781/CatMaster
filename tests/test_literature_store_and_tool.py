from __future__ import annotations

import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from catmaster.runtime.literature import (
    LiteratureContextPack,
    LiteratureStore,
    LiteratureSubagent,
    PaperRecord,
)
from catmaster.llm.config import LiteratureRuntimeConfig
from catmaster.runtime.literature.tools import run_literature_research
from catmaster.tools.base import workspace_scope


def test_literature_store_persists_records_queries_and_memos(tmp_path: Path) -> None:
    store = LiteratureStore(workspace=tmp_path)
    pack = LiteratureContextPack(
        query="CO adsorption on Fe(110)",
        depth="quick",
        topic="fe-co",
        summary="Representative papers identified.",
        key_papers=[],
        evidence_table=[],
        citations=[],
        followup_questions=[],
        confidence="medium",
        sources_used=["semantic_scholar"],
    )
    query_path = store.persist_query_cache(query=pack.query, depth=pack.depth, candidate_paper_ids=["paper-1"])
    memo_path = store.persist_memo(pack)

    assert Path(query_path).is_file()
    assert Path(memo_path).is_file()
    assert json.loads(Path(memo_path).read_text(encoding="utf-8"))["summary"] == "Representative papers identified."


def test_run_literature_research_returns_compact_summary(monkeypatch, tmp_path: Path) -> None:
    pack = LiteratureContextPack(
        query="CO adsorption on Fe(110)",
        depth="quick",
        topic=None,
        summary="Representative adsorption papers were identified.",
        key_papers=[
            {
                "paper_id": "p1",
                "title": "CO adsorption on Fe surfaces",
                "year": 2023,
                "venue": "J. Catal.",
                "url": "https://example.org/p1",
                "doi": "10.1234/example",
                "abstract": None,
                "authors": ["A. Author"],
                "citation_count": 12,
                "influential_citation_count": 3,
                "source": "semantic_scholar",
                "snippet": "CO adsorption benchmark.",
            }
        ],
        evidence_table=[],
        citations=[],
        followup_questions=[],
        confidence="medium",
        sources_used=["semantic_scholar"],
    )

    class _FakeSubagent:
        def run(self, **kwargs):
            _ = kwargs
            return pack

    monkeypatch.setattr(
        "catmaster.runtime.literature.tools.LiteratureSubagent.create_default",
        lambda: _FakeSubagent(),
    )

    with workspace_scope(tmp_path):
        content, artifact = run_literature_research({"query": "CO adsorption on Fe(110)", "depth": "quick"})

    assert "Literature research depth: quick" in content
    assert "CO adsorption on Fe surfaces" in content
    assert artifact["tool_name"] == "run_literature_research"
    assert artifact["data"]["summary"] == "Representative adsorption papers were identified."


def test_literature_subagent_run_persists_agent_result(monkeypatch, tmp_path: Path) -> None:
    pack = LiteratureContextPack(
        query="CO adsorption Fe(110) representative papers",
        depth="quick",
        topic="fe110",
        summary="Recovered from internal literature agent.",
        key_papers=[
            PaperRecord(
                paper_id="fallback-1",
                title="Fallback paper proxy",
                year=2024,
                venue="Web supplement",
                url="https://example.org/fe110",
                source="public_web",
                snippet="Public web fallback result.",
            )
        ],
        evidence_table=[],
        citations=[],
        followup_questions=[],
        confidence="low",
        sources_used=["public_web"],
    )

    monkeypatch.setattr(
        LiteratureSubagent,
        "_invoke_research_agent",
        lambda self, **kwargs: pack,
    )

    subagent = LiteratureSubagent(
        semanticscholar=object(),
        online_search=object(),
        store=LiteratureStore(workspace=tmp_path),
        config=LiteratureRuntimeConfig(),
    )

    result = subagent.run(
        query="CO adsorption Fe(110) representative papers",
        requested_depth="quick",
        topic="fe110",
    )

    assert result.summary == "Recovered from internal literature agent."
    memo_files = list((LiteratureStore(workspace=tmp_path).memos_dir).glob("*.json"))
    assert memo_files


def test_literature_subagent_uses_configured_budgets(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    class _RecordingSemanticScholar:
        def search_papers(self, query: str, limit: int = 6):
            observed["search_limit"] = limit
            return []

        def get_paper(self, ident: str):
            raise AssertionError("seed lookup should not be used in this test")

        def get_recommendations(self, seed_paper_ids, limit: int = 4):
            observed["recommendation_limit"] = limit
            observed["seed_ids"] = list(seed_paper_ids)
            return []

    class _RecordingOnlineSearch:
        def search_public_web(self, query: str, max_results: int = 5):
            observed["public_web_limit"] = max_results
            return type("_Result", (), {"results": []})()

    config = LiteratureRuntimeConfig.from_dict(
        {
            "budgets": {
                "standard": {
                    "search_limit": 7,
                    "recommendation_limit": 2,
                    "recommendation_seed_count": 1,
                    "public_web_limit": 4,
                    "use_public_web": True,
                }
            }
        }
    )
    subagent = LiteratureSubagent(
        semanticscholar=_RecordingSemanticScholar(),
        online_search=_RecordingOnlineSearch(),
        store=LiteratureStore(workspace=tmp_path),
        config=config,
    )
    tools = {tool.name: tool for tool in subagent._build_search_tools(budget=config.budget_for_depth("standard"), topic="fe")}

    json.loads(tools["search_semantic_scholar"].invoke({"query": "CO adsorption Fe surfaces"}))
    json.loads(tools["recommend_semantic_scholar"].invoke({"seed_paper_ids": ["S2-1"]}))
    json.loads(tools["search_public_web"].invoke({"query": "CO adsorption Fe surfaces"}))

    assert observed["search_limit"] == 7
    assert observed["recommendation_limit"] == 2
    assert observed["seed_ids"] == ["S2-1"]
    assert observed["public_web_limit"] == 4


def test_literature_subagent_uses_openalex_as_primary_search_source(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    class _FakeOpenAlex:
        def search_works(self, query: str, limit: int):
            observed["openalex_limit"] = limit
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

        def get_work(self, ident: str):
            raise AssertionError("seed lookup should not be used in this test")

    class _RecordingSemanticScholar:
        def search_papers(self, query: str, limit: int = 6):
            observed["s2_limit"] = limit
            return []

        def get_paper(self, ident: str):
            raise AssertionError("seed lookup should not be used in this test")

        def get_recommendations(self, seed_paper_ids, limit: int = 4):
            return []

    subagent = LiteratureSubagent(
        openalex=_FakeOpenAlex(),
        semanticscholar=_RecordingSemanticScholar(),
        online_search=type("_Online", (), {"search_public_web": lambda self, query, max_results=5: type("_Result", (), {"results": []})()})(),
        store=LiteratureStore(workspace=tmp_path),
        config=LiteratureRuntimeConfig.from_dict(
            {
                "budgets": {
                    "quick": {
                        "search_limit": 3,
                        "recommendation_limit": 0,
                        "recommendation_seed_count": 0,
                        "public_web_limit": 0,
                        "use_public_web": False,
                    }
                }
            }
        ),
    )
    tools = {tool.name: tool for tool in subagent._build_search_tools(budget=subagent.config.budget_for_depth("quick"), topic="fe110")}
    payload = json.loads(tools["search_openalex"].invoke({"query": "CO adsorption Fe(110)"}))

    assert observed["openalex_limit"] == 3
    assert payload["source"] == "openalex"
    assert payload["count"] == 1
    assert payload["papers"][0]["source"] == "openalex"


def test_literature_subagent_builds_agentic_toolset_with_topic_hint(tmp_path: Path) -> None:
    subagent = LiteratureSubagent(
        openalex=None,
        semanticscholar=object(),
        online_search=object(),
        store=LiteratureStore(workspace=tmp_path),
        config=LiteratureRuntimeConfig(),
    )
    tools = {tool.name: tool for tool in subagent._build_search_tools(budget=subagent.config.budget_for_depth("quick"), topic="CO adsorption on Fe(110)")}
    assert "short scholarly queries" in tools["search_openalex"].description
    assert "Topic hint: CO adsorption on Fe(110)" in tools["search_openalex"].description
    assert "open_public_page" in tools
    assert "find_in_page" in tools


def test_literature_subagent_invoke_research_agent_uses_internal_agent(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}
    pack = LiteratureContextPack(
        query="original query",
        depth="quick",
        topic="CO adsorption on Fe(110)",
        summary="Representative papers identified.",
        key_papers=[
            PaperRecord(
                paper_id="W1",
                title="Representative Fe(110) paper",
                year=2024,
                venue="J. Catal.",
                source="openalex",
            )
        ],
        evidence_table=[],
        citations=[],
        followup_questions=[],
        confidence="medium",
        sources_used=["openalex"],
    )

    class _FakeAgent:
        def invoke(self, payload, config=None):
            observed["payload"] = payload
            observed["config"] = config
            return {"structured_response": pack}

    def fake_create_agent(**kwargs):
        observed["create_kwargs"] = kwargs
        return _FakeAgent()

    def fake_tool_strategy(schema, handle_errors=False):
        observed["schema"] = schema
        observed["handle_errors"] = handle_errors
        return ("tool_strategy", schema.__name__, handle_errors)

    monkeypatch.setattr("catmaster.runtime.literature.subagent.build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr("catmaster.runtime.literature.subagent._load_create_agent", lambda: fake_create_agent)
    monkeypatch.setattr("catmaster.runtime.literature.subagent._load_tool_strategy", lambda: fake_tool_strategy)

    subagent = LiteratureSubagent(
        openalex=None,
        semanticscholar=object(),
        online_search=object(),
        store=LiteratureStore(workspace=tmp_path),
        config=LiteratureRuntimeConfig(),
    )

    result = subagent._invoke_research_agent(
        query="Quick literature pass: representative papers on CO adsorption on Fe(110). Return 5-6 key papers.",
        depth="quick",
        topic="CO adsorption on Fe(110)",
        seed_papers=["10.1234/example"],
        budget=subagent.config.budget_for_depth("quick"),
    )

    assert result.summary == "Representative papers identified."
    create_kwargs = observed["create_kwargs"]
    assert create_kwargs["name"] == "literature_research_subagent"
    assert "Rewrite scholarly queries into short database-friendly phrases" in create_kwargs["system_prompt"]
    assert "start with public web search for broad orientation" in create_kwargs["system_prompt"]
    assert "Only use OpenAlex or Semantic Scholar when you need paper-level metadata" in create_kwargs["system_prompt"]
    assert create_kwargs["response_format"] == ("tool_strategy", "LiteratureContextPack", False)
    assert observed["schema"] is LiteratureContextPack
    assert observed["handle_errors"] is False
    payload = observed["payload"]
    assert isinstance(payload["messages"], list)
    assert isinstance(payload["messages"][0], HumanMessage)
    content = payload["messages"][0].content
    assert "Research intent:" in content
    assert "Quick literature pass: representative papers on CO adsorption on Fe(110)." in content
    assert "Topic hint: CO adsorption on Fe(110)" in content
    assert "Source-routing guidance:" in content
    assert "web-first for broad orientation or public-page summaries" in content
    assert "Stopping guidance:" in content
    assert "- internal tool-call budget: 4" in content
    assert observed["config"] == {"recursion_limit": 12}
