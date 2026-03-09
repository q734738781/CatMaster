from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.messages import HumanMessage
from langchain_core.tools import StructuredTool

from catmaster.llm.config import LLMProfile, LiteratureRuntimeConfig
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.tool_output_adapter import tool_error_to_message

from .depth_policy import resolve_depth
from .models import LiteratureContextPack, PaperRecord, ResearchDepth
from .openalex_client import OpenAlexClient
from .online_search_adapter import OnlineSearchAdapter
from .semanticscholar_client import SemanticScholarClient
from .store import LiteratureStore


def _load_create_agent():
    from langchain.agents import create_agent as _create_agent

    return _create_agent


def _load_tool_strategy():
    from langchain.agents.structured_output import ToolStrategy as _ToolStrategy

    return _ToolStrategy


def _load_agent_middleware():
    from langchain.agents.middleware import AgentMiddleware as _AgentMiddleware

    return _AgentMiddleware


def _make_internal_tool_budget_middleware(*, max_tool_calls: int) -> list[Any]:
    AgentMiddleware = _load_agent_middleware()
    tool_limit = max(1, int(max_tool_calls))
    counter: dict[str, int] = {"used": 0}

    class _ResetToolCallCounterMiddleware(AgentMiddleware):
        def before_agent(self, state: dict, runtime: Any) -> dict[str, Any] | None:
            _ = (state, runtime)
            counter["used"] = 0
            return None

        async def abefore_agent(self, state: dict, runtime: Any) -> dict[str, Any] | None:
            _ = (state, runtime)
            counter["used"] = 0
            return None

    class _ToolCallBudgetMiddleware(AgentMiddleware):
        @staticmethod
        def _request_info(request: Any) -> tuple[int, str, str]:
            used = int(counter.get("used", 0))
            tool_call = getattr(request, "tool_call", None) or {}
            tool_name = str(tool_call.get("name") or "unknown")
            tool_call_id = str(tool_call.get("id") or "")
            return used, tool_name, tool_call_id

        @staticmethod
        def _budget_exceeded_message(*, used: int, tool_name: str, tool_call_id: str):
            return tool_error_to_message(
                exc=RuntimeError(
                    f"literature subagent tool-call budget exceeded: executed={used}, "
                    f"limit={tool_limit}, next_tool={tool_name}"
                ),
                tool_name=tool_name,
                tool_call_id=tool_call_id,
            )

        def wrap_tool_call(self, request: Any, handler: Any) -> Any:
            used, tool_name, tool_call_id = self._request_info(request)
            if used >= tool_limit:
                return self._budget_exceeded_message(
                    used=used,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
            try:
                result = handler(request)
                if inspect.isawaitable(result):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise RuntimeError("Synchronous literature middleware received awaitable handler.")
                return result
            finally:
                counter["used"] = used + 1

        async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
            used, tool_name, tool_call_id = self._request_info(request)
            if used >= tool_limit:
                return self._budget_exceeded_message(
                    used=used,
                    tool_name=tool_name,
                    tool_call_id=tool_call_id,
                )
            try:
                result = handler(request)
                if inspect.isawaitable(result):
                    result = await result
                return result
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                raise
            finally:
                counter["used"] = used + 1

    return [_ResetToolCallCounterMiddleware(), _ToolCallBudgetMiddleware()]


@dataclass
class LiteratureSubagent:
    semanticscholar: SemanticScholarClient
    online_search: OnlineSearchAdapter
    store: LiteratureStore
    config: LiteratureRuntimeConfig
    openalex: OpenAlexClient | None = None

    @classmethod
    def create_default(cls) -> "LiteratureSubagent":
        profile = LLMProfile.from_env_or_file()
        return cls(
            openalex=OpenAlexClient(),
            semanticscholar=SemanticScholarClient(
                retry_429_attempts=profile.literature.semantic_scholar_retry_429_attempts,
                retry_429_wait_seconds=profile.literature.semantic_scholar_retry_429_wait_seconds,
            ),
            online_search=OnlineSearchAdapter(),
            store=LiteratureStore(),
            config=profile.literature,
        )

    @staticmethod
    def _dedupe_papers(papers: Iterable[PaperRecord]) -> list[PaperRecord]:
        out: list[PaperRecord] = []
        seen: set[str] = set()
        for paper in papers:
            key = paper.paper_id or paper.doi or paper.title.lower().strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(paper)
        return out

    @staticmethod
    def _semantic_scholar_seed_ids(papers: Iterable[PaperRecord]) -> list[str]:
        ids: list[str] = []
        for paper in papers:
            paper_id = str(paper.paper_id or "").strip()
            source = str(paper.source or "").strip().lower()
            if not paper_id:
                continue
            if source.startswith("semantic_scholar"):
                ids.append(paper_id)
        return ids

    @staticmethod
    def _paper_payload(paper: PaperRecord) -> dict:
        return paper.model_dump()

    def _build_search_tools(self, *, budget, topic: str | None) -> list[StructuredTool]:
        search_limit = budget.search_limit
        rec_limit = budget.recommendation_limit
        rec_seed_count = budget.recommendation_seed_count
        web_limit = budget.public_web_limit
        allow_public_web = bool(budget.use_public_web or self.config.public_web_on_search_failure)

        def search_openalex(query: str, limit: int | None = None) -> str:
            actual_limit = max(1, min(int(limit or search_limit or 1), max(search_limit, 1)))
            try:
                hits = self.openalex.search_works(query, limit=actual_limit) if self.openalex is not None else []
            except Exception as exc:
                return f"OpenAlex search failed: {exc}"
            papers = [hit.paper for hit in hits]
            return json.dumps(
                {
                    "status": "ok",
                    "source": "openalex",
                    "query": query,
                    "count": len(papers),
                    "papers": [self._paper_payload(paper) for paper in papers],
                },
                ensure_ascii=False,
            )

        def search_semantic_scholar(query: str, limit: int | None = None) -> str:
            actual_limit = max(1, min(int(limit or search_limit or 1), max(search_limit, 1)))
            try:
                hits = self.semanticscholar.search_papers(query, limit=actual_limit)
            except Exception as exc:
                return f"Semantic Scholar search failed: {exc}"
            papers = [hit.paper for hit in hits]
            return json.dumps(
                {
                    "status": "ok",
                    "source": "semantic_scholar",
                    "query": query,
                    "count": len(papers),
                    "papers": [self._paper_payload(paper) for paper in papers],
                },
                ensure_ascii=False,
            )

        def get_openalex_record(work_id_or_doi: str) -> str:
            try:
                record = self.openalex.get_work(work_id_or_doi) if self.openalex is not None else None
            except Exception as exc:
                return f"OpenAlex record fetch failed: {exc}"
            if record is None:
                return "OpenAlex record fetch failed: OpenAlex client unavailable"
            return json.dumps(
                {"status": "ok", "source": "openalex", "paper": self._paper_payload(record)},
                ensure_ascii=False,
            )

        def get_semantic_scholar_record(paper_id_or_doi: str) -> str:
            try:
                record = self.semanticscholar.get_paper(paper_id_or_doi)
            except Exception as exc:
                return f"Semantic Scholar record fetch failed: {exc}"
            return json.dumps(
                {"status": "ok", "source": "semantic_scholar", "paper": self._paper_payload(record)},
                ensure_ascii=False,
            )

        def recommend_semantic_scholar(seed_paper_ids: list[str], limit: int | None = None) -> str:
            cleaned = [str(item).strip() for item in (seed_paper_ids or []) if str(item).strip()]
            if not cleaned:
                return "Semantic Scholar recommendations skipped: no valid semantic scholar seed ids provided."
            actual_limit = max(1, min(int(limit or rec_limit or 1), max(rec_limit, 1)))
            try:
                hits = self.semanticscholar.get_recommendations(cleaned, limit=actual_limit)
            except Exception as exc:
                return f"Semantic Scholar recommendations failed: {exc}"
            papers = [hit.paper for hit in hits]
            return json.dumps(
                {
                    "status": "ok",
                    "source": "semantic_scholar_recommendation",
                    "seed_paper_ids": cleaned,
                    "count": len(papers),
                    "papers": [self._paper_payload(paper) for paper in papers],
                },
                ensure_ascii=False,
            )

        def search_public_web(query: str, max_results: int | None = None) -> str:
            actual_limit = max(1, min(int(max_results or web_limit or 1), max(web_limit, 1)))
            try:
                hits = self.online_search.search_public_web(query, max_results=actual_limit).results
            except Exception as exc:
                return f"Public web search failed: {exc}"
            return json.dumps(
                {
                    "status": "ok",
                    "source": "public_web",
                    "query": query,
                    "count": len(hits),
                    "hits": [hit.model_dump() for hit in hits],
                },
                ensure_ascii=False,
            )

        def open_public_page(url: str, max_chars: int | None = None) -> str:
            try:
                page = self.online_search.open_public_page(url, max_chars=max_chars or 12000)
            except Exception as exc:
                return f"Open public page failed: {exc}"
            return json.dumps(
                {
                    "status": "ok",
                    "source": "public_web_page",
                    "page": page.model_dump(),
                },
                ensure_ascii=False,
            )

        def find_in_page(
            url: str,
            pattern: str,
            max_matches: int | None = None,
            context_chars: int | None = None,
        ) -> str:
            try:
                result = self.online_search.find_in_page(
                    url,
                    pattern,
                    max_matches=max_matches or 5,
                    context_chars=context_chars or 240,
                )
            except Exception as exc:
                return f"Find in page failed: {exc}"
            return json.dumps(
                {
                    "status": "ok",
                    "source": "public_web_page",
                    "result": result.model_dump(),
                },
                ensure_ascii=False,
            )

        tools: list[StructuredTool] = [
            StructuredTool.from_function(
                func=search_openalex,
                name="search_openalex",
                description=(
                    f"Search OpenAlex for scholarly works. Use short scholarly queries, not reporting instructions. "
                    f"This is the primary metadata search tool. Default/cap limit={search_limit}. Topic hint: {topic or '(none)'}."
                ),
            ),
            StructuredTool.from_function(
                func=search_semantic_scholar,
                name="search_semantic_scholar",
                description=(
                    f"Search Semantic Scholar to supplement OpenAlex with paper metadata and abstract/tldr hints. "
                    f"Use short scholarly queries. Default/cap limit={search_limit}."
                ),
            ),
            StructuredTool.from_function(
                func=get_openalex_record,
                name="get_openalex_record",
                description="Fetch a single OpenAlex work by OpenAlex id or DOI when a specific paper needs richer metadata.",
            ),
            StructuredTool.from_function(
                func=get_semantic_scholar_record,
                name="get_semantic_scholar_record",
                description="Fetch a single Semantic Scholar paper by paper id or DOI when a specific paper needs confirmation or extra metadata.",
            ),
        ]
        if rec_limit > 0:
            tools.append(
                StructuredTool.from_function(
                    func=recommend_semantic_scholar,
                    name="recommend_semantic_scholar",
                    description=(
                        f"Get Semantic Scholar recommendations from semantic scholar seed ids only. "
                        f"Use only after you already found promising semantic scholar papers. Default/cap limit={rec_limit};"
                        f" seed count should stay small (typically <= {rec_seed_count})."
                    ),
                )
            )
        if allow_public_web and web_limit > 0:
            tools.append(
                StructuredTool.from_function(
                    func=search_public_web,
                    name="search_public_web",
                    description=(
                        f"Search public web for broad background, NIH/landing-page summaries, or context beyond metadata. "
                        f"Use this for broad orientation or when scholarly APIs are sparse. Default/cap results={web_limit}."
                    ),
                )
            )
            tools.append(
                StructuredTool.from_function(
                    func=open_public_page,
                    name="open_public_page",
                    description=(
                        "Open a public http(s) page returned from search and extract normalized visible text, title, "
                        "description, and final URL. Use this when snippets are not enough and you need page-level evidence."
                    ),
                )
            )
            tools.append(
                StructuredTool.from_function(
                    func=find_in_page,
                    name="find_in_page",
                    description=(
                        "Search normalized text within a public page previously identified by URL. Use this to confirm "
                        "whether a page mentions a specific method, adsorbate, material, reference-state convention, or keyword."
                    ),
                )
            )
        return tools

    def _invoke_research_agent(
        self,
        *,
        query: str,
        depth: ResearchDepth,
        topic: str | None,
        seed_papers: list[str] | None,
        budget,
    ) -> LiteratureContextPack:
        profile = LLMProfile.from_env_or_file()
        cfg = profile.config_for_role("literature_deep_research" if depth == "deep_report" else "literature_synthesizer")
        model = build_chat_model(cfg)
        create_agent = _load_create_agent()
        ToolStrategy = _load_tool_strategy()
        tools = self._build_search_tools(budget=budget, topic=topic)
        system_prompt = (
            "You are a literature research agent. "
            "The outer caller provides a research intent, not database-ready search strings. "
            "You must decide how to search. Rewrite scholarly queries into short database-friendly phrases. "
            "Do not pass full reporting instructions verbatim into scholarly databases. "
            "Source routing rule: start with public web search for broad orientation, public summaries, landing-page abstracts, and lightweight evidence checks. "
            "Only use OpenAlex or Semantic Scholar when you need paper-level metadata, citations, DOI/year/venue details, recommendation expansion, or explicit literature grounding. "
            "Do not automatically call both scholarly search and public web in the same first pass unless the request clearly needs both. "
            "Use OpenAlex as the primary scholarly metadata source when scholarly grounding is needed. Use Semantic Scholar as a supplementary scholarly source and for recommendations. "
            "Use public web search for broad public background, landing pages, NIH summaries, or when scholarly sources are sparse. "
            "Stop as soon as you have an adequate evidence pack for the requested depth. "
            "Do not repeat near-identical searches on the same source just to be more complete. "
            "For quick or focused requests, prefer a small number of high-value retrieval steps over exhaustive coverage. "
            "Stay within the requested depth and return a LiteratureContextPack only. "
            "Ground claims in retrieved evidence. Do not invent papers or citations."
        )
        agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=system_prompt,
            response_format=ToolStrategy(LiteratureContextPack, handle_errors=False),
            middleware=_make_internal_tool_budget_middleware(max_tool_calls=budget.agent_step_budget),
            name="literature_research_subagent",
        )
        human = HumanMessage(
            content=(
                f"Research intent:\n{query}\n\n"
                f"Resolved depth: {depth}\n"
                f"Topic hint: {topic or '(none)'}\n"
                f"Seed papers: {json.dumps(seed_papers or [], ensure_ascii=False)}\n\n"
                "Depth guidance:\n"
                "- quick: small representative set, minimal notes\n"
                "- standard: conventions / benchmark framing / concise evidence summary\n"
                "- focused: targeted evidence for one narrow question\n"
                "- deep_report: broader benchmark landscape and disagreements\n\n"
                "Source-routing guidance:\n"
                "- web-first for broad orientation or public-page summaries\n"
                "- scholarly search only when paper metadata / citations / explicit literature grounding are needed\n"
                "- avoid doing OpenAlex and public web together in the first pass unless clearly justified\n\n"
                "Stopping guidance:\n"
                "- stop once the evidence pack is adequate for the requested depth\n"
                "- do not issue repeated near-identical searches against the same source\n"
                "- quick/focused should stay small and selective, not exhaustive\n\n"
                f"Current search budget:\n"
                f"- internal tool-call budget: {budget.agent_step_budget}\n"
                f"- scholarly search limit: {budget.search_limit}\n"
                f"- recommendation limit: {budget.recommendation_limit}\n"
                f"- recommendation seed count: {budget.recommendation_seed_count}\n"
                f"- public web limit: {budget.public_web_limit}\n"
                f"- use public web by default: {budget.use_public_web}\n"
            )
        )
        result = agent.invoke(
            {"messages": [human]},
            config={"recursion_limit": max(12, int(budget.agent_step_budget) * 3)},
        )
        raw = result.get("structured_response") if isinstance(result, dict) else None
        if isinstance(raw, LiteratureContextPack):
            return raw.model_copy(update={"query": query, "depth": depth, "topic": topic})
        if raw is None:
            raise RuntimeError("literature subagent missing structured_response")
        pack = LiteratureContextPack.model_validate(raw)
        return pack.model_copy(update={"query": query, "depth": depth, "topic": topic})

    def run(
        self,
        *,
        query: str,
        requested_depth: str = "auto",
        topic: str | None = None,
        seed_papers: list[str] | None = None,
        matched_skills: Iterable[str] | None = None,
        flags: dict | None = None,
        role: str | None = None,
    ) -> LiteratureContextPack:
        depth = resolve_depth(
            query,
            requested_depth=requested_depth,
            role=role,
            matched_skills=matched_skills,
            flags=flags,
            config=self.config,
        )
        if depth == "none":
            pack = LiteratureContextPack(
                query=query,
                depth=depth,
                topic=topic,
                summary="Literature module was not needed for this request.",
                key_papers=[],
                evidence_table=[],
                citations=[],
                followup_questions=[],
                confidence="low",
                sources_used=[],
            )
            self.store.persist_memo(pack)
            return pack

        budget = self.config.budget_for_depth(depth)
        pack = self._invoke_research_agent(
            query=query,
            depth=depth,
            topic=topic,
            seed_papers=seed_papers,
            budget=budget,
        )
        self.store.persist_records(pack.key_papers)
        self.store.persist_query_cache(
            query=query,
            depth=depth,
            candidate_paper_ids=[paper.paper_id for paper in pack.key_papers if paper.paper_id],
        )
        self.store.persist_memo(pack)
        return pack


__all__ = ["LiteratureSubagent"]
