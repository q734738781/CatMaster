from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model

from .models import LiteratureContextPack, PaperRecord, PublicWebHit, ResearchDepth


def _resolve_synth_config(*, deep: bool, model_override: str | None = None):
    profile = LLMProfile.from_env_or_file()
    role = "literature_deep_research" if deep else "literature_synthesizer"
    base_cfg = profile.config_for_role(role)
    override = str(model_override or "").strip()
    if not override:
        return base_cfg
    if override in profile.models:
        return profile.models[override]
    return replace(base_cfg, model=override)


def _paper_block(paper: PaperRecord, idx: int) -> str:
    authors = ", ".join(paper.authors[:4]) if paper.authors else "(authors unavailable)"
    citation = paper.doi or paper.url or "(no DOI/url)"
    return (
        f"[{idx}] {paper.title}\n"
        f"year={paper.year} venue={paper.venue or '(unknown)'} citations={paper.citation_count}\n"
        f"authors={authors}\n"
        f"source={paper.source}\n"
        f"doi_or_url={citation}\n"
        f"abstract={paper.abstract or '(none)'}\n"
        f"snippet={paper.snippet or '(none)'}"
    )


def _web_block(hit: PublicWebHit, idx: int) -> str:
    return f"[{idx}] {hit.title}\nurl={hit.url or '(none)'}\nsnippet={hit.snippet}"


def synthesize_standard(
    *,
    query: str,
    depth: ResearchDepth,
    topic: str | None,
    papers: Sequence[PaperRecord],
    public_web_hits: Sequence[PublicWebHit],
    model_override: str | None = None,
) -> LiteratureContextPack:
    cfg = _resolve_synth_config(deep=False, model_override=model_override)
    model = build_chat_model(cfg).with_structured_output(LiteratureContextPack)
    papers_text = "\n\n".join(_paper_block(paper, idx) for idx, paper in enumerate(papers, start=1)) or "(none)"
    web_text = "\n\n".join(_web_block(hit, idx) for idx, hit in enumerate(public_web_hits, start=1)) or "(none)"
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a literature synthesis agent. Produce a compact LiteratureContextPack. "
                    "Ground claims in the retrieved papers first; use public-web hits only as supplementary context. "
                    "Do not invent papers or overstate certainty. Keep summaries concise and execution-useful."
                )
            ),
            HumanMessage(
                content=(
                    f"Query: {query}\n"
                    f"Resolved depth: {depth}\n"
                    f"Topic: {topic or '(none)'}\n\n"
                    "Retrieved papers:\n"
                    f"{papers_text}\n\n"
                    "Supplementary public web hits:\n"
                    f"{web_text}\n"
                )
            ),
        ]
    )
    pack = response if isinstance(response, LiteratureContextPack) else LiteratureContextPack.model_validate(response)
    return pack.model_copy(update={"query": query, "depth": depth, "topic": topic})


def synthesize_deep_report(
    *,
    query: str,
    depth: ResearchDepth,
    topic: str | None,
    papers: Sequence[PaperRecord],
    public_web_hits: Sequence[PublicWebHit],
    model_override: str | None = None,
) -> LiteratureContextPack:
    cfg = _resolve_synth_config(deep=True, model_override=model_override)
    model = build_chat_model(cfg).with_structured_output(LiteratureContextPack)
    papers_text = "\n\n".join(_paper_block(paper, idx) for idx, paper in enumerate(papers, start=1)) or "(none)"
    web_text = "\n\n".join(_web_block(hit, idx) for idx, hit in enumerate(public_web_hits, start=1)) or "(none)"
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a deep literature synthesis agent. Build a high-quality LiteratureContextPack that captures "
                    "the main consensus, important disagreements, benchmark conventions, and open questions. "
                    "Ground the pack in the retrieved scholarly evidence; public-web hits are secondary."
                )
            ),
            HumanMessage(
                content=(
                    f"Query: {query}\n"
                    f"Resolved depth: {depth}\n"
                    f"Topic: {topic or '(none)'}\n\n"
                    "Retrieved papers:\n"
                    f"{papers_text}\n\n"
                    "Supplementary public web hits:\n"
                    f"{web_text}\n"
                )
            ),
        ]
    )
    pack = response if isinstance(response, LiteratureContextPack) else LiteratureContextPack.model_validate(response)
    return pack.model_copy(update={"query": query, "depth": depth, "topic": topic})


__all__ = ["synthesize_standard", "synthesize_deep_report"]
