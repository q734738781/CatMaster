from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.research import ResearchStore


class WritingEvidenceChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(...)
    path: str = Field(...)
    section: str = Field(...)
    line_range: list[int] = Field(default_factory=lambda: [0, 0])
    text: str = Field("")
    score: float = Field(0.0)


class ReadResearchPackInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_campaign_id: str | None = Field(None)
    kind: Literal["dossier", "board", "conclusion", "literature_latest", "experiment_latest"] = Field(...)


class ReviewResearchContextInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_campaign_id: str | None = Field(None)
    query: str = Field(...)
    section_id: str | None = Field(None)
    max_chunks: int = Field(6, ge=1, le=12)


@dataclass
class WritingToolDeps:
    workspace: Any
    memory_store: MemoryStore
    project_id: str

    def source_store(self, campaign_id: str) -> ResearchStore:
        return ResearchStore(workspace=self.workspace, campaign_id=campaign_id)


def _chunk_md(title: str, chunks: list[WritingEvidenceChunk]) -> str:
    if not chunks:
        return f"{title}:\n- (none)"
    lines = [f"{title}:"]
    for chunk in chunks:
        lines.append(f"- {chunk.path} [{chunk.section}]: {' '.join(str(chunk.text or '').split())}")
    return "\n".join(lines)


def make_read_research_pack_tool(deps: WritingToolDeps) -> StructuredTool:
    def _run(*, source_campaign_id: str | None = None, kind: str) -> tuple[str, dict[str, Any]]:
        source_campaign_id = str(source_campaign_id or "").strip() or None
        if source_campaign_id is None:
            return f"{kind}: (source campaign not provided)", {"kind": kind, "data": None}
        store = deps.source_store(source_campaign_id)
        if kind == "dossier":
            payload = store.load_dossier()
        elif kind == "board":
            payload = store.load_board()
        elif kind == "conclusion":
            payload = store.load_conclusion()
        elif kind == "literature_latest":
            packs = store.load_literature_packs()
            payload = packs[-1] if packs else None
        elif kind == "experiment_latest":
            packs = store.load_experiment_packs()
            payload = packs[-1] if packs else None
        else:
            payload = None
        if payload is None:
            return f"{kind}: (missing)", {"kind": kind, "data": None}
        data = payload.model_dump() if hasattr(payload, "model_dump") else payload
        text = json.dumps(data, ensure_ascii=False, indent=2)
        return text, {"kind": kind, "data": data, "suppress_content_offload_ref": True}

    return StructuredTool.from_function(
        func=_run,
        name="read_research_pack",
        description="Read a structured research campaign pack such as dossier, board, conclusion, or latest literature/experiment pack.",
        args_schema=ReadResearchPackInput,
        response_format="content_and_artifact",
    )


def make_review_research_context_tool(deps: WritingToolDeps) -> StructuredTool:
    async def _arun(*, source_campaign_id: str | None = None, query: str, section_id: str | None = None, max_chunks: int = 6) -> tuple[str, dict[str, Any]]:
        source_campaign_id = str(source_campaign_id or "").strip() or None
        store = deps.source_store(source_campaign_id) if source_campaign_id is not None else None
        board = store.load_board() if store is not None else None
        chunks: list[WritingEvidenceChunk] = []
        for topic in ("FACTS", "CONSTRAINTS", "QUESTIONS", "FILES"):
            try:
                raw = deps.memory_store.read_topic(topic)
            except Exception:
                continue
            text = str(raw or "").strip()
            if not text:
                continue
            chunks.append(
                WritingEvidenceChunk(
                    run_id="memory",
                    path=f"MEMORY/topics/{topic}.md",
                    section="summary",
                    line_range=[0, 0],
                    text=text[:800],
                    score=0.0,
                )
            )
        if board is not None:
            for ref in list(board.action_refs)[-8:]:
                chunks.append(
                    WritingEvidenceChunk(
                        run_id="workspace",
                        path=ref.ref_path,
                        section=f"{ref.kind}/{ref.status}",
                        line_range=[0, 0],
                        text=ref.summary,
                        score=0.0,
                )
            )
        selected = chunks[:max_chunks]
        citations = []
        confidence = 0.0
        memory_chunks = [item for item in selected if item.run_id == "memory"]
        workspace_chunks = [item for item in selected if item.run_id == "workspace"]
        text = "\n\n".join(
            [
                f"Query: {query}",
                f"Section: {section_id or '(none)'}",
                _chunk_md("Relevant durable memory", memory_chunks),
                _chunk_md("Relevant workspace refs", workspace_chunks),
            ]
        ).strip()
        artifact = {
            "query": query,
            "section_id": section_id,
            "confidence": confidence,
            "citations": citations,
            "selected_chunks": [item.model_dump() for item in selected],
            "suppress_content_offload_ref": True,
        }
        return text, artifact

    return StructuredTool.from_function(
        coroutine=_arun,
        name="review_research_context",
        description="Review the most relevant durable memory and research workspace refs for a writing section query.",
        args_schema=ReviewResearchContextInput,
        response_format="content_and_artifact",
    )


__all__ = [
    "ReadResearchPackInput",
    "ReviewResearchContextInput",
    "WritingToolDeps",
    "make_read_research_pack_tool",
    "make_review_research_context_tool",
]
