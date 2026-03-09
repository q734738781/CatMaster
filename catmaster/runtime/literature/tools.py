from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from catmaster.llm.config import LLMProfile
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError

from .subagent import LiteratureSubagent


class RunLiteratureResearchInput(BaseModel):
    """Run a precise literature-grounding workflow and return a compact context pack."""

    query: str = Field(..., description="Literature question or retrieval topic to investigate.")
    depth: str = Field(
        "auto",
        description="Research depth: auto | none | quick | standard | focused | deep_report.",
    )
    topic: str | None = Field(
        None,
        description="Optional short topic/search phrase. If provided, scholarly databases use this instead of the full natural-language request.",
    )
    seed_papers: list[str] | None = Field(
        None,
        description="Optional Semantic Scholar ids / DOI-like seeds to anchor recommendations.",
    )
    role: str | None = Field(
        None,
        description="Internal caller role used for depth-policy gating. Usually injected automatically.",
    )


def _pack_summary_text(data: dict[str, Any]) -> str:
    summary_limit = max(1, LLMProfile.from_env_or_file().literature.summary_key_paper_count)
    lines = [
        f"Literature research depth: {data.get('depth', 'unknown')}",
        f"Summary: {data.get('summary', '').strip()}",
        "Key papers:",
    ]
    key_papers = data.get("key_papers") or []
    if key_papers:
        for idx, paper in enumerate(key_papers[:summary_limit], start=1):
            if not isinstance(paper, dict):
                continue
            title = str(paper.get("title") or "Untitled paper").strip()
            year = str(paper.get("year") or "?").strip()
            venue = str(paper.get("venue") or "").strip()
            doi = str(paper.get("doi") or paper.get("url") or "").strip()
            oa_pdf = str(paper.get("open_access_pdf_url") or "").strip()
            abstract_flag = paper.get("has_abstract")
            suffix = f" ({venue})" if venue else ""
            cite = f" - {doi}" if doi else ""
            extra_parts = []
            if abstract_flag is True:
                extra_parts.append("abstract")
            if oa_pdf:
                extra_parts.append("oa_pdf")
            extra = f" [{', '.join(extra_parts)}]" if extra_parts else ""
            lines.append(f"- [{idx}] {title} [{year}]{suffix}{cite}{extra}")
    else:
        lines.append("- (none)")
    return "\n".join(lines)


def run_literature_research(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "run_literature_research"
    try:
        params = RunLiteratureResearchInput(**payload)
        subagent = LiteratureSubagent.create_default()
        pack = subagent.run(
            query=params.query,
            requested_depth=params.depth,
            topic=params.topic,
            seed_papers=params.seed_papers,
            role=params.role,
        )
        data = pack.model_dump()
        return _pack_summary_text(data), {
            "tool_name": tool_name,
            "data": data,
            # Keep representative citations visible in tool content even when the
            # full artifact is offloaded to disk.
            "suppress_content_offload_ref": True,
        }
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} failed: {exc}",
            artifact={"tool_name": tool_name, "data": {"query": payload.get("query")}},
            error_code="run_literature_research_failed",
        ) from exc


__all__ = ["RunLiteratureResearchInput", "run_literature_research"]
