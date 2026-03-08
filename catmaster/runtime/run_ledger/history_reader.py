from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from catmaster.agents.llm_utils import llm_text
from catmaster.runtime.run_ledger.hybrid_search import HybridRunLedgerSearcher
from catmaster.runtime.run_ledger.models import (
    HistoricalRunsContextPack,
    RunEvidenceChunk,
    RunLedgerEntry,
    RunSearchHit,
)
from catmaster.runtime.run_ledger.store import RunLedgerStore

logger = logging.getLogger(__name__)


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _safe_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_path(system_root: Path, rel_or_abs: str) -> Path:
    raw = str(rel_or_abs or "").strip()
    if not raw:
        return system_root / "__missing__"
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (system_root / p).resolve()


def _to_project_relpath(system_root: Path, path: Path, *, fallback: str = "") -> str:
    hint = str(fallback or "").strip()
    try:
        rel = path.resolve().relative_to(system_root)
        rel_text = rel.as_posix().strip()
        if rel_text:
            return rel_text
    except Exception:
        pass
    if hint:
        return hint
    name = str(path.name or "").strip()
    return name or str(path)


def _line_chunks_from_markdown(text: str, *, run_id: str, path: str) -> List[RunEvidenceChunk]:
    lines = text.splitlines()
    chunks: List[RunEvidenceChunk] = []
    heading = "ROOT"
    para_start = 1
    para_lines: List[str] = []

    def flush(end_line: int) -> None:
        nonlocal para_lines, para_start
        if not para_lines:
            return
        content = "\n".join(para_lines).strip()
        if content:
            chunks.append(
                RunEvidenceChunk(
                    run_id=run_id,
                    path=path,
                    section=heading,
                    line_range=[para_start, end_line],
                    text=content,
                    score=0.0,
                )
            )
        para_lines = []

    for idx, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if stripped.startswith("#"):
            flush(idx - 1)
            heading = stripped.lstrip("#").strip() or "ROOT"
            para_start = idx + 1
            continue
        if not stripped:
            flush(idx - 1)
            para_start = idx + 1
            continue
        if not para_lines:
            para_start = idx
        para_lines.append(raw)
    flush(len(lines))
    return chunks


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9_./:+-]+", _normalize_text(text).lower())


def _overlap_score(query: str, chunk_text: str) -> float:
    q = set(_tokenize(query))
    if not q:
        return 0.0
    c = set(_tokenize(chunk_text))
    if not c:
        return 0.0
    overlap = len(q.intersection(c))
    return float(overlap) / float(max(1, len(q)))


class HistoryReader:
    """Mini-model specialist for run-ledger retrieval and evidence packing."""

    def __init__(
        self,
        *,
        searcher: HybridRunLedgerSearcher,
        run_ledger_store: RunLedgerStore,
        system_root: Path,
        rerank_model: Optional[BaseChatModel] = None,
        max_candidate_runs: int = 6,
        max_candidate_chunks: int = 24,
        max_selected_chunks: int = 6,
    ) -> None:
        self.searcher = searcher
        self.run_ledger_store = run_ledger_store
        self.system_root = Path(system_root).expanduser().resolve()
        self.rerank_model = rerank_model
        self.max_candidate_runs = max(1, int(max_candidate_runs))
        self.max_candidate_chunks = max(4, int(max_candidate_chunks))
        self.max_selected_chunks = max(1, int(max_selected_chunks))

    async def aindex_entry(self, entry: RunLedgerEntry) -> None:
        """Update dense index for a newly written run-ledger entry."""
        try:
            query_vector = await self.searcher.embeddings.aembed_query(entry.search_blob_text)
            metadata = {
                "project_id": entry.project_id,
                "lane": entry.lane,
                "status": entry.status,
                "request": entry.request,
                "answer_summary": entry.answer_summary,
                "final_report_relpath": entry.final_report_relpath,
                "run_export_relpath": entry.run_export_relpath,
            }
            await asyncio.to_thread(
                self.searcher.vector_index.upsert,
                entry.run_id,
                query_vector,
                metadata,
            )
        except Exception as exc:
            logger.warning("history_reader index update failed for run_id=%s: %s", entry.run_id, exc)

    async def _model_select_indices(
        self,
        *,
        query: str,
        chunks: List[RunEvidenceChunk],
        max_pick: int,
    ) -> tuple[List[int], float]:
        if self.rerank_model is None:
            return [], 0.0
        lines: List[str] = []
        for i, ch in enumerate(chunks):
            snippet = _normalize_text(ch.text)
            if len(snippet) > 420:
                snippet = snippet[:417] + "..."
            line_range = ch.line_range if isinstance(ch.line_range, list) else [0, 0]
            lines.append(
                f"[{i}] run_id={ch.run_id} section={ch.section} "
                f"line_range={line_range} text={snippet}"
            )
        prompt = (
            "You are a retrieval specialist. Select the most relevant evidence chunks for the query.\n"
            "Return strict JSON only:\n"
            '{"selected_indices":[int,...],"confidence":0.0-1.0}\n'
            f"Max selected indices: {max_pick}\n\n"
            f"Query:\n{query}\n\nCandidates:\n" + "\n".join(lines)
        )
        try:
            resp = await self.rerank_model.ainvoke([HumanMessage(content=prompt)])
            text = llm_text(resp)
            payload = json.loads(text)
            idxs = payload.get("selected_indices") if isinstance(payload, dict) else []
            conf = float(payload.get("confidence", 0.0)) if isinstance(payload, dict) else 0.0
            out: List[int] = []
            if isinstance(idxs, list):
                for raw in idxs:
                    try:
                        val = int(raw)
                    except Exception:
                        continue
                    if 0 <= val < len(chunks) and val not in out:
                        out.append(val)
                    if len(out) >= max_pick:
                        break
            return out, max(0.0, min(1.0, conf))
        except Exception as exc:
            logger.warning("history_reader model rerank failed, fallback to deterministic: %s", exc)
            return [], 0.0

    @staticmethod
    def _context_text_from_chunks(chunks: List[RunEvidenceChunk], *, limit_chars: int = 3500) -> str:
        if not chunks:
            return "(none)"
        lines: List[str] = ["Relevant historical runs (auto-retrieved):"]
        for ch in chunks:
            snippet = _normalize_text(ch.text)
            if len(snippet) > 520:
                snippet = snippet[:517] + "..."
            lines.append(
                f"- [{ch.run_id}] {ch.section}: {snippet}"
            )
        text = "\n".join(lines)
        if len(text) > limit_chars:
            text = text[: limit_chars - 20].rstrip() + "\n...[truncated]"
        return text

    def _citations(self, chunks: List[RunEvidenceChunk]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[tuple[str, str, str, int, int]] = set()
        for ch in chunks:
            line_range = ch.line_range if isinstance(ch.line_range, list) and len(ch.line_range) >= 2 else [0, 0]
            key = (ch.run_id, ch.path, ch.section, int(line_range[0]), int(line_range[1]))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                {
                    "run_id": ch.run_id,
                    "path": ch.path,
                    "section": ch.section,
                    "line_range": [int(line_range[0]), int(line_range[1])],
                }
            )
        return out

    def _build_candidates(self, entries: List[RunLedgerEntry], query: str) -> List[RunEvidenceChunk]:
        chunks: List[RunEvidenceChunk] = []
        for entry in entries:
            report_path = _resolve_path(self.system_root, entry.final_report_relpath)
            report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
            if not report_text:
                export_path = _resolve_path(self.system_root, entry.run_export_relpath)
                export = _safe_json(export_path)
                report_text = str(export.get("answer_summary") or entry.answer_summary or "")
            rel_path = _to_project_relpath(
                self.system_root,
                report_path,
                fallback=entry.final_report_relpath,
            )
            md_chunks = _line_chunks_from_markdown(report_text, run_id=entry.run_id, path=rel_path)
            for ch in md_chunks:
                score = _overlap_score(query, ch.text)
                chunks.append(replace(ch, score=score))
        chunks.sort(key=lambda item: item.score, reverse=True)
        return chunks[: self.max_candidate_chunks]

    async def aload_candidate_chunks(
        self,
        *,
        query: str,
        project_id: str,
        limit: int = 12,
        lane: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[RunEvidenceChunk]:
        hits = await self.searcher.asearch(
            query=query,
            project_id=project_id,
            limit=max(1, int(limit)),
            sparse_k=max(12, int(limit) * 2),
            dense_k=max(12, int(limit) * 2),
            lane=lane,
            status=status,
        )
        if not hits:
            return []

        selected_hits: List[RunSearchHit] = []
        seen: set[str] = set()
        for hit in hits:
            rid = str(hit.run_id or "").strip()
            if not rid or rid in seen:
                continue
            seen.add(rid)
            selected_hits.append(hit)
            if len(selected_hits) >= self.max_candidate_runs:
                break
        entries = self.run_ledger_store.get_entries([hit.run_id for hit in selected_hits])
        if not entries:
            return []

        candidates = self._build_candidates(entries, query)
        if candidates:
            return candidates

        fallback: List[RunEvidenceChunk] = []
        for entry in entries:
            summary = _normalize_text(entry.answer_summary or entry.request)
            if not summary:
                continue
            fallback.append(
                RunEvidenceChunk(
                    run_id=entry.run_id,
                    path=entry.final_report_relpath or entry.run_export_relpath or "",
                    section="summary",
                    line_range=[0, 0],
                    text=summary,
                    score=_overlap_score(query, summary),
                )
            )
        fallback.sort(key=lambda item: item.score, reverse=True)
        return fallback[: self.max_candidate_chunks]

    async def aselect_relevant_chunks(
        self,
        *,
        query: str,
        chunks: List[RunEvidenceChunk],
        max_pick: Optional[int] = None,
    ) -> tuple[List[RunEvidenceChunk], float]:
        if not chunks:
            return [], 0.0
        rescored = [replace(chunk, score=_overlap_score(query, chunk.text)) for chunk in chunks]
        rescored.sort(key=lambda item: item.score, reverse=True)
        limit = max(1, int(max_pick or self.max_selected_chunks))
        chosen_idx, model_conf = await self._model_select_indices(
            query=query,
            chunks=rescored[: self.max_candidate_chunks],
            max_pick=limit,
        )
        if chosen_idx:
            selected = [rescored[i] for i in chosen_idx[:limit] if 0 <= i < len(rescored)]
            return selected, float(max(0.35, model_conf))
        selected = rescored[:limit]
        confidence = min(0.8, max(0.25, sum(item.score for item in selected) / max(1, len(selected))))
        return selected, float(max(0.0, min(1.0, confidence)))

    def citations_from_chunks(self, chunks: List[RunEvidenceChunk]) -> List[Dict[str, Any]]:
        return self._citations(chunks)

    async def aload_context(
        self,
        *,
        query: str,
        project_id: str,
        limit: int = 12,
        lane: Optional[str] = None,
        status: Optional[str] = None,
    ) -> HistoricalRunsContextPack:
        candidates = await self.aload_candidate_chunks(
            query=query,
            project_id=project_id,
            limit=limit,
            lane=lane,
            status=status,
        )
        if not candidates:
            return HistoricalRunsContextPack(
                query=query,
                selected_runs=[],
                context_text="(none)",
                citations=[],
                confidence=0.0,
            )
        selected, confidence = await self.aselect_relevant_chunks(query=query, chunks=candidates)

        context_text = self._context_text_from_chunks(selected)
        citations = self.citations_from_chunks(selected)
        selected_runs = []
        seen_runs: set[str] = set()
        for ch in selected:
            if ch.run_id in seen_runs:
                continue
            seen_runs.add(ch.run_id)
            selected_runs.append(ch.run_id)

        return HistoricalRunsContextPack(
            query=query,
            selected_runs=selected_runs,
            context_text=context_text,
            citations=citations,
            confidence=float(max(0.0, min(1.0, confidence))),
        )


__all__ = ["HistoryReader"]
