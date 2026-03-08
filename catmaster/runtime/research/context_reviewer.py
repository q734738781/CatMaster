from __future__ import annotations

from dataclasses import dataclass

from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.run_ledger.history_reader import HistoryReader
from catmaster.runtime.run_ledger.models import RunEvidenceChunk

from .models import ResearchBoard, ResearchContextReviewPack
from .store import ResearchStore


def _normalize(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _trim(text: str, *, max_chars: int = 260) -> str:
    compact = _normalize(text)
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)] + "..."


def _extract_section(text: str, heading: str) -> str:
    lines = str(text or "").splitlines()
    out: list[str] = []
    capture = False
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("## "):
            if capture:
                break
            capture = stripped.lower() == heading.lower()
            continue
        if capture:
            out.append(raw.rstrip())
    return "\n".join(out).strip()


@dataclass
class ResearchContextReviewer:
    history_reader: HistoryReader
    store: ResearchStore
    memory_store: MemoryStore
    project_id: str
    max_selected_chunks: int = 6

    async def areview(self, *, board: ResearchBoard) -> ResearchContextReviewPack:
        query = self._build_query(board)
        candidates = await self._history_chunks(query=query)
        candidates.extend(self._memory_chunks(board))
        candidates.extend(self._workspace_chunks(board))
        selected, confidence = await self.history_reader.aselect_relevant_chunks(
            query=query,
            chunks=candidates,
            max_pick=self.max_selected_chunks,
        )
        return ResearchContextReviewPack(
            query=query,
            history_focus_md=self._render_group(selected, source="history", title="Reviewed historical runs"),
            durable_memory_focus_md=self._render_group(selected, source="memory", title="Reviewed durable memory"),
            workspace_focus_md=self._render_group(selected, source="workspace", title="Reviewed workspace/artifacts"),
            citations=self.history_reader.citations_from_chunks(selected),
            confidence=confidence,
        )

    @staticmethod
    def render(pack: ResearchContextReviewPack) -> str:
        parts: list[str] = []
        if pack.history_focus_md:
            parts.append(pack.history_focus_md)
        if pack.durable_memory_focus_md:
            parts.append(pack.durable_memory_focus_md)
        if pack.workspace_focus_md:
            parts.append(pack.workspace_focus_md)
        if pack.citations:
            parts.append(
                "\n".join(
                    [
                        "Reviewed context citations:",
                        *[
                            f"- {item.get('run_id', '')}: {item.get('path', '')} [{item.get('section', '')}]"
                            for item in pack.citations[:8]
                        ],
                    ]
                )
            )
        if not parts:
            return "(none)"
        return "\n\n".join(parts).strip()

    def _build_query(self, board: ResearchBoard) -> str:
        lines = [f"Research question: {board.question}"]
        hypothesis_rows = self._hypothesis_focus(board)
        if hypothesis_rows:
            lines.append("Hypothesis focus: " + " | ".join(hypothesis_rows))
        if board.current_best_answer_md:
            lines.append(f"Current best answer: {board.current_best_answer_md}")
        latest_literature = self._latest_literature_signal()
        if latest_literature:
            lines.append("Latest literature: " + latest_literature)
        latest_experiment = self._latest_experiment_signal()
        if latest_experiment:
            lines.append("Latest experiment: " + latest_experiment)
        frontier = self._frontier_signal(board)
        if frontier:
            lines.append("Current frontier: " + " | ".join(frontier))
        if board.open_questions:
            lines.append("Open questions: " + " | ".join(board.open_questions[:4]))
        if board.human_feedback_summary:
            lines.append("Human feedback: " + board.human_feedback_summary)
        if board.latest_human_questions:
            lines.append("Ask-human focus: " + " | ".join(board.latest_human_questions[:4]))
        return "\n".join(lines).strip()

    @staticmethod
    def _hypothesis_focus(board: ResearchBoard, *, limit: int = 4) -> list[str]:
        prioritized = [
            item for item in board.hypotheses if item.status in {"active", "open", "seed", "supported", "weakened"}
        ]
        if not prioritized:
            prioritized = list(board.hypotheses)
        rows: list[str] = []
        for item in prioritized[:limit]:
            rows.append(f"{item.hypothesis_id}[{item.status}] {item.text}")
        return rows

    def _latest_literature_signal(self) -> str:
        packs = self.store.load_literature_packs()
        if not packs:
            return ""
        pack = packs[-1]
        rows = [f"query={pack.query}", f"depth={pack.depth}", f"summary={_trim(pack.summary, max_chars=180)}"]
        return " ; ".join(rows)

    def _latest_experiment_signal(self) -> str:
        packs = self.store.load_experiment_packs()
        if not packs:
            return ""
        pack = packs[-1]
        rows = [
            f"id={pack.experiment_id}",
            f"lane={pack.lane}",
            f"goal={_trim(pack.brief.goal, max_chars=120)}",
            f"summary={_trim(pack.summary, max_chars=180)}",
        ]
        return " ; ".join(rows)

    @staticmethod
    def _frontier_signal(board: ResearchBoard, *, limit: int = 4) -> list[str]:
        frontier: list[str] = []
        for item in list(board.action_refs)[-limit:]:
            frontier.append(f"{item.action_id}:{item.kind}/{item.status}")
        if board.used_literature_queries < board.max_literature_queries:
            frontier.append("literature_budget_available")
        if board.used_fast_runs < board.max_fast_runs:
            frontier.append("fast_budget_available")
        if board.used_standard_runs < board.max_standard_runs:
            frontier.append("standard_budget_available")
        return frontier[:limit]

    async def _history_chunks(self, *, query: str) -> list[RunEvidenceChunk]:
        return await self.history_reader.aload_candidate_chunks(
            query=query,
            project_id=self.project_id,
            lane=None,
        )

    def _memory_chunks(self, board: ResearchBoard) -> list[RunEvidenceChunk]:
        chunks: list[RunEvidenceChunk] = []
        topics = ["CONSTRAINTS"]
        if board.hypotheses or board.current_best_answer_md:
            topics.append("FACTS")
        if board.open_questions or board.latest_human_questions:
            topics.append("QUESTIONS")
        for topic in topics:
            try:
                raw = self.memory_store.read_topic(topic)
            except FileNotFoundError:
                continue
            summary = _extract_section(raw, "## TL;DR") or _normalize(raw)
            if not summary:
                continue
            chunks.append(
                RunEvidenceChunk(
                    run_id="memory",
                    path=f"MEMORY/topics/{topic}.md",
                    section="TL;DR",
                    line_range=[0, 0],
                    text=summary,
                    score=0.0,
                )
            )
        return chunks

    def _workspace_chunks(self, board: ResearchBoard) -> list[RunEvidenceChunk]:
        chunks: list[RunEvidenceChunk] = []
        for ref in list(board.action_refs)[-6:]:
            chunks.append(
                RunEvidenceChunk(
                    run_id="workspace",
                    path=ref.ref_path,
                    section=f"{ref.kind}/{ref.status}",
                    line_range=[0, 0],
                    text=f"{ref.action_id}: {ref.summary}",
                    score=0.0,
                )
            )
        for ref_path in (board.latest_literature_ref, board.latest_experiment_ref, board.latest_writer_ref):
            path = str(ref_path or "").strip()
            if not path:
                continue
            chunks.append(
                RunEvidenceChunk(
                    run_id="workspace",
                    path=path,
                    section="latest-ref",
                    line_range=[0, 0],
                    text=path,
                    score=0.0,
                )
            )
        return chunks

    def _render_group(self, chunks: list[RunEvidenceChunk], *, source: str, title: str) -> str:
        rows: list[str] = []
        for chunk in chunks:
            if source == "history" and chunk.run_id in {"memory", "workspace"}:
                continue
            if source == "memory" and chunk.run_id != "memory":
                continue
            if source == "workspace" and chunk.run_id != "workspace":
                continue
            rows.append(f"- {chunk.path} [{chunk.section}]: {_trim(chunk.text)}")
        if not rows:
            return ""
        return "\n".join([f"{title}:", *rows]).strip()


__all__ = ["ResearchContextReviewer"]
