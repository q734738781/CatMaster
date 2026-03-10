from __future__ import annotations

from dataclasses import dataclass, field

from catmaster.runtime.literature.models import LiteratureContextPack
from catmaster.runtime.memory_store import MemoryStore

from .models import ExperimentRunPack, ResearchBoard, ResearchPlannerContextPack
from .store import ResearchStore


def _clean(text: str, *, max_chars: int) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)] + "..."


def _bullet_list(items: list[str], *, empty: str = "(none)") -> str:
    cleaned = [item for item in (" ".join(str(raw).split()).strip() for raw in items) if item]
    if not cleaned:
        return f"- {empty}"
    return "\n".join(f"- {item}" for item in cleaned)


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


def _first_nonempty_lines(text: str, *, limit: int = 8) -> str:
    out: list[str] = []
    for raw in str(text or "").splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        out.append(stripped)
        if len(out) >= limit:
            break
    return "\n".join(out).strip()


def _campaign_goal_text(question: str) -> str:
    return (
        "Conduct research about the following request and converge on the best supported answer.\n"
        f"Request: {question}"
    ).strip()


@dataclass(frozen=True)
class ResearchContextBudget:
    memory_index_max_chars: int = 1800
    constraints_tldr_max_chars: int = 480
    facts_tldr_max_chars: int = 420
    questions_tldr_max_chars: int = 420
    files_tldr_max_chars: int = 420
    history_max_chars: int = 1400
    recent_actions_limit: int = 6
    recent_artifacts_limit: int = 6
    hypothesis_limit: int = 8
    open_questions_limit: int = 8
    latest_literature_max_chars: int = 600
    latest_experiment_max_chars: int = 600
    current_best_answer_max_chars: int = 900
    human_feedback_max_chars: int = 1200


@dataclass
class ResearchContextBuilder:
    store: ResearchStore
    memory_store: MemoryStore
    budget: ResearchContextBudget = field(default_factory=ResearchContextBudget)

    def build_planner_context(
        self,
        *,
        board: ResearchBoard,
        latest_literature: LiteratureContextPack | None = None,
        latest_experiment: ExperimentRunPack | None = None,
        session_context: str = "",
    ) -> ResearchPlannerContextPack:
        literature_packs = self.store.load_literature_packs()
        experiment_packs = self.store.load_experiment_packs()
        latest_lit = latest_literature or (literature_packs[-1] if literature_packs else None)
        latest_exp = latest_experiment or (experiment_packs[-1] if experiment_packs else None)
        return ResearchPlannerContextPack(
            question=board.question,
            campaign_summary_md=self._campaign_summary(board, literature_packs, experiment_packs),
            hypothesis_snapshot_md=self._hypothesis_snapshot(board),
            recent_actions_md=self._recent_actions(board),
            budget_snapshot_md=self._budget_snapshot(board),
            durable_memory_summary_md=self._durable_memory_summary(board),
            session_context_md=_clean(
                session_context or "(empty)",
                max_chars=self.budget.history_max_chars,
            ),
            human_feedback_md=self._human_feedback(board),
            workspace_summary_md=self._workspace_summary(board, experiment_packs),
            latest_literature_summary_md=self._latest_literature_summary(latest_lit),
            latest_experiment_summary_md=self._latest_experiment_summary(latest_exp),
            current_best_answer_md=_clean(
                board.current_best_answer_md or "(empty)",
                max_chars=self.budget.current_best_answer_max_chars,
            ),
            open_questions_md=_bullet_list(list(board.open_questions)[: self.budget.open_questions_limit], empty="(none)"),
        )

    @staticmethod
    def render(pack: ResearchPlannerContextPack) -> str:
        return "\n\n".join(
            [
                f"Question:\n{pack.question}",
                f"Campaign goal:\n{_campaign_goal_text(pack.question)}",
                f"Campaign summary:\n{pack.campaign_summary_md}",
                f"Hypothesis snapshot:\n{pack.hypothesis_snapshot_md}",
                f"Recent actions:\n{pack.recent_actions_md}",
                f"Budget snapshot:\n{pack.budget_snapshot_md}",
                f"Durable memory summary:\n{pack.durable_memory_summary_md}",
                f"Chat session context:\n{pack.session_context_md or '(empty)'}",
                f"Human feedback:\n{pack.human_feedback_md or '(none)'}",
                f"Workspace summary:\n{pack.workspace_summary_md}",
                f"Latest literature:\n{pack.latest_literature_summary_md}",
                f"Latest experiment:\n{pack.latest_experiment_summary_md}",
                f"Current best answer:\n{pack.current_best_answer_md}",
                f"Open questions:\n{pack.open_questions_md}",
            ]
        ).strip()

    def _campaign_summary(
        self,
        board: ResearchBoard,
        literature_packs: list[LiteratureContextPack],
        experiment_packs: list[ExperimentRunPack],
    ) -> str:
        lines = [
            f"- status: {board.status}",
            f"- cycle: {board.cycle_index}/{board.max_cycles}",
            f"- hypotheses: {len(board.hypotheses)}",
            f"- literature packs: {len(literature_packs)}",
            f"- experiment packs: {len(experiment_packs)}",
        ]
        if board.memory_promotion_candidates:
            lines.append(f"- provisional memory candidates: {len(board.memory_promotion_candidates)}")
        return "\n".join(lines)

    def _hypothesis_snapshot(self, board: ResearchBoard) -> str:
        rows = [
            f"- {item.hypothesis_id} [{item.status}]: {item.text}" + (f" | note: {item.note}" if item.note else "")
            for item in board.hypotheses[: self.budget.hypothesis_limit]
        ]
        return "\n".join(rows) if rows else "- (none)"

    def _recent_actions(self, board: ResearchBoard) -> str:
        rows = [
            f"- {item.action_id} [{item.kind}/{item.status}]: {item.summary} | ref={item.ref_path}"
            for item in list(board.action_refs)[-self.budget.recent_actions_limit :]
        ]
        return "\n".join(rows) if rows else "- (none)"

    def _budget_snapshot(self, board: ResearchBoard) -> str:
        return "\n".join(
            [
                f"- cycles: {board.cycle_index}/{board.max_cycles}",
                f"- literature: {board.used_literature_queries}/{board.max_literature_queries}",
                f"- fast: {board.used_fast_runs}/{board.max_fast_runs}",
                f"- standard: {board.used_standard_runs}/{board.max_standard_runs}",
            ]
        )

    def _durable_memory_summary(self, board: ResearchBoard) -> str:
        parts = [
            "MEMORY.md index:",
            _clean(
                self.memory_store.read_index(max_chars=self.budget.memory_index_max_chars) or "(empty)",
                max_chars=self.budget.memory_index_max_chars,
            ),
            "",
            "CONSTRAINTS.md TL;DR:",
            self._topic_tldr("CONSTRAINTS", max_chars=self.budget.constraints_tldr_max_chars),
        ]
        if board.hypotheses or board.current_best_answer_md:
            parts.extend(
                [
                    "",
                    "FACTS.md TL;DR:",
                    self._topic_tldr("FACTS", max_chars=self.budget.facts_tldr_max_chars),
                ]
            )
        if board.open_questions or board.latest_human_questions:
            parts.extend(
                [
                    "",
                    "QUESTIONS.md TL;DR:",
                    self._topic_tldr("QUESTIONS", max_chars=self.budget.questions_tldr_max_chars),
                ]
            )
        return "\n".join(parts).strip()

    def _workspace_summary(self, board: ResearchBoard, experiment_packs: list[ExperimentRunPack]) -> str:
        artifact_rows: list[str] = []
        seen: set[str] = set()
        for pack in experiment_packs[-5:]:
            for artifact in pack.key_artifacts[: self.budget.recent_artifacts_limit]:
                path = str(artifact.path or "").strip()
                if not path or path in seen:
                    continue
                seen.add(path)
                artifact_rows.append(f"- {path}: {artifact.description}")
                if len(artifact_rows) >= self.budget.recent_artifacts_limit:
                    break
            if len(artifact_rows) >= self.budget.recent_artifacts_limit:
                break
        files_excerpt = self._topic_tldr("FILES", max_chars=self.budget.files_tldr_max_chars)
        lines = [
            "Recent artifact refs:",
            *(artifact_rows or ["- (none)"]),
            "",
            "Latest persisted refs:",
            f"- literature: {board.latest_literature_ref or '(none)'}",
            f"- experiment: {board.latest_experiment_ref or '(none)'}",
            f"- writer: {board.latest_writer_ref or '(none)'}",
            "",
            "Board refs:",
            *([f"- {item.ref_path}" for item in list(board.action_refs)[-self.budget.recent_actions_limit :]] or ["- (none)"]),
            "",
            "FILES.md TL;DR:",
            files_excerpt,
        ]
        dossier_path = self.store.files_root / "dossier" / "RESEARCH_DOSSIER.md"
        if dossier_path.exists():
            lines.extend(["", f"Dossier path: research/{board.campaign_id}/dossier/RESEARCH_DOSSIER.md"])
        if board.latest_writer_ref:
            lines.extend(["", f"Latest writer output: {board.latest_writer_ref}"])
        return "\n".join(lines).strip()

    def _human_feedback(self, board: ResearchBoard) -> str:
        if not board.latest_human_questions and not board.human_feedback_summary:
            return "(none)"
        lines: list[str] = []
        if board.latest_human_questions:
            lines.extend(
                [
                    "Latest ask-human questions:",
                    *[f"- {item}" for item in board.latest_human_questions[: self.budget.open_questions_limit]],
                ]
            )
        if board.human_feedback_summary:
            if lines:
                lines.append("")
            lines.extend(
                [
                    "Human feedback summary:",
                    _clean(board.human_feedback_summary, max_chars=self.budget.human_feedback_max_chars),
                ]
            )
        return "\n".join(lines).strip() or "(none)"

    def _topic_tldr(self, topic: str, *, max_chars: int) -> str:
        try:
            text = self.memory_store.read_topic(topic)
        except FileNotFoundError:
            return "(missing)"
        section = _extract_section(text, "## TL;DR") or _first_nonempty_lines(text)
        return _clean(section or "(empty)", max_chars=max_chars)

    def _latest_literature_summary(self, pack: LiteratureContextPack | None) -> str:
        if pack is None:
            return "(none)"
        lines = [
            f"- query: {pack.query}",
            f"- depth: {pack.depth}",
            f"- confidence: {pack.confidence}",
            f"- summary: {_clean(pack.summary, max_chars=self.budget.latest_literature_max_chars)}",
        ]
        if pack.citations:
            lines.append(f"- citations: {pack.citations[:5]}")
        return "\n".join(lines)

    def _latest_experiment_summary(self, pack: ExperimentRunPack | None) -> str:
        if pack is None:
            return "(none)"
        lines = [
            f"- id: {pack.experiment_id}",
            f"- lane: {pack.lane}",
            f"- status: {pack.status}",
            f"- summary: {_clean(pack.summary, max_chars=self.budget.latest_experiment_max_chars)}",
            f"- run ref: {pack.run_dir}",
        ]
        if pack.key_artifacts:
            lines.append(f"- artifacts: {[item.path for item in pack.key_artifacts[:5]]}")
        if pack.child_memory_candidates:
            lines.append(f"- provisional memory candidates: {len(pack.child_memory_candidates)}")
        return "\n".join(lines)


__all__ = ["ResearchContextBudget", "ResearchContextBuilder"]
