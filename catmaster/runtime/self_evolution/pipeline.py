from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from catmaster.llm.config import LLMProfile

from .agents import ProposerAgent, ReviewerAgent, build_self_evolution_agents, prepare_candidate_workspace
from .gate import CandidateGate
from .models import LearningCandidate, SKILL_GROUPS, SelfEvolutionJob
from .promotion import PromotionConflict, PromotionManager
from .settings import (
    SelfEvolutionMode,
    resolve_self_evolution_mode,
    self_evolution_enqueue_enabled,
    self_evolution_promotion_enabled,
)
from .storage import SelfEvolutionStore, hash_text, hash_tree, stable_id, utc_now
from .trace import TERMINAL_STATUSES, collect_turn_trace


logger = logging.getLogger(__name__)


class SelfEvolutionCoordinator:
    def __init__(
        self,
        *,
        workspace: Path | str,
        project_id: str = "",
        model_config: str = "",
        mode: str | None = None,
        proposer: ProposerAgent | Any | None = None,
        reviewer: ReviewerAgent | Any | None = None,
        repo_root: Path | str | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.project_id = str(project_id or self.workspace.name).strip() or self.workspace.name
        self.model_config = str(model_config or "").strip()
        self.mode: SelfEvolutionMode = resolve_self_evolution_mode(mode)
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[3]).expanduser().resolve()
        self.store = SelfEvolutionStore(self.workspace, project_id=self.project_id)
        self.gate = CandidateGate(self.store)
        self.promotion = PromotionManager(self.store, repo_root=self.repo_root)
        self._proposer = proposer
        self._reviewer = reviewer

    def enqueue_post_run(
        self,
        *,
        run_id: str,
        thread_id: str = "",
        message_id: str = "",
        entrypoint: str = "",
        terminal_status: str,
        run_dir: Path | str,
        payload: dict[str, Any] | None = None,
        model_config: str = "",
    ) -> SelfEvolutionJob | None:
        if not self_evolution_enqueue_enabled(self.mode):
            return None
        status = str(terminal_status or "").strip().lower()
        if status not in TERMINAL_STATUSES:
            return None
        return self.store.enqueue_job(
            trigger_kind="post_run",
            run_id=run_id,
            run_dir=run_dir,
            thread_id=thread_id,
            payload={
                **dict(payload or {}),
                "message_id": str(message_id or ""),
                "entrypoint": str(entrypoint or ""),
                "terminal_status": status,
            },
            model_config=str(model_config or self.model_config),
        )

    def process_pending_jobs(self, *, limit: int = 4) -> list[SelfEvolutionJob]:
        if not self_evolution_enqueue_enabled(self.mode):
            return []
        processed: list[SelfEvolutionJob] = []
        for job in self.store.claim_jobs(limit=limit):
            try:
                processed.append(self._process_job(job))
            except Exception as exc:
                logger.exception("Self-evolution job %s failed", job.job_id)
                processed.append(self.store.finish_job(job, status="error", error=f"{type(exc).__name__}: {exc}"))
        return processed

    def _agents(self, job: SelfEvolutionJob) -> tuple[Any, Any]:
        if self._proposer is not None and self._reviewer is not None:
            return self._proposer, self._reviewer
        profile = LLMProfile.from_env_or_file(job.model_config or self.model_config or None)
        return build_self_evolution_agents(profile, workspace=self.workspace)

    def _process_job(self, job: SelfEvolutionJob) -> SelfEvolutionJob:
        trace = collect_turn_trace(
            run_dir=job.run_dir,
            fallback={"run_id": job.run_id, "thread_id": job.thread_id, **dict(job.payload)},
        )
        if not trace.has_user_content():
            return self.store.finish_job(job, status="done")

        candidate_id = "sec_" + stable_id(self.project_id, job.job_id, length=28)
        candidate_root = prepare_candidate_workspace(
            store=self.store,
            candidate_id=candidate_id,
            trace=trace,
            repo_root=self.repo_root,
        )
        proposer, reviewer = self._agents(job)
        proposal, proposer_meta = proposer.propose(candidate_root=candidate_root)
        if proposal.action == "ignore":
            shutil.rmtree(candidate_root)
            self.store.append_audit_event(
                {
                    "event": "proposer_ignored",
                    "job_id": job.job_id,
                    "run_id": job.run_id,
                    "rationale": proposal.rationale,
                    "proposer": proposer_meta,
                }
            )
            return self.store.finish_job(job, status="done")

        if proposal.action == "skill":
            self._discard_unchanged_memory_candidate(candidate_root)
        group = str(proposal.group or "").strip() if proposal.action == "skill" else ""
        name = str(proposal.name or "").strip() if proposal.action == "skill" else ""
        base_hash = self._base_target_hash(
            action=proposal.action,
            group=group,
            name=name,
            candidate_root=candidate_root,
        )
        bundle_hash = (
            hash_text((candidate_root / "memories" / "AGENTS.md").read_text(encoding="utf-8", errors="replace"))
            if proposal.action == "memory" and (candidate_root / "memories" / "AGENTS.md").is_file()
            else hash_tree(candidate_root / "proposed" / group / name)
            if proposal.action == "skill" and group in SKILL_GROUPS and name
            else ""
        )
        candidate = LearningCandidate(
            candidate_id=candidate_id,
            project_id=self.project_id,
            run_id=job.run_id,
            thread_id=job.thread_id,
            action=proposal.action,
            group=group,
            name=name,
            rationale=proposal.rationale,
            base_target_hash=base_hash,
            bundle_hash=bundle_hash,
            review={"proposer": proposer_meta},
            created_at=utc_now(),
        )
        self.store.write_candidate(candidate)
        report = self.gate.run(candidate)
        candidate.validation = report.to_dict()
        self.store.write_validation_report(report)
        if not report.valid:
            candidate.status = "invalid"
            self.store.write_candidate(candidate)
            self._cleanup_review_context(candidate_root)
            self.store.append_audit_event(
                {
                    "event": "candidate_invalid",
                    "candidate_id": candidate_id,
                    "errors": report.errors,
                }
            )
            return self.store.finish_job(job, status="done", candidate_id=candidate_id)

        try:
            review, reviewer_meta = reviewer.review(
                candidate_root=candidate_root,
                action=candidate.action,
                group=candidate.group,
                name=candidate.name,
                rationale=candidate.rationale,
                validation=report.to_dict(),
            )
        except Exception as exc:
            candidate.review = {
                "decision": "unavailable",
                "error": f"{type(exc).__name__}: {exc}",
                "proposer": proposer_meta,
            }
            self.store.write_candidate(candidate)
            self._cleanup_review_context(candidate_root)
            return self.store.finish_job(
                job,
                status="error",
                candidate_id=candidate_id,
                error=candidate.review["error"],
            )
        candidate.review = {
            "decision": review.decision,
            "rationale": review.rationale,
            "proposer": proposer_meta,
            "reviewer": reviewer_meta,
        }
        (candidate_root / "review.json").write_text(
            json.dumps(candidate.review, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        if review.decision == "reject":
            candidate.status = "rejected"
            self.store.write_candidate(candidate)
            self._cleanup_review_context(candidate_root)
            self.store.append_audit_event(
                {
                    "event": "candidate_rejected",
                    "candidate_id": candidate_id,
                    "rationale": review.rationale,
                }
            )
            return self.store.finish_job(job, status="done", candidate_id=candidate_id)

        candidate.status = "approved"
        self.store.write_candidate(candidate)
        try:
            if self_evolution_promotion_enabled(self.mode):
                candidate = self.promotion.promote(candidate, report)
        except PromotionConflict as exc:
            candidate.status = "conflict"
            candidate.promotion = {**dict(candidate.promotion), "error": str(exc)}
            self.store.write_candidate(candidate)
        except Exception as exc:
            candidate.promotion = {**dict(candidate.promotion), "error": f"{type(exc).__name__}: {exc}"}
            self.store.write_candidate(candidate)
            return self.store.finish_job(
                job,
                status="error",
                candidate_id=candidate_id,
                error=str(candidate.promotion["error"]),
            )
        finally:
            self._cleanup_review_context(candidate_root)
        return self.store.finish_job(job, status="done", candidate_id=candidate_id)

    def _base_target_hash(self, *, action: str, group: str, name: str, candidate_root: Path) -> str:
        if action == "memory":
            frozen = candidate_root / "current" / "AGENTS.md"
            if frozen.is_file():
                return hash_text(frozen.read_text(encoding="utf-8", errors="replace"))
            return self.store.memory_hash()
        if group not in SKILL_GROUPS or not name:
            return ""
        frozen_target = candidate_root / "current" / "target"
        if frozen_target.is_dir():
            return hash_tree(frozen_target)
        workspace_target = self.store.self_develop_skills_dir / group / name
        if workspace_target.is_dir():
            return hash_tree(workspace_target)
        return hash_tree(self.repo_root / "skills" / group / name)

    @staticmethod
    def _discard_unchanged_memory_candidate(candidate_root: Path) -> None:
        current = candidate_root / "current" / "AGENTS.md"
        proposed = candidate_root / "memories" / "AGENTS.md"
        if not current.is_file() or not proposed.is_file():
            return
        if current.read_bytes() != proposed.read_bytes():
            return
        proposed.unlink()
        proposed.parent.rmdir()

    @staticmethod
    def _cleanup_review_context(candidate_root: Path) -> None:
        current = candidate_root / "current"
        if current.exists():
            shutil.rmtree(current)


__all__ = ["SelfEvolutionCoordinator"]
