from __future__ import annotations

import json
import logging
import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from catmaster.llm.config import LLMProfile

from .agents import (
    ProposerAgent,
    ReviewerAgent,
    build_self_evolution_agents,
    prepare_candidate_workspace,
)
from .consolidation import ConsolidationService, EvidenceBatch
from .gate import CandidateGate
from .models import (
    CandidateRevision,
    LearningCandidate,
    Observation,
    ProposerResult,
    ReflectionResult,
    SKILL_GROUPS,
    SelfEvolutionJob,
    ValidationReport,
)
from .promotion import PromotionManager
from .settings import SelfEvolutionMode, resolve_self_evolution_mode, self_evolution_enqueue_enabled
from .storage import SelfEvolutionStore, hash_text, hash_tree, stable_id, utc_now
from .telemetry import finalize_skill_run_telemetry
from .trace import (
    TERMINAL_STATUSES,
    TurnTrace,
    collect_turn_trace,
)


logger = logging.getLogger(__name__)


def _run_state(run_dir: Path | str) -> dict[str, Any]:
    try:
        value = json.loads(
            (Path(run_dir).expanduser().resolve() / "run_state.json").read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


class SelfEvolutionCoordinator:
    """Workspace evidence governance and human-controlled release coordinator."""

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
        worker_id: str = "",
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.project_id = str(project_id or self.workspace.name).strip() or self.workspace.name
        self.model_config = str(model_config or "").strip()
        self.mode: SelfEvolutionMode = resolve_self_evolution_mode(mode)
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[3]).expanduser().resolve()
        self.store = SelfEvolutionStore(self.workspace, project_id=self.project_id)
        self.gate = CandidateGate(self.store)
        self.promotion = PromotionManager(self.store, repo_root=self.repo_root)
        self.consolidation = ConsolidationService(self.store)
        self._proposer = proposer
        self._reviewer = reviewer
        self.worker_id = (
            str(worker_id or "").strip()
            or f"self-evolution-{os.getpid()}-{stable_id(id(self), length=8)}"
        )

    # -- enqueue ---------------------------------------------------------

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
        supplied = dict(payload or {})
        state = _run_state(run_dir)
        if not str(
            state.get("user_prompt")
            or state.get("resume_guidance")
            or supplied.get("prompt")
            or supplied.get("resume_guidance")
            or ""
        ).strip():
            return None
        return self.store.enqueue_job(
            trigger_kind="post_run",
            run_id=run_id,
            run_dir=run_dir,
            thread_id=thread_id,
            payload={
                **supplied,
                "message_id": str(message_id or ""),
                "entrypoint": str(entrypoint or ""),
                "execution_status": status,
                "task_outcome": str(supplied.get("task_outcome") or ""),
                "outcome_ref": str(supplied.get("outcome_ref") or ""),
            },
            model_config=str(model_config or self.model_config),
        )

    def enqueue_explicit_learn(
        self,
        *,
        run_id: str,
        run_dir: Path | str,
        note: str,
        thread_id: str = "",
        model_config: str = "",
        actor: str = "",
    ) -> SelfEvolutionJob:
        correction = str(note or "").strip()
        if not correction:
            raise ValueError("a concrete durable correction is required")
        state = _run_state(run_dir)
        job = self.store.enqueue_job(
            trigger_kind="explicit_learn",
            run_id=run_id,
            run_dir=run_dir,
            thread_id=(
                thread_id
                or str(state.get("webui_thread_id") or state.get("thread_id") or "")
            ),
            payload={
                "note": correction,
                "actor": str(actor or "").strip(),
                "execution_status": str(state.get("status") or "unknown"),
            },
            model_config=str(model_config or self.model_config),
        )
        self.store.append_audit_event(
            {
                "event": "explicit_learn_queued",
                "job_id": job.job_id,
                "run_id": job.run_id,
                "thread_id": job.thread_id,
                "actor": str(actor or "").strip(),
            }
        )
        return job

    def enqueue_revision(
        self,
        *,
        candidate_id: str,
        expected_revision: int,
        guidance: str,
        actor: str,
        model_config: str = "",
    ) -> SelfEvolutionJob:
        candidate = self.store.read_candidate(candidate_id)
        if candidate is None:
            raise ValueError("candidate not found")
        if candidate.revision != int(expected_revision):
            raise ValueError("candidate revision changed; reopen the exact revision")
        if candidate.status != "revision":
            raise ValueError("request the revision before enqueueing revision work")
        return self.store.enqueue_job(
            trigger_kind="candidate_revision",
            run_id=f"revision-{candidate_id}-r{expected_revision:04d}",
            run_dir=self.store.revision_dir(candidate_id, expected_revision),
            thread_id=candidate.thread_id,
            payload={
                "candidate_id": candidate_id,
                "expected_revision": int(expected_revision),
                "guidance": str(guidance or "").strip(),
                "actor": str(actor or "").strip(),
            },
            model_config=str(model_config or self.model_config),
        )

    # -- worker ----------------------------------------------------------

    def process_pending_jobs(self, *, limit: int = 4) -> list[SelfEvolutionJob]:
        if not self_evolution_enqueue_enabled(self.mode):
            return []
        processed: list[SelfEvolutionJob] = []
        jobs = self.store.claim_jobs(
            limit=limit,
            project_id=self.project_id,
            owner=self.worker_id,
            lease_seconds=600,
        )
        for job in jobs:
            try:
                with self._lease_heartbeat(job):
                    processed.append(self._process_job(job))
            except FileExistsError as exc:
                logger.warning("Self-evolution job %s needs recovery review: %s", job.job_id, exc)
                processed.append(
                    self.store.finish_job(
                        job,
                        status="recovery_review",
                        error=f"Immutable revision already exists: {exc}",
                        owner=self.worker_id,
                    )
                )
            except Exception as exc:
                logger.exception("Self-evolution job %s failed", job.job_id)
                processed.append(
                    self.store.finish_job(
                        job,
                        status="error",
                        candidate_id=job.candidate_id,
                        error=f"{type(exc).__name__}: {exc}",
                        owner=self.worker_id,
                    )
                )
        return processed

    @contextmanager
    def _lease_heartbeat(self, job: SelfEvolutionJob) -> Iterator[None]:
        stopped = threading.Event()

        def beat() -> None:
            while not stopped.wait(30):
                if not self.store.heartbeat_job(
                    job.job_id,
                    owner=self.worker_id,
                    lease_seconds=600,
                ):
                    logger.error("Lost self-evolution lease for %s", job.job_id)
                    return

        thread = threading.Thread(target=beat, name=f"lease-{job.job_id}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(timeout=2)

    def _process_job(self, job: SelfEvolutionJob) -> SelfEvolutionJob:
        if job.trigger_kind == "candidate_revision":
            candidate_id = self._process_revision_job(job)
            return self.store.finish_job(
                job,
                status="done",
                candidate_id=candidate_id,
                owner=self.worker_id,
            )

        trace = collect_turn_trace(
            run_dir=job.run_dir,
            fallback={
                "run_id": job.run_id,
                "thread_id": job.thread_id,
                **dict(job.payload),
            },
        )
        self._finalize_run_telemetry(job, trace)
        proposer, _reviewer = self._agents(job)
        reflection, _reflection_meta = proposer.reflect(
            trajectory_markdown=trace.to_markdown(),
            skill_catalog=self._skill_catalog(),
            prior_targets=self.store.list_observation_targets(),
        )
        observation = self._observation_from_reflection(
            job=job,
            trace=trace,
            reflection=reflection,
        )
        if observation is None:
            return self.store.finish_job(job, status="done", owner=self.worker_id)
        observation = self.store.write_observation(observation)
        batch = self.consolidation.batch_for(observation)
        route, owner_group, owner_name = self._target_details(
            batch.target
        )
        candidate_id = "sec_" + stable_id("target", batch.target, length=28)
        existing = self.store.read_candidate(candidate_id)
        if existing is not None and observation.observation_id in {
            str(item) for item in existing.evidence_ids
        }:
            self.store.set_observation_status(
                [observation.observation_id],
                "consolidated",
            )
            return self.store.finish_job(
                job,
                status="done",
                candidate_id=existing.candidate_id,
                owner=self.worker_id,
            )
        evidence_markdown = self.consolidation.evidence_markdown(
            batch,
            traces=self._traces_for_batch(
                batch,
                current_observation_id=observation.observation_id,
                current_trace=trace,
            ),
        )
        candidate = self._build_candidate_revision(
            job=job,
            candidate_id=candidate_id,
            revision=(existing.revision + 1 if existing is not None else 1),
            route=route,
            evidence_ids=list(batch.evidence_ids),
            evidence_markdown=evidence_markdown,
            owner_group=owner_group,
            owner_name=owner_name,
            run_id=(existing.run_id if existing is not None else job.run_id),
            thread_id=(existing.thread_id if existing is not None else job.thread_id),
        )
        if candidate is not None:
            self.store.set_observation_status(
                list(batch.evidence_ids),
                "consolidated",
            )
        return self.store.finish_job(
            job,
            status="done",
            candidate_id=(candidate.candidate_id if candidate is not None else ""),
            owner=self.worker_id,
        )

    def _finalize_run_telemetry(
        self,
        job: SelfEvolutionJob,
        trace: TurnTrace,
    ) -> None:
        if job.trigger_kind != "post_run":
            return
        records = finalize_skill_run_telemetry(
            store=self.store,
            run_id=job.run_id,
            run_dir=job.run_dir,
            task_outcome=trace.task_outcome,
            outcome_ref=trace.outcome_ref,
        )
        for record in records:
            failure_reason = ""
            if record.false_activation:
                failure_reason = (
                    "The exact selected canary falsely activated in this run."
                )
            elif record.used and record.outcome == "verified_failure":
                failure_reason = (
                    "The exact selected canary was used in a verified failed run."
                )
            if failure_reason:
                self.promotion.stop_canary_on_failure(
                    skill_name=record.skill_name,
                    skill_version=record.skill_version,
                    run_id=record.run_id,
                    reason=failure_reason,
                )

    def _observation_from_reflection(
        self,
        *,
        job: SelfEvolutionJob,
        trace: TurnTrace,
        reflection: ReflectionResult,
    ) -> Observation | None:
        if reflection.kind in {"no_change", "execution_lapse"}:
            return None
        group = str(reflection.group or "").strip()
        name = str(reflection.name or "").strip()
        if reflection.kind == "workspace_preference":
            if not name or any(
                char
                not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                for char in name
            ):
                return None
            target = f"memory/{name}"
            signal_kind = "workspace_preference"
        else:
            if (
                group not in SKILL_GROUPS
                or not name
                or any(
                    char
                    not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-"
                    for char in name
                )
            ):
                return None
            target = f"{group}/{name}"
            exists = self._skill_target_exists(group=group, name=name)
            if reflection.kind == "skill_revision" and not exists:
                return None
            if reflection.kind == "skill_discovery" and exists:
                return None
            signal_kind = reflection.kind

        change = str(reflection.change or "").strip()
        if not change:
            return None
        available_refs = {
            str(item.get("source_ref") or "").strip()
            for item in trace.events
            if isinstance(item, dict)
            and str(item.get("source_ref") or "").strip()
        }
        run_ref = f"run:{job.run_id}"
        selected_refs = [
            str(item).strip()
            for item in reflection.evidence_refs
            if str(item).strip() in available_refs
        ]
        if not selected_refs:
            selected_refs = [run_ref]
        evidence_refs = [
            {
                "source_ref": item,
                "reason": (
                    "explicit_user_correction"
                    if job.trigger_kind == "explicit_learn"
                    else "semantic_reflection"
                ),
            }
            for item in dict.fromkeys(selected_refs)
        ]
        if job.trigger_kind == "explicit_learn":
            evidence_refs.insert(
                0,
                {
                    "source_ref": f"job:{job.job_id}",
                    "reason": "explicit_user_correction",
                },
            )
        return Observation(
            observation_id="seo_"
            + stable_id(job.job_id, target, change, length=28),
            run_id=job.run_id,
            thread_id=job.thread_id or trace.thread_id,
            signal_kind=signal_kind,  # type: ignore[arg-type]
            target=target,
            claim=change,
            evidence_refs=evidence_refs,
            outcome_ref=trace.outcome_ref,
            created_at=utc_now(),
        )

    # -- candidate revisions --------------------------------------------

    def _agents(self, job: SelfEvolutionJob) -> tuple[Any, Any]:
        if self._proposer is not None and self._reviewer is not None:
            return self._proposer, self._reviewer
        profile = LLMProfile.from_env_or_file(job.model_config or self.model_config or None)
        return build_self_evolution_agents(profile, workspace=self.workspace)

    def _skill_catalog(self) -> str:
        effective: dict[tuple[str, str], Path] = {}
        for root in (self.repo_root / "skills", self.store.self_develop_skills_dir):
            if not root.is_dir():
                continue
            for skill_md in root.glob("*/*/SKILL.md"):
                group = skill_md.parent.parent.name
                name = skill_md.parent.name
                if group in SKILL_GROUPS:
                    effective[(group, name)] = skill_md
        lines = [
            (
                f"- `{group}/{name}`: "
                f"{self._skill_description(skill_md) or 'No description recorded.'}"
            )
            for (group, name), skill_md in sorted(effective.items())
        ]
        return "\n".join(lines) if lines else "No skills are currently available."

    @staticmethod
    def _skill_description(skill_md: Path) -> str:
        try:
            lines = skill_md.read_text(
                encoding="utf-8",
                errors="replace",
            ).splitlines()
        except OSError:
            return ""
        for line in lines[:40]:
            if line.startswith("description:"):
                return line.split(":", 1)[1].strip().strip("'\"")
        return ""

    def _skill_target_exists(self, *, group: str, name: str) -> bool:
        return any(
            (root / group / name / "SKILL.md").is_file()
            for root in (
                self.store.self_develop_skills_dir,
                self.repo_root / "skills",
            )
        )

    def _target_details(self, target: str) -> tuple[str, str, str]:
        if target.startswith("memory/") and target.partition("/")[2]:
            return "workspace_preference", "", ""
        group, separator, name = str(target or "").partition("/")
        if (
            not separator
            or group not in SKILL_GROUPS
            or not name
        ):
            raise ValueError(f"invalid reflected skill target: {target!r}")
        if self._skill_target_exists(group=group, name=name):
            return "amend_existing_skill", group, name
        return "new_skill", group, name

    def _traces_for_batch(
        self,
        batch: EvidenceBatch,
        *,
        current_observation_id: str = "",
        current_trace: TurnTrace | None = None,
    ) -> dict[str, TurnTrace]:
        traces: dict[str, TurnTrace] = {}
        for observation in batch.observations:
            if (
                current_trace is not None
                and observation.observation_id == current_observation_id
            ):
                traces[observation.observation_id] = current_trace
                continue
            run_dir = self.store.run_dir_for(observation.run_id)
            if run_dir is None:
                continue
            traces[observation.observation_id] = collect_turn_trace(
                run_dir=run_dir,
                fallback={
                    "run_id": observation.run_id,
                    "thread_id": observation.thread_id,
                    "note": self._explicit_correction_for(observation),
                },
            )
        return traces

    def _explicit_correction_for(self, observation: Observation) -> str:
        for ref in observation.evidence_refs:
            if not isinstance(ref, dict):
                continue
            source_ref = str(ref.get("source_ref") or "").strip()
            if not source_ref.startswith("job:"):
                continue
            job = self.store.read_job(source_ref.removeprefix("job:"))
            if job is not None and job.trigger_kind == "explicit_learn":
                return str(job.payload.get("note") or "").strip()
        return ""

    def _process_revision_job(self, job: SelfEvolutionJob) -> str:
        candidate_id = str(job.payload.get("candidate_id") or "").strip()
        expected_revision = int(job.payload.get("expected_revision") or 0)
        guidance = str(job.payload.get("guidance") or "").strip()
        if not candidate_id or not guidance:
            raise ValueError("candidate revision job is missing candidate_id or guidance")
        current = self.store.read_candidate(candidate_id)
        if current is None:
            raise ValueError("candidate not found")
        if current.revision != expected_revision or current.status != "revision":
            raise ValueError("candidate changed before revision work began")
        old_root = self.store.revision_dir(candidate_id, expected_revision)
        old_evidence = (old_root / "evidence.md").read_text(
            encoding="utf-8",
            errors="replace",
        )
        old_review = self._read_json(old_root / "review.json")
        evidence_markdown = "\n".join(
            [
                old_evidence.rstrip(),
                "",
                "## Human revision guidance",
                "",
                guidance,
                "",
                "## Prior reviewer concerns",
                "",
                *[f"- {item}" for item in list(old_review.get("concerns") or [])[:20]],
                "",
            ]
        )
        revised = self._build_candidate_revision(
            job=job,
            candidate_id=candidate_id,
            revision=expected_revision + 1,
            route=current.route,
            evidence_ids=list(current.evidence_ids),
            evidence_markdown=evidence_markdown,
            owner_group=current.group,
            owner_name=current.name,
            run_id=current.run_id,
            thread_id=current.thread_id,
        )
        if revised is None:
            raise ValueError("the revision proposer found no justified candidate")
        self.store.append_audit_event(
            {
                "event": "candidate_revision_created",
                "candidate_id": candidate_id,
                "from_revision": expected_revision,
                "to_revision": revised.revision,
                "actor": str(job.payload.get("actor") or ""),
            }
        )
        return candidate_id

    def _build_candidate_revision(
        self,
        *,
        job: SelfEvolutionJob,
        candidate_id: str,
        revision: int,
        route: str,
        evidence_ids: list[str],
        evidence_markdown: str,
        owner_group: str,
        owner_name: str,
        run_id: str,
        thread_id: str,
    ) -> LearningCandidate | None:
        candidate_root = prepare_candidate_workspace(
            store=self.store,
            candidate_id=candidate_id,
            repo_root=self.repo_root,
            revision=revision,
            evidence_markdown=evidence_markdown,
            owner_group=owner_group,
            owner_name=owner_name,
        )
        proposer, reviewer = self._agents(job)
        proposal, _proposer_meta = proposer.propose(candidate_root=candidate_root)
        if proposal.action == "ignore":
            shutil.rmtree(candidate_root)
            self.store.append_audit_event(
                {
                    "event": "proposer_declined_candidate",
                    "candidate_id": candidate_id,
                    "revision": revision,
                    "reason": proposal.rationale,
                }
            )
            return None

        action = "memory" if route == "workspace_preference" else "skill"
        group = str(owner_group or "").strip() if action == "skill" else ""
        name = str(owner_name or "").strip() if action == "skill" else ""
        if action == "skill":
            self._discard_unchanged_memory_copy(candidate_root)
        base_hash = self._base_target_hash(
            action=action,
            group=group,
            name=name,
            candidate_root=candidate_root,
        )
        bundle_hash = self._bundle_hash(
            action=action,
            group=group,
            name=name,
            candidate_root=candidate_root,
        )
        candidate = LearningCandidate(
            candidate_id=candidate_id,
            project_id=self.project_id,
            run_id=run_id,
            thread_id=thread_id,
            action=action,  # type: ignore[arg-type]
            status="pending",
            route=route,  # type: ignore[arg-type]
            group=group,
            name=name,
            rationale=proposal.rationale,
            evidence_ids=list(evidence_ids),
            revision=revision,
            base_target_hash=base_hash,
            bundle_hash=bundle_hash,
            created_at=utc_now(),
        )
        self._seal_payload(candidate_root)
        self.store.write_candidate(candidate)
        job.candidate_id = candidate_id

        route_errors = self._route_contract_errors(
            route=route,
            proposal=proposal,
            owner_group=owner_group,
            owner_name=owner_name,
        )
        report = self.gate.run(candidate)
        if route_errors:
            report.errors.extend(route_errors)
            report.valid = False
        self.store.write_validation_report(report, revision=revision)
        candidate.validation = report.to_dict()

        revision_record = CandidateRevision(
            candidate_id=candidate_id,
            revision=revision,
            route=route,  # type: ignore[arg-type]
            target=(
                {"path": "/memories/AGENTS.md"}
                if action == "memory"
                else {"group": group, "name": name}
            ),
            delta_operation=proposal.delta_operation,
            evidence_ids=tuple(evidence_ids),
            applicability_boundary=tuple(proposal.applicability_boundary),
            non_applicability=tuple(proposal.non_applicability),
            expected_step_change=proposal.expected_step_change,
            created_at=utc_now(),
        )
        proposal_artifact = {
            **revision_record.to_dict(),
            "rationale": proposal.rationale,
        }
        self.store.write_revision_json(candidate_id, revision, "proposal.json", proposal_artifact)
        if not report.valid:
            self.store.update_candidate_status(candidate_id, "revision")
            self.store.append_audit_event(
                {
                    "event": "candidate_invalid",
                    "candidate_id": candidate_id,
                    "revision": revision,
                    "errors": list(report.errors),
                }
            )
            return self.store.read_candidate(candidate_id)

        review, _reviewer_meta = reviewer.review(
            candidate_root=candidate_root,
            action=candidate.action,
            group=candidate.group,
            name=candidate.name,
            rationale=candidate.rationale,
            validation=report.to_dict(),
        )
        review_payload = review.model_dump(mode="json")
        self.store.write_revision_json(
            candidate_id,
            revision,
            "review.json",
            review_payload,
        )
        self.store.append_audit_event(
            {
                "event": "reviewer_recommendation",
                "candidate_id": candidate_id,
                "revision": revision,
                "recommendation": review.recommendation,
            }
        )
        return self.store.update_candidate_status(
            candidate_id,
            "review",
        )

    def review_candidate(
        self,
        *,
        candidate_id: str,
        expected_revision: int,
        model_config: str = "",
    ) -> LearningCandidate:
        """Run one advisory scope review for an exact immutable candidate."""

        candidate = self.store.read_candidate(candidate_id)
        if candidate is None:
            raise ValueError(f"candidate not found: {candidate_id}")
        if candidate.revision != int(expected_revision):
            raise ValueError("candidate revision changed before independent review")
        if candidate.review:
            return candidate
        if self._reviewer is not None:
            reviewer = self._reviewer
        else:
            profile = LLMProfile.from_env_or_file(
                model_config or self.model_config or None
            )
            _proposer, reviewer = build_self_evolution_agents(
                profile,
                workspace=self.workspace,
            )
        report = self.gate.run(candidate)
        if not report.valid:
            raise ValueError("candidate no longer passes static validation")
        candidate_root = self.store.revision_dir(
            candidate.candidate_id,
            candidate.revision,
        )
        review, _reviewer_meta = reviewer.review(
            candidate_root=candidate_root,
            action=candidate.action,
            group=candidate.group,
            name=candidate.name,
            rationale=candidate.rationale,
            validation=report.to_dict(),
        )
        self.store.write_revision_json(
            candidate.candidate_id,
            candidate.revision,
            "review.json",
            review.model_dump(mode="json"),
        )
        self.store.append_audit_event(
            {
                "event": "reviewer_recommendation",
                "candidate_id": candidate.candidate_id,
                "revision": candidate.revision,
                "recommendation": review.recommendation,
            }
        )
        return self.store.update_candidate_status(candidate.candidate_id, "review")

    # -- content and route contracts ------------------------------------

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return value if isinstance(value, dict) else {}

    def _base_target_hash(
        self,
        *,
        action: str,
        group: str,
        name: str,
        candidate_root: Path,
    ) -> str:
        if action == "memory":
            frozen = candidate_root / "current" / "AGENTS.md"
            return (
                hash_text(frozen.read_text(encoding="utf-8", errors="replace"))
                if frozen.is_file()
                else self.store.memory_hash()
            )
        frozen_target = candidate_root / "current" / "target"
        if frozen_target.is_dir():
            return hash_tree(frozen_target)
        workspace_target = self.store.self_develop_skills_dir / group / name
        if workspace_target.is_dir():
            return hash_tree(workspace_target)
        return hash_tree(self.repo_root / "skills" / group / name)

    @staticmethod
    def _bundle_hash(
        *,
        action: str,
        group: str,
        name: str,
        candidate_root: Path,
    ) -> str:
        if action == "memory":
            path = candidate_root / "memories" / "AGENTS.md"
            return (
                hash_text(path.read_text(encoding="utf-8", errors="replace"))
                if path.is_file()
                else ""
            )
        return hash_tree(candidate_root / "proposed" / group / name)

    @staticmethod
    def _discard_unchanged_memory_copy(candidate_root: Path) -> None:
        current = candidate_root / "current" / "AGENTS.md"
        proposed = candidate_root / "memories" / "AGENTS.md"
        if not current.is_file() or not proposed.is_file():
            return
        if current.read_bytes() != proposed.read_bytes():
            return
        proposed.unlink()
        try:
            proposed.parent.rmdir()
        except OSError:
            pass

    def _route_contract_errors(
        self,
        *,
        route: str,
        proposal: ProposerResult,
        owner_group: str,
        owner_name: str,
    ) -> list[str]:
        errors: list[str] = []
        if route == "workspace_preference" and proposal.action != "memory":
            errors.append("workspace preferences must revise memory, not create a workflow skill")
        if route in {"amend_existing_skill", "new_skill"} and proposal.action != "skill":
            errors.append(f"{route} must produce exactly one complete skill bundle")
        if route in {"amend_existing_skill", "new_skill"} and (
            proposal.group != owner_group or proposal.name != owner_name
        ):
            errors.append(
                f"the reflected target is {owner_group}/{owner_name}; "
                "the proposer cannot redirect the revision"
            )
        if route == "new_skill" and proposal.action == "skill":
            existing_targets = (
                self.repo_root / "skills" / proposal.group / proposal.name,
                self.store.self_develop_skills_dir / proposal.group / proposal.name,
            )
            if any(path.is_dir() for path in existing_targets):
                errors.append(
                    "the proposed target already exists; route this evidence to an amendment instead of a duplicate skill"
                )
        if proposal.action == "skill" and (
            proposal.group not in SKILL_GROUPS or not proposal.name
        ):
            errors.append("skill target is missing or outside the supported CatMaster skill groups")
        if not proposal.applicability_boundary:
            errors.append("the candidate must state a concrete applicability boundary")
        if not proposal.non_applicability:
            errors.append("the candidate must state at least one non-applicability boundary")
        if not proposal.expected_step_change.strip():
            errors.append("the candidate must state which decision or step it changes")
        return errors

    @staticmethod
    def _seal_payload(candidate_root: Path) -> None:
        targets = [
            candidate_root / "current",
            candidate_root / "memories",
            candidate_root / "proposed",
        ]
        for target in targets:
            if not target.exists():
                continue
            paths = [target, *target.rglob("*")] if target.is_dir() else [target]
            for path in reversed(paths):
                try:
                    path.chmod(0o555 if path.is_dir() else 0o444)
                except OSError:
                    continue


__all__ = ["SelfEvolutionCoordinator"]
