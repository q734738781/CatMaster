from __future__ import annotations

import json
import inspect
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
from .effective import EffectiveSkillsManager, candidate_target, candidate_version
from .gate import CandidateGate, read_skill_frontmatter
from .models import (
    CandidateRevision,
    LearningCandidate,
    Observation,
    ProposerResult,
    ReflectionBatch,
    ReflectionResult,
    SKILL_GROUPS,
    SelfEvolutionJob,
    TextResult,
    ValidationReport,
)
from .promotion import PromotionManager
from .query import EvolutionHistoryScope, EvolutionTraceScope
from .settings import SelfEvolutionMode, resolve_self_evolution_mode, self_evolution_enqueue_enabled
from .storage import SelfEvolutionStore, hash_text, hash_tree, stable_id, utc_now
from .telemetry import finalize_skill_run_telemetry
from .trace import (
    TERMINAL_STATUSES,
    TurnTrace,
    collect_turn_trace,
    read_thread_turn,
)


logger = logging.getLogger(__name__)


def _invoke_agent_method(method: Any, **kwargs: Any) -> Any:
    """Pass new trace-scope arguments without breaking injected legacy test agents."""

    signature = inspect.signature(method)
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return method(**kwargs)
    accepted = {
        name: value
        for name, value in kwargs.items()
        if name in signature.parameters
    }
    return method(**accepted)


def _response_text(response: Any, metadata: dict[str, Any]) -> str:
    recorded = str(metadata.get("response_evidence_text") or "")
    if recorded:
        return recorded
    if isinstance(response, TextResult):
        return response.text
    return json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2)


def _run_state(run_dir: Path | str, *, run_id: str = "", thread_id: str = "") -> dict[str, Any]:
    current = read_thread_turn(run_dir, run_id=run_id, thread_id=thread_id)
    if current:
        return current
    path = Path(run_dir).expanduser().resolve() / "run_state.json"
    if not path.is_file():
        return {
            "_read_diagnostic": {
                "error_type": "FileNotFoundError",
                "error": f"run state is missing: {path}",
                "path": str(path),
            }
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {
            "_read_diagnostic": {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "path": str(path),
            }
        }
    if not isinstance(value, dict):
        return {
            "_read_diagnostic": {
                "error_type": "TypeError",
                "error": "run_state.json must contain a JSON object",
                "path": str(path),
            }
        }
    return value


class SelfEvolutionCoordinator:
    """Workspace evidence governance and reviewer-directed activation coordinator."""

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
        self.repo_root = Path(repo_root or Path(__file__).resolve().parents[3]).expanduser().resolve()
        self.store = SelfEvolutionStore(self.workspace, project_id=self.project_id)
        self.effective = EffectiveSkillsManager(self.store, repo_root=self.repo_root)
        self.mode: SelfEvolutionMode = self.effective.workspace_mode(mode)
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
        episode_id: str = "",
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
        status = {"success": "done", "timeout": "error"}.get(status, status)
        if status not in TERMINAL_STATUSES and status != "steered":
            return None
        supplied = dict(payload or {})
        state = _run_state(run_dir, run_id=run_id, thread_id=thread_id)
        if status in {"error", "interrupted", "interrupted_paused"} and bool(
            state.get("checkpoint_resume_available")
        ):
            return None
        episode = str(
            episode_id or state.get("episode_id") or message_id or run_id
        ).strip()
        return self.store.enqueue_job(
            trigger_kind="post_run",
            run_id=run_id,
            run_dir=run_dir,
            thread_id=thread_id,
            episode_id=episode,
            payload={
                **supplied,
                "episode_id": episode,
                "episode_prompt": str(
                    state.get("episode_prompt")
                    or state.get("user_prompt")
                    or supplied.get("prompt")
                    or ""
                ),
                "message_id": str(state.get("message_id") or message_id or ""),
                "entrypoint": str(entrypoint or ""),
                "execution_status": status,
                "task_outcome": str(supplied.get("task_outcome") or ""),
                "outcome_ref": str(supplied.get("outcome_ref") or ""),
                **(
                    {"run_state_diagnostic": dict(state["_read_diagnostic"])}
                    if isinstance(state.get("_read_diagnostic"), dict)
                    else {}
                ),
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
        episode_id: str = "",
        model_config: str = "",
        actor: str = "",
    ) -> SelfEvolutionJob:
        correction = str(note or "").strip()
        if not correction:
            raise ValueError("a concrete durable correction is required")
        state = _run_state(run_dir, run_id=run_id, thread_id=thread_id)
        episode = str(
            episode_id or state.get("episode_id") or run_id
        ).strip()
        job = self.store.enqueue_job(
            trigger_kind="explicit_learn",
            run_id=run_id,
            run_dir=run_dir,
            thread_id=(
                thread_id
                or str(state.get("webui_thread_id") or state.get("thread_id") or "")
            ),
            episode_id=episode,
            payload={
                "note": correction,
                "actor": str(actor or "").strip(),
                "execution_status": str(state.get("status") or "unknown"),
                "episode_id": episode,
                **(
                    {"run_state_diagnostic": dict(state["_read_diagnostic"])}
                    if isinstance(state.get("_read_diagnostic"), dict)
                    else {}
                ),
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
        run_dir = self.store.run_dir_for(candidate.run_id)
        if run_dir is None:
            raise ValueError("candidate anchor run is no longer available")
        return self.store.enqueue_job(
            trigger_kind="candidate_revision",
            run_id=candidate.run_id,
            run_dir=run_dir,
            thread_id=candidate.thread_id,
            episode_id=candidate.episode_id,
            payload={
                "candidate_id": candidate_id,
                "expected_revision": int(expected_revision),
                "guidance": str(guidance or "").strip(),
                "actor": str(actor or "").strip(),
            },
            model_config=str(model_config or self.model_config),
        )

    def enqueue_selected_retry(
        self,
        *,
        job_id: str,
        item_ref: str = "",
        actor: str,
        model_config: str = "",
    ) -> SelfEvolutionJob:
        """Queue one explicit retry without modifying the predecessor or run evidence."""

        predecessor = self.store.read_job(str(job_id or "").strip())
        if predecessor is None:
            raise ValueError("self-evolution job not found")
        selected_ref = str(item_ref or "").strip()
        reflection_items = [
            dict(item)
            for item in list(predecessor.outcome.get("reflection_items") or [])
            if isinstance(item, dict)
        ]
        failed_items = [
            item
            for item in reflection_items
            if str(item.get("status") or "") == "error"
            or bool(str(item.get("error") or "").strip())
        ]
        selected: dict[str, Any] | None = None
        if selected_ref:
            selected = next(
                (
                    item
                    for item in failed_items
                    if str(item.get("item_ref") or "") == selected_ref
                ),
                None,
            )
            if selected is None:
                raise ValueError("the selected failed reflection item was not found")
        elif failed_items:
            if len(failed_items) != 1:
                raise ValueError("select one failed reflection item to retry")
            selected = failed_items[0]
            selected_ref = str(selected.get("item_ref") or "").strip()
        elif predecessor.status not in {"error", "recovery_review"}:
            raise ValueError("this job has no failed reflection item to retry")
        payload = {
            **dict(predecessor.payload),
            "retry_of": predecessor.job_id,
            "original_trigger_kind": str(
                predecessor.payload.get("original_trigger_kind")
                or predecessor.trigger_kind
            ),
            **(
                {"selected_finding": dict(selected.get("finding") or {})}
                if selected is not None
                else {}
            ),
        }
        if selected is not None and not payload.get("selected_finding"):
            raise ValueError("the failed item has no replayable reflection finding")
        retry = self.store.enqueue_job(
            trigger_kind="selected_retry",
            run_id=predecessor.run_id,
            run_dir=predecessor.run_dir,
            thread_id=predecessor.thread_id,
            episode_id=predecessor.episode_id,
            selected_item_ref=selected_ref,
            payload=payload,
            model_config=str(model_config or predecessor.model_config or self.model_config),
            predecessor_job_id=predecessor.job_id,
        )
        self.store.append_audit_event(
            {
                "event": "self_evolution_job_retry_queued",
                "job_id": retry.job_id,
                "predecessor_job_id": predecessor.job_id,
                "selected_item_ref": selected_ref,
                "actor": str(actor or "").strip(),
            }
        )
        return retry

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
        effective_trigger = str(
            job.payload.get("original_trigger_kind") or job.trigger_kind
        ).strip()
        if effective_trigger == "candidate_revision":
            candidate_id = self._process_revision_job(job)
            latest = self.store.read_candidate(candidate_id)
            return self.store.finish_job(
                job,
                status="done",
                candidate_id=candidate_id,
                outcome={
                    **job.outcome,
                    "reflection_items": [],
                    "no_change": False,
                    "revision_candidate_id": candidate_id,
                    "revision_status": (
                        "completed" if job.outcome.get("text_responses")
                        else self._candidate_outcome_status(latest) if latest is not None else "error"
                    ),
                },
                owner=self.worker_id,
            )

        trace = collect_turn_trace(
            run_dir=job.run_dir,
            fallback={
                "run_id": job.run_id,
                "thread_id": job.thread_id,
                **dict(job.payload),
            },
            include_events=False,
        )
        self._finalize_run_telemetry(job, trace)
        trace_scope = EvolutionTraceScope(
            {job.run_id: (Path(job.run_dir), trace)},
            anchor_run_id=job.run_id,
            history_scope=EvolutionHistoryScope(db_path=self.store.db_path),
        )
        proposer, _reviewer = self._agents(job)
        skill_catalog = self._skill_catalog()
        prior_targets: list[str] = []
        trace_index = trace_scope.index(
            context_label=(
                "explicit durable correction"
                if self._is_explicit_learning_job(job)
                else "completed CatMaster run"
            ),
            skill_context="\n".join(
                [
                    "Current skill catalog:",
                    skill_catalog,
                ]
            ),
        )
        selected_finding = job.payload.get("selected_finding")
        if isinstance(selected_finding, dict):
            reflection = ReflectionBatch(
                items=[ReflectionResult.model_validate(selected_finding)]
            )
        else:
            reflection_feedback: list[str] = []
            reflection = ReflectionBatch(items=[ReflectionResult(kind="no_change")])
            for _correction_round in range(3):
                reflection_value, _reflection_meta = _invoke_agent_method(
                    proposer.reflect,
                    trajectory_markdown=trace_index,
                    skill_catalog=self._skill_catalog(),
                    prior_targets=prior_targets,
                    trace_scope=trace_scope,
                    effective_skills=self.effective,
                    correction_feedback=reflection_feedback,
                )
                if isinstance(reflection_value, TextResult):
                    return self.store.finish_job(
                        job, status="done", owner=self.worker_id,
                        outcome={
                            "reflection_items": [], "no_change": False,
                            "text_responses": [{"stage": "reflection", "text": reflection_value.text}],
                        },
                    )
                reflection = (
                    reflection_value
                    if isinstance(reflection_value, ReflectionBatch)
                    else ReflectionBatch.model_validate(
                        reflection_value.model_dump(mode="json")
                        if isinstance(reflection_value, ReflectionResult)
                        else reflection_value
                    )
                )
                reflection_feedback = self._reflection_contract_diagnostics(
                    reflection,
                    trace_scope=trace_scope,
                )
                if not reflection_feedback:
                    break
        outcome: dict[str, Any] = {
            "reflection_items": [],
            "no_change": bool(
                len(reflection.items) == 1
                and reflection.items[0].kind == "no_change"
            ),
        }
        job.outcome = outcome
        candidate_ids: list[str] = []
        actionable_count = 0
        error_count = 0
        pending_targets: dict[
            str,
            list[tuple[Observation, dict[str, Any]]],
        ] = {}
        for item_index, finding in enumerate(reflection.items):
            item_ref = (
                job.selected_item_ref
                or self._reflection_item_ref(
                    job=job,
                    finding=finding,
                    index=item_index,
                )
            )
            item_outcome: dict[str, Any] = {
                "item_ref": item_ref,
                "kind": finding.kind,
                "target": self._reflection_target(finding),
                "finding": finding.model_dump(mode="json"),
                "status": "pending",
                "observation_id": "",
                "candidate_id": "",
            }
            outcome["reflection_items"].append(item_outcome)
            if finding.kind == "no_change":
                item_outcome["status"] = "no_change"
                continue
            if finding.kind == "execution_lapse":
                item_outcome["status"] = "execution_lapse"
                continue
            actionable_count += 1
            try:
                observation = self._observation_from_reflection(
                    job=job,
                    trace=trace,
                    trace_scope=trace_scope,
                    reflection=finding,
                    item_ref=item_ref,
                )
                if observation is None:
                    raise ValueError("reflection finding does not identify a valid durable target")
                observation = self.store.write_observation(observation)
                item_outcome["observation_id"] = observation.observation_id
                pending_targets.setdefault(observation.target, []).append(
                    (observation, item_outcome)
                )
            except Exception as exc:
                error_count += 1
                item_outcome["status"] = "error"
                item_outcome["error"] = f"{type(exc).__name__}: {exc}"
                self.store.append_audit_event(
                    {
                        "event": "reflection_item_error",
                        "job_id": job.job_id,
                        "kind": finding.kind,
                        "target": item_outcome["target"],
                        "error": item_outcome["error"],
                    }
                )
                logger.exception(
                    "Self-evolution target %s failed independently for job %s",
                    item_outcome["target"],
                    job.job_id,
                )

        # Materialize once per exact target after every finding in this reflection
        # batch has been persisted. Consolidation can therefore present all
        # same-target evidence to one proposer/reviewer pass instead of creating
        # a redundant immutable revision for each finding in the same episode.
        for target_items in pending_targets.values():
            anchor = target_items[-1][0]
            try:
                candidate, decision, decision_reason, resolved_target = self._candidate_for_observation(
                    job=job,
                    observation=anchor,
                    current_trace=trace,
                )
                for _observation, item_outcome in target_items:
                    item_outcome["resolved_target"] = resolved_target
                if candidate is not None:
                    candidate_ids.append(candidate.candidate_id)
                    review_error = (
                        dict(candidate.review)
                        if isinstance(candidate.review, dict)
                        and str(candidate.review.get("error") or "").strip()
                        else {}
                    )
                    for _observation, item_outcome in target_items:
                        item_outcome["candidate_id"] = candidate.candidate_id
                        if review_error:
                            error_count += 1
                            item_outcome["status"] = "error"
                            item_outcome["error"] = (
                                f"{review_error.get('error_type') or 'ReviewerError'}: "
                                f"{review_error.get('error')}"
                            )
                        else:
                            item_outcome["status"] = self._candidate_outcome_status(candidate)
                            if item_outcome["status"] == "text_completed":
                                item_outcome["reason"] = str(candidate.review.get("text") or "")
                            if item_outcome["status"] == "needs_revision":
                                item_outcome["reason"] = (
                                    "The bounded automatic repair budget ended on reviewer "
                                    "feedback. The latest revision remains dormant."
                                )
                else:
                    for _observation, item_outcome in target_items:
                        item_outcome["status"] = (
                            "text_completed" if decision == "text"
                            else "deferred" if decision == "defer" else "ignored"
                        )
                        if decision_reason:
                            item_outcome["reason"] = decision_reason
            except Exception as exc:
                error_count += len(target_items)
                for _observation, item_outcome in target_items:
                    item_outcome["status"] = "error"
                    item_outcome["error"] = f"{type(exc).__name__}: {exc}"
                    self.store.append_audit_event(
                        {
                            "event": "reflection_item_error",
                            "job_id": job.job_id,
                            "kind": item_outcome["kind"],
                            "target": item_outcome["target"],
                            "error": item_outcome["error"],
                        }
                    )
                logger.exception(
                    "Self-evolution target %s failed for %s findings in job %s",
                    anchor.target,
                    len(target_items),
                    job.job_id,
                )
        terminal_status = (
            "error"
            if actionable_count > 0 and error_count == actionable_count
            else "done"
        )
        return self.store.finish_job(
            job,
            status=terminal_status,
            candidate_id=(candidate_ids[0] if candidate_ids else ""),
            outcome=outcome,
            error=(
                "All durable reflection items failed; retry an exact failed item."
                if terminal_status == "error"
                else ""
            ),
            owner=self.worker_id,
        )

    def _candidate_outcome_status(self, candidate: LearningCandidate) -> str:
        review = candidate.review if isinstance(candidate.review, dict) else {}
        if review.get("format") == "text":
            return "text_completed"
        recommendation = str(review.get("recommendation") or "").strip()
        if recommendation == "needs_revision":
            return "needs_revision"
        if recommendation == "reject" or candidate.status == "rejected":
            return "rejected"
        if recommendation != "approve":
            return "completed"
        human_checks = [
            str(item).strip()
            for item in list(review.get("human_checks") or [])
            if str(item).strip()
        ]
        if human_checks:
            return "waiting_for_clarification"
        try:
            state = self.effective.target_summary(candidate_target(candidate))
        except Exception:
            return "dormant"
        return (
            "activated"
            if state.get("selected_version") == candidate_version(candidate)
            else "dormant"
        )

    def _finalize_run_telemetry(
        self,
        job: SelfEvolutionJob,
        trace: TurnTrace,
    ) -> None:
        if job.trigger_kind != "post_run":
            return
        finalize_skill_run_telemetry(
            store=self.store,
            run_id=job.run_id,
            run_dir=job.run_dir,
            task_outcome=trace.task_outcome,
            outcome_ref=trace.outcome_ref,
        )

    def _observation_from_reflection(
        self,
        *,
        job: SelfEvolutionJob,
        trace: TurnTrace,
        trace_scope: EvolutionTraceScope,
        reflection: ReflectionResult,
        item_ref: str,
    ) -> Observation | None:
        if reflection.kind in {"no_change", "execution_lapse"}:
            return None
        group = str(reflection.group or "").strip()
        name = str(reflection.name or "").strip()
        if reflection.kind == "workspace_preference":
            target = f"memory/{name}" if name else f"memory/{item_ref}"
            signal_kind = "workspace_preference"
        else:
            target = f"{group}/{name}".strip("/") or f"unresolved/{item_ref}"
            signal_kind = reflection.kind

        change = str(reflection.change or "").strip()
        selected_refs = [
            str(item).strip()
            for item in reflection.evidence_refs
            if str(item).strip()
        ]
        evidence_refs = [
            {
                "source_ref": item,
                "reason": (
                    "explicit_user_correction"
                    if self._is_explicit_learning_job(job)
                    else "semantic_reflection"
                    if trace_scope.contains_event(item)
                    else "unavailable_ref_from_reflection"
                ),
            }
            for item in dict.fromkeys(selected_refs)
        ]
        if self._is_explicit_learning_job(job):
            evidence_refs.insert(
                0,
                {
                    "source_ref": f"job:{job.job_id}",
                    "reason": "explicit_user_correction",
                },
            )
        return Observation(
            observation_id="seo_"
            + stable_id(job.episode_id or job.run_id, item_ref, length=28),
            run_id=job.run_id,
            thread_id=job.thread_id or trace.thread_id,
            job_id=job.job_id,
            episode_id=job.episode_id,
            item_ref=item_ref,
            signal_kind=signal_kind,  # type: ignore[arg-type]
            target=target,
            claim=change,
            evidence_refs=evidence_refs,
            outcome_ref=trace.outcome_ref,
            created_at=utc_now(),
        )

    @staticmethod
    def _reflection_contract_diagnostics(
        reflection: ReflectionBatch,
        *,
        trace_scope: EvolutionTraceScope,
    ) -> list[str]:
        """Return exact, model-correctable evidence-handle diagnostics."""

        diagnostics: list[str] = []
        for index, item in enumerate(reflection.items, start=1):
            if item.kind in {"no_change", "execution_lapse"}:
                continue
            if not str(item.change or "").strip():
                diagnostics.append(f"finding {index} has no proposed behavior change")
            if item.kind == "workspace_preference":
                if not str(item.name or "").strip():
                    diagnostics.append(f"finding {index} has no memory topic name")
            elif not str(item.group or "").strip() or not str(item.name or "").strip():
                diagnostics.append(f"finding {index} has no skill group/name anchor")
            selected_refs = [
                str(value).strip()
                for value in item.evidence_refs
                if str(value).strip()
            ]
            if not selected_refs:
                diagnostics.append(
                    f"finding {index} cites no reopenable run:...#event:... evidence handle"
                )
                continue
            invalid_refs = [
                value for value in selected_refs if not trace_scope.contains_event(value)
            ]
            if invalid_refs:
                diagnostics.append(
                    f"finding {index} cites unavailable handles: {', '.join(invalid_refs)}"
                )
        return diagnostics

    @staticmethod
    def _reflection_item_ref(
        *,
        job: SelfEvolutionJob,
        finding: ReflectionResult,
        index: int,
    ) -> str:
        return "sei_" + stable_id(
            job.job_id,
            str(index),
            json.dumps(
                finding.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
            ),
            length=28,
        )

    @staticmethod
    def _is_explicit_learning_job(job: SelfEvolutionJob) -> bool:
        return bool(
            job.trigger_kind == "explicit_learn"
            or str(job.payload.get("original_trigger_kind") or "")
            == "explicit_learn"
        )

    @staticmethod
    def _reflection_target(reflection: ReflectionResult) -> str:
        if reflection.kind == "workspace_preference":
            name = str(reflection.name or "").strip()
            return f"memory/{name}" if name else ""
        if reflection.kind in {"skill_revision", "skill_discovery"}:
            group = str(reflection.group or "").strip()
            name = str(reflection.name or "").strip()
            return f"{group}/{name}" if group and name else ""
        return ""

    def _candidate_for_observation(
        self,
        *,
        job: SelfEvolutionJob,
        observation: Observation,
        current_trace: TurnTrace,
    ) -> tuple[LearningCandidate | None, str, str, str]:
        target = observation.target
        visited_targets: set[str] = set()
        carried_observations: dict[str, Observation] = {}
        for _reroute_round in range(4):
            if target in visited_targets:
                raise ValueError(f"candidate owner reroute formed a cycle at {target!r}")
            visited_targets.add(target)
            canonical_id = "sec_" + stable_id("target", target, length=28)
            existing_for_target = self.store.read_candidate_for_target(target)
            candidate_id = (
                existing_for_target.candidate_id
                if existing_for_target is not None
                else canonical_id
            )
            with self.store.candidate_lock(candidate_id):
                batch = self.consolidation.batch_for(observation, target=target)
                carried_observations.update(
                    (item.observation_id, item) for item in batch.observations
                )
                batch = EvidenceBatch(
                    target=target,
                    observations=tuple(
                        sorted(
                            carried_observations.values(),
                            key=lambda item: (item.created_at, item.observation_id),
                        )
                    ),
                )
                _route, owner_group, owner_name = self._target_details(batch.target)
                existing = self.store.read_candidate_for_target(target)
                if existing is None:
                    existing = self.store.read_candidate(candidate_id)
                if existing is not None and observation.observation_id in {
                    str(item) for item in existing.evidence_ids
                }:
                    explicit_failed_item_retry = bool(
                        job.trigger_kind == "selected_retry"
                        and isinstance(existing.review, dict)
                        and str(existing.review.get("error") or "").strip()
                    )
                    if not explicit_failed_item_retry:
                        resolved_existing_target = self._candidate_resolved_target(
                            existing
                        )
                        self.store.set_observation_resolved_target(
                            [observation.observation_id],
                            resolved_existing_target,
                        )
                        self.store.set_observation_status(
                            [observation.observation_id],
                            "consolidated",
                        )
                        return existing, "candidate", "", resolved_existing_target
                evidence_markdown = self.consolidation.evidence_markdown(batch)
                trace_scope = self._trace_scope_for_batch(
                    batch,
                    current_observation_id=observation.observation_id,
                    current_trace=current_trace,
                    anchor_run_id=job.run_id,
                )
                revision = existing.revision + 1 if existing is not None else 1
                continue_failed_draft = bool(
                    job.trigger_kind == "selected_retry"
                    and existing is not None
                    and isinstance(existing.review, dict)
                    and str(existing.review.get("error") or "").strip()
                )
                content_parent_version = (
                    candidate_version(existing)
                    if continue_failed_draft and existing is not None
                    else self._active_parent_version(target)
                )
                prior_revision_root = (
                    self.store.revision_dir(existing.candidate_id, existing.revision)
                    if continue_failed_draft and existing is not None
                    else None
                )
                candidate, decision, decision_reason, resolved_target = (
                    self._build_candidate_with_review_repairs(
                        job=job,
                        candidate_id=candidate_id,
                        revision=revision,
                        evidence_ids=list(batch.evidence_ids),
                        evidence_markdown=evidence_markdown,
                        owner_group=owner_group,
                        owner_name=owner_name,
                        expected_target=target,
                        content_parent_version=content_parent_version,
                        run_id=job.run_id,
                        thread_id=job.thread_id,
                        trace_scope=trace_scope,
                        prior_revision_root=prior_revision_root,
                    )
                )
                if decision == "reroute":
                    target = resolved_target
                    continue
                resolved_owner = str(resolved_target or target).strip()
                self.store.set_observation_resolved_target(
                    list(batch.evidence_ids),
                    resolved_owner,
                )
                if candidate is not None or decision == "ignore":
                    self.store.set_observation_status(
                        list(batch.evidence_ids),
                        "consolidated",
                    )
                return candidate, decision, decision_reason, resolved_owner
        raise ValueError("candidate owner did not converge after four proposer reroutes")

    def _active_parent_version(self, target: str) -> str:
        lookup_target = (
            "/memories/AGENTS.md" if str(target).startswith("memory/") else str(target)
        )
        try:
            summary = self.effective.target_summary(lookup_target)
        except (TypeError, ValueError):
            return ""
        return str(summary.get("selected_version") or "")

    def _candidate_resolved_target(self, candidate: LearningCandidate) -> str:
        """Return the semantic owner rather than the memory activation path."""

        if candidate.action != "memory":
            return candidate_target(candidate)
        for evidence_id in reversed(candidate.evidence_ids):
            observation = self.store.read_observation(str(evidence_id))
            if observation is None:
                continue
            target = str(
                observation.resolved_target or observation.target or ""
            ).strip()
            if target.startswith("memory/"):
                return target
        return candidate_target(candidate)

    # -- candidate revisions --------------------------------------------

    def _agents(self, job: SelfEvolutionJob) -> tuple[Any, Any]:
        if self._proposer is not None and self._reviewer is not None:
            return self._proposer, self._reviewer
        profile = LLMProfile.from_env_or_file(job.model_config or self.model_config or None)
        return build_self_evolution_agents(profile, workspace=self.workspace, usage_context={
            "job_id": job.job_id, "attempt": job.attempt_count,
            "source_run_id": job.run_id, "source_thread_id": job.thread_id,
        })

    def _skill_catalog(self) -> str:
        count = self.effective.target_count()
        return (
            f"{count} effective skill or workspace-guidance target"
            f"{'s are' if count != 1 else ' is'} available. Use query_effective_skills "
            "with its returned cursor for complete model-selected discovery; target names "
            "are not serialized into this starting index."
        )

    @staticmethod
    def _skill_description(skill_md: Path) -> str:
        try:
            return str(
                read_skill_frontmatter(skill_md).get("description") or ""
            ).strip()
        except Exception as exc:
            return f"Load diagnostic: {type(exc).__name__}: {exc}"

    def _skill_target_exists(self, *, group: str, name: str) -> bool:
        target = f"{group}/{name}"
        try:
            summary = self.effective.target_summary(target)
        except ValueError:
            return False
        return bool(
            summary.get("selected_version")
            or int(summary.get("revision_count") or 0) > 0
        )

    def _target_details(self, target: str) -> tuple[str, str, str]:
        if target.startswith("memory/") and target.partition("/")[2]:
            return "workspace_preference", "", ""
        group, separator, name = str(target or "").partition("/")
        if (
            not separator
            or group not in SKILL_GROUPS
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
        ):
            # Keep the reflection as an evidence anchor without letting an early
            # incomplete owner choice constrain or escape the proposer workspace.
            return "new_skill", "", ""
        if self._skill_target_exists(group=group, name=name):
            return "amend_existing_skill", group, name
        return "new_skill", group, name

    def _trace_scope_for_batch(
        self,
        batch: EvidenceBatch,
        *,
        current_observation_id: str = "",
        current_trace: TurnTrace | None = None,
        anchor_run_id: str = "",
    ) -> EvolutionTraceScope | None:
        runs: dict[str, tuple[Path, TurnTrace]] = {}
        if current_trace is not None:
            current_run_dir = self.store.run_dir_for(current_trace.run_id)
            if current_run_dir is not None:
                runs[current_trace.run_id] = (current_run_dir, current_trace)
        else:
            for observation in batch.observations:
                run_dir = self.store.run_dir_for(observation.run_id)
                if run_dir is None:
                    continue
                trace = collect_turn_trace(
                    run_dir=run_dir,
                    fallback={
                        "run_id": observation.run_id,
                        "thread_id": observation.thread_id,
                        "note": self._explicit_correction_for(observation),
                    },
                    include_events=False,
                )
                runs[trace.run_id] = (run_dir, trace)
        resolved_anchor = str(
            anchor_run_id
            or (current_trace.run_id if current_trace is not None else "")
            or (batch.observations[-1].run_id if batch.observations else "")
        ).strip()
        if resolved_anchor and resolved_anchor not in runs:
            run_dir = self.store.run_dir_for(resolved_anchor)
            if run_dir is not None:
                trace = collect_turn_trace(
                    run_dir=run_dir,
                    fallback={"run_id": resolved_anchor},
                    include_events=False,
                )
                runs[resolved_anchor] = (run_dir, trace)
        if not runs:
            return None
        if resolved_anchor not in runs:
            resolved_anchor = next(iter(runs))
        return EvolutionTraceScope(
            runs,
            anchor_run_id=resolved_anchor,
            history_scope=EvolutionHistoryScope(
                db_path=self.store.db_path,
                target=batch.target,
            ),
        )

    def _trace_scope_for_evidence_ids(
        self,
        evidence_ids: list[str],
        *,
        anchor_run_id: str = "",
        target: str = "",
    ) -> EvolutionTraceScope | None:
        observations = [
            observation
            for evidence_id in evidence_ids
            if (observation := self.store.read_observation(str(evidence_id)))
            is not None
        ]
        missing = sorted(
            set(str(item) for item in evidence_ids)
            - {observation.observation_id for observation in observations}
        )
        if missing:
            raise ValueError(
                "candidate evidence observations are unavailable: " + ", ".join(missing)
            )
        if not observations:
            return None
        batch = EvidenceBatch(
            target=str(target or observations[0].target).strip(),
            observations=tuple(observations),
        )
        return self._trace_scope_for_batch(
            batch,
            anchor_run_id=anchor_run_id,
        )

    def _explicit_correction_for(self, observation: Observation) -> str:
        for ref in observation.evidence_refs:
            if not isinstance(ref, dict):
                continue
            source_ref = str(ref.get("source_ref") or "").strip()
            if not source_ref.startswith("job:"):
                continue
            job = self.store.read_job(source_ref.removeprefix("job:"))
            if job is not None and self._is_explicit_learning_job(job):
                return str(job.payload.get("note") or "").strip()
        return ""

    def _process_revision_job(self, job: SelfEvolutionJob) -> str:
        candidate_id = str(job.payload.get("candidate_id") or "").strip()
        expected_revision = int(job.payload.get("expected_revision") or 0)
        if not candidate_id:
            raise ValueError("candidate revision job is missing candidate_id")
        with self.store.candidate_lock(candidate_id):
            return self._process_revision_job_locked(job)

    def _process_revision_job_locked(self, job: SelfEvolutionJob) -> str:
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
        old_evidence = (old_root / "evidence.md").read_text(encoding="utf-8")
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
                "Read the complete predecessor review at `/prior_review.json`.",
                "",
            ]
        )
        revised, decision, decision_reason, resolved_target = (
            self._build_candidate_with_review_repairs(
                job=job,
                candidate_id=candidate_id,
                revision=expected_revision + 1,
                evidence_ids=list(current.evidence_ids),
                evidence_markdown=evidence_markdown,
                owner_group=current.group,
                owner_name=current.name,
                expected_target=self._candidate_resolved_target(current),
                content_parent_version=candidate_version(current),
                run_id=current.run_id,
                thread_id=current.thread_id,
                trace_scope=self._trace_scope_for_evidence_ids(
                    list(current.evidence_ids),
                    anchor_run_id=current.run_id,
                    target=self._candidate_resolved_target(current),
                ),
                prior_review=old_review,
                prior_revision_root=old_root,
            )
        )
        if revised is None:
            if decision == "text":
                # A textual closeout is complete, but did not submit a new
                # revision. Keep the prior candidate and its review unchanged.
                return candidate_id
            raise ValueError(
                "the revision proposer declined to create another candidate"
                + (
                    f" and selected owner {resolved_target!r}"
                    if decision == "reroute" and resolved_target
                    else ""
                )
                + (f": {decision_reason}" if decision_reason else "")
            )
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

    def _build_candidate_with_review_repairs(
        self,
        *,
        job: SelfEvolutionJob,
        candidate_id: str,
        revision: int,
        evidence_ids: list[str],
        evidence_markdown: str,
        owner_group: str,
        owner_name: str,
        expected_target: str,
        content_parent_version: str,
        run_id: str,
        thread_id: str,
        trace_scope: EvolutionTraceScope | None,
        prior_review: dict[str, Any] | None = None,
        prior_revision_root: Path | None = None,
        max_revisions: int = 3,
    ) -> tuple[LearningCandidate | None, str, str, str]:
        """Apply bounded reviewer-directed repair without creating human work."""

        next_revision = max(1, int(revision))
        current_prior_review = dict(prior_review or {}) or None
        current_prior_root = prior_revision_root
        current_parent_version = str(content_parent_version or "")
        current_target = str(expected_target or "").strip()
        base_evidence = str(evidence_markdown or "").rstrip()
        for repair_round in range(1, max(1, int(max_revisions)) + 1):
            round_evidence = base_evidence
            if current_prior_review:
                diagnosis = self._review_revision_diagnosis(current_prior_review)
                round_evidence = "\n".join(
                    [
                        base_evidence,
                        "",
                        "## Automatic reviewer revision request",
                        "",
                        diagnosis,
                        "",
                        "The predecessor files and full reviewer result are available under "
                        "`/current/prior_revision` and `/prior_review.json`. Revise the exact "
                        "candidate against the same evidence; do not broaden its scope.",
                    ]
                ).strip()
            candidate, decision, decision_reason, resolved_target = self._build_candidate_revision(
                job=job,
                candidate_id=candidate_id,
                revision=next_revision,
                evidence_ids=evidence_ids,
                evidence_markdown=round_evidence,
                owner_group=owner_group,
                owner_name=owner_name,
                expected_target=current_target,
                content_parent_version=current_parent_version,
                run_id=run_id,
                thread_id=thread_id,
                trace_scope=trace_scope,
                prior_review=current_prior_review,
                prior_revision_root=current_prior_root,
            )
            if candidate is None:
                return None, decision, decision_reason, resolved_target
            recommendation = str(candidate.review.get("recommendation") or "")
            if recommendation != "needs_revision":
                return candidate, "candidate", decision_reason, resolved_target
            if repair_round >= max(1, int(max_revisions)):
                self.store.append_audit_event(
                    {
                        "event": "automatic_revision_budget_exhausted",
                        "candidate_id": candidate.candidate_id,
                        "revision": candidate.revision,
                        "job_id": job.job_id,
                        "review": dict(candidate.review),
                    }
                )
                return candidate, "candidate", decision_reason, resolved_target
            self.store.append_audit_event(
                {
                    "event": "automatic_revision_requested",
                    "candidate_id": candidate.candidate_id,
                    "from_revision": candidate.revision,
                    "to_revision": candidate.revision + 1,
                    "job_id": job.job_id,
                    "repair_round": repair_round,
                }
            )
            current_prior_review = dict(candidate.review)
            current_prior_root = self.store.revision_dir(
                candidate.candidate_id,
                candidate.revision,
            )
            current_parent_version = candidate_version(candidate)
            current_target = resolved_target
            next_revision = candidate.revision + 1
        raise RuntimeError("automatic review repair loop ended without a terminal result")

    @staticmethod
    def _review_revision_diagnosis(review: dict[str, Any]) -> str:
        parts = [
            str(review.get("rationale") or "").strip(),
            str(review.get("scope_assessment") or "").strip(),
            *(
                str(item).strip()
                for item in list(review.get("concerns") or [])
                if str(item).strip()
            ),
        ]
        return "\n".join(f"- {part}" for part in parts if part) or (
            "- The reviewer requested a narrower or better-supported revision."
        )

    def _build_candidate_revision(
        self,
        *,
        job: SelfEvolutionJob,
        candidate_id: str,
        revision: int,
        evidence_ids: list[str],
        evidence_markdown: str,
        owner_group: str,
        owner_name: str,
        expected_target: str,
        content_parent_version: str,
        run_id: str,
        thread_id: str,
        trace_scope: EvolutionTraceScope | None,
        prior_review: dict[str, Any] | None = None,
        prior_revision_root: Path | None = None,
    ) -> tuple[LearningCandidate | None, str, str, str]:
        candidate_root = prepare_candidate_workspace(
            store=self.store,
            candidate_id=candidate_id,
            repo_root=self.repo_root,
            revision=revision,
            evidence_markdown=evidence_markdown,
            owner_group=owner_group,
            owner_name=owner_name,
            prior_revision_root=prior_revision_root,
        )
        if prior_review is not None:
            self.store.write_revision_json(
                candidate_id,
                revision,
                "prior_review.json",
                prior_review,
            )
        proposer, reviewer = self._agents(job)
        correction_feedback: list[str] = []
        proposer_attempts: list[str] = []
        validation_feedback_rounds: list[dict[str, Any]] = []
        proposal: ProposerResult | None = None
        candidate: LearningCandidate | None = None
        report: ValidationReport | None = None
        for correction_round in range(1, 4):
            proposal, proposer_meta = _invoke_agent_method(
                proposer.propose,
                candidate_root=candidate_root,
                trace_scope=trace_scope,
                correction_feedback=correction_feedback,
            )
            response_text = _response_text(proposal, proposer_meta)
            proposer_attempts.append(response_text)
            if isinstance(proposal, TextResult):
                # Keep all edits as an unsubmitted draft, outside immutable
                # revision slots, so a later real proposal can use this number.
                evidence_path = self.store.write_job_evidence(
                    job.job_id,
                    f"{candidate_id}.r{revision:04d}.a{job.attempt_count}.proposer_response.txt",
                    response_text,
                )
                draft_path = evidence_path.with_suffix(".draft")
                if draft_path.exists():
                    raise FileExistsError(draft_path)
                shutil.move(str(candidate_root), draft_path)
                job.outcome.setdefault("text_responses", []).append({
                    "stage": "proposal", "text": proposal.text,
                    "draft_path": str(draft_path.relative_to(self.workspace)),
                })
                self.store.append_audit_event({
                    "event": "proposer_text_response", "job_id": job.job_id,
                    "candidate_id": candidate_id, "revision": revision,
                    "draft_path": str(draft_path.relative_to(self.workspace)),
                })
                return None, "text", proposal.text, str(expected_target or "")
            if proposal.action in {"defer", "ignore"}:
                shutil.rmtree(candidate_root)
                self.store.append_audit_event(
                    {
                        "event": "proposer_declined_candidate",
                        "candidate_id": candidate_id,
                        "revision": revision,
                        "decision": proposal.action,
                        "reason": proposal.rationale,
                    }
                )
                return (
                    None,
                    proposal.action,
                    str(proposal.rationale or "").strip(),
                    str(expected_target or "").strip(),
                )

            action = proposal.action
            group = str(proposal.group or "").strip() if action == "skill" else ""
            name = str(proposal.name or "").strip() if action == "skill" else ""
            resolved_target = (
                f"{group}/{name}"
                if action == "skill" and group and name
                else str(expected_target or "").strip()
            )
            if resolved_target != str(expected_target or "").strip():
                shutil.rmtree(candidate_root)
                return (
                    None,
                    "reroute",
                    str(proposal.rationale or "").strip(),
                    resolved_target,
                )
            selected_route = (
                "workspace_preference"
                if action == "memory"
                else "amend_existing_skill"
                if self._skill_target_exists(group=group, name=name)
                else "new_skill"
            )
            if action == "skill":
                self._discard_unchanged_memory_copy(candidate_root)

            candidate = LearningCandidate(
                candidate_id=candidate_id,
                project_id=self.project_id,
                run_id=run_id,
                thread_id=thread_id,
                action=action,
                episode_id=job.episode_id,
                status="pending",
                route=selected_route,
                group=group,
                name=name,
                rationale=proposal.rationale,
                evidence_ids=list(evidence_ids),
                revision=revision,
                created_at=utc_now(),
            )
            preliminary = self.gate.run(candidate)
            if preliminary.valid:
                candidate.base_target_hash = self._base_target_hash(
                    action=action,
                    group=group,
                    name=name,
                    candidate_root=candidate_root,
                )
                candidate.bundle_hash = self._bundle_hash(
                    action=action,
                    group=group,
                    name=name,
                    candidate_root=candidate_root,
                )
                report = self.gate.run(candidate)
            else:
                report = preliminary
            if not report.repair_required:
                break
            correction_feedback = [*report.errors, *report.diagnostics]
            validation_feedback_rounds.append(
                {
                    "round": correction_round,
                    "report": report.to_dict(),
                }
            )

        assert proposal is not None and candidate is not None and report is not None
        proposer_evidence = proposer_attempts[-1]
        job_evidence_name = (
            f"{candidate_id}.r{revision:04d}.a{max(1, int(job.attempt_count or 1))}."
            "proposer_response.txt"
        )
        self.store.write_job_evidence(job.job_id, job_evidence_name, proposer_evidence)
        self.store.write_revision_text(
            candidate_id,
            revision,
            "proposer_response.txt",
            proposer_evidence,
        )
        proposer_attempt_ref = ""
        if len(proposer_attempts) > 1:
            proposer_attempt_ref = "proposer_attempts.json"
            self.store.write_revision_json(
                candidate_id,
                revision,
                proposer_attempt_ref,
                {
                    "attempts": [
                        {"round": index, "response_text": attempt_text}
                        for index, attempt_text in enumerate(proposer_attempts, start=1)
                    ]
                },
            )
        validation_feedback_ref = ""
        if validation_feedback_rounds:
            validation_feedback_ref = "validation_feedback.json"
            self.store.write_revision_json(
                candidate_id,
                revision,
                validation_feedback_ref,
                {"rounds": validation_feedback_rounds},
            )
        self.store.write_validation_report(report, revision=revision)
        candidate.validation = report.to_dict()

        revision_record = CandidateRevision(
            candidate_id=candidate_id,
            revision=revision,
            route=candidate.route,
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
            "content_parent_version": str(content_parent_version or ""),
            "raw_response_ref": "proposer_response.txt",
            **(
                {"attempt_response_ref": proposer_attempt_ref}
                if proposer_attempt_ref
                else {}
            ),
            **(
                {"validation_feedback_ref": validation_feedback_ref}
                if validation_feedback_ref
                else {}
            ),
        }
        self.store.write_revision_json(candidate_id, revision, "proposal.json", proposal_artifact)
        try:
            review, reviewer_meta = _invoke_agent_method(
                reviewer.review,
                candidate_root=candidate_root,
                action=candidate.action,
                group=candidate.group,
                name=candidate.name,
                rationale=candidate.rationale,
                validation=report.to_dict(),
                trace_scope=trace_scope,
            )
        except Exception as exc:
            review_payload = {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "recommendation": "unavailable",
                "summary": "The semantic reviewer did not return a validated structured result.",
            }
            self.store.write_revision_json(
                candidate_id,
                revision,
                "review.json",
                review_payload,
            )
            candidate.status = "revision"
            candidate.review = review_payload
            self.store.write_candidate(candidate)
            job.candidate_id = candidate_id
            self.store.append_audit_event(
                {
                    "event": "reviewer_error",
                    "candidate_id": candidate_id,
                    "revision": revision,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return (
                self.store.read_candidate(candidate_id),
                "candidate",
                "",
                resolved_target,
            )
        reviewer_evidence = _response_text(review, reviewer_meta)
        job_evidence_name = (
            f"{candidate_id}.r{revision:04d}.a{max(1, int(job.attempt_count or 1))}."
            "reviewer_response.txt"
        )
        self.store.write_job_evidence(job.job_id, job_evidence_name, reviewer_evidence)
        self.store.write_revision_text(
            candidate_id,
            revision,
            "reviewer_response.txt",
            reviewer_evidence,
        )
        review_payload = {
            **review.model_dump(mode="json"),
            "raw_response_ref": "reviewer_response.txt",
        }
        recommendation = "" if isinstance(review, TextResult) else review.recommendation
        if isinstance(review, TextResult):
            review_payload.update(format="text", summary=review.text)
            job.outcome.setdefault("text_responses", []).append({
                "stage": "review", "text": review.text, "candidate_id": candidate_id,
            })
        self.store.write_revision_json(
            candidate_id,
            revision,
            "review.json",
            review_payload,
        )
        candidate.status = (
            "rejected"
            if recommendation == "reject"
            else "revision"
            if recommendation == "needs_revision"
            else "review"
        )
        candidate.review = review_payload
        self.store.write_candidate(candidate)
        job.candidate_id = candidate_id
        self.store.append_audit_event(
            {
                "event": "reviewer_text_response" if isinstance(review, TextResult) else "reviewer_recommendation",
                "candidate_id": candidate_id,
                "revision": revision,
                "recommendation": recommendation,
            }
        )
        if recommendation == "approve":
            try:
                activation = self.effective.record_review_result(
                    candidate,
                    report,
                    review_payload,
                    mode=self.mode,
                )
            except Exception as exc:
                activation = {
                    "eligible": True,
                    "auto_head_advanced": False,
                    "selected": False,
                    "held_reason": f"{type(exc).__name__}: {exc}",
                }
                self.store.append_audit_event(
                    {
                        "event": "automatic_selection_error",
                        "candidate_id": candidate_id,
                        "revision": revision,
                        "target": (
                            "/memories/AGENTS.md"
                            if candidate.action == "memory"
                            else f"{candidate.group}/{candidate.name}"
                        ),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            if activation.get("selected"):
                candidate = self.store.update_candidate_status(candidate_id, "stable")
        return (
            self.store.read_candidate(candidate_id),
            "candidate",
            "",
            resolved_target,
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
                usage_context={"candidate_id": candidate_id, "revision": expected_revision,
                               "source_run_id": candidate.run_id, "source_thread_id": candidate.thread_id,
                               "trigger_kind": "review_candidate"},
            )
        report = self.gate.run(candidate)
        self.store.write_validation_report(report, revision=candidate.revision)
        candidate_root = self.store.revision_dir(
            candidate.candidate_id,
            candidate.revision,
        )
        review, reviewer_meta = _invoke_agent_method(
            reviewer.review,
            candidate_root=candidate_root,
            action=candidate.action,
            group=candidate.group,
            name=candidate.name,
            rationale=candidate.rationale,
            validation=report.to_dict(),
            trace_scope=self._trace_scope_for_evidence_ids(
                list(candidate.evidence_ids),
                anchor_run_id=candidate.run_id,
                target=self._candidate_resolved_target(candidate),
            ),
        )
        reviewer_evidence = _response_text(review, reviewer_meta)
        self.store.write_revision_text(
            candidate.candidate_id,
            candidate.revision,
            "reviewer_response.txt",
            reviewer_evidence,
        )
        self.store.write_revision_json(
            candidate.candidate_id,
            candidate.revision,
            "review.json",
            {
                **review.model_dump(mode="json"),
                "raw_response_ref": "reviewer_response.txt",
                **({"format": "text", "summary": review.text} if isinstance(review, TextResult) else {}),
            },
        )
        self.store.append_audit_event(
            {
                "event": "reviewer_text_response" if isinstance(review, TextResult) else "reviewer_recommendation",
                "candidate_id": candidate.candidate_id,
                "revision": candidate.revision,
                "recommendation": "" if isinstance(review, TextResult) else review.recommendation,
            }
        )
        return self.store.update_candidate_status(candidate.candidate_id, "review")

    # -- content and route contracts ------------------------------------

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "_read_error_type": type(exc).__name__,
                "_read_error": str(exc),
                "_path": str(path),
            }
        if not isinstance(value, dict):
            return {
                "_read_error_type": "TypeError",
                "_read_error": "expected a JSON object",
                "_path": str(path),
            }
        return value

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
                hash_text(frozen.read_text(encoding="utf-8"))
                if frozen.is_file()
                else self.store.memory_hash()
            )
        frozen_target = candidate_root / "current" / "skills" / group / name
        if frozen_target.is_dir():
            return hash_tree(frozen_target)
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
                hash_text(path.read_text(encoding="utf-8"))
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

__all__ = ["SelfEvolutionCoordinator"]
