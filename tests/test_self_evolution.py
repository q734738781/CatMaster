from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.runtime.observability_store import ObservabilityStore
from catmaster.runtime.self_evolution import (
    CandidateGate,
    LearningCandidate,
    Observation,
    PromotionManager,
    ProposerResult,
    ReflectionResult,
    ReviewerResult,
    SelfEvolutionCoordinator,
    SelfEvolutionStore,
    SkillRun,
    normalize_candidate_status,
)
from catmaster.runtime.self_evolution.consolidation import ConsolidationService
from catmaster.runtime.self_evolution.storage import (
    hash_text,
    hash_tree,
)
from catmaster.runtime.self_evolution.telemetry import write_skill_version_manifest
from catmaster.runtime.self_evolution.trace import collect_turn_trace
from catmaster.specialists.runtime import build_specialist_runner
from catmaster.tools.base import ensure_project_space_layout


def _skill_text(name: str, marker: str) -> str:
    return "\n".join(
        [
            "---",
            f"name: {name}",
            f"description: {marker} surface termination repair selection workflow.",
            "license: project-local",
            "compatibility: local",
            "---",
            f"# {name}",
            "",
            "## Overview",
            marker,
            "",
            "## Quick Start",
            "Use the explicit surface index only for a matching termination-selection failure.",
            "",
            "## Workflow",
            "Repair the bounded surface termination decision.",
            "",
            "## Method-critical defaults",
            "Do not add checksum or recovery-grade QC to ordinary work.",
            "",
            "## Output Contract",
            "Return the selected termination and evidence.",
            "",
            "## References",
            "No external references.",
            "",
        ]
    )


def _write_skill(root: Path, *, group: str, name: str, marker: str) -> Path:
    path = root / group / name if group else root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "SKILL.md").write_text(_skill_text(name, marker), encoding="utf-8")
    return path


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "skills").mkdir(parents=True)
    (root / "skills" / "AGENTS.MD").write_text(
        "# Skill authoring\n\nKeep changes bounded.\n",
        encoding="utf-8",
    )
    return root


def _run_dir(
    workspace: Path,
    run_id: str,
    *,
    prompt: str = "Complete the task.",
    status: str = "done",
    final_answer: str = "",
    resume_guidance: str = "",
) -> Path:
    path = workspace / "metadata" / "runs" / run_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "thread_id": "",
                "entrypoint": "experiment",
                "status": status,
                "user_prompt": prompt,
                "summary": "Task finished." if status == "done" else "Task failed.",
                "final_answer": final_answer,
                "resume_guidance": resume_guidance,
            }
        ),
        encoding="utf-8",
    )
    (path / "meta.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    return path


def _record_run_event(
    run_dir: Path,
    *,
    name: str,
    payload: dict,
    source: str = "test",
    ts: float = 1.0,
) -> None:
    ObservabilityStore(run_dir).record_event(
        source=source,
        channel="test",
        name=name,
        category="test",
        ts=ts,
        seq=None,
        run_id=run_dir.name,
        task_id="",
        step_id=None,
        payload=payload,
    )


class _MemoryProposer:
    def __init__(self, suffix: str = "- Prefer Chinese reports.\n") -> None:
        self.suffix = suffix

    def reflect(self, **_kwargs):
        return (
            ReflectionResult(
                kind="workspace_preference",
                name="report_language",
                change="Prefer Chinese for future workspace reports.",
                evidence_refs=[],
                rationale="The user explicitly established a durable preference.",
            ),
            {},
        )

    def propose(self, *, candidate_root: Path):
        path = candidate_root / "memories" / "AGENTS.md"
        current = path.read_text(encoding="utf-8")
        path.write_text(current.rstrip() + "\n" + self.suffix, encoding="utf-8")
        return (
            ProposerResult(
                action="memory",
                rationale="The user explicitly requested a durable workspace preference.",
                delta_operation="merge",
                applicability_boundary=["Future reports in this workspace."],
                non_applicability=["Other workspaces and verbatim source quotations."],
                expected_step_change="Use Chinese for future workspace reports.",
            ),
            {},
        )


class _SkillProposer:
    def __init__(
        self,
        *,
        group: str = "materials_worker",
        name: str = "surface-repair",
        marker: str = "candidate revision",
        reflection_kind: str = "skill_revision",
        proposed_group: str = "",
        proposed_name: str = "",
    ) -> None:
        self.group = group
        self.name = name
        self.marker = marker
        self.reflection_kind = reflection_kind
        self.proposed_group = proposed_group or group
        self.proposed_name = proposed_name or name

    def reflect(self, **_kwargs):
        return (
            ReflectionResult(
                kind=self.reflection_kind,
                group=self.group,
                name=self.name,
                change=(
                    "Repair the explicit surface-index decision only for a "
                    "matching termination-selection failure."
                ),
                evidence_refs=[],
                rationale="The complete trajectory demonstrates a missing bounded instruction.",
            ),
            {},
        )

    def propose(self, *, candidate_root: Path):
        _write_skill(
            candidate_root / "proposed",
            group=self.proposed_group,
            name=self.proposed_name,
            marker=self.marker,
        )
        return (
            ProposerResult(
                action="skill",
                group=self.proposed_group,
                name=self.proposed_name,
                rationale="Repeated verified failures and a counterexample support a narrow owner-skill amendment.",
                delta_operation="merge",
                applicability_boundary=["Verified surface termination-selection failures."],
                non_applicability=["Ordinary analysis without a termination-selection failure."],
                expected_step_change="Repair the explicit surface-index decision without adding general QC.",
            ),
            {},
        )


class _Reviewer:
    def __init__(self, recommendation: str = "approve") -> None:
        self.recommendation = recommendation

    def review(self, **_kwargs):
        return (
            ReviewerResult(
                recommendation=self.recommendation,
                summary="A bounded change with explicit non-applicability.",
                evidence_sufficiency="Independent supporting and counterexample evidence is present.",
                scope_assessment="The scope is limited to the observed decision.",
                counterexamples=["Ordinary analysis must not activate the skill."],
                concerns=[],
                human_checks=["Confirm the exact target and canary scope."],
                rationale="The evidence and bounded scope support human canary review.",
            ),
            {},
        )


class _SequenceReviewer:
    def __init__(self, *recommendations: str) -> None:
        self.recommendations = list(recommendations)

    def review(self, **_kwargs):
        return _Reviewer(self.recommendations.pop(0)).review()


class _FailingReviewer:
    def review(self, **_kwargs):
        raise RuntimeError("review service unavailable")


class _NoChangeProposer:
    def __init__(self, kind: str = "no_change") -> None:
        self.kind = kind
        self.trajectories: list[str] = []

    def reflect(self, *, trajectory_markdown: str, **_kwargs):
        self.trajectories.append(trajectory_markdown)
        return ReflectionResult(kind=self.kind), {}

    def propose(self, **_kwargs):
        raise AssertionError("non-actionable reflection must not invoke proposal")


def _explicit_preference_candidate(
    workspace: Path,
    repo: Path,
    *,
    reviewer=None,
    proposer=None,
) -> tuple[SelfEvolutionCoordinator, LearningCandidate]:
    run_dir = _run_dir(
        workspace,
        "run-preference",
        prompt="Use Chinese for future workspace reports.",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=proposer or _MemoryProposer(),
        reviewer=reviewer or _Reviewer(),
    )
    coordinator.enqueue_explicit_learn(
        run_id="run-preference",
        run_dir=run_dir,
        thread_id="thread-a",
        note="From now on, always prefer Chinese for reports in this workspace.",
    )
    job = coordinator.process_pending_jobs()[0]
    candidate = coordinator.store.read_candidate(job.candidate_id)
    assert candidate is not None
    return coordinator, candidate


def _recurrent_skill_candidate(
    workspace: Path,
    repo: Path,
    *,
    reviewer=None,
    proposer=None,
) -> tuple[SelfEvolutionCoordinator, LearningCandidate]:
    _write_skill(
        repo / "skills",
        group="materials_worker",
        name="surface-repair",
        marker="surface termination repair selection failure explicit index",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=proposer or _SkillProposer(),
        reviewer=reviewer or _Reviewer(),
    )
    claim = "surface termination repair selection failure needs an explicit surface index"
    inputs = [
        ("run-fail-a", "thread-a", "error", "verifier:run-fail-a", "supporting"),
        ("run-fail-b", "thread-b", "error", "verifier:run-fail-b", "supporting"),
        ("run-success-c", "thread-b", "done", "outcome:surface-control", "counterexample"),
    ]
    for run_id, thread_id, status, outcome_ref, role in inputs:
        run_dir = _run_dir(workspace, run_id, prompt=claim, status=status)
        coordinator.enqueue_post_run(
            run_id=run_id,
            thread_id=thread_id,
            terminal_status=status,
            run_dir=run_dir,
            payload={
                "learning_claim": claim,
                "task_outcome": "failure" if status == "error" else "success",
                "outcome_ref": outcome_ref,
                "evidence_role": role,
            },
        )
        coordinator.process_pending_jobs()
    candidates = coordinator.store.list_candidates()
    assert len(candidates) == 1
    return coordinator, candidates[0]


def _actual_owner_candidate(
    workspace: Path,
    *,
    run_id: str,
    note: str,
    marker: str,
) -> tuple[SelfEvolutionCoordinator, LearningCandidate]:
    repo = Path(__file__).resolve().parents[1]
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=_SkillProposer(
            name="surface-and-termination-screening",
            marker=marker,
        ),
        reviewer=_Reviewer(),
    )
    run_dir = _run_dir(workspace, run_id, prompt=note)
    coordinator.enqueue_explicit_learn(
        run_id=run_id,
        run_dir=run_dir,
        thread_id=f"thread-{run_id}",
        note=note,
    )
    job = coordinator.process_pending_jobs()[0]
    candidate = coordinator.store.read_candidate(job.candidate_id)
    assert candidate is not None and candidate.status == "review"
    assert candidate.name == "surface-and-termination-screening"
    return coordinator, candidate


def test_store_uses_only_four_domain_tables_and_minimal_observation_columns(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = SelfEvolutionStore(workspace, project_id="demo")
    with sqlite3.connect(store.db_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(observations)")
        }
        job_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(jobs)")
        }
    assert tables == {"jobs", "observations", "candidates", "skill_runs"}
    assert columns == {
        "observation_id",
        "run_id",
        "thread_id",
        "signal_kind",
        "target",
        "claim",
        "evidence_refs_json",
        "outcome_ref",
        "status",
        "created_at",
    }
    assert not {"confidence", "importance", "model", "tokens", "checksum"} & columns
    assert {"payload_json", "owner", "lease_until", "updated_at"} <= job_columns
    assert not {"heartbeat_at", "input_ref"} & job_columns


@pytest.mark.parametrize(
    "unsupported",
    ["proposed", "approved", "reviewed", "promoted", "rolled_back"],
)
def test_candidate_status_rejects_abandoned_lifecycle_values(
    unsupported: str,
) -> None:
    with pytest.raises(ValueError, match="unsupported candidate status"):
        normalize_candidate_status(unsupported)


def test_store_uses_filesystem_safe_sqlite_journal_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CATMASTER_WORKSPACE_SQLITE_JOURNAL_MODE", "DELETE")
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    with store._connect() as connection:
        mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
    assert mode == "delete"


def test_skill_run_reports_bounded_actual_use_without_claiming_causal_credit() -> None:
    presented_only = SkillRun(
        run_id="run-presented",
        skill_name="materials_worker/surface-repair",
        skill_version="sec_test@r0001",
        presented=True,
    )
    read = SkillRun(
        run_id="run-read",
        skill_name="materials_worker/surface-repair",
        skill_version="sec_test@r0001",
        presented=True,
        read=True,
    )
    assert presented_only.to_dict()["used"] is False
    assert read.to_dict()["used"] is True


def test_jobs_keep_one_payload_copy_and_enforce_same_owner_finish(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(workspace, "run-one")
    store = SelfEvolutionStore(workspace, project_id="demo")
    job = store.enqueue_job(
        trigger_kind="explicit_learn",
        run_id="run-one",
        run_dir=run_dir,
        payload={"note": "durable preference"},
    )
    assert job.payload == {"note": "durable preference"}
    assert not (store.root / "job_inputs").exists()
    claimed = store.claim_jobs(owner="worker-a", lease_seconds=60)
    assert len(claimed) == 1 and claimed[0].owner == "worker-a"
    with pytest.raises(RuntimeError, match="lease is no longer owned"):
        store.finish_job(claimed[0], status="done", owner="worker-b")
    finished = store.finish_job(claimed[0], status="done", owner="worker-a")
    assert finished.status == "done"
    assert finished.payload == {"note": "durable preference"}


def test_restart_requeues_only_expired_leases(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(workspace, "run-one")
    store = SelfEvolutionStore(workspace, project_id="demo")
    store.enqueue_job(trigger_kind="post_run", run_id="run-one", run_dir=run_dir)
    claimed = store.claim_jobs(owner="worker-a", lease_seconds=600)[0]
    assert store.requeue_expired_jobs() == 0
    with sqlite3.connect(store.db_path) as connection:
        connection.execute(
            "UPDATE jobs SET lease_until = '2000-01-01T00:00:00+00:00' WHERE job_id = ?",
            (claimed.job_id,),
        )
    assert store.requeue_expired_jobs() == 1
    assert store.list_jobs()[0].status == "queued"


def test_trace_preserves_complete_semantic_events_and_excludes_transport_duplicates(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(
        workspace,
        "run-one",
        status="error",
        final_answer="The final answer retained the repaired output.",
        resume_guidance="Continue from the approved repair boundary.",
    )
    for index in range(12):
        _record_run_event(
            run_dir,
            name="LLM_RAW_RESPONSE",
            ts=float(index + 1),
            payload={
                "callback_run_id": f"llm-{index}",
                "generations": [
                    {
                        "reasoning_text": f"reasoning-{index}",
                        "response_text": f"complete-model-result-{index}",
                        "parsed_tool_calls": [],
                        "response_content_raw": (
                            [{"type": "text", "text": "raw-content-block"}]
                            if index == 0
                            else []
                        ),
                        "invalid_tool_calls": (
                            [{"name": "broken_call", "error": "invalid arguments"}]
                            if index == 0
                            else []
                        ),
                    }
                ],
            },
        )
    _record_run_event(
        run_dir,
        name="LLM_RAW_REQUEST",
        ts=20,
        payload={"messages": ["TRANSPORT-DUPLICATE-SHOULD-NOT-APPEAR"]},
    )
    _record_run_event(
        run_dir,
        name="TASK_DECISION",
        ts=20.5,
        payload={"decision": "repair", "reason": "tool result contradicted the first branch"},
    )
    long_result = "full-tool-result-" + ("x" * 6_000)
    raw_tool_result = {
        "content": long_result,
        "artifact": {"complete_detail": "raw-artifact-content"},
    }
    _record_run_event(
        run_dir,
        name="TOOL_RAW_OUTPUT",
        ts=21,
        payload={
            "callback_run_id": "tool-1",
            "tool": "surface_select",
            "status": "success",
            "projection": {"content_text": "condensed projection"},
            "raw_output": raw_tool_result,
        },
    )
    trace = collect_turn_trace(run_dir=run_dir)
    markdown = trace.to_markdown()
    assert len(trace.events) == 14
    assert "complete-model-result-0" in markdown
    assert "complete-model-result-11" in markdown
    assert "raw-content-block" in markdown
    assert "broken_call" in markdown
    assert long_result in markdown
    assert "raw-artifact-content" in markdown
    assert "Continue from the approved repair boundary." in markdown
    assert "The final answer retained the repaired output." in markdown
    assert "Task failed." in markdown
    assert "tool result contradicted the first branch" in markdown
    assert "TRANSPORT-DUPLICATE-SHOULD-NOT-APPEAR" not in markdown
    assert "omitted_count" not in trace.to_dict()
    assert trace.run_id == "run-one"


def test_execution_error_and_unreferenced_task_end_are_not_verified_outcomes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(workspace, "run-error", status="error")
    _record_run_event(
        run_dir,
        name="TASK_END",
        payload={"outcome": "failure", "summary": "The task failed."},
    )

    trace = collect_turn_trace(run_dir=run_dir)

    assert trace.status == "error"
    assert trace.task_outcome == ""
    assert trace.outcome_ref == ""

    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=_NoChangeProposer(),
        reviewer=_Reviewer(),
    )
    coordinator.enqueue_post_run(
        run_id="run-error",
        thread_id="thread-a",
        terminal_status="error",
        run_dir=run_dir,
        payload={"learning_claim": "Do not infer a task verdict from process failure."},
    )
    coordinator.process_pending_jobs()
    assert coordinator.store.list_observations() == []


def test_formal_task_end_requires_explicit_verifier_reference(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(workspace, "run-verified", status="done")
    _record_run_event(
        run_dir,
        name="TASK_END",
        source="host_verifier",
        payload={
            "task_outcome": "success",
            "outcome_ref": "verifier:run-verified",
            "summary": "Host verifier passed.",
        },
    )

    trace = collect_turn_trace(run_dir=run_dir)

    assert trace.task_outcome == "verified_success"
    assert trace.outcome_ref == "verifier:run-verified"


def test_complete_failure_and_repair_trajectory_reaches_reflector_and_candidate(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(
        workspace,
        "run-repaired",
        prompt="Repair the failed surface selection call.",
        status="done",
    )
    _record_run_event(
        run_dir,
        name="TOOL_RAW_INPUT",
        ts=1,
        payload={
            "callback_run_id": "call-failed",
            "tool": "surface_select",
            "params_compact": '{"surface_index": 4}',
        },
    )
    _record_run_event(
        run_dir,
        name="TOOL_CALL_END",
        ts=2,
        payload={
            "callback_run_id": "call-failed",
            "tool": "surface_select",
            "status": "error",
            "error": "Index 4 is outside the generated set.",
        },
    )
    _record_run_event(
        run_dir,
        name="TOOL_RAW_INPUT",
        ts=3,
        payload={
            "callback_run_id": "call-repair",
            "tool": "surface_select",
            "params_compact": '{"surface_index": 2}',
        },
    )
    _record_run_event(
        run_dir,
        name="TOOL_CALL_END",
        ts=4,
        payload={
            "callback_run_id": "call-repair",
            "tool": "surface_select",
            "status": "success",
            "projection": {"content_preview": "Selected surface 2."},
        },
    )
    proposer = _SkillProposer()
    repo = _repo(tmp_path)
    _write_skill(
        repo / "skills",
        group="materials_worker",
        name="surface-repair",
        marker="current surface repair workflow",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=proposer,
        reviewer=_Reviewer(),
    )
    coordinator.enqueue_post_run(
        run_id="run-repaired",
        thread_id="thread-repair",
        terminal_status="done",
        run_dir=run_dir,
    )
    coordinator.process_pending_jobs()

    observations = coordinator.store.list_observations()
    assert len(observations) == 1
    assert observations[0].signal_kind == "skill_revision"
    assert observations[0].target == "materials_worker/surface-repair"
    candidate = coordinator.store.list_candidates()[0]
    evidence = (
        coordinator.store.revision_dir(candidate.candidate_id, candidate.revision)
        / "evidence.md"
    ).read_text(encoding="utf-8")
    assert "Index 4 is outside the generated set." in evidence
    assert "Selected surface 2." in evidence


def test_successful_terminal_run_does_not_create_observation_or_candidate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    repo = _repo(tmp_path)
    proposer = _NoChangeProposer()
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=proposer,
        reviewer=_Reviewer(),
    )
    run_dir = _run_dir(workspace, "run-one", prompt="The agent voluntarily computed a checksum.")
    job = coordinator.enqueue_post_run(
        run_id="run-one",
        thread_id="thread-a",
        terminal_status="done",
        run_dir=run_dir,
    )
    assert job is not None
    processed = coordinator.process_pending_jobs()
    assert len(processed) == 1 and processed[0].status == "done"
    assert len(proposer.trajectories) == 1
    assert "voluntarily computed a checksum" in proposer.trajectories[0]
    assert len(coordinator.store.list_jobs()) == 1
    assert coordinator.store.list_observations() == []
    assert coordinator.store.list_candidates() == []


def test_new_skill_proposer_cannot_redirect_the_reflected_exact_target(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    repo = _repo(tmp_path)
    proposer = _SkillProposer(
        group="materials_worker",
        name="new-surface-method",
        reflection_kind="skill_discovery",
        proposed_name="unrelated-method",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=proposer,
        reviewer=_Reviewer(),
    )
    run_dir = _run_dir(
        workspace,
        "run-discovery",
        prompt="The complete episode demonstrates one independent reusable surface method.",
    )
    coordinator.enqueue_post_run(
        run_id="run-discovery",
        thread_id="thread-discovery",
        terminal_status="done",
        run_dir=run_dir,
    )
    coordinator.process_pending_jobs()

    observation = coordinator.store.list_observations()[0]
    candidate = coordinator.store.list_candidates()[0]
    assert observation.target == "materials_worker/new-surface-method"
    assert candidate.route == "new_skill"
    assert candidate.status == "revision"
    assert any(
        "cannot redirect" in error
        for error in candidate.validation["errors"]
    )


def test_new_skill_keeps_the_reflected_exact_target_through_review(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    repo = _repo(tmp_path)
    proposer = _SkillProposer(
        group="materials_worker",
        name="new-surface-method",
        reflection_kind="skill_discovery",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=proposer,
        reviewer=_Reviewer(),
    )
    run_dir = _run_dir(workspace, "run-discovery")
    coordinator.enqueue_post_run(
        run_id="run-discovery",
        thread_id="thread-discovery",
        terminal_status="done",
        run_dir=run_dir,
    )
    coordinator.process_pending_jobs()

    candidate = coordinator.store.list_candidates()[0]
    assert candidate.route == "new_skill"
    assert candidate.group == "materials_worker"
    assert candidate.name == "new-surface-method"
    assert candidate.status == "review"


def test_explicit_durable_preference_skips_recurrence_but_stays_human_controlled(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _explicit_preference_candidate(workspace, _repo(tmp_path))
    assert candidate.route == "workspace_preference"
    assert candidate.action == "memory"
    assert candidate.status == "review"
    assert coordinator.store.read_memory_text() == ""
    assert (
        coordinator.store.revision_dir(candidate.candidate_id, 1)
        / "review.json"
    ).is_file()
    evidence = (
        coordinator.store.revision_dir(candidate.candidate_id, 1)
        / "evidence.md"
    ).read_text(encoding="utf-8")
    assert "From now on, always prefer Chinese for reports in this workspace." in evidence
    assert candidate.candidate_id not in coordinator.store.read_active_skills()["skills"]


def test_exact_target_combines_complete_cross_thread_evidence_without_threshold(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _recurrent_skill_candidate(workspace, _repo(tmp_path))
    assert candidate.route == "amend_existing_skill"
    assert candidate.group == "materials_worker"
    assert candidate.name == "surface-repair"
    assert candidate.status == "review"
    assert candidate.revision == 3
    observations = coordinator.store.list_observations(status="consolidated")
    assert len(observations) == 3
    assert len({item.thread_id for item in observations}) == 2
    proposal = json.loads(
        (
            coordinator.store.revision_dir(candidate.candidate_id, 3)
            / "proposal.json"
        ).read_text(encoding="utf-8")
    )
    assert proposal["evidence_ids"] == [
        item.observation_id for item in reversed(observations)
    ]
    assert "supporting_evidence_ids" not in proposal
    assert "counterexample_ids" not in proposal
    evidence = (
        coordinator.store.revision_dir(candidate.candidate_id, 3)
        / "evidence.md"
    ).read_text(encoding="utf-8")
    assert evidence.count("# Complete episode trajectory") == 3


def test_human_rejection_does_not_create_a_regex_blocklist_for_future_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _recurrent_skill_candidate(workspace, _repo(tmp_path))
    coordinator.promotion.reject(
        candidate,
        actor="alice",
        rationale="This proposal generalizes beyond the observed failure.",
    )
    rejected_revision = candidate.revision
    run_dir = _run_dir(
        workspace,
        "run-later",
        prompt="A later complete episode supplies new evidence for the same exact target.",
    )
    coordinator.enqueue_post_run(
        run_id="run-later",
        thread_id="thread-later",
        terminal_status="done",
        run_dir=run_dir,
    )
    coordinator.process_pending_jobs()
    revised = coordinator.store.read_candidate(candidate.candidate_id)
    assert revised is not None
    assert revised.revision == rejected_revision + 1
    assert revised.status == "review"
    audit = coordinator.store.audit_log_path.read_text(encoding="utf-8")
    assert "rejection_signature" not in audit


@pytest.mark.parametrize(
    "reflection_kind",
    [
        "no_change",
        "execution_lapse",
    ],
)
def test_non_actionable_reflections_never_invoke_candidate_proposal(
    tmp_path: Path,
    reflection_kind: str,
) -> None:
    workspace = tmp_path / reflection_kind
    run_dir = _run_dir(workspace, "run-route")
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path / reflection_kind),
        proposer=_NoChangeProposer(reflection_kind),
        reviewer=_Reviewer(),
    )
    coordinator.enqueue_explicit_learn(
        run_id="run-route",
        run_dir=run_dir,
        note="Inspect the complete episode before deciding whether anything is reusable.",
    )
    job = coordinator.process_pending_jobs()[0]
    assert job.status == "done"
    assert coordinator.store.list_candidates() == []
    assert coordinator.store.list_observations() == []


def test_ai_review_is_advisory_and_never_removes_human_release_actions(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "reject"
    coordinator, candidate = _explicit_preference_candidate(
        workspace,
        _repo(tmp_path / "reject"),
        reviewer=_Reviewer("reject"),
    )
    assert candidate.status == "review"
    assert coordinator.promotion.allowed_actions(candidate) == [
        "request_revision",
        "promote_stable",
        "reject",
    ]

    workspace_revision = tmp_path / "needs-revision"
    coordinator2, candidate2 = _explicit_preference_candidate(
        workspace_revision,
        _repo(tmp_path / "needs-revision"),
        reviewer=_Reviewer("needs_revision"),
    )
    assert candidate2.status == "review"
    assert coordinator2.promotion.allowed_actions(candidate2) == [
        "request_revision",
        "promote_stable",
        "reject",
    ]


def test_request_revision_creates_r0002_without_mutating_r0001(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    proposer = _MemoryProposer("- Prefer Chinese reports.\n")
    coordinator, candidate = _explicit_preference_candidate(
        workspace,
        _repo(tmp_path),
        reviewer=_SequenceReviewer("needs_revision", "approve"),
        proposer=proposer,
    )
    root1 = coordinator.store.revision_dir(candidate.candidate_id, 1)
    before = {
        path.relative_to(root1).as_posix(): path.read_bytes()
        for path in root1.rglob("*")
        if path.is_file()
    }
    updated = coordinator.promotion.request_revision(
        candidate,
        actor="alice",
        rationale="Limit the preference to generated reports, not quoted source text.",
    )
    coordinator.enqueue_revision(
        candidate_id=updated.candidate_id,
        expected_revision=1,
        guidance="Limit the preference to generated reports, not quoted source text.",
        actor="alice",
    )
    job = coordinator.process_pending_jobs()[0]
    revised = coordinator.store.read_candidate(candidate.candidate_id)
    assert job.status == "done"
    assert revised is not None and revised.revision == 2
    assert revised.status == "review"
    assert coordinator.store.revision_dir(candidate.candidate_id, 2).is_dir()
    after = {
        path.relative_to(root1).as_posix(): path.read_bytes()
        for path in root1.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_reviewer_failure_keeps_linked_immutable_candidate_and_error_job(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=_MemoryProposer(),
        reviewer=_FailingReviewer(),
    )
    run_dir = _run_dir(workspace, "run-review")
    coordinator.enqueue_explicit_learn(
        run_id="run-review",
        run_dir=run_dir,
        note="From now on always prefer Chinese workspace reports.",
    )
    job = coordinator.process_pending_jobs()[0]
    assert job.status == "error"
    assert job.candidate_id
    candidate = coordinator.store.read_candidate(job.candidate_id)
    assert candidate is not None
    assert candidate.status == "pending"
    assert not (
        coordinator.store.revision_dir(candidate.candidate_id, 1)
        / "review.json"
    ).exists()


def test_memory_requires_human_promotion_and_rollback_restores_exact_parent(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    store = SelfEvolutionStore(workspace, project_id="demo")
    before = "# Persistent Instruction Memory\n\n- Prefer English reports.\n"
    assert store.compare_and_swap_memory(
        expected_hash=store.memory_hash(),
        new_text=before,
    )[0]
    coordinator, candidate = _explicit_preference_candidate(workspace, _repo(tmp_path))
    assert coordinator.store.read_memory_text() == before
    report = coordinator.gate.run(candidate)
    promoted = coordinator.promotion.promote_stable(
        candidate,
        report,
        actor="alice",
        rationale="The explicit preference and exact memory diff were reviewed.",
    )
    assert promoted.status == "stable"
    assert "Chinese" in coordinator.store.read_memory_text()
    rolled_back = coordinator.promotion.rollback(
        promoted,
        actor="alice",
        rationale="The preference is no longer desired.",
    )
    assert rolled_back.status == "inactive"
    assert coordinator.store.read_memory_text() == before


def test_skill_rollback_restores_exact_previous_stable_revision(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, revision_one = _actual_owner_candidate(
        workspace,
        run_id="stable-revision-one",
        note=(
            "Surface termination screening must preserve the established slab "
            "generation and freezing policy."
        ),
        marker="stable revision one",
    )
    report_one = coordinator.gate.run(revision_one)
    canary_one = coordinator.promotion.start_canary(
        revision_one,
        report_one,
        actor="alice",
        run_ids=["canary-revision-one"],
    )
    version_one = f"{revision_one.candidate_id}@r{revision_one.revision:04d}"
    coordinator.store.upsert_skill_run(
        SkillRun(
            run_id="canary-revision-one",
            skill_name="materials_worker/surface-and-termination-screening",
            skill_version=version_one,
            presented=True,
            read=True,
            outcome="verified_success",
        )
    )
    coordinator.promotion.promote_stable(
        canary_one,
        report_one,
        actor="alice",
        rationale="The first exact revision passed its selected canary.",
    )

    coordinator_two, revision_two = _actual_owner_candidate(
        workspace,
        run_id="stable-revision-two",
        note=(
            "Surface termination screening must also preserve the bounded "
            "neighbor criterion in the same workflow."
        ),
        marker="stable revision two",
    )
    assert revision_two.candidate_id == revision_one.candidate_id
    assert revision_two.revision == revision_one.revision + 1
    report_two = coordinator_two.gate.run(revision_two)
    canary_two = coordinator_two.promotion.start_canary(
        revision_two,
        report_two,
        actor="alice",
        run_ids=["canary-revision-two"],
    )
    version_two = f"{revision_two.candidate_id}@r{revision_two.revision:04d}"
    coordinator_two.store.upsert_skill_run(
        SkillRun(
            run_id="canary-revision-two",
            skill_name="materials_worker/surface-and-termination-screening",
            skill_version=version_two,
            presented=True,
            read=True,
            outcome="verified_success",
        )
    )
    stable_two = coordinator_two.promotion.promote_stable(
        canary_two,
        report_two,
        actor="alice",
        rationale="The second exact revision passed its selected canary.",
    )
    materialized = (
        coordinator_two.store.self_develop_skills_dir
        / "materials_worker"
        / "surface-and-termination-screening"
        / "SKILL.md"
    )
    assert "stable revision two" in materialized.read_text(encoding="utf-8")

    rolled_back = coordinator_two.promotion.rollback(
        stable_two,
        actor="alice",
        rationale="Restore the previous exact stable revision.",
    )
    active = coordinator_two.store.read_active_skills()["skills"][
        "materials_worker/surface-and-termination-screening"
    ]
    assert rolled_back.status == "inactive"
    assert active == {"stable": version_one}
    restored = materialized.read_text(encoding="utf-8")
    assert "stable revision one" in restored
    assert "stable revision two" not in restored


def test_skill_must_canary_exact_scope_and_actual_use_before_stable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _recurrent_skill_candidate(workspace, _repo(tmp_path))
    report = coordinator.gate.run(candidate)
    assert coordinator.promotion.allowed_actions(candidate) == [
        "request_revision",
        "start_canary",
        "reject",
    ]
    with pytest.raises(ValueError, match="at least one explicit thread or run"):
        coordinator.promotion.start_canary(
            candidate,
            report,
            actor="alice",
        )
    canary = coordinator.promotion.start_canary(
        candidate,
        report,
        actor="alice",
        thread_ids=["thread-canary"],
        rationale="Low-risk canary thread.",
    )
    pointer = coordinator.store.read_active_skills()["skills"][
        "materials_worker/surface-repair"
    ]["canary"]
    exact_version = f"{candidate.candidate_id}@r{candidate.revision:04d}"
    assert pointer == {
        "version": exact_version,
        "thread_ids": ["thread-canary"],
        "run_ids": [],
    }
    assert "promote_stable" not in coordinator.promotion.allowed_actions(canary)
    coordinator.store.upsert_skill_run(
        SkillRun(
            run_id="run-canary",
            skill_name="materials_worker/surface-repair",
            skill_version=exact_version,
            presented=True,
            read=True,
            outcome="success",
        )
    )
    assert "promote_stable" not in coordinator.promotion.allowed_actions(canary)
    coordinator.store.upsert_skill_run(
        SkillRun(
            run_id="run-canary",
            skill_name="materials_worker/surface-repair",
            skill_version=exact_version,
            presented=True,
            read=True,
            outcome="verified_success",
        )
    )
    assert "promote_stable" in coordinator.promotion.allowed_actions(canary)
    stable = coordinator.promotion.promote_stable(
        canary,
        report,
        actor="alice",
        rationale="The exact canary revision improved its selected run.",
    )
    active = coordinator.store.read_active_skills()["skills"][
        "materials_worker/surface-repair"
    ]
    assert active == {"stable": exact_version}
    assert stable.status == "stable"


def test_false_activation_stops_only_exact_canary_and_preserves_stable(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _recurrent_skill_candidate(workspace, _repo(tmp_path))
    report = coordinator.gate.run(candidate)
    canary = coordinator.promotion.start_canary(
        candidate,
        report,
        actor="alice",
        run_ids=["run-canary"],
    )
    active = coordinator.store.read_active_skills()
    active["skills"]["materials_worker/surface-repair"]["stable"] = "older@r0001"
    coordinator.store.write_active_skills(active)
    exact_version = f"{candidate.candidate_id}@r{candidate.revision:04d}"
    stopped = coordinator.promotion.stop_canary_on_failure(
        skill_name="materials_worker/surface-repair",
        skill_version=exact_version,
        run_id="run-canary",
        reason="Hard-negative false activation.",
    )
    assert stopped is True
    pointers = coordinator.store.read_active_skills()["skills"][
        "materials_worker/surface-repair"
    ]
    assert pointers == {"stable": "older@r0001"}
    assert coordinator.store.read_candidate(canary.candidate_id).status == "inactive"


def test_host_verified_false_activation_uses_actual_exact_skill_read_and_stops_canary(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _recurrent_skill_candidate(workspace, _repo(tmp_path))
    report = coordinator.gate.run(candidate)
    coordinator.promotion.start_canary(
        candidate,
        report,
        actor="alice",
        run_ids=["run-observed-false-activation"],
    )
    active = coordinator.store.read_active_skills()
    active["skills"]["materials_worker/surface-repair"]["stable"] = "older@r0001"
    coordinator.store.write_active_skills(active)

    run_id = "run-observed-false-activation"
    run_dir = _run_dir(
        workspace,
        run_id,
        prompt="Perform an ordinary analysis with no surface-selection failure.",
        status="error",
    )
    exact_version = f"{candidate.candidate_id}@r{candidate.revision:04d}"
    virtual_path = "/.deepagents/skills/materials_worker/surface-repair"
    write_skill_version_manifest(
        run_dir=run_dir,
        run_id=run_id,
        entries=[
            {
                "skill_name": "materials_worker/surface-repair",
                "skill_version": exact_version,
                "virtual_path": virtual_path,
            }
        ],
    )
    _record_run_event(
        run_dir,
        name="TOOL_RAW_INPUT",
        ts=1,
        payload={
            "callback_run_id": "read-skill",
            "tool": "read_file",
            "params_compact": f'{{"path":"{virtual_path}/SKILL.md"}}',
        },
    )
    _record_run_event(
        run_dir,
        name="TOOL_CALL_END",
        ts=2,
        payload={
            "callback_run_id": "read-skill",
            "tool": "read_file",
            "status": "success",
        },
    )
    _record_run_event(
        run_dir,
        name="SKILL_OUTCOME",
        ts=3,
        source="host_verifier",
        payload={
            "skill_name": "materials_worker/surface-repair",
            "skill_version": exact_version,
            "outcome": "failure",
            "false_activation": True,
            "outcome_ref": "verifier:false-activation",
        },
    )
    coordinator._proposer = _NoChangeProposer()
    coordinator.enqueue_post_run(
        run_id=run_id,
        thread_id="thread-ordinary",
        terminal_status="error",
        run_dir=run_dir,
    )
    coordinator.process_pending_jobs()

    records = coordinator.store.list_skill_runs(
        skill_name="materials_worker/surface-repair",
        run_id=run_id,
    )
    assert len(records) == 1
    assert records[0].presented is True
    assert records[0].read is True
    assert records[0].helper_used is False
    assert records[0].outcome == "verified_failure"
    assert records[0].false_activation is True
    assert coordinator.store.read_active_skills()["skills"][
        "materials_worker/surface-repair"
    ] == {"stable": "older@r0001"}
    assert coordinator.store.read_candidate(candidate.candidate_id).status == "inactive"


def test_verified_failed_canary_run_stops_exact_pointer_without_false_activation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _recurrent_skill_candidate(workspace, _repo(tmp_path))
    report = coordinator.gate.run(candidate)
    run_id = "run-verified-canary-failure"
    coordinator.promotion.start_canary(
        candidate,
        report,
        actor="alice",
        run_ids=[run_id],
    )
    active = coordinator.store.read_active_skills()
    active["skills"]["materials_worker/surface-repair"]["stable"] = "older@r0001"
    coordinator.store.write_active_skills(active)

    run_dir = _run_dir(
        workspace,
        run_id,
        prompt="Use the surface repair procedure.",
        status="error",
    )
    exact_version = f"{candidate.candidate_id}@r{candidate.revision:04d}"
    virtual_path = "/.deepagents/skills/materials_worker/surface-repair"
    write_skill_version_manifest(
        run_dir=run_dir,
        run_id=run_id,
        entries=[
            {
                "skill_name": "materials_worker/surface-repair",
                "skill_version": exact_version,
                "virtual_path": virtual_path,
            }
        ],
    )
    _record_run_event(
        run_dir,
        name="TOOL_RAW_INPUT",
        ts=1,
        payload={
            "callback_run_id": "read-skill",
            "tool": "read_file",
            "params_compact": f'{{"path":"{virtual_path}/SKILL.md"}}',
        },
    )
    _record_run_event(
        run_dir,
        name="TOOL_CALL_END",
        ts=2,
        payload={
            "callback_run_id": "read-skill",
            "tool": "read_file",
            "status": "success",
        },
    )
    _record_run_event(
        run_dir,
        name="TASK_END",
        ts=3,
        source="host_verifier",
        payload={
            "task_outcome": "failure",
            "outcome_ref": "verifier:canary-task-failure",
            "summary": "The required surface selection remained wrong.",
        },
    )
    coordinator._proposer = _NoChangeProposer()
    coordinator.enqueue_post_run(
        run_id=run_id,
        thread_id="thread-canary-failure",
        terminal_status="error",
        run_dir=run_dir,
    )
    coordinator.process_pending_jobs()

    record = coordinator.store.list_skill_runs(
        skill_name="materials_worker/surface-repair",
        run_id=run_id,
    )[0]
    assert record.used is True
    assert record.outcome == "verified_failure"
    assert record.false_activation is False
    assert coordinator.store.read_active_skills()["skills"][
        "materials_worker/surface-repair"
    ] == {"stable": "older@r0001"}
    assert coordinator.store.read_candidate(candidate.candidate_id).status == "inactive"


def test_runtime_pins_exact_canary_and_stable_in_distinct_immutable_snapshots(tmp_path: Path) -> None:
    class _Profile:
        def config_for_role(self, role: str):
            return SimpleNamespace(model=f"{role}-model", provider="langchain", base_url=None)

    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, stable_candidate = _actual_owner_candidate(
        workspace,
        run_id="stable-source",
        note=(
            "Surface termination screening slab generation must preserve one fixed "
            "freezing policy and uniform VASP ranking."
        ),
        marker="stable revision",
    )
    stable_report = coordinator.gate.run(stable_candidate)
    stable_canary = coordinator.promotion.start_canary(
        stable_candidate,
        stable_report,
        actor="alice",
        run_ids=["stable-canary"],
    )
    coordinator.store.upsert_skill_run(
        SkillRun(
            run_id="stable-canary",
            skill_name="materials_worker/surface-and-termination-screening",
            skill_version=(
                f"{stable_candidate.candidate_id}@r{stable_candidate.revision:04d}"
            ),
            presented=True,
            read=True,
            outcome="verified_success",
        )
    )
    stable_candidate = coordinator.promotion.promote_stable(
        stable_canary,
        stable_report,
        actor="alice",
        rationale="Selected canary succeeded.",
    )
    coordinator2, canary_candidate = _actual_owner_candidate(
        workspace,
        run_id="canary-source",
        note=(
            "The controlled slab termination screening set must keep the neighbor "
            "criterion fixed alongside slab generation, freezing policy, lateral "
            "expansion, and standardized VASP ranking runs."
        ),
        marker="canary revision with fixed neighbor criterion",
    )
    canary_report = coordinator2.gate.run(canary_candidate)
    canary = coordinator2.promotion.start_canary(
        canary_candidate,
        canary_report,
        actor="alice",
        thread_ids=["thread-canary"],
    )

    canary_runner = build_specialist_runner(
        workspace=workspace,
        llm_profile=_Profile(),
        reporter=None,
        run_control=None,
        project_id="demo",
        preferred_entrypoint="experiment",
    ).runner
    stable_runner = build_specialist_runner(
        workspace=workspace,
        llm_profile=_Profile(),
        reporter=None,
        run_control=None,
        project_id="demo",
        preferred_entrypoint="experiment",
    ).runner
    canary_runner._stage_deepagent_assets(workspace / "files", thread_id="thread-canary")
    stable_runner._stage_deepagent_assets(workspace / "files", thread_id="thread-stable")
    assert canary_runner._skill_snapshot_root.is_dir()
    assert stable_runner._skill_snapshot_root.is_dir()
    assert (
        canary_runner._skill_snapshot_root
        / "skills/materials_worker/surface-and-termination-screening/SKILL.md"
    ).is_file()
    assert canary_runner._skill_snapshot_root != stable_runner._skill_snapshot_root
    assert canary_runner._skill_version_entries
    assert {
        item["skill_name"]
        for item in canary_runner._skill_version_entries
    } == {"materials_worker/surface-and-termination-screening"}
    assert {
        item["skill_name"]
        for item in stable_runner._skill_version_entries
    } == {"materials_worker/surface-and-termination-screening"}
    canary_version = next(
        item["skill_version"]
        for item in canary_runner._skill_version_entries
        if item["skill_name"] == "materials_worker/surface-and-termination-screening"
    )
    stable_version = next(
        item["skill_version"]
        for item in stable_runner._skill_version_entries
        if item["skill_name"] == "materials_worker/surface-and-termination-screening"
    )
    assert canary_version == (
        f"{canary_candidate.candidate_id}@r{canary_candidate.revision:04d}"
    )
    assert stable_version == (
        f"{stable_candidate.candidate_id}@r{stable_candidate.revision:04d}"
    )
    assert canary.status == "canary"


def test_builtin_drift_requests_a_new_revision(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator, candidate = _actual_owner_candidate(
        workspace,
        run_id="drift-source",
        note=(
            "Surface termination screening slab generation must preserve one fixed "
            "freezing policy and uniform VASP ranking."
        ),
        marker="candidate before builtin drift",
    )
    report = coordinator.gate.run(candidate)
    coordinator.promotion.start_canary(
        candidate,
        report,
        actor="alice",
        thread_ids=["thread-drift"],
    )
    snapshot_skill = (
        coordinator.store.revision_dir(candidate.candidate_id, 1)
        / "current/builtin_target/SKILL.md"
    )
    snapshot_skill.chmod(0o644)
    snapshot_skill.write_text(
        snapshot_skill.read_text(encoding="utf-8") + "\nLegacy snapshot marker.\n",
        encoding="utf-8",
    )
    runner = build_specialist_runner(
        workspace=workspace,
        llm_profile=SimpleNamespace(
            config_for_role=lambda role: SimpleNamespace(
                model=f"{role}-model",
                provider="langchain",
                base_url=None,
            )
        ),
        reporter=None,
        run_control=None,
        project_id="demo",
        preferred_entrypoint="experiment",
    ).runner
    selected = runner._active_skill_sources(thread_id="thread-drift")
    assert "materials_worker/surface-and-termination-screening" not in selected
    assert coordinator.store.read_candidate(candidate.candidate_id).status == "revision"


def test_newest_first_cursor_pagination_for_candidates_and_observations(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = SelfEvolutionStore(workspace, project_id="demo")
    for index in range(4):
        observation = Observation(
            observation_id=f"obs-{index}",
            run_id=f"run-{index}",
            thread_id=f"thread-{index % 2}",
            signal_kind="workspace_preference",
            target=f"memory/preference-{index}",
            claim=f"failure {index}",
            evidence_refs=[],
            outcome_ref=f"run:{index}",
            created_at=f"2026-07-29T00:00:0{index}+00:00",
        )
        store.write_observation(observation)
        candidate = LearningCandidate(
            candidate_id=f"candidate-{index}",
            project_id="demo",
            run_id=f"run-{index}",
            thread_id="thread",
            action="memory",
            route="workspace_preference",
            evidence_ids=[observation.observation_id],
            created_at=f"2026-07-29T00:00:0{index}+00:00",
        )
        root = store.reset_candidate_dir(candidate.candidate_id)
        (root / "memories").mkdir()
        (root / "memories/AGENTS.md").write_text(f"rule {index}", encoding="utf-8")
        candidate.bundle_hash = hash_text(f"rule {index}")
        store.write_candidate(candidate)
    first = store.list_candidates(limit=2)
    second = store.list_candidates(limit=2, before=first[-1].candidate_id)
    assert [item.candidate_id for item in first] == ["candidate-3", "candidate-2"]
    assert [item.candidate_id for item in second] == ["candidate-1", "candidate-0"]
    observations = store.list_observations(limit=2)
    assert [item.observation_id for item in observations] == ["obs-3", "obs-2"]


def test_candidate_gate_validates_final_registry_and_duplicate_existing_target(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = SelfEvolutionStore(workspace, project_id="demo")
    existing = store.self_develop_skills_dir / "materials_worker/other-directory"
    _write_skill(
        existing.parent,
        group="",
        name="other-directory",
        marker="existing",
    )
    text = (existing / "SKILL.md").read_text(encoding="utf-8")
    (existing / "SKILL.md").write_text(
        text.replace("name: other-directory", "name: duplicate-name"),
        encoding="utf-8",
    )
    candidate = LearningCandidate(
        candidate_id="candidate-duplicate",
        project_id="demo",
        run_id="run",
        thread_id="thread",
        action="skill",
        group="materials_worker",
        name="duplicate-name",
    )
    root = store.reset_candidate_dir(candidate.candidate_id)
    proposed = _write_skill(
        root / "proposed",
        group="materials_worker",
        name="duplicate-name",
        marker="candidate",
    )
    candidate.bundle_hash = hash_tree(proposed)
    report = CandidateGate(store).run(candidate)
    assert report.valid is False
    assert any("already exists in directory" in item for item in report.errors)


def test_candidate_gate_rejects_empty_scaffold_and_wrong_audience_tool(
    tmp_path: Path,
) -> None:
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    scaffold = LearningCandidate(
        candidate_id="candidate-scaffold",
        project_id="demo",
        run_id="run",
        thread_id="thread",
        action="skill",
        group="materials_worker",
        name="empty-scaffold",
    )
    scaffold_root = store.reset_candidate_dir(scaffold.candidate_id)
    skill_root = (
        scaffold_root
        / "proposed"
        / scaffold.group
        / scaffold.name
    )
    skill_root.mkdir(parents=True)
    (skill_root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {scaffold.name}",
                "description: Replace this with what the skill is for and when to use it.",
                "---",
                f"# {scaffold.name}",
                "",
                "## Overview",
                "",
                "## Quick Start",
                "",
                "## Workflow",
                "",
                "## Method-critical defaults",
                "",
                "## Output Contract",
                "",
                "## References",
                "",
            ]
        ),
        encoding="utf-8",
    )
    scaffold.bundle_hash = hash_tree(skill_root)
    scaffold_report = CandidateGate(
        store,
        allowed_tool_names=set(),
    ).run(scaffold)
    assert scaffold_report.valid is False
    assert any("placeholder" in item for item in scaffold_report.errors)
    assert any("substantive content" in item for item in scaffold_report.errors)

    wrong_audience = LearningCandidate(
        candidate_id="candidate-wrong-audience",
        project_id="demo",
        run_id="run",
        thread_id="thread",
        action="skill",
        group="writing_specialist",
        name="vasp-writing",
    )
    wrong_root = store.reset_candidate_dir(wrong_audience.candidate_id)
    proposed = _write_skill(
        wrong_root / "proposed",
        group=wrong_audience.group,
        name=wrong_audience.name,
        marker="bounded manuscript editing",
    )
    skill_path = proposed / "SKILL.md"
    skill_path.write_text(
        skill_path.read_text(encoding="utf-8").replace(
            "compatibility: local",
            "compatibility: local\nallowed-tools: vasp_prepare",
        ),
        encoding="utf-8",
    )
    wrong_audience.bundle_hash = hash_tree(proposed)
    wrong_report = CandidateGate(store).run(wrong_audience)
    assert wrong_report.valid is False
    assert any(
        "final writing_specialist specialist/worker surface" in item
        and "vasp_prepare" in item
        for item in wrong_report.errors
    )


def test_exact_target_consolidation_does_not_group_by_wording(tmp_path: Path) -> None:
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    same_target_a = store.write_observation(
        Observation(
            observation_id="obs-surface-a",
            run_id="run-a",
            thread_id="thread-a",
            signal_kind="skill_revision",
            target="materials_worker/surface-repair",
            claim="Require the valid surface index after a selection failure.",
            evidence_refs=[{"source_ref": "run:run-a"}],
            created_at="2026-07-27T00:00:00+00:00",
        )
    )
    same_target_b = store.write_observation(
        Observation(
            observation_id="obs-surface-b",
            run_id="run-b",
            thread_id="thread-b",
            signal_kind="skill_revision",
            target="materials_worker/surface-repair",
            claim="表面选择报错后，只修正这一次的终止面编号。",
            evidence_refs=[{"source_ref": "run:run-b"}],
            created_at="2026-07-28T00:00:00+00:00",
        )
    )
    same_words_other_target = store.write_observation(
        Observation(
            observation_id="obs-other-target",
            run_id="run-c",
            thread_id="thread-c",
            signal_kind="skill_revision",
            target="materials_worker/adsorbate-placement",
            claim=same_target_a.claim,
            evidence_refs=[{"source_ref": "run:run-c"}],
            created_at="2026-07-29T00:00:00+00:00",
        )
    )

    batch = ConsolidationService(store).batch_for(same_target_b)

    assert batch.target == "materials_worker/surface-repair"
    assert batch.evidence_ids == (
        same_target_a.observation_id,
        same_target_b.observation_id,
    )
    assert same_words_other_target.observation_id not in batch.evidence_ids


def test_consolidation_exposes_no_similarity_or_recurrence_decision_api() -> None:
    public_methods = {
        name
        for name in dir(ConsolidationService)
        if not name.startswith("_")
        and callable(getattr(ConsolidationService, name))
    }

    assert public_methods == {"batch_for", "evidence_markdown"}
    assert not {
        "decide",
        "eligible",
        "cluster_for",
        "similarity",
        "route",
    } & public_methods


def test_new_same_target_evidence_creates_immutable_revision_without_fixed_gate(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    repo = _repo(tmp_path)
    _write_skill(
        repo / "skills",
        group="materials_worker",
        name="surface-repair",
        marker="current surface repair workflow",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=_SkillProposer(),
        reviewer=_Reviewer(),
    )

    first_run = _run_dir(
        workspace,
        "run-first",
        prompt="First complete episode for the surface-repair target.",
    )
    coordinator.enqueue_post_run(
        run_id="run-first",
        thread_id="thread-a",
        terminal_status="done",
        run_dir=first_run,
    )
    first_job = coordinator.process_pending_jobs()[0]
    first = coordinator.store.read_candidate(first_job.candidate_id)
    assert first is not None and first.revision == 1
    revision_one = coordinator.store.revision_dir(first.candidate_id, 1)
    revision_one_before = {
        path.relative_to(revision_one).as_posix(): path.read_bytes()
        for path in revision_one.rglob("*")
        if path.is_file()
    }

    second_run = _run_dir(
        workspace,
        "run-second",
        prompt="A differently worded second episode maps to the same exact target.",
    )
    coordinator.enqueue_post_run(
        run_id="run-second",
        thread_id="thread-b",
        terminal_status="done",
        run_dir=second_run,
    )
    second_job = coordinator.process_pending_jobs()[0]
    revised = coordinator.store.read_candidate(first.candidate_id)

    assert second_job.candidate_id == first.candidate_id
    assert revised is not None and revised.revision == 2
    assert len(revised.evidence_ids) == 2
    proposal = json.loads(
        (
            coordinator.store.revision_dir(revised.candidate_id, 2)
            / "proposal.json"
        ).read_text(encoding="utf-8")
    )
    assert proposal["evidence_ids"] == revised.evidence_ids
    assert "supporting_evidence_ids" not in proposal
    assert "counterexample_ids" not in proposal
    revision_one_after = {
        path.relative_to(revision_one).as_posix(): path.read_bytes()
        for path in revision_one.rglob("*")
        if path.is_file()
    }
    assert revision_one_after == revision_one_before

    duplicate = coordinator.enqueue_post_run(
        run_id="run-second",
        thread_id="thread-b",
        terminal_status="done",
        run_dir=second_run,
    )
    assert duplicate.status == "done"
    assert coordinator.process_pending_jobs() == []
    assert coordinator.store.read_candidate(first.candidate_id).revision == 2


def test_later_revision_restores_each_explicit_correction_from_its_job(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(
        workspace,
        "run-shared",
        prompt="Prepare the workspace report.",
    )
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=_MemoryProposer(),
        reviewer=_Reviewer(),
    )
    first_note = "Use Chinese for generated workspace reports."
    second_note = "Keep verbatim source quotations in their original language."
    for note in (first_note, second_note):
        coordinator.enqueue_explicit_learn(
            run_id="run-shared",
            run_dir=run_dir,
            thread_id="thread-a",
            note=note,
        )
        coordinator.process_pending_jobs()

    candidate = coordinator.store.list_candidates()[0]
    evidence = (
        coordinator.store.revision_dir(candidate.candidate_id, 2)
        / "evidence.md"
    ).read_text(encoding="utf-8")

    assert first_note in evidence
    assert second_note in evidence
    assert evidence.count("## Explicit durable correction") == 2


def test_legacy_running_job_without_lease_moves_to_recovery_review(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    root = workspace / "metadata" / "self_evolution"
    root.mkdir(parents=True, exist_ok=True)
    database = root / "jobs.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE jobs (
                job_id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                run_dir TEXT NOT NULL,
                thread_id TEXT NOT NULL DEFAULT '',
                trigger_kind TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                candidate_id TEXT NOT NULL DEFAULT '',
                model_config TEXT NOT NULL DEFAULT '',
                payload_json TEXT NOT NULL DEFAULT '{}',
                error TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO jobs(
                job_id, project_id, run_id, run_dir, trigger_kind, status,
                created_at, updated_at
            ) VALUES (
                'legacy-running', 'demo', 'run-old', '/tmp/run-old',
                'post_run', 'running', '2026-07-20T00:00:00+00:00',
                '2026-07-20T00:00:00+00:00'
            )
            """
        )
    store = SelfEvolutionStore(workspace, project_id="demo")
    recovered = store.list_jobs()[0]
    assert recovered.status == "recovery_review"
    assert recovered.owner == ""
    assert recovered.lease_until == ""
    assert "no verifiable lease" in recovered.error
    assert store.claim_jobs(owner="worker-a") == []


def test_candidate_cursor_pagination_promotes_updated_old_candidate(
    tmp_path: Path,
) -> None:
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    for index in range(30):
        candidate = LearningCandidate(
            candidate_id=f"candidate-{index:02d}",
            project_id="demo",
            run_id=f"run-{index:02d}",
            thread_id="thread",
            action="memory",
            route="workspace_preference",
            created_at=f"2020-01-{index + 1:02d}T00:00:00+00:00",
        )
        root = store.reset_candidate_dir(candidate.candidate_id)
        (root / "memories").mkdir()
        text = f"rule {index}"
        (root / "memories/AGENTS.md").write_text(text, encoding="utf-8")
        candidate.bundle_hash = hash_text(text)
        store.write_candidate(candidate)
    with sqlite3.connect(store.db_path) as connection:
        for index in range(30):
            connection.execute(
                "UPDATE candidates SET updated_at = ? WHERE candidate_id = ?",
                (
                    f"2020-01-{index + 1:02d}T00:00:00+00:00",
                    f"candidate-{index:02d}",
                ),
            )
    store.update_candidate_status("candidate-00", "review")

    observed: list[str] = []
    before = ""
    while True:
        page = store.list_candidates(limit=11, before=before)
        if not page:
            break
        observed.extend(item.candidate_id for item in page)
        before = page[-1].candidate_id
    assert len(observed) == 30
    assert len(set(observed)) == 30
    assert observed[0] == "candidate-00"
