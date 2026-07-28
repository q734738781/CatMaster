from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage

from catmaster.runtime.self_evolution import (
    CandidateGate,
    LearningCandidate,
    PromotionConflict,
    PromotionManager,
    ProposerResult,
    ReviewerResult,
    SelfEvolutionCoordinator,
    SelfEvolutionStore,
)
from catmaster.runtime.self_evolution.agents import (
    _prepare_skill_tool,
    _recover_proposer_result_from_files,
    _recover_reviewer_result_from_text,
    prepare_candidate_workspace,
)
from catmaster.runtime.self_evolution.storage import hash_text, hash_tree, utc_now
from catmaster.runtime.self_evolution.settings import resolve_self_evolution_mode
from catmaster.runtime.self_evolution.trace import collect_turn_trace
from catmaster.runtime.observability_store import ObservabilityStore
from catmaster.specialists.runtime import SpecialistRunner, build_specialist_runner
from catmaster.tools.base import ensure_project_space_layout


def _skill_text(name: str, marker: str) -> str:
    return f"""---
name: {name}
description: Use this skill for durable test workflows and bundle validation.
license: project-local
compatibility: local
---
# {name}

## Overview

{marker}

## Quick Start

Read the helper and run the bounded workflow.

## Workflow

1. Inspect inputs.
2. Run the helper.

## Method-critical defaults

Keep the test marker explicit.

## Output Contract

Return the marker and generated artifact path.

## References

[Helper](scripts/helper.py)
"""


def _write_skill(root: Path, *, group: str, name: str, marker: str) -> Path:
    skill = root / group / name
    (skill / "scripts").mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(_skill_text(name, marker), encoding="utf-8")
    (skill / "scripts" / "helper.py").write_text(f"MARKER = {marker!r}\n", encoding="utf-8")
    return skill


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "skills").mkdir(parents=True)
    (repo / "skills" / "AGENTS.MD").write_text("Skill authoring rules.", encoding="utf-8")
    return repo


def _run_dir(workspace: Path, *, run_id: str = "run-one", prompt: str = "Please handle this task.") -> Path:
    run_dir = workspace / "metadata" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "thread_id": "thread-one",
                "entrypoint": "experiment",
                "status": "done",
                "user_prompt": prompt,
                "final_answer": "Completed the requested task.",
                "summary": "done",
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_self_evolution_defaults_to_human_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CATMASTER_SELF_EVOLUTION_MODE", raising=False)

    assert resolve_self_evolution_mode() == "observe"
    assert resolve_self_evolution_mode("auto") == "auto"


def test_default_human_approval_keeps_reviewed_candidate_unpromoted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CATMASTER_SELF_EVOLUTION_MODE", raising=False)
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=_BundleProposer(),
        reviewer=_Reviewer("approve"),
    )
    coordinator.enqueue_post_run(
        run_id="run-one",
        thread_id="thread-one",
        terminal_status="done",
        run_dir=_run_dir(workspace, prompt="Correct this reusable workflow."),
    )

    job = coordinator.process_pending_jobs()[0]
    candidate = coordinator.store.read_candidate(job.candidate_id)

    assert candidate is not None and candidate.status == "reviewed"
    assert not (coordinator.store.self_develop_skills_dir / "materials_worker" / "demo-workflow").exists()
    assert (
        coordinator.store.candidate_dir(candidate.candidate_id)
        / "proposed"
        / "materials_worker"
        / "demo-workflow"
        / "SKILL.md"
    ).is_file()
    assert (coordinator.store.candidate_dir(candidate.candidate_id) / "current" / "catalog.md").is_file()


def test_job_claims_use_the_workspace_id_recorded_at_enqueue_time(tmp_path: Path) -> None:
    workspace = tmp_path / "users" / "alice" / "project-one"
    ensure_project_space_layout(workspace, create=True)
    short_store = SelfEvolutionStore(workspace, project_id="project-one")
    global_store = SelfEvolutionStore(workspace, project_id="users/alice/project-one")
    short_store.enqueue_job(trigger_kind="post_run", run_id="run-short", run_dir=_run_dir(workspace, run_id="run-short"))
    global_store.enqueue_job(trigger_kind="post_run", run_id="run-global", run_dir=_run_dir(workspace, run_id="run-global"))

    assert short_store.queued_project_ids() == ["project-one", "users/alice/project-one"]
    claimed = short_store.claim_jobs(limit=4, project_id="project-one")

    assert [job.project_id for job in claimed] == ["project-one"]
    assert [job.project_id for job in global_store.claim_jobs(limit=4, project_id="users/alice/project-one")] == [
        "users/alice/project-one"
    ]


def test_trace_projection_keeps_full_tool_sequence_without_raw_llm_payloads(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    run_dir = _run_dir(workspace)
    store = ObservabilityStore(run_dir)

    store.record_raw_callback(
        "TOOL_RAW_INPUT",
        category="tool",
        payload={
            "callback_run_id": "tool-early",
            "agent_name": "materials_worker",
            "tool": "build_slab",
            "params_compact": '{"miller_index": [1, 0, 0]}',
        },
    )
    store.record_raw_callback(
        "TOOL_CALL_END",
        category="tool",
        payload={
            "callback_run_id": "tool-early",
            "agent_name": "materials_worker",
            "tool": "build_slab",
            "status": "success",
            "projection": {"content_preview": "created slab"},
        },
    )
    for index in range(600):
        store.record_raw_callback(
            "LLM_RAW_RESPONSE",
            category="llm",
            payload={"callback_run_id": f"raw-{index}", "raw_response": "x" * 2_000},
        )
    store.record_raw_callback(
        "TOOL_RAW_INPUT",
        category="tool",
        payload={
            "callback_run_id": "tool-late",
            "agent_name": "materials_worker",
            "tool": "build_slab",
            "params_compact": '{"miller_index": [0, 0, 1]}',
        },
    )
    store.record_raw_callback(
        "TOOL_CALL_END",
        category="tool",
        payload={
            "callback_run_id": "tool-late",
            "agent_name": "materials_worker",
            "tool": "build_slab",
            "status": "error",
            "error": "termination selection failed",
            "projection": {"content_preview": "termination selection failed"},
        },
    )

    trace = collect_turn_trace(run_dir=run_dir)
    markdown = trace.to_markdown()
    sequence = next(event for event in trace.events if event["name"] == "tool_sequence")
    errors = [event for event in trace.events if event["name"] == "tool_error"]

    assert sequence["payload"]["calls"] == [
        "1: materials_worker/build_slab [success]",
        "2: materials_worker/build_slab [error]",
    ]
    assert len(errors) == 1
    assert "termination selection failed" in errors[0]["payload"]["result"]
    assert "LLM_RAW_RESPONSE" not in markdown
    assert len(markdown) < 50_000


class _IgnoreProposer:
    def propose(self, *, candidate_root: Path):
        assert (candidate_root / "trace.md").is_file()
        return ProposerResult(action="ignore", rationale="One-turn objective only."), {"model_label": "fake-proposer"}


class _BundleProposer:
    def __init__(self, *, marker: str = "workspace version") -> None:
        self.marker = marker

    def propose(self, *, candidate_root: Path):
        _write_skill(
            candidate_root / "proposed",
            group="materials_worker",
            name="demo-workflow",
            marker=self.marker,
        )
        return (
            ProposerResult(
                action="skill",
                group="materials_worker",
                name="demo-workflow",
                rationale="The trace explicitly corrects a reusable workflow.",
            ),
            {"model_label": "fake-proposer"},
        )


class _MemoryFileProposer:
    def propose(self, *, candidate_root: Path):
        path = candidate_root / "memories" / "AGENTS.md"
        current = path.read_text(encoding="utf-8")
        path.write_text(current.replace("Prefer English", "Prefer Chinese"), encoding="utf-8")
        return ProposerResult(action="memory", rationale="The user changed a durable preference."), {
            "model_label": "fake-proposer"
        }


class _Reviewer:
    def __init__(self, decision: str = "approve") -> None:
        self.decision = decision

    def review(self, *, candidate_root: Path, **kwargs):
        _ = kwargs
        assert (candidate_root / "trace.md").is_file()
        if kwargs["action"] == "skill":
            assert (candidate_root / "proposed" / "materials_worker" / "demo-workflow" / "scripts" / "helper.py").is_file()
        if kwargs["action"] == "memory":
            assert (candidate_root / "memories" / "AGENTS.md").is_file()
            assert (candidate_root / "current" / "AGENTS.md").is_file()
        return ReviewerResult(
            decision=self.decision,
            summary="Correct the reusable workflow without broadening its activation.",
            change_points=[
                {
                    "title": "Correct helper behavior",
                    "before": "The workspace used the built-in helper behavior.",
                    "after": "The workspace uses the corrected helper behavior.",
                    "evidence": "The user explicitly corrected the reusable workflow.",
                    "evidence_source": "user",
                    "impact": "Future matching runs use the corrected helper.",
                }
            ],
            scope_assessment="The change stays within the existing materials workflow.",
            proportionality_assessment={"status": "pass", "explanation": "No extra audit work is added."},
            concerns=[],
            human_checks=["Confirm the corrected helper matches the intended workflow."],
            rationale=f"Independent {self.decision}.",
        ), {"model_label": "fake-reviewer"}


class _FailingReviewer:
    def review(self, **kwargs):
        _ = kwargs
        raise RuntimeError("review provider unavailable")


def test_native_deepagents_later_source_overrides_same_name(tmp_path: Path) -> None:
    root = tmp_path / "files"
    built_in = _write_skill(root / "built_in", group="materials_worker", name="demo", marker="built-in")
    override = _write_skill(root / "override", group="materials_worker", name="demo", marker="workspace")
    middleware = SkillsMiddleware(
        backend=FilesystemBackend(root_dir=root, virtual_mode=True),
        sources=["/built_in/materials_worker", "/override/materials_worker"],
    )

    update = middleware.before_agent({}, None, {})

    assert update is not None
    skills = list(update["skills_metadata"])
    assert len(skills) == 1
    assert skills[0]["name"] == "demo"
    assert skills[0]["path"] == "/override/materials_worker/demo/SKILL.md"
    assert built_in.is_dir() and override.is_dir()


def test_skill_source_order_keeps_all_workspace_layers_last() -> None:
    assert SpecialistRunner._skill_roots_for_groups("materials_worker", "execution") == [
        "/.deepagents/skills/materials_worker",
        "/.deepagents/skills/execution",
        "/.deepagents/self_develop_skills/materials_worker",
        "/.deepagents/self_develop_skills/execution",
    ]


def test_same_user_thread_next_run_reloads_new_workspace_override(tmp_path: Path) -> None:
    class _Profile:
        def config_for_role(self, role: str):
            return SimpleNamespace(model=f"{role}-model", provider="langchain", base_url=None)

    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    first = build_specialist_runner(
        workspace=workspace,
        llm_profile=_Profile(),
        reporter=None,
        run_control=None,
        project_id="demo",
        preferred_entrypoint="experiment",
    )
    first.runner._stage_deepagent_assets(workspace / "files")
    first_checkpoint = first.runner._deepagent_checkpoint_thread_id("thread-one")
    assert first_checkpoint == first.runner._deepagent_checkpoint_thread_id("thread-one")

    override = workspace / "metadata" / "self_evolution" / "self_develop_skills" / "materials_worker"
    _write_skill(override, group="", name="slab-construction-and-surface-modeling", marker="workspace override")
    second = build_specialist_runner(
        workspace=workspace,
        llm_profile=_Profile(),
        reporter=None,
        run_control=None,
        project_id="demo",
        preferred_entrypoint="experiment",
    )
    second.runner._stage_deepagent_assets(workspace / "files")
    second_checkpoint = second.runner._deepagent_checkpoint_thread_id("thread-one")
    middleware = SkillsMiddleware(
        backend=FilesystemBackend(root_dir=workspace / "files", virtual_mode=True),
        sources=second.runner._skill_roots_for_group("materials_worker"),
    )

    update = middleware.before_agent({}, None, {})

    assert first_checkpoint != second_checkpoint
    skill = next(
        item
        for item in update["skills_metadata"]
        if item["name"] == "slab-construction-and-surface-modeling"
    )
    assert skill["path"].startswith("/.deepagents/self_develop_skills/materials_worker/")


def test_workspace_skill_override_is_shared_across_threads_but_isolated_between_workspaces(tmp_path: Path) -> None:
    class _Profile:
        def config_for_role(self, role: str):
            return SimpleNamespace(model=f"{role}-model", provider="langchain", base_url=None)

    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    for workspace in (alpha, beta):
        ensure_project_space_layout(workspace, create=True)
    override = alpha / "metadata" / "self_evolution" / "self_develop_skills" / "materials_worker"
    _write_skill(override, group="", name="slab-construction-and-surface-modeling", marker="alpha workspace override")

    alpha_runner = build_specialist_runner(
        workspace=alpha,
        llm_profile=_Profile(),
        reporter=None,
        run_control=None,
        project_id="alpha",
        preferred_entrypoint="experiment",
    ).runner
    beta_runner = build_specialist_runner(
        workspace=beta,
        llm_profile=_Profile(),
        reporter=None,
        run_control=None,
        project_id="beta",
        preferred_entrypoint="experiment",
    ).runner
    alpha_runner._stage_deepagent_assets(alpha / "files")
    beta_runner._stage_deepagent_assets(beta / "files")

    def _skill_path(runner: SpecialistRunner, workspace: Path) -> str:
        middleware = SkillsMiddleware(
            backend=FilesystemBackend(root_dir=workspace / "files", virtual_mode=True),
            sources=runner._skill_roots_for_group("materials_worker"),
        )
        update = middleware.before_agent({}, None, {})
        return next(
            item["path"]
            for item in update["skills_metadata"]
            if item["name"] == "slab-construction-and-surface-modeling"
        )

    assert alpha_runner._deepagent_checkpoint_thread_id("thread-a") != alpha_runner._deepagent_checkpoint_thread_id("thread-b")
    assert _skill_path(alpha_runner, alpha).startswith("/.deepagents/self_develop_skills/materials_worker/")
    assert _skill_path(beta_runner, beta).startswith("/.deepagents/skills/materials_worker/")


def test_prepare_skill_tool_copies_complete_effective_bundle(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    store = SelfEvolutionStore(workspace, project_id="demo")
    repo = _repo(tmp_path)
    _write_skill(repo / "skills", group="materials_worker", name="demo-workflow", marker="built-in")
    trace = collect_turn_trace(run_dir=_run_dir(workspace))
    root = prepare_candidate_workspace(
        store=store,
        candidate_id="sec_prepare",
        trace=trace,
        repo_root=repo,
    )

    result = _prepare_skill_tool(
        root,
        repo_root=repo,
        self_develop_root=store.self_develop_skills_dir,
    ).invoke({"group": "materials_worker", "name": "demo-workflow"})

    assert "complete current built_in bundle" in str(result)
    assert (root / "proposed" / "materials_worker" / "demo-workflow" / "scripts" / "helper.py").is_file()


def test_candidate_workspace_exposes_an_isolated_complete_memory_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    store = SelfEvolutionStore(workspace, project_id="demo")
    initial = "# Persistent Instruction Memory\n\n- Prefer English reports.\n"
    swapped, _ = store.compare_and_swap_memory(expected_hash=store.memory_hash(), new_text=initial)
    assert swapped

    root = prepare_candidate_workspace(
        store=store,
        candidate_id="sec_memory_prepare",
        trace=collect_turn_trace(run_dir=_run_dir(workspace)),
        repo_root=_repo(tmp_path),
    )

    assert (root / "current" / "AGENTS.md").read_text(encoding="utf-8") == initial
    assert (root / "memories" / "AGENTS.md").read_text(encoding="utf-8") == initial
    assert store.read_memory_text() == initial


def test_completed_memory_file_recovers_when_control_output_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    (root / "current").mkdir(parents=True)
    (root / "memories").mkdir(parents=True)
    (root / "proposed").mkdir(parents=True)
    (root / "current" / "AGENTS.md").write_text("- Prefer English.\n", encoding="utf-8")
    (root / "memories" / "AGENTS.md").write_text("- Prefer Chinese.\n", encoding="utf-8")

    result = _recover_proposer_result_from_files(result={"messages": []}, candidate_root=root)

    assert result.action == "memory"


def test_completed_single_bundle_recovers_when_control_output_is_missing(tmp_path: Path) -> None:
    root = tmp_path / "candidate"
    _write_skill(root / "proposed", group="materials_worker", name="demo-workflow", marker="workspace")

    result = _recover_proposer_result_from_files(result={"messages": []}, candidate_root=root)

    assert result.action == "skill"
    assert result.group == "materials_worker"
    assert result.name == "demo-workflow"


def test_gate_rejects_duplicate_frontmatter_name_within_workspace_layer(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    store = SelfEvolutionStore(workspace, project_id="demo")
    existing = store.self_develop_skills_dir / "materials_worker" / "other-directory"
    (existing / "scripts").mkdir(parents=True)
    (existing / "SKILL.md").write_text(_skill_text("demo-workflow", "existing"), encoding="utf-8")
    (existing / "scripts" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    candidate = LearningCandidate(
        candidate_id="sec_duplicate",
        project_id="demo",
        run_id="run-one",
        thread_id="thread-one",
        action="skill",
        group="materials_worker",
        name="demo-workflow",
    )
    root = store.reset_candidate_dir(candidate.candidate_id)
    proposed = _write_skill(root / "proposed", group="materials_worker", name="demo-workflow", marker="candidate")
    candidate.bundle_hash = hash_tree(proposed)

    report = CandidateGate(store).run(candidate)

    assert not report.valid
    assert any("already exists in directory" in error for error in report.errors)


def test_reviewer_control_fallback_requires_one_exact_decision_line() -> None:
    approved = _recover_reviewer_result_from_text(
        {"messages": [AIMessage(content="The reviewed bundle is grounded.\nDECISION: APPROVE")]}
    )
    ambiguous = _recover_reviewer_result_from_text(
        {"messages": [AIMessage(content="I might approve this after another change.")]}
    )

    assert approved.recommendation == "approve"
    assert approved.change_points == []
    assert approved.concerns
    assert ambiguous.recommendation == "reject"


def test_reviewer_result_schema_is_non_nullable_and_accepts_legacy_nulls() -> None:
    strategy = ToolStrategy(ReviewerResult)
    schema_text = json.dumps(strategy.schema_specs[0].json_schema, sort_keys=True)

    assert '"type": "null"' not in schema_text
    assert '"recommendation"' in schema_text
    assert '"change_points"' in schema_text
    assert '"proportionality_assessment"' in schema_text

    result = ReviewerResult.model_validate(
        {
            "decision": "approve",
            "summary": None,
            "change_points": None,
            "scope_assessment": None,
            "proportionality_assessment": None,
            "concerns": None,
            "human_checks": None,
            "rationale": None,
        }
    )

    assert result.recommendation == "approve"
    assert result.summary == ""
    assert result.change_points == []
    assert result.concerns == []
    assert result.proportionality_assessment.status == "warning"
    assert "decision" not in result.model_dump()


def test_pipeline_ignore_creates_no_candidate_or_workspace_write(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    repo = _repo(tmp_path)
    run_dir = _run_dir(workspace, prompt="For this answer only, use a short table.")
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=_IgnoreProposer(),
        reviewer=_Reviewer(),
        mode="auto",
    )
    first = coordinator.enqueue_post_run(
        run_id="run-one",
        thread_id="thread-one",
        terminal_status="done",
        run_dir=run_dir,
    )
    second = coordinator.enqueue_post_run(
        run_id="run-one",
        thread_id="thread-one",
        terminal_status="done",
        run_dir=run_dir,
    )

    assert first is not None and second is not None and first.job_id == second.job_id
    processed = coordinator.process_pending_jobs()
    assert processed[0].status == "done"
    assert coordinator.store.list_candidates() == []
    assert not any(coordinator.store.self_develop_skills_dir.rglob("SKILL.md"))


def test_worker_restart_requeues_only_running_jobs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    store = SelfEvolutionStore(workspace, project_id="demo")
    run_dir = _run_dir(workspace)
    store.enqueue_job(trigger_kind="post_run", run_id="run-one", run_dir=run_dir)
    claimed = store.claim_jobs(limit=1)
    assert claimed[0].status == "running"

    assert store.requeue_running_jobs() == 1
    assert store.list_jobs()[0].status == "queued"


def test_pipeline_promotes_complete_memory_file_edit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    store = SelfEvolutionStore(workspace, project_id="demo")
    initial = "# Persistent Instruction Memory\n\n- Prefer English reports.\n- Preserve this unrelated fact.\n"
    swapped, _ = store.compare_and_swap_memory(expected_hash=store.memory_hash(), new_text=initial)
    assert swapped
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=_MemoryFileProposer(),
        reviewer=_Reviewer("approve"),
        mode="auto",
    )
    coordinator.enqueue_post_run(
        run_id="run-one",
        thread_id="thread-one",
        terminal_status="done",
        run_dir=_run_dir(workspace, prompt="Use Chinese for all future workspace reports."),
    )

    job = coordinator.process_pending_jobs()[0]
    candidate = coordinator.store.read_candidate(job.candidate_id)

    assert job.status == "done"
    assert candidate is not None and candidate.status == "promoted"
    assert "Prefer Chinese reports" in store.read_memory_text()
    assert "Prefer English reports" not in store.read_memory_text()
    assert "Preserve this unrelated fact" in store.read_memory_text()


def test_pipeline_never_auto_promotes_skill_bundle_and_reviewer_reject_is_noop(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    repo = _repo(tmp_path)
    _write_skill(repo / "skills", group="materials_worker", name="demo-workflow", marker="built-in")
    run_dir = _run_dir(workspace, prompt="The previous workflow was wrong; this reusable method needs a corrected helper.")
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=_BundleProposer(),
        reviewer=_Reviewer("approve"),
        mode="auto",
    )
    coordinator.enqueue_post_run(
        run_id="run-one",
        thread_id="thread-one",
        terminal_status="done",
        run_dir=run_dir,
    )

    processed = coordinator.process_pending_jobs()

    candidate = coordinator.store.read_candidate(processed[0].candidate_id)
    assert candidate is not None and candidate.status == "reviewed"
    target = coordinator.store.self_develop_skills_dir / "materials_worker" / "demo-workflow"
    assert not target.exists()
    reviewed = coordinator.store.candidate_dir(candidate.candidate_id) / "proposed" / "materials_worker" / "demo-workflow"
    assert "workspace version" in (reviewed / "SKILL.md").read_text(encoding="utf-8")
    assert (reviewed / "scripts" / "helper.py").is_file()
    assert hash_tree(reviewed) == candidate.bundle_hash
    persisted_review = json.loads(
        (coordinator.store.candidate_dir(candidate.candidate_id) / "review.json").read_text(encoding="utf-8")
    )
    assert persisted_review["recommendation"] == "approve"
    assert persisted_review["summary"]
    assert persisted_review["change_points"][0]["evidence_source"] == "user"
    audit_events = [
        json.loads(line) for line in coordinator.store.audit_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["event"] == "reviewer_recommendation" for event in audit_events)
    assert not any(event["event"] == "human_decision" for event in audit_events)

    rejected_workspace = tmp_path / "rejected"
    ensure_project_space_layout(rejected_workspace, create=True)
    rejected_run = _run_dir(rejected_workspace, run_id="run-two")
    rejected = SelfEvolutionCoordinator(
        workspace=rejected_workspace,
        project_id="rejected",
        repo_root=repo,
        proposer=_BundleProposer(marker="must not promote"),
        reviewer=_Reviewer("reject"),
        mode="auto",
    )
    rejected.enqueue_post_run(
        run_id="run-two",
        thread_id="thread-two",
        terminal_status="done",
        run_dir=rejected_run,
    )
    rejected_job = rejected.process_pending_jobs()[0]
    rejected_candidate = rejected.store.read_candidate(rejected_job.candidate_id)
    assert rejected_candidate is not None and rejected_candidate.status == "rejected"
    assert not (rejected.store.self_develop_skills_dir / "materials_worker" / "demo-workflow").exists()


def test_needs_revision_remains_visible_for_human_decision(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=_repo(tmp_path),
        proposer=_BundleProposer(),
        reviewer=_Reviewer("needs_revision"),
        mode="auto",
    )
    coordinator.enqueue_post_run(
        run_id="run-one",
        thread_id="thread-one",
        terminal_status="done",
        run_dir=_run_dir(workspace, prompt="Narrow this reusable workflow correction."),
    )

    job = coordinator.process_pending_jobs()[0]
    candidate = coordinator.store.read_candidate(job.candidate_id)

    assert candidate is not None and candidate.status == "reviewed"
    assert candidate.review["recommendation"] == "needs_revision"
    assert not any(coordinator.store.self_develop_skills_dir.rglob("SKILL.md"))


def test_reviewer_failure_keeps_candidate_linked_without_promoting(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    repo = _repo(tmp_path)
    run_dir = _run_dir(workspace)
    coordinator = SelfEvolutionCoordinator(
        workspace=workspace,
        project_id="demo",
        repo_root=repo,
        proposer=_BundleProposer(),
        reviewer=_FailingReviewer(),
        mode="auto",
    )
    coordinator.enqueue_post_run(
        run_id="run-one",
        thread_id="thread-one",
        terminal_status="done",
        run_dir=run_dir,
    )

    job = coordinator.process_pending_jobs()[0]
    candidate = coordinator.store.read_candidate(job.candidate_id)

    assert job.status == "error"
    assert candidate is not None and candidate.status == "proposed"
    assert candidate.review["recommendation"] == "unavailable"
    assert not any(coordinator.store.self_develop_skills_dir.rglob("SKILL.md"))


def test_memory_file_promotion_replaces_and_rollback_restores(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    store = SelfEvolutionStore(workspace, project_id="demo")
    before = "# Persistent Instruction Memory\n\n- Prefer English reports.\n- Keep citations concise.\n"
    after = "# Persistent Instruction Memory\n\n- Prefer Chinese reports.\n- Keep citations concise.\n"
    swapped, _ = store.compare_and_swap_memory(expected_hash=store.memory_hash(), new_text=before)
    assert swapped
    candidate = LearningCandidate(
        candidate_id="sec_memory",
        project_id="demo",
        run_id="run-one",
        thread_id="thread-one",
        action="memory",
        status="approved",
        rationale="Stable preference.",
        base_target_hash=hash_text(before),
        bundle_hash=hash_text(after),
        created_at=utc_now(),
    )
    root = store.reset_candidate_dir(candidate.candidate_id)
    (root / "memories").mkdir(parents=True)
    (root / "memories" / "AGENTS.md").write_text(after, encoding="utf-8")
    store.write_candidate(candidate)
    report = CandidateGate(store).run(candidate)
    assert report.valid

    manager = PromotionManager(store, repo_root=_repo(tmp_path))
    promoted = manager.promote(candidate, report, decision_source="human", actor="alice", rationale="Reviewed exact diff.")
    assert store.read_memory_text() == after
    assert "Prefer English" not in store.read_memory_text()

    manager.rollback(promoted)
    assert store.read_memory_text() == before


def test_memory_file_promotion_rejects_concurrent_parent_edit(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    store = SelfEvolutionStore(workspace, project_id="demo")
    before = "# Memory\n\n- Prefer English.\n"
    proposed = "# Memory\n\n- Prefer Chinese.\n"
    parent_edit = before + "- Parent added an independent fact.\n"
    swapped, _ = store.compare_and_swap_memory(expected_hash=store.memory_hash(), new_text=before)
    assert swapped
    candidate = LearningCandidate(
        candidate_id="sec_memory_conflict",
        project_id="demo",
        run_id="run-one",
        thread_id="thread-one",
        action="memory",
        status="approved",
        base_target_hash=hash_text(before),
        bundle_hash=hash_text(proposed),
        created_at=utc_now(),
    )
    root = store.reset_candidate_dir(candidate.candidate_id)
    (root / "memories").mkdir(parents=True)
    (root / "memories" / "AGENTS.md").write_text(proposed, encoding="utf-8")
    store.write_candidate(candidate)
    report = CandidateGate(store).run(candidate)
    assert report.valid
    swapped, _ = store.compare_and_swap_memory(expected_hash=hash_text(before), new_text=parent_edit)
    assert swapped

    with pytest.raises(PromotionConflict, match="changed after proposal"):
        PromotionManager(store, repo_root=_repo(tmp_path)).promote(candidate, report)

    assert store.read_memory_text() == parent_edit
    stored = store.read_candidate(candidate.candidate_id)
    assert stored is not None and stored.status == "conflict"
    events = [json.loads(line) for line in store.audit_log_path.read_text(encoding="utf-8").splitlines()]
    assert any(event["event"] == "promotion_conflict" for event in events)


def test_promotion_readiness_explains_stale_skill_candidate(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    repo = _repo(tmp_path)
    built_in = _write_skill(repo / "skills", group="materials_worker", name="demo-workflow", marker="built-in")
    store = SelfEvolutionStore(workspace, project_id="demo")
    candidate = LearningCandidate(
        candidate_id="sec_stale",
        project_id="demo",
        run_id="run-one",
        thread_id="thread-one",
        action="skill",
        status="approved",
        group="materials_worker",
        name="demo-workflow",
        base_target_hash=hash_tree(built_in),
        created_at=utc_now(),
    )
    root = store.reset_candidate_dir(candidate.candidate_id)
    proposed = _write_skill(root / "proposed", group="materials_worker", name="demo-workflow", marker="workspace")
    candidate.bundle_hash = hash_tree(proposed)
    store.write_candidate(candidate)
    manager = PromotionManager(store, repo_root=repo)

    assert manager.promotion_readiness(candidate)["ready"] is True
    (built_in / "SKILL.md").write_text(_skill_text("demo-workflow", "new built-in version"), encoding="utf-8")
    readiness = manager.promotion_readiness(candidate)

    assert readiness["ready"] is False
    assert readiness["bundle_unchanged"] is True
    assert readiness["target_unchanged"] is False
    assert "regenerate" in readiness["reason"]


def test_skill_rollback_removes_new_override_and_reveals_builtin(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    repo = _repo(tmp_path)
    built_in = _write_skill(repo / "skills", group="materials_worker", name="demo-workflow", marker="built-in")
    store = SelfEvolutionStore(workspace, project_id="demo")
    candidate = LearningCandidate(
        candidate_id="sec_skill",
        project_id="demo",
        run_id="run-one",
        thread_id="thread-one",
        action="skill",
        status="approved",
        group="materials_worker",
        name="demo-workflow",
        base_target_hash=hash_tree(built_in),
        created_at=utc_now(),
    )
    root = store.reset_candidate_dir(candidate.candidate_id)
    proposed = _write_skill(root / "proposed", group="materials_worker", name="demo-workflow", marker="workspace")
    candidate.bundle_hash = hash_tree(proposed)
    store.write_candidate(candidate)
    report = CandidateGate(store).run(candidate)
    manager = PromotionManager(store, repo_root=repo)
    with pytest.raises(ValueError, match="explicit human decision"):
        manager.promote(candidate, report)
    promoted = manager.promote(
        candidate,
        report,
        decision_source="human",
        actor="alice",
        rationale="The exact skill diff was reviewed.",
    )

    rolled_back = manager.rollback(promoted)

    assert rolled_back.status == "rolled_back"
    assert not (store.self_develop_skills_dir / "materials_worker" / "demo-workflow").exists()
    assert "built-in" in (built_in / "SKILL.md").read_text(encoding="utf-8")
    events = [json.loads(line) for line in store.audit_log_path.read_text(encoding="utf-8").splitlines()]
    human_decision = next(event for event in events if event["event"] == "human_decision")
    assert human_decision["action"] == "promote"
    assert human_decision["actor"] == "alice"
    assert human_decision["candidate_hash"] == candidate.bundle_hash
    assert human_decision["rationale"] == "The exact skill diff was reviewed."
    assert any(event["event"] == "candidate_promoted" for event in events)
    assert any(event["event"] == "candidate_rolled_back" for event in events)


def test_manual_rejection_records_human_actor_rationale_and_candidate_hash(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    ensure_project_space_layout(workspace, create=True)
    store = SelfEvolutionStore(workspace, project_id="demo")
    candidate = LearningCandidate(
        candidate_id="sec_reject",
        project_id="demo",
        run_id="run-one",
        thread_id="thread-one",
        action="skill",
        status="reviewed",
        group="materials_worker",
        name="demo-workflow",
        bundle_hash="sha256:reviewed",
        created_at=utc_now(),
    )
    store.reset_candidate_dir(candidate.candidate_id)
    store.write_candidate(candidate)

    manager = PromotionManager(store, repo_root=_repo(tmp_path))
    rejected = manager.reject(
        candidate,
        actor="alice",
        rationale="The scope is broader than the demonstrated failure.",
    )

    assert rejected.status == "rejected"
    assert rejected.review["human_decision"]["actor"] == "alice"
    events = [json.loads(line) for line in store.audit_log_path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event"] == "human_decision"
    assert events[-1]["action"] == "reject"
    assert events[-1]["actor"] == "alice"
    assert events[-1]["candidate_hash"] == "sha256:reviewed"
    assert events[-1]["rationale"] == "The scope is broader than the demonstrated failure."
    with pytest.raises(ValueError, match="already final"):
        manager.reject(candidate, actor="bob", rationale="Conflicting second decision.")
