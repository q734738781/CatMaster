from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from deepagents.backends import FilesystemBackend
from deepagents.middleware.skills import SkillsMiddleware
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
from catmaster.runtime.self_evolution.trace import collect_turn_trace
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
        return ReviewerResult(decision=self.decision, rationale=f"Independent {self.decision}."), {"model_label": "fake-reviewer"}


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

    assert approved.decision == "approve"
    assert ambiguous.decision == "reject"


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


def test_pipeline_promotes_complete_skill_bundle_and_reviewer_reject_is_noop(tmp_path: Path) -> None:
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
    assert candidate is not None and candidate.status == "promoted"
    target = coordinator.store.self_develop_skills_dir / "materials_worker" / "demo-workflow"
    assert "workspace version" in (target / "SKILL.md").read_text(encoding="utf-8")
    assert (target / "scripts" / "helper.py").is_file()
    assert hash_tree(target) == candidate.bundle_hash

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
    assert candidate.review["decision"] == "unavailable"
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
    promoted = manager.promote(candidate, report)
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
    promoted = manager.promote(candidate, report)

    rolled_back = manager.rollback(promoted)

    assert rolled_back.status == "rolled_back"
    assert not (store.self_develop_skills_dir / "materials_worker" / "demo-workflow").exists()
    assert "built-in" in (built_in / "SKILL.md").read_text(encoding="utf-8")
