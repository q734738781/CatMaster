from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.runtime.self_evolution.effective import (
    BASE_VERSION,
    EffectiveSkillConflict,
    EffectiveSkillsManager,
    candidate_version,
)
from catmaster.runtime.self_evolution.models import LearningCandidate, ValidationReport
from catmaster.runtime.self_evolution.settings import resolve_self_evolution_mode
from catmaster.runtime.self_evolution.storage import SelfEvolutionStore, hash_tree, utc_now
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.specialists.runtime import build_specialist_runner
from catmaster.tools.base import workspace_scope
from catmaster.tools.misc.effective_skills import manage_effective_skills
from catmaster.tools.registry import get_tool_registry


GROUP = "materials_worker"
NAME = "effective-demo"
TARGET = f"{GROUP}/{NAME}"


def _write_skill(root: Path, marker: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                f"name: {NAME}",
                f"description: {marker}",
                "---",
                "",
                f"# {marker}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _repo(tmp_path: Path, *, with_base: bool = True) -> Path:
    root = tmp_path / "repo"
    if with_base:
        _write_skill(root / "skills" / GROUP / NAME, "repository base")
    return root


def test_unconfigured_workspace_defaults_to_auto_and_preserves_explicit_observe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CATMASTER_SELF_EVOLUTION_MODE", raising=False)
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    manager = EffectiveSkillsManager(store, repo_root=_repo(tmp_path))

    assert resolve_self_evolution_mode() == "auto"
    assert manager.mode_info() == {
        "mode": "auto",
        "source": "deployment_default",
    }

    manager.set_workspace_mode("observe", actor="alice", expected_mode="auto")
    assert manager.mode_info() == {
        "mode": "observe",
        "source": "workspace",
    }


def _approved_candidate(
    store: SelfEvolutionStore,
    *,
    revision: int,
    marker: str,
    recommendation: str = "approve",
    human_checks: list[str] | None = None,
) -> tuple[LearningCandidate, ValidationReport]:
    candidate_id = "sec_effective_demo"
    proposed = (
        store.revision_dir(candidate_id, revision)
        / "proposed"
        / GROUP
        / NAME
    )
    _write_skill(proposed, marker)
    report = ValidationReport(
        candidate_id=candidate_id,
        valid=True,
        loadable=True,
    )
    candidate = LearningCandidate(
        candidate_id=candidate_id,
        project_id=store.project_id,
        run_id=f"run-{revision}",
        thread_id="thread-one",
        action="skill",
        status="review",
        route="amend_existing_skill",
        group=GROUP,
        name=NAME,
        revision=revision,
        bundle_hash=hash_tree(proposed),
        review={
            "recommendation": recommendation,
            "summary": f"review {revision}",
            "human_checks": list(human_checks or []),
        },
        validation=report.to_dict(),
        created_at=utc_now(),
    )
    store.write_candidate(candidate)
    return candidate, report


def test_auto_selects_approved_exact_revision_without_overriding_pin_or_disable(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    store = SelfEvolutionStore(workspace, project_id="demo")
    manager = EffectiveSkillsManager(store, repo_root=_repo(tmp_path))
    manager.set_workspace_mode("auto", actor="alice", expected_mode="auto")

    first, first_report = _approved_candidate(store, revision=1, marker="revision one")
    first_result = manager.record_review_result(
        first,
        first_report,
        first.review,
    )
    assert first_result == {
        "eligible": True,
        "auto_head_advanced": True,
        "selected": True,
        "held_reason": "",
    }
    first_version = candidate_version(first)
    assert manager.target_summary(TARGET)["selected_version"] == first_version
    audit_events = [
        json.loads(line)
        for line in store.audit_log_path.read_text(encoding="utf-8").splitlines()
    ]
    activation_events = [
        event
        for event in audit_events
        if event.get("candidate_id") == first.candidate_id
        and event.get("revision") == first.revision
    ]
    assert [event["event"] for event in activation_events] == [
        "review_approved_auto_head"
    ]
    assert activation_events[0]["selected"] is True
    assert activation_events[0]["state_after"]["selected_version"] == first_version

    pinned = manager.update_target(
        TARGET,
        actor="alice",
        selected_version=BASE_VERSION,
        expected_selected_version=first_version,
    )
    assert pinned["selected_version"] == BASE_VERSION
    assert pinned["update_policy"] == "pinned"

    second, second_report = _approved_candidate(store, revision=2, marker="revision two")
    manager.record_review_result(second, second_report, second.review)
    second_version = candidate_version(second)
    pinned_after_review = manager.target_summary(TARGET)
    assert pinned_after_review["selected_version"] == BASE_VERSION
    assert pinned_after_review["auto_head"] == second_version

    following = manager.update_target(
        TARGET,
        actor="alice",
        update_policy="follow_auto",
        expected_selected_version=BASE_VERSION,
    )
    assert following["selected_version"] == second_version

    manager.update_target(
        TARGET,
        actor="alice",
        enabled=False,
        expected_selected_version=second_version,
    )
    third, third_report = _approved_candidate(store, revision=3, marker="revision three")
    manager.record_review_result(third, third_report, third.review)
    third_version = candidate_version(third)
    disabled = manager.target_summary(TARGET)
    assert disabled["enabled"] is False
    assert disabled["selected_version"] == second_version
    assert disabled["auto_head"] == third_version
    selected, disabled_targets = manager.runtime_overrides(include_canary=False)
    assert TARGET not in selected
    assert TARGET in disabled_targets


def test_observe_keeps_new_workspace_only_skill_dormant_until_auto(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    store = SelfEvolutionStore(workspace, project_id="demo")
    manager = EffectiveSkillsManager(store, repo_root=_repo(tmp_path, with_base=False))
    manager.set_workspace_mode("observe", actor="alice", expected_mode="auto")
    candidate, report = _approved_candidate(store, revision=1, marker="new workspace skill")

    result = manager.record_review_result(candidate, report, candidate.review)
    version = candidate_version(candidate)

    assert result["selected"] is False
    summary = manager.target_summary(TARGET)
    assert summary["selected_version"] == ""
    assert summary["auto_head"] == version
    preview = manager.preview_workspace_mode("auto")
    assert preview["changes"] == [
        {
            "target": TARGET,
            "selected_version_before": "",
            "selected_version_after": version,
        }
    ]
    applied = manager.set_workspace_mode(
        "auto",
        actor="alice",
        expected_mode="observe",
    )
    assert applied["changes"] == preview["changes"]
    assert manager.target_summary(TARGET)["selected_version"] == version


def test_human_boundary_requires_an_exact_chat_resolution_and_reject_stays_ineligible(
    tmp_path: Path,
) -> None:
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    manager = EffectiveSkillsManager(store, repo_root=_repo(tmp_path))
    manager.set_workspace_mode("auto", actor="alice", expected_mode="auto")
    held, report = _approved_candidate(
        store,
        revision=1,
        marker="held revision",
        human_checks=["Ask the user which policy is intended."],
    )
    result = manager.record_review_result(held, report, held.review)
    assert result["eligible"] is False
    assert "Ask the user" in result["held_reason"]
    with pytest.raises(ValueError, match="reviewer-approved"):
        manager.update_target(
            TARGET,
            actor="alice",
            selected_version=candidate_version(held),
            expected_selected_version=BASE_VERSION,
        )
    assert manager.waiting_for_clarification_count() == 1
    resolved = manager.resolve_human_boundary(
        TARGET,
        candidate_version(held),
        actor="chat:thread-one",
        message_id="msg_user_choice",
        thread_id="thread-one",
        resolution="Use the narrower recovery-only policy.",
        expected_selected_version=BASE_VERSION,
    )
    assert resolved["selected_version"] == candidate_version(held)
    assert resolved["auto_head"] == candidate_version(held)
    assert manager.waiting_for_clarification_count() == 0
    assert manager.target_detail(TARGET)["versions"][0]["eligible"] is True
    history = manager.target_detail(TARGET)["history"]
    assert history[0]["event"] == "human_boundary_resolved"
    assert history[0]["message_id"] == "msg_user_choice"

    rejected, _ = _approved_candidate(
        store,
        revision=2,
        marker="rejected revision",
        recommendation="reject",
    )
    with pytest.raises(ValueError, match="reviewer-approved"):
        manager.update_target(
            TARGET,
            actor="alice",
            selected_version=candidate_version(rejected),
            expected_selected_version=candidate_version(held),
        )


def test_chat_skill_control_requires_and_audits_current_user_message(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    store = SelfEvolutionStore(workspace, project_id=workspace.name)
    manager = EffectiveSkillsManager(store)
    manager.set_workspace_mode("auto", actor="alice", expected_mode="auto")
    held, report = _approved_candidate(
        store,
        revision=1,
        marker="chat-authorized revision",
        human_checks=["Confirm whether this should apply to recovery runs."],
    )
    manager.record_review_result(held, report, held.review)

    with workspace_scope(workspace):
        with pytest.raises(CatMasterToolExecutionError, match="trusted current thread"):
            manage_effective_skills(
                {
                    "operation": "resolve_boundary",
                    "target": TARGET,
                    "selected_version": candidate_version(held),
                    "expected_selected_version": "",
                    "resolution": "Apply only to recovery runs.",
                }
            )
        with toolcall_context(
            "tool-chat-skill",
            context={
                "thread_id": "thread-one",
                "user_message_id": "msg_user_resolution",
            },
        ):
            _content, artifact = manage_effective_skills(
                {
                    "operation": "resolve_boundary",
                    "target": TARGET,
                    "selected_version": candidate_version(held),
                    "expected_selected_version": "",
                    "resolution": "Apply only to recovery runs.",
                }
            )

    assert artifact["data"]["source_ref"] == "message:msg_user_resolution"
    active = store.read_active_skills()["skills"][TARGET]
    assert active["selected_version"] == candidate_version(held)
    assert active["human_authorizations"][candidate_version(held)]["message_id"] == (
        "msg_user_resolution"
    )
    schema = next(
        tool
        for tool in get_tool_registry().as_openai_tools(
            allowlist=["manage_effective_skills"]
        )
        if tool["name"] == "manage_effective_skills"
    )["parameters"]
    assert '"type": "null"' not in json.dumps(schema)


def test_legacy_stable_pointer_migrates_to_same_selected_pinned_version(
    tmp_path: Path,
) -> None:
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    manager = EffectiveSkillsManager(store, repo_root=_repo(tmp_path))
    candidate, _report = _approved_candidate(store, revision=1, marker="legacy stable")
    version = candidate_version(candidate)
    store.write_active_skills({"skills": {TARGET: {"stable": version}}})

    payload = manager.ensure_migrated()

    assert payload["skills"][TARGET] == {
        "enabled": True,
        "selected_version": version,
        "update_policy": "pinned",
        "auto_head": version,
        "stable": version,
    }
    with pytest.raises(EffectiveSkillConflict, match="changed after"):
        manager.update_target(
            TARGET,
            actor="alice",
            enabled=False,
            expected_selected_version="stale-version",
        )


def test_catalog_is_complete_and_cursor_paged_without_hiding_ineligible_history(
    tmp_path: Path,
) -> None:
    store = SelfEvolutionStore(tmp_path / "workspace", project_id="demo")
    repo = _repo(tmp_path)
    _write_skill(repo / "skills" / GROUP / "another-skill", "another repository skill")
    manager = EffectiveSkillsManager(store, repo_root=repo)
    candidate, _report = _approved_candidate(
        store,
        revision=1,
        marker="needs another revision",
        recommendation="needs_revision",
    )

    first_page, cursor = manager.list_targets(limit=1)
    remaining, final_cursor = manager.list_targets(after=cursor, limit=20)
    targets = [item["target"] for item in [*first_page, *remaining]]

    assert targets == sorted({"/memories/AGENTS.md", f"{GROUP}/another-skill", TARGET})
    assert final_cursor == ""
    detail = manager.target_detail(TARGET)
    version = next(item for item in detail["versions"] if item["candidate_id"])
    assert version["version"] == candidate_version(candidate)
    assert version["status"] == "needs_revision"
    assert version["eligible"] is False
    assert version["files"] == ["SKILL.md"]


def test_fresh_specialist_snapshot_uses_auto_selected_version_and_honors_disable(
    tmp_path: Path,
) -> None:
    class _Profile:
        @staticmethod
        def config_for_role(role: str):
            return SimpleNamespace(
                model=f"{role}-model",
                provider="langchain",
                base_url=None,
            )

    workspace = tmp_path / "workspace"
    store = SelfEvolutionStore(workspace, project_id="demo")
    manager = EffectiveSkillsManager(store)
    manager.set_workspace_mode("auto", actor="alice", expected_mode="auto")
    candidate, report = _approved_candidate(
        store,
        revision=1,
        marker="automatically selected runtime version",
    )
    manager.record_review_result(candidate, report, candidate.review)

    runner = build_specialist_runner(
        workspace=workspace,
        llm_profile=_Profile(),
        reporter=None,
        run_control=None,
        project_id="demo",
        preferred_entrypoint="experiment",
    ).runner
    runner._stage_deepagent_assets(workspace / "files", thread_id="thread-one")
    staged = runner._skill_snapshot_root / "skills" / GROUP / NAME / "SKILL.md"
    assert "automatically selected runtime version" in staged.read_text(encoding="utf-8")
    selected_entries = [
        item
        for item in runner._skill_version_entries
        if "@r" in item["skill_version"]
    ]
    assert selected_entries == [
        {
            "skill_name": TARGET,
            "skill_version": candidate_version(candidate),
            "virtual_path": f"/.deepagents/skills/{TARGET}",
        }
    ]
    assert any(
        item["skill_version"].startswith("base@")
        for item in runner._skill_version_entries
    )

    manager.update_target(
        TARGET,
        actor="alice",
        enabled=False,
        expected_selected_version=candidate_version(candidate),
    )
    disabled_runner = build_specialist_runner(
        workspace=workspace,
        llm_profile=_Profile(),
        reporter=None,
        run_control=None,
        project_id="demo",
        preferred_entrypoint="experiment",
    ).runner
    disabled_runner._stage_deepagent_assets(
        workspace / "files",
        thread_id="thread-two",
    )
    assert not (
        disabled_runner._skill_snapshot_root / "skills" / GROUP / NAME
    ).exists()
