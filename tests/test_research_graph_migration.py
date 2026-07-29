from __future__ import annotations

import json
from pathlib import Path

import pytest

import catmaster.research.knowledge_graph.migration as migration_module
from catmaster.research.knowledge_graph.migration import ResearchGraphMigrator
from catmaster.research.knowledge_graph.store import ResearchGraphStore
from catmaster.tools.base import ensure_project_space_layout
from catmaster.webui.thread_store import ThreadStore


def _v4_state() -> dict:
    return {
        "schema_version": 4,
        "revision": 2,
        "question": "Which mechanism survives?",
        "hypotheses": [
            {
                "id": "h1",
                "claim": "Mechanism one.",
                "rationale": "Reason one.",
                "predictions": ["Prediction one."],
                "derived_from": [],
                "status": "open",
            }
        ],
        "actions": [
            {
                "id": "a1",
                "executor": "literature",
                "question": "Check mechanism one.",
                "task": "Read the primary source.",
                "target_hypotheses": ["h1"],
                "decision_rule": "Matching evidence supports.",
                "prerequisite_action_ids": [],
                "information_value": "high",
                "cost": "low",
                "status": "completed",
                "failure_reason": "",
            }
        ],
        "evidence": [
            {
                "action_id": "a1",
                "summary": "The source supports mechanism one.",
                "source": "https://example.org/source",
                "effects": [
                    {
                        "hypothesis_id": "h1",
                        "verdict": "supports",
                        "reason": "The prediction is observed.",
                    }
                ],
            }
        ],
        "active_action_id": "",
    }


def _active_v4_state() -> dict:
    payload = _v4_state()
    payload["actions"][0]["status"] = "running"
    payload["evidence"] = []
    payload["active_action_id"] = "a1"
    return payload


def test_migration_dry_run_apply_idempotency_and_rollback(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    campaign = tmp_path / "files" / "research_hypothesis_engines" / "campaign"
    campaign.mkdir(parents=True)
    source = campaign / "state.json"
    source.write_text(json.dumps(_v4_state()), encoding="utf-8")
    incomplete = (
        tmp_path / "files" / "research_hypothesis_engines" / "legacy-v2"
    )
    incomplete.mkdir(parents=True)
    (incomplete / "state.json").write_text(
        json.dumps({"schema_version": 2, "question": "Incomplete"}),
        encoding="utf-8",
    )
    truncated = (
        tmp_path / "files" / "research_hypothesis_engines" / "broken"
    )
    truncated.mkdir(parents=True)
    (truncated / "state.json").write_text("{broken", encoding="utf-8")

    migrator = ResearchGraphMigrator(tmp_path)
    dry = migrator.dry_run()
    assert dry["totals"]["graphs"] == 1
    assert dry["totals"]["review_items"] == 1
    assert dry["totals"]["quarantined_files"] == 1
    assert source.is_file()

    applied = migrator.apply()
    graph_id = applied["graphs"][0]
    snapshot = ResearchGraphStore(tmp_path).get_snapshot(graph_id)
    assert len(snapshot["nodes"]) == 3
    assert {edge["relation"] for edge in snapshot["edges"]} == {
        "tests",
        "produces",
        "supports",
    }
    assert not source.exists()
    manifest = tmp_path / applied["manifest"]
    assert manifest.is_file()

    second = migrator.apply()
    assert second["counts"]["graphs"] == 0
    assert ResearchGraphStore(tmp_path).list_graphs(include_archived=True)[0][
        "graph_id"
    ] == graph_id

    rolled_back = migrator.rollback(manifest)
    assert graph_id in rolled_back["deleted_graphs"]
    assert source.is_file()
    assert ResearchGraphStore(tmp_path).list_graphs(include_archived=True) == []


def test_v3_campaign_upgrades_without_mode_fields(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    payload = _v4_state()
    payload["schema_version"] = 3
    payload["mode"] = "interactive"
    payload["paused"] = True
    campaign = tmp_path / "files" / "research_hypothesis_engines" / "v3"
    campaign.mkdir(parents=True)
    (campaign / "state.json").write_text(json.dumps(payload), encoding="utf-8")

    migrator = ResearchGraphMigrator(tmp_path)
    report = migrator.dry_run()
    assert report["campaigns"][0]["status"] == "ready"
    applied = migrator.apply()
    snapshot = ResearchGraphStore(tmp_path).get_snapshot(applied["graphs"][0])
    assert snapshot["graph"]["question"] == "Which mechanism survives?"
    assert len(snapshot["nodes"]) == 3


def test_failed_import_rolls_back_the_partial_graph_transaction(
    tmp_path: Path,
) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    campaign = tmp_path / "files" / "research_hypothesis_engines" / "broken-plan"
    campaign.mkdir(parents=True)
    source = campaign / "state.json"
    source.write_text(json.dumps(_v4_state()), encoding="utf-8")
    migrator = ResearchGraphMigrator(tmp_path)
    graph_id = "graph_transaction_rollback"

    with pytest.raises(KeyError):
        migrator._import_campaign(
            path=source,
            mapping={
                "graph_id": graph_id,
                "hypotheses": {"h1": "hyp_transaction_rollback"},
                "experiments": {},
                "results": {"a1": "res_transaction_rollback"},
                "blocked_results": {},
            },
        )

    with pytest.raises(KeyError):
        ResearchGraphStore(tmp_path).get_graph(graph_id)


def test_apply_resumes_after_crash_between_legacy_move_and_manifest_update(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    campaign = tmp_path / "files" / "research_hypothesis_engines" / "resume"
    campaign.mkdir(parents=True)
    source = campaign / "state.json"
    source.write_text(json.dumps(_v4_state()), encoding="utf-8")
    threads = ThreadStore(workspace=tmp_path, workspace_id="default")
    bound_thread = threads.create_thread(
        title="Legacy campaign",
        deepagent_thread_id="resume",
    )
    migrator = ResearchGraphMigrator(tmp_path)
    original_atomic_json = migration_module._atomic_json
    calls = 0

    def _crash_on_first_post_move_manifest(path, payload):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("simulated process crash")
        original_atomic_json(path, payload)

    monkeypatch.setattr(
        migration_module,
        "_atomic_json",
        _crash_on_first_post_move_manifest,
    )
    with pytest.raises(RuntimeError, match="simulated process crash"):
        migrator.apply()
    assert not source.exists()
    assert migrator.progress_path.is_file()
    # Simulate the graph commit surviving while the non-transactional thread
    # binding is absent. The legacy source has already moved, so resume must
    # repair the binding from the stable plan rather than reread state.json.
    threads.update_thread(
        bound_thread.thread_id,
        active_research_graph_id="",
        research_focus_node_id="",
    )

    monkeypatch.setattr(migration_module, "_atomic_json", original_atomic_json)
    resumed = ResearchGraphMigrator(tmp_path).apply()
    assert resumed["completed"] is True
    assert resumed["counts"]["graphs"] == 1
    assert resumed["counts"]["nodes"] == 3
    assert not migrator.progress_path.exists()
    assert (tmp_path / resumed["manifest"]).is_file()
    rebound = threads.get_thread(bound_thread.thread_id)
    assert rebound.active_research_graph_id == resumed["graphs"][0]


def test_apply_repairs_binding_after_crash_immediately_after_graph_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    campaign = (
        tmp_path
        / "files"
        / "research_hypothesis_engines"
        / "binding-crash"
    )
    campaign.mkdir(parents=True)
    source = campaign / "state.json"
    source.write_text(json.dumps(_v4_state()), encoding="utf-8")
    threads = ThreadStore(workspace=tmp_path, workspace_id="default")
    legacy_thread = threads.create_thread(
        title="Legacy source",
        deepagent_thread_id="binding-crash",
    )
    migrator = ResearchGraphMigrator(tmp_path)

    def _crash_before_binding(**_kwargs):
        raise RuntimeError("crash after graph commit")

    monkeypatch.setattr(
        migrator,
        "_bind_campaign_threads",
        _crash_before_binding,
    )
    with pytest.raises(RuntimeError, match="crash after graph commit"):
        migrator.apply()
    graphs = ResearchGraphStore(tmp_path).list_graphs(include_archived=True)
    assert len(graphs) == 1
    assert source.is_file()
    assert (
        threads.get_thread(legacy_thread.thread_id).active_research_graph_id
        == ""
    )

    resumed = ResearchGraphMigrator(tmp_path).apply()
    assert resumed["completed"] is True
    assert (
        threads.get_thread(legacy_thread.thread_id).active_research_graph_id
        == graphs[0]["graph_id"]
    )
    assert len(
        ResearchGraphStore(tmp_path).list_graphs(include_archived=True)
    ) == 1


def test_legacy_active_action_without_live_child_is_released_and_reported(
    tmp_path: Path,
) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    campaign = (
        tmp_path
        / "files"
        / "research_hypothesis_engines"
        / "orphan-active"
    )
    campaign.mkdir(parents=True)
    source = campaign / "state.json"
    source.write_text(json.dumps(_active_v4_state()), encoding="utf-8")
    threads = ThreadStore(workspace=tmp_path, workspace_id="default")
    terminal_child = threads.create_thread(
        title="Finished legacy child",
        meta={
            "research_campaign_id": "orphan-active",
            "research_map_action_id": "a1",
        },
    )

    migrator = ResearchGraphMigrator(tmp_path)
    dry_run = migrator.dry_run()
    assert dry_run["totals"]["launches"] == 0
    assert dry_run["totals"]["review_items"] == 1
    assert "release the Experiment to ready" in dry_run["campaigns"][0][
        "issues"
    ][0]

    applied = migrator.apply()
    snapshot = ResearchGraphStore(tmp_path).get_snapshot(applied["graphs"][0])
    experiment = next(
        node for node in snapshot["nodes"] if node["kind"] == "experiment"
    )
    assert experiment["state"] == "ready"
    assert snapshot["launches"] == []
    assert any(
        "release the Experiment to ready" in issue
        for row in applied["review_queue"]
        for issue in row.get("issues", [])
    )
    rebound = threads.get_thread(terminal_child.thread_id)
    assert rebound.active_research_graph_id == applied["graphs"][0]
    assert rebound.research_focus_node_id == experiment["node_id"]
