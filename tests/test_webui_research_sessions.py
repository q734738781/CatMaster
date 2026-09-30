from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from catmaster.research.knowledge_graph.models import (
    ExperimentCreateRequest,
    GraphCreateRequest,
    GraphPatchRequest,
    ResearchExperimentPairOutcomeDraft,
    ResearchRefInput,
    ResultCreateRequest,
)
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.research.knowledge_graph.store import ResearchGraphStore
from catmaster.tools.base import ensure_project_space_layout
from catmaster.webui.agent_loop import ThreadAgentLoopService
from catmaster.webui.server import create_app
from catmaster.webui.thread_models import (
    MessagePart,
    ThreadMessage,
    ThreadRole,
    ThreadStatus,
)
from catmaster.webui.thread_store import ThreadStore


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    return workspace


@pytest.mark.parametrize("status, expected", [
    (ThreadStatus.RUNNING, "running"),
    (ThreadStatus.ERROR, "operationally_incomplete"),
    (ThreadStatus.IDLE, "waiting_continue"),
])
def test_native_research_root_activity_follows_actual_run(tmp_path, status, expected):
    from catmaster.webui.projections.research_sessions import project_research_session_activity

    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace, workspace_id="default")
    root = service.thread_store.create_thread(title="Literature recommendations", entrypoint="persistent_research")
    service.create_graph(GraphCreateRequest(question="Which Cu experiments are informative?", orchestration_mode="auto"),
                         session_root_thread_id=root.thread_id)
    service.thread_store.update_thread(root.thread_id, status=status)
    root = service.thread_store.get_thread(root.thread_id)
    activity = project_research_session_activity(root, threads=[root], graph_service=service, workspace=workspace)
    assert activity.state == expected
    assert activity.active_child_thread_id == ""
    assert activity.decision_round_count == 0


def test_native_research_activity_exposes_reconsideration_and_open_premises(tmp_path):
    from catmaster.webui.projections.research_sessions import project_research_session_activity

    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace, workspace_id="default")
    root = service.thread_store.create_thread(title="Literature recommendations", entrypoint="persistent_research")
    graph = service.create_graph(GraphCreateRequest(question="Which Cu experiments are informative?", orchestration_mode="auto"),
                                 session_root_thread_id=root.thread_id)
    root = service.thread_store.get_thread(root.thread_id)
    gid = graph["graph"]["graph_id"]
    body = dict(disposition="stalled", problem="A decisive electrolyte condition is unclear",
                reason="Accessible sources disagree", authorized_scope="Literature only")
    decision = service.store.record_disposition(gid, thread_id=root.thread_id, run_id="r", body=body)
    activity = project_research_session_activity(root, threads=[root], graph_service=service, workspace=workspace)
    assert activity.state == "reviewing"
    service.store.record_review(gid, review=dict(decision_id=decision["decision_id"],
        assessment="The original methods are unavailable; the conditional proposal remains useful.",
        resume_when="Methods become available"))
    service.store.record_disposition(gid, thread_id=root.thread_id, run_id="r",
        body={**body, "decision_id": decision["decision_id"], "disposition": "parked", "resume_when": "Methods become available"})
    activity = project_research_session_activity(root, threads=[root], graph_service=service, workspace=workspace)
    assert activity.state == "parked"
    assert "conditional proposal" in activity.decision_summary


def test_research_activity_uses_child_lifecycle_not_historical_cards(tmp_path):
    from catmaster.webui.projections.research_sessions import project_research_session_activity

    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace, workspace_id="default")
    root = service.thread_store.create_thread(title="Cu literature", entrypoint="persistent_research")
    graph = service.create_graph(GraphCreateRequest(question="Which Cu experiments?", orchestration_mode="auto"),
                                session_root_thread_id=root.thread_id)
    root = service.thread_store.get_thread(root.thread_id)
    service.thread_store.append_message(ThreadMessage(id="msg_native_async", thread_id=root.thread_id,
        role="assistant", status="completed", parts=[MessagePart(id="part_native_async", type="subagent", status="running",
            meta={"source": "litreview_agent", "task_id": "native-literature", "native_status": "running"})]))
    activity = project_research_session_activity(root, threads=[root], graph_service=service, workspace=workspace)
    assert activity.state == "waiting_continue"
    assert activity.active_parts == []
    child = service.thread_store.create_thread(thread_id="native-literature", title="Literature",
        parent_thread_id=root.thread_id,
        meta={"background_task": True, "agent_name": "litreview_agent"})
    child = service.thread_store.update_thread(child.thread_id, status="running")
    activity = project_research_session_activity(root, threads=[root, child], graph_service=service, workspace=workspace)
    assert activity.state == "running"
    assert activity.current_title == "Literature review"
    assert activity.active_child_thread_id == child.thread_id
    assert activity.active_child_role == "primary"
    assert activity.execution_count == 1
    assert activity.active_parts == []  # The execution-backed snapshot owns cards.
    service.store.update_graph(graph['graph']['graph_id'], expected_revision=graph['graph']['revision'], changes={'completed': True})
    completed = project_research_session_activity(root, threads=[root], graph_service=service, workspace=workspace)
    assert completed.state == "completed"


def _append_user_message(store: ThreadStore, thread_id: str, text: str) -> ThreadMessage:
    suffix = len(store.list_messages(thread_id)) + 1
    message = ThreadMessage(
        id=f"msg_test_user_{suffix}",
        thread_id=thread_id,
        role="user",
        status="completed",
        parts=[
            MessagePart(
                id=f"part_test_user_{suffix}",
                type="text",
                text=text,
                status="completed",
            )
        ],
    )
    store.append_message(message)
    return message


def _active_research_session(tmp_path: Path) -> tuple[Path, ResearchGraphService, object, object, dict]:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace, workspace_id="default")
    root = service.thread_store.create_thread(
        title=(
            "Example research study with a deliberately long "
            "session title that remains recoverable"
        ),
        entrypoint="persistent_research",
    )
    graph = service.create_graph(
        GraphCreateRequest(
            question="Which ligand descriptor controls asymmetric ketone selectivity?",
            orchestration_mode="auto",
            initial_hypotheses=[
                {"claim": "An isolated ligand descriptor controls selectivity."}
            ],
        ),
        session_root_thread_id=root.thread_id,
    )
    root = service.thread_store.get_thread(root.thread_id)
    hypothesis_id = graph["nodes"][0]["node_id"]
    first = service.add_experiment(
        graph["graph"]["graph_id"],
        ExperimentCreateRequest(
            expected_revision=graph["graph"]["revision"],
            title="Validate the baseline ligand descriptor",
            objective="Measure the baseline descriptor.",
            plan_summary="Compute it for the matched ligand set.",
            decision_rule="A stable correlation supports the descriptor.",
            execution_lane="experiment",
            state="ready",
            tests_hypothesis_ids=[hypothesis_id],
        ),
    )
    second = service.add_experiment(
        graph["graph"]["graph_id"],
        ExperimentCreateRequest(
            expected_revision=first["graph"]["revision"],
            title=(
                "Validate ligand selectivity descriptor across the complete "
                "isolated ligand set"
            ),
            objective="Measure the descriptor across the complete isolated ligand set.",
            plan_summary="Compute descriptors under matched settings.",
            decision_rule="A robust trend distinguishes the hypothesis.",
            execution_lane="experiment",
            state="ready",
            tests_hypothesis_ids=[hypothesis_id],
        ),
    )
    first_execution = service.thread_store.create_thread(
        title="Run: Validate the baseline ligand descriptor",
        entrypoint="experiment",
        parent_thread_id=root.thread_id,
        thread_role=ThreadRole.RESEARCH_EXECUTION,
    )
    service.thread_store.update_thread(
        first_execution.thread_id,
        active_research_graph_id=graph["graph"]["graph_id"],
        research_focus_node_id=first["node"]["node_id"],
    )
    active_execution = service.thread_store.create_thread(
        title=(
            "Run: Validate ligand selectivity descriptor across the complete "
            "isolated ligand set"
        ),
        entrypoint="experiment",
        parent_thread_id=root.thread_id,
        thread_role=ThreadRole.RESEARCH_EXECUTION,
    )
    service.thread_store.update_thread(
        active_execution.thread_id,
        active_research_graph_id=graph["graph"]["graph_id"],
        research_focus_node_id=second["node"]["node_id"],
    )
    launch, claimed = service.store.claim_launch(
        graph["graph"]["graph_id"],
        second["node"]["node_id"],
        expected_revision=second["graph"]["revision"],
        replicate=False,
        lease_owner="test-session-fixture",
        session_root_thread_id=root.thread_id,
    )
    assert claimed is True
    service.store.update_launch(
        launch["launch_id"],
        status="running",
        thread_id=active_execution.thread_id,
        lease_owner="",
        lease_until=0,
    )
    service.thread_store.append_message(
        ThreadMessage(
            id="msg_session_user",
            thread_id=active_execution.thread_id,
            role="user",
            status="completed",
            parts=[
                MessagePart(
                    id="part_session_user",
                    type="text",
                    text="Run the descriptor calculation.",
                    status="completed",
                )
            ],
        )
    )
    service.thread_store.append_message(
        ThreadMessage(
            id="msg_session_active",
            thread_id=active_execution.thread_id,
            role="assistant",
            status="streaming",
            parts=[
                MessagePart(
                    id="part_session_progress",
                    type="tool-call",
                    status="completed",
                    meta={
                        "tool": "notify_progress",
                        "input": {
                            "summary": (
                                "Computing descriptors for the isolated ligand set"
                            )
                        },
                    },
                ),
                MessagePart(
                    id="part_session_remote",
                    type="tool-call",
                    status="running",
                    meta={
                        "tool": "remote_submission",
                        "input": {"task": "Ligand descriptor calculation"},
                        "agent_name": "materials_worker",
                        "started_at": 1_700_000_000.0,
                    },
                ),
            ],
        )
    )
    service.thread_store.update_thread(
        active_execution.thread_id,
        status=ThreadStatus.RUNNING,
    )
    return workspace, service, root, active_execution, second


def test_thread_roles_persist_and_legacy_internal_roles_are_migrated(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    root = store.create_thread(title="Root", entrypoint="persistent_research")
    assert root.thread_role is ThreadRole.RESEARCH_ROOT
    for role in (
        ThreadRole.PRIMARY,
        ThreadRole.RESEARCH_EXECUTION,
        ThreadRole.RESEARCH_PLANNING,
        ThreadRole.RESEARCH_COMPARISON,
    ):
        thread = store.create_thread(
            title=role.value,
            parent_thread_id=(root.thread_id if role is not ThreadRole.PRIMARY else ""),
            thread_role=role,
        )
        recovered = ThreadStore(
            workspace=workspace,
            workspace_id="default",
        ).get_thread(thread.thread_id)
        assert recovered.thread_role is role
        assert recovered.parent_thread_id == (
            root.thread_id if role is not ThreadRole.PRIMARY else ""
        )

    legacy = store.create_thread(title="Legacy internal planning")
    payload = legacy.model_dump(mode="json")
    payload.pop("thread_role")
    payload.pop("parent_thread_id")
    payload["meta"] = {"internal_kind": "research_graph_planning"}
    store.thread_path(legacy.thread_id).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    changed = store.migrate_research_thread_roles()
    assert legacy.thread_id in {thread.thread_id for thread in changed}
    assert store.get_thread(legacy.thread_id).thread_role is ThreadRole.RESEARCH_PLANNING


def test_role_migration_demotes_stale_research_root_and_restores_graph_owner(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    valid_root = store.create_thread(
        title="Persistent owner",
        entrypoint="persistent_research",
    )
    stale_root = store.create_thread(
        title="Accidentally inherited Persistent Research",
        entrypoint="persistent_research",
    )
    graph_store = ResearchGraphStore(workspace)
    graph = graph_store.create_graph(
        title="Shared scientific context",
        question="Which experiment should run?",
        orchestration_mode="auto",
        orchestration_thread_id=stale_root.thread_id,
    )
    graph_id = graph["graph_id"]
    store.update_thread(valid_root.thread_id, active_research_graph_id=graph_id)
    store.update_thread(stale_root.thread_id, active_research_graph_id=graph_id)

    stale_payload = stale_root.model_dump(mode="json")
    stale_payload["entrypoint"] = "research"
    stale_payload["active_research_graph_id"] = graph_id
    store.thread_path(stale_root.thread_id).write_text(
        json.dumps(stale_payload),
        encoding="utf-8",
    )

    ResearchGraphService(workspace=workspace, workspace_id="default")

    recovered = store.get_thread(stale_root.thread_id)
    assert recovered.entrypoint == "research"
    assert recovered.thread_role is ThreadRole.PRIMARY
    assert recovered.active_research_graph_id == graph_id
    assert graph_store.get_graph(graph_id)["orchestration_thread_id"] == valid_root.thread_id


def test_ordinary_research_binding_keeps_persistent_orchestration_owner(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace, workspace_id="default")
    root = service.thread_store.create_thread(
        title="Persistent owner",
        entrypoint="persistent_research",
    )
    graph = service.create_graph(
        GraphCreateRequest(
            question="Which experiment should run next?",
            orchestration_mode="auto",
            initial_hypotheses=[{"claim": "A bounded calculation can test it."}],
        ),
        session_root_thread_id=root.thread_id,
    )
    graph_id = graph["graph"]["graph_id"]
    ordinary = service.thread_store.create_thread(
        title="Independent user-launched research",
        entrypoint="research",
    )

    bound = service.bind_thread(
        ordinary.thread_id,
        graph_id=graph_id,
        focus_node_id=graph["nodes"][0]["node_id"],
    )

    assert bound.active_research_graph_id == graph_id
    assert bound.thread_role is ThreadRole.PRIMARY
    assert service.store.get_graph(graph_id)["orchestration_thread_id"] == root.thread_id


def test_switching_persistent_root_to_research_keeps_graph_context_and_uses_normal_turn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace, workspace_id="default")
    valid_root = service.thread_store.create_thread(
        title="Existing Persistent Research",
        entrypoint="persistent_research",
    )
    graph = service.create_graph(
        GraphCreateRequest(
            question="Which bounded experiment should run?",
            orchestration_mode="auto",
            initial_hypotheses=[{"claim": "Experiment A may distinguish the routes."}],
        ),
        session_root_thread_id=valid_root.thread_id,
    )
    graph_id = graph["graph"]["graph_id"]
    focus_node_id = graph["nodes"][0]["node_id"]
    accidental_root = service.thread_store.create_thread(
        title="Independent user-launched research",
        entrypoint="persistent_research",
    )
    service.bind_thread(
        accidental_root.thread_id,
        graph_id=graph_id,
        focus_node_id=focus_node_id,
    )
    assert service.store.get_graph(graph_id)["orchestration_thread_id"] == accidental_root.thread_id

    calls: list[tuple[str, str, str]] = []

    async def fake_submit(self, *, thread_id, payload):
        calls.append((thread_id, payload.entrypoint, payload.text))
        message = _append_user_message(self.store, thread_id, payload.text)
        return {
            "queued": False,
            "thread": self.store.get_thread(thread_id),
            "message": message,
        }

    monkeypatch.setattr(ThreadAgentLoopService, "submit", fake_submit)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    patched = client.patch(
        f"/api/threads/{accidental_root.thread_id}",
        json={"entrypoint": "research"},
    )
    assert patched.status_code == 200
    changed = patched.json()["thread"]
    assert changed["entrypoint"] == "research"
    assert changed["thread_role"] == "primary"
    assert changed["active_research_graph_id"] == graph_id
    assert changed["research_focus_node_id"] == focus_node_id
    assert service.store.get_graph(graph_id)["orchestration_thread_id"] == valid_root.thread_id

    submitted = client.post(
        f"/api/threads/{accidental_root.thread_id}/submit",
        json={"text": "Analyze the failed managed calculation.", "entrypoint": "research"},
    )
    assert submitted.status_code == 200
    assert submitted.json()["queued"] is False
    assert calls == [
        (
            accidental_root.thread_id,
            "research",
            "Analyze the failed managed calculation.",
        )
    ]


def test_thread_list_projects_one_research_folder_and_keeps_internal_threads_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, service, root, active_execution, _second = _active_research_session(
        tmp_path
    )
    for index in range(2):
        service.thread_store.create_thread(
            title=f"Plan next step: round {index + 1}",
            entrypoint="research",
            parent_thread_id=root.thread_id,
            thread_role=ThreadRole.RESEARCH_PLANNING,
            meta={"internal_kind": "research_graph_planning"},
        )
    for index in range(8):
        service.thread_store.create_thread(
            title="Compare ready Experiments",
            entrypoint="research",
            parent_thread_id=root.thread_id,
            thread_role=ThreadRole.RESEARCH_COMPARISON,
            meta={"internal_kind": "research_graph_experiment_comparison"},
        )

    monkeypatch.setenv("CATMASTER_WEBUI_DEVELOPER_DIAGNOSTICS", "1")
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    response = client.get("/api/workspaces/default/threads")
    assert response.status_code == 200
    threads = response.json()["threads"]
    assert {thread["thread_role"] for thread in threads} == {
        "research_root",
        "research_execution",
    }
    assert len(threads) == 3
    projected_root = next(
        thread for thread in threads if thread["thread_id"] == root.thread_id
    )
    activity = projected_root["research_activity"]
    assert projected_root["status"] == "idle"
    assert activity["state"] == "running"
    assert activity["active_child_thread_id"] == active_execution.thread_id
    assert activity["execution_count"] == 2
    assert activity["decision_round_count"] == 2
    assert activity["latest_progress"].startswith("Computing descriptors")
    assert [part["id"] for part in activity["active_parts"]] == [
        "part_session_remote"
    ]
    assert all(
        thread["title"] != "Compare ready Experiments" for thread in threads
    )

    diagnostic = client.get(
        "/api/workspaces/default/threads",
        params={"include_internal": "true"},
    )
    assert diagnostic.status_code == 200
    diagnostic_roles = [thread["thread_role"] for thread in diagnostic.json()["threads"]]
    assert diagnostic_roles.count("research_planning") == 2
    assert diagnostic_roles.count("research_comparison") == 8

    graph = service.store.get_graph(root.active_research_graph_id)
    service.patch_graph(
        graph["graph_id"],
        GraphPatchRequest(
            expected_revision=graph["revision"],
            orchestration_mode="manual",
        ),
        session_root_thread_id=root.thread_id,
    )
    paused = client.get("/api/workspaces/default/threads").json()["threads"]
    paused_activity = next(
        thread for thread in paused if thread["thread_id"] == root.thread_id
    )["research_activity"]
    assert paused_activity["state"] == "running"
    assert paused_activity["automation_paused"] is True

    service.thread_store.update_thread(
        active_execution.thread_id,
        status=ThreadStatus.INTERRUPTED,
    )
    waiting = client.get("/api/workspaces/default/threads").json()["threads"]
    waiting_activity = next(
        thread for thread in waiting if thread["thread_id"] == root.thread_id
    )["research_activity"]
    assert waiting_activity["state"] == "waiting_review"
    assert waiting_activity["action_required"] is True


def test_result_writes_milestone_and_latest_report_to_research_root(
    tmp_path: Path,
) -> None:
    workspace, service, root, active_execution, experiment = _active_research_session(
        tmp_path
    )
    report_path = workspace / "files/calculations/descriptor/scientific_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("# Descriptor result\n\nDetailed evidence.\n", encoding="utf-8")
    child_report = service.artifact_registry.register_path(
        "files/calculations/descriptor/scientific_report.md",
        thread_id=active_execution.thread_id,
        summary="Detailed research report",
    )
    launch = service.store.find_active_launch_by_thread(active_execution.thread_id)
    assert launch is not None
    graph = service.store.get_graph(root.active_research_graph_id)

    recorded = service.record_result(
        graph["graph_id"],
        ResultCreateRequest(
            expected_revision=graph["revision"],
            title="Ligand descriptor separates ee",
            summary="The low-complexity ligand descriptor improves held-out ee ranking.",
            experiment_node_id=experiment["node"]["node_id"],
            refs=[
                ResearchRefInput(
                    ref_kind="note",
                    ref_id="calculations/descriptor/scientific_report.md",
                ),
                ResearchRefInput(
                    ref_kind="thread",
                    ref_id=active_execution.thread_id,
                ),
            ],
        ),
        launch_id=launch["launch_id"],
        run_id="run_descriptor_result",
    )

    root_messages = service.thread_store.list_messages(root.thread_id)
    milestone = next(
        message
        for message in root_messages
        if message.meta.get("research_session_event") == "result"
    )
    assert recorded["node"]["title"] in milestone.parts[0].text
    artifact_parts = [part for part in milestone.parts if part.type == "artifact"]
    assert len(artifact_parts) == 1
    assert artifact_parts[0].path == "files/calculations/descriptor/scientific_report.md"
    assert artifact_parts[0].artifact_id != child_report.artifact_id
    assert {
        artifact.artifact_id
        for artifact in service.artifact_registry.list_artifacts(
            thread_id=root.thread_id
        )
    } == {artifact_parts[0].artifact_id}

    projected = service.latest_session_result(
        graph["graph_id"],
        session_root_thread_id=root.thread_id,
    )
    assert projected["title"] == "Ligand descriptor separates ee"
    assert projected["report_artifact_id"] == artifact_parts[0].artifact_id
    assert projected["report_path"] == artifact_parts[0].path

    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    threads = client.get("/api/workspaces/default/threads").json()["threads"]
    activity = next(
        thread for thread in threads if thread["thread_id"] == root.thread_id
    )["research_activity"]
    assert activity["latest_result_title"] == "Ligand descriptor separates ee"
    assert activity["latest_report_artifact_id"] == artifact_parts[0].artifact_id
    assert activity["latest_report_path"] == artifact_parts[0].path


def test_scientific_wait_is_visible_and_written_back_as_unfinished(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace, workspace_id="default")
    root = service.thread_store.create_thread(
        title="Persistent wait visibility",
        entrypoint="persistent_research",
    )
    graph = service.create_graph(
        GraphCreateRequest(
            question="Which bounded descriptor experiment should run next?",
            orchestration_mode="auto",
            initial_hypotheses=[{"claim": "Descriptor A may be discriminating."}],
        ),
        session_root_thread_id=root.thread_id,
    )
    experiment = service.add_experiment(
        graph["graph"]["graph_id"],
        ExperimentCreateRequest(
            expected_revision=graph["graph"]["revision"],
            title="Test descriptor A",
            objective="Test descriptor A.",
            plan_summary="Run a bounded matched comparison.",
            decision_rule="A stable held-out improvement supports descriptor A.",
            state="ready",
            tests_hypothesis_ids=[graph["nodes"][0]["node_id"]],
        ),
    )
    revision = experiment["graph"]["revision"]
    planning, claimed = service.store.claim_planning(
        graph["graph"]["graph_id"],
        expected_revision=revision,
        session_root_thread_id=root.thread_id,
    )
    assert claimed is True
    comparison = service.thread_store.create_thread(
        title="Compare ready Experiments",
        entrypoint="research",
        parent_thread_id=root.thread_id,
        thread_role=ThreadRole.RESEARCH_COMPARISON,
        meta={"internal_kind": "research_graph_experiment_comparison"},
    )
    selection = service._initial_selection_state(
        revision=revision,
        candidate_ids=[experiment["node"]["node_id"]],
    )
    selection.update(
        {
            "status": "comparing",
            "phase": "wait_reverse",
            "first_wait_winner_id": "__wait__",
            "active_comparison": {
                "purpose": "wait_reverse",
                "candidate_a_id": "__wait__",
                "candidate_b_id": experiment["node"]["node_id"],
                "thread_id": comparison.thread_id,
                "sequence": 1,
            },
        }
    )
    service.store.set_planning_preview(
        graph["graph"]["graph_id"],
        planning["planning_id"],
        start_revision=revision,
        preview={"selection": selection},
    )

    outcome = service.record_pair_comparison(
        comparison_thread_id=comparison.thread_id,
        outcome=ResearchExperimentPairOutcomeDraft(
            outcome="a",
            reason="Every bounded route has a concrete model-domain failure.",
        ),
    )
    assert outcome["status"] == "wait"
    wait_message = next(
        message
        for message in service.thread_store.list_messages(root.thread_id)
        if message.meta.get("research_session_event") == "waiting"
    )
    assert "not completed" in wait_message.parts[0].text
    assert "concrete model-domain failure" in wait_message.parts[0].text

    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    threads = client.get("/api/workspaces/default/threads").json()["threads"]
    activity = next(
        thread for thread in threads if thread["thread_id"] == root.thread_id
    )["research_activity"]
    assert activity["state"] == "waiting"
    assert activity["current_title"] == "Waiting — research is not complete"
    assert "concrete model-domain failure" in activity["decision_summary"]


def test_root_submit_remains_a_root_turn_while_an_execution_is_active(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, service, root, active_execution, _second = _active_research_session(
        tmp_path
    )
    active_message_ids = [
        message.id
        for message in service.thread_store.list_messages(
            active_execution.thread_id
        )
    ]
    calls: list[tuple[str, str, str]] = []

    async def fake_submit(self, *, thread_id, payload):
        calls.append((thread_id, payload.entrypoint, payload.text))
        message = _append_user_message(self.store, thread_id, payload.text)
        return {
            "accepted": True,
            "queued": False,
            "thread": self.store.get_thread(thread_id),
            "message": message,
        }

    monkeypatch.setattr(ThreadAgentLoopService, "submit", fake_submit)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    response = client.post(
        f"/api/threads/{root.thread_id}/submit",
        json={
            "text": "Prioritize the ligand family with the strongest enantioselectivity.",
            "entrypoint": "persistent_research",
        },
    )
    assert response.status_code == 200
    assert response.json()["queued"] is False
    assert calls == [
        (
            root.thread_id,
            "persistent_research",
            "Prioritize the ligand family with the strongest enantioselectivity.",
        )
    ]
    recovered_root = service.thread_store.get_thread(root.thread_id)
    assert recovered_root.status is ThreadStatus.IDLE
    root_message = service.thread_store.list_messages(root.thread_id)[-1]
    assert root_message.role == "user"
    assert root_message.meta.get("kind") != "research_session_steering"
    assert [
        message.id
        for message in service.thread_store.list_messages(
            active_execution.thread_id
        )
    ] == active_message_ids


def test_claimed_launch_before_child_materialization_keeps_root_projection_running(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    service = ResearchGraphService(workspace=workspace, workspace_id="default")
    root = service.thread_store.create_thread(
        title="Persistent root awaiting child materialization",
        entrypoint="persistent_research",
    )
    graph = service.create_graph(
        GraphCreateRequest(
            question="Which descriptor should be measured first?",
            orchestration_mode="auto",
            initial_hypotheses=[{"claim": "Descriptor A is discriminating."}],
        ),
        session_root_thread_id=root.thread_id,
    )
    experiment = service.add_experiment(
        graph["graph"]["graph_id"],
        ExperimentCreateRequest(
            expected_revision=graph["graph"]["revision"],
            title="Measure descriptor A",
            objective="Measure descriptor A.",
            plan_summary="Compute descriptor A for the matched set.",
            decision_rule="A stable trend supports the hypothesis.",
            state="ready",
            tests_hypothesis_ids=[graph["nodes"][0]["node_id"]],
        ),
    )
    launch, claimed = service.store.claim_launch(
        graph["graph"]["graph_id"],
        experiment["node"]["node_id"],
        expected_revision=experiment["graph"]["revision"],
        replicate=False,
        lease_owner="projection-test",
        session_root_thread_id=root.thread_id,
    )
    assert claimed is True
    assert launch["thread_id"] == ""

    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    response = client.get("/api/workspaces/default/threads")
    assert response.status_code == 200
    projected_root = next(
        thread
        for thread in response.json()["threads"]
        if thread["thread_id"] == root.thread_id
    )
    assert projected_root["research_activity"]["state"] == "running"
    assert projected_root["research_activity"]["active_child_thread_id"] == ""
    assert projected_root["research_activity"]["current_title"] == "Measure descriptor A"


def test_root_submit_stays_on_the_root_when_an_execution_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace_path, service, root, active_execution, _second = (
        _active_research_session(tmp_path)
    )
    service.thread_store.update_thread(
        active_execution.thread_id,
        status=ThreadStatus.INTERRUPTED,
    )
    calls: list[str] = []

    async def fake_submit(self, *, thread_id, payload):
        calls.append(thread_id)
        message = _append_user_message(self.store, thread_id, payload.text)
        return {
            "accepted": True,
            "queued": False,
            "thread": self.store.get_thread(thread_id),
            "message": message,
        }

    monkeypatch.setattr(ThreadAgentLoopService, "submit", fake_submit)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    response = client.post(
        f"/api/threads/{root.thread_id}/submit",
        json={"text": "Do not use the racemic reference as training evidence."},
    )
    assert response.status_code == 200
    assert calls == [root.thread_id]
    root_message = service.thread_store.list_messages(root.thread_id)[-1]
    assert root_message.role == "user"
    assert root_message.meta.get("kind") != "research_session_steering"


def test_historical_planning_comparison_and_execution_children_share_the_explicit_root(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    submissions: dict[str, str] = {}

    class _Loop:
        async def submit(self, *, thread_id, payload):
            submissions[thread_id] = payload.text
            return {}

    service = ResearchGraphService(
        workspace=workspace,
        workspace_id="default",
        agent_loop_factory=lambda _workspace, _workspace_id: _Loop(),
    )
    root = service.thread_store.create_thread(
        title="Persistent research root",
        entrypoint="persistent_research",
    )
    graph = service.create_graph(
        GraphCreateRequest(
            question="Which of two descriptor tests is discriminating?",
            initial_hypotheses=[{"claim": "One descriptor is discriminating."}],
        ),
        session_root_thread_id=root.thread_id,
    )
    hypothesis_id = graph["nodes"][0]["node_id"]
    first = service.add_experiment(
        graph["graph"]["graph_id"],
        ExperimentCreateRequest(
            expected_revision=graph["graph"]["revision"],
            objective="Measure descriptor one.",
            plan_summary="Compute descriptor one.",
            decision_rule="A robust trend supports descriptor one.",
            state="ready",
            tests_hypothesis_ids=[hypothesis_id],
        ),
    )
    second = service.add_experiment(
        graph["graph"]["graph_id"],
        ExperimentCreateRequest(
            expected_revision=first["graph"]["revision"],
            objective="Measure descriptor two.",
            plan_summary="Compute descriptor two.",
            decision_rule="A robust trend supports descriptor two.",
            state="ready",
            tests_hypothesis_ids=[hypothesis_id],
        ),
    )
    _started, planning_thread = asyncio.run(
        service._launch_planning_child(
            graph["graph"]["graph_id"],
            revision=second["graph"]["revision"],
            session_root_thread_id=root.thread_id,
        )
    )
    planning = service.store.find_planning_by_thread(planning_thread.thread_id)
    assert planning is not None
    selection = service._initial_selection_state(
        revision=second["graph"]["revision"],
        candidate_ids=[first["node"]["node_id"], second["node"]["node_id"]],
        initial_incumbent_id=first["node"]["node_id"],
    )
    service.store.set_planning_preview(
        graph["graph"]["graph_id"],
        planning["planning_id"],
        start_revision=planning["revision"],
        preview={"selection": selection},
    )
    planning = service.store.get_planning(
        graph["graph"]["graph_id"], planning["planning_id"]
    )
    started, comparison_thread = asyncio.run(
        service._launch_pair_comparison_child(
            graph_id=graph["graph"]["graph_id"],
            planning=planning,
        )
    )
    assert started is True
    assert comparison_thread is not None

    launched = asyncio.run(
        service.launch_experiment(
            graph["graph"]["graph_id"],
            first["node"]["node_id"],
            expected_revision=second["graph"]["revision"],
            session_root_thread_id=root.thread_id,
        )
    )
    execution_thread = launched["thread"]

    assert planning_thread.thread_role is ThreadRole.RESEARCH_PLANNING
    assert comparison_thread.thread_role is ThreadRole.RESEARCH_COMPARISON
    assert execution_thread.thread_role is ThreadRole.RESEARCH_EXECUTION
    assert {
        planning_thread.parent_thread_id,
        comparison_thread.parent_thread_id,
        execution_thread.parent_thread_id,
    } == {root.thread_id}
    assert planning["session_root_thread_id"] == root.thread_id
    assert service.store.get_launch(launched["launch"]["launch_id"])[
        "session_root_thread_id"
    ] == root.thread_id
    assert "Use the matched solvent model" not in submissions[execution_thread.thread_id]
