from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from catmaster.research.hypothesis_engine import (
    EvidenceEffect,
    EvidenceJudgment,
    ExecutionLane,
    Hypothesis,
    HypothesisEngine,
    HypothesisEngineState,
    VerificationAction,
)
from catmaster.research.hypothesis_engine.storage import (
    engine_path,
    load_engine,
    save_engine,
)
from catmaster.tools.base import ensure_project_space_layout
from catmaster.webui.research_map_service import (
    ACTION_ID_META,
    AUTOPILOT_ENABLED_META,
    CAMPAIGN_ID_META,
    LAUNCH_MODE_META,
    SOURCE_THREAD_ID_META,
    ResearchMapService,
)
from catmaster.webui.server import create_app
from catmaster.webui.thread_models import MessagePart, ThreadMessage, ThreadStatus
from catmaster.webui.thread_store import ThreadStore, new_id


def _hypothesis(hypothesis_id: str, *, parent: str = "") -> Hypothesis:
    return Hypothesis(
        id=hypothesis_id,
        claim=f"Scientific claim {hypothesis_id}.",
        rationale=f"Scientific rationale {hypothesis_id}.",
        predictions=[f"Discriminating prediction {hypothesis_id}."],
        derived_from=[parent] if parent else [],
    )


def _action(
    action_id: str,
    *,
    targets: list[str],
    prerequisites: list[str] | None = None,
    cost: str = "low",
) -> VerificationAction:
    return VerificationAction(
        id=action_id,
        executor=ExecutionLane.LITERATURE,
        question=f"Check {action_id}.",
        task=f"Search primary evidence for {action_id}.",
        target_hypotheses=targets,
        decision_rule="A matching prediction supports; a conflict opposes; otherwise inconclusive.",
        prerequisite_action_ids=prerequisites or [],
        information_value="high",
        cost=cost,
    )


def _create_research_thread(client: TestClient) -> dict:
    created = client.post(
        "/api/workspaces/default/threads",
        json={"title": "Hypothesis map", "entrypoint": "research"},
    )
    assert created.status_code == 200
    return created.json()["thread"]


def test_thread_endpoint_exposes_lean_operational_controller(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    client = TestClient(
        create_app(project_space_root=str(tmp_path), no_login=True)
    )
    thread = _create_research_thread(client)
    thread_id = thread["thread_id"]

    missing = client.get(f"/api/threads/{thread_id}/hypothesis-engine")
    assert missing.status_code == 200
    assert missing.json()["available"] is False

    engine = HypothesisEngine(
        HypothesisEngineState(
            question="Which explanation survives the source check?",
            hypotheses=[_hypothesis("h1"), _hypothesis("h2")],
            actions=[
                _action("source-check", targets=["h1", "h2"]),
                _action(
                    "follow-up",
                    targets=["h1", "h2"],
                    prerequisites=["source-check"],
                ),
            ],
        )
    )
    first_packet = engine.advance("source-check")
    assert first_packet is not None
    save_engine(workspace / "files", thread["deepagent_thread_id"], engine)

    response = client.get(f"/api/threads/{thread_id}/hypothesis-engine")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available"] is True
    assert payload["controller"]["status"] == "execution_required"
    assert payload["controller"]["active_packet"]["action_id"] == "source-check"
    assert payload["controller"]["active_packet"]["hypotheses"][0]["predictions"]
    assert payload["graph"]["controller"] == payload["controller"]
    assert payload["automation"]["enabled"] is False
    assert payload["automation"]["status"] == "off"
    assert {node["kind"] for node in payload["graph"]["nodes"]} == {
        "hypothesis",
        "action",
    }
    assert {"tested_by", "unlocks"}.issubset(
        {edge["kind"] for edge in payload["graph"]["edges"]}
    )
    assert "runs" not in payload["state"]
    assert "budget" not in payload["state"]


def test_thread_endpoint_projects_proposer_revision_and_judged_evidence(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    client = TestClient(
        create_app(project_space_root=str(tmp_path), no_login=True)
    )
    thread = _create_research_thread(client)
    engine = HypothesisEngine(
        HypothesisEngineState(
            question="Can evidence motivate a separately proposed branch?",
            hypotheses=[_hypothesis("h1"), _hypothesis("h2")],
            actions=[_action("source-check", targets=["h1", "h2"])],
        )
    )
    engine.advance("source-check")
    engine.record_result(
        "source-check",
        outcome="completed",
        judgment=EvidenceJudgment(
            action_id="source-check",
            summary="The source favors h1 and leaves h2 unresolved.",
            source="doi:10.1021/example",
            effects=[
                EvidenceEffect(
                    hypothesis_id="h1",
                    verdict="supports",
                    reason="The prediction is observed.",
                ),
                EvidenceEffect(
                    hypothesis_id="h2",
                    verdict="inconclusive",
                    reason="The source does not test the alternative directly.",
                ),
            ],
        ),
    )
    engine.extend(
        hypotheses=[_hypothesis("h3", parent="h1")],
        actions=[
            _action(
                "derived-check",
                targets=["h2", "h3"],
                prerequisites=["source-check"],
            )
        ],
    )
    save_engine(
        workspace / "files",
        thread["deepagent_thread_id"],
        engine,
    )

    response = client.get(
        f"/api/threads/{thread['thread_id']}/hypothesis-engine"
    )

    assert response.status_code == 200
    graph = response.json()["graph"]
    node_ids = {node["id"] for node in graph["nodes"]}
    assert {
        "hypothesis:h3",
        "action:derived-check",
        "evidence:source-check",
    }.issubset(node_ids)
    assert {"derives", "unlocks", "produces", "supports", "inconclusive"}.issubset(
        {edge["kind"] for edge in graph["edges"]}
    )
    assert not any(node["kind"] == "run" for node in graph["nodes"])


def test_thread_endpoint_is_scoped_and_reports_invalid_persisted_state(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    other_workspace = tmp_path / "other"
    ensure_project_space_layout(other_workspace, create=True)
    client = TestClient(
        create_app(project_space_root=str(tmp_path), no_login=True)
    )
    thread = _create_research_thread(client)

    path = engine_path(
        workspace / "files",
        thread["deepagent_thread_id"],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not-json", encoding="utf-8")

    invalid = client.get(
        f"/api/threads/{thread['thread_id']}/hypothesis-engine"
    )
    assert invalid.status_code == 409
    assert "state is invalid" in invalid.json()["detail"]

    unknown = client.get("/api/threads/not-a-thread/hypothesis-engine")
    assert unknown.status_code == 404


def test_autopilot_routes_toggle_operational_metadata_not_scientific_state(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    client = TestClient(
        create_app(project_space_root=str(tmp_path), no_login=True)
    )
    thread = _create_research_thread(client)
    engine = HypothesisEngine(
        HypothesisEngineState(
            question="Which explanation survives?",
            hypotheses=[_hypothesis("h1"), _hypothesis("h2")],
            actions=[_action("source-check", targets=["h1", "h2"])],
        )
    )
    save_engine(workspace / "files", thread["deepagent_thread_id"], engine)

    started = client.post(
        f"/api/threads/{thread['thread_id']}/hypothesis-engine/autopilot/start",
        json={},
    )

    assert started.status_code == 200
    assert started.json()["automation"]["enabled"] is True
    persisted = load_engine(workspace / "files", thread["deepagent_thread_id"])
    assert "mode" not in persisted.state.model_dump()
    assert "autopilot" not in persisted.state.model_dump()
    thread_record = ThreadStore(
        workspace=workspace,
        workspace_id="default",
    ).get_thread(thread["thread_id"])
    assert thread_record.meta[AUTOPILOT_ENABLED_META] is True

    stopped = client.post(
        f"/api/threads/{thread['thread_id']}/hypothesis-engine/autopilot/stop",
        json={},
    )
    assert stopped.status_code == 200
    assert stopped.json()["automation"]["enabled"] is False


class _FakeBroker:
    def emit(self, *_args, **_kwargs):
        return None


class _FakeResearchLoop:
    def __init__(self, store: ThreadStore) -> None:
        self.store = store
        self.broker = _FakeBroker()
        self.submissions: list[tuple[str, object]] = []

    async def submit(self, *, thread_id: str, payload):
        self.submissions.append((thread_id, payload))
        message = ThreadMessage(
            id=new_id("msg"),
            thread_id=thread_id,
            role="user",
            status="completed",
            parts=[
                MessagePart(
                    id=new_id("part_text"),
                    type="text",
                    text=payload.text,
                    status="completed",
                )
            ],
        )
        self.store.append_message(message)
        thread = self.store.update_thread(thread_id, status=ThreadStatus.RUNNING)
        return {"thread": thread, "message": message}


def test_manual_map_launch_reserves_action_and_starts_an_isolated_research_thread(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    source = store.create_thread(
        title="Source Research",
        entrypoint="research",
        meta={"permission_mode": "hitl"},
    )
    engine = HypothesisEngine(
        HypothesisEngineState(
            question="Which mechanism survives?",
            hypotheses=[_hypothesis("h1"), _hypothesis("h2")],
            actions=[
                _action("decisive", targets=["h1", "h2"], cost="high")
            ],
        )
    )
    save_engine(workspace / "files", source.deepagent_thread_id, engine)
    loop = _FakeResearchLoop(store)
    service = ResearchMapService(agent_loop_factory=lambda *_args: loop)

    result = asyncio.run(
        service.launch_action(
            workspace=workspace,
            workspace_id="default",
            source_thread_id=source.thread_id,
            action_id="decisive",
            expected_revision=0,
            launch_mode="manual",
        )
    )

    child = store.get_thread(result["thread"]["thread_id"])
    child_meta = child.meta
    assert child.thread_id != source.thread_id
    assert child.deepagent_thread_id != source.deepagent_thread_id
    assert child_meta[CAMPAIGN_ID_META] == source.deepagent_thread_id
    assert child_meta[SOURCE_THREAD_ID_META] == source.thread_id
    assert child_meta[ACTION_ID_META] == "decisive"
    assert child_meta[LAUNCH_MODE_META] == "manual"
    assert child_meta["permission_mode"] == "hitl"
    assert store.list_messages(source.thread_id) == []
    submitted_prompt = loop.submissions[0][1].text
    assert "<research-map-request>" not in submitted_prompt
    assert "ordinary CatMaster Research turn" in submitted_prompt
    assert source.deepagent_thread_id in submitted_prompt
    persisted = load_engine(workspace / "files", source.deepagent_thread_id)
    assert persisted.state.active_action_id == "decisive"
    assert persisted.state.actions[0].cost.value == "high"


def test_manual_map_launch_waits_for_the_source_research_turn(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    source = store.create_thread(
        title="Busy source Research",
        entrypoint="research",
    )
    store.update_thread(source.thread_id, status=ThreadStatus.RUNNING)
    engine = HypothesisEngine(
        HypothesisEngineState(
            question="Which mechanism survives?",
            hypotheses=[_hypothesis("h1"), _hypothesis("h2")],
            actions=[_action("decisive", targets=["h1", "h2"])],
        )
    )
    save_engine(workspace / "files", source.deepagent_thread_id, engine)
    loop = _FakeResearchLoop(store)
    service = ResearchMapService(agent_loop_factory=lambda *_args: loop)

    try:
        asyncio.run(
            service.launch_action(
                workspace=workspace,
                workspace_id="default",
                source_thread_id=source.thread_id,
                action_id="decisive",
                expected_revision=0,
                launch_mode="manual",
            )
        )
    except ValueError as exc:
        assert "source Research thread is still running" in str(exc)
    else:
        raise AssertionError("Map launch should not compete with a running source turn")

    persisted = load_engine(workspace / "files", source.deepagent_thread_id)
    assert persisted.state.active_action_id == ""
    assert service.related_children(store, source.thread_id) == []


def test_async_map_worker_skips_human_action_and_launches_ranked_research_thread(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    source = store.create_thread(
        title="Automatic campaign",
        entrypoint="research",
        meta={
            "permission_mode": "auto",
            AUTOPILOT_ENABLED_META: True,
        },
    )
    engine = HypothesisEngine(
        HypothesisEngineState(
            question="Which check should run next?",
            hypotheses=[_hypothesis("h1"), _hypothesis("h2")],
            actions=[
                VerificationAction(
                    id="ask-user",
                    executor=ExecutionLane.HUMAN,
                    question="What did the user observe?",
                    task="Ask for the missing observation.",
                    target_hypotheses=["h1", "h2"],
                    decision_rule="The supplied observation distinguishes the hypotheses.",
                    information_value="high",
                    cost="low",
                ),
                _action(
                    "automatic-check",
                    targets=["h1", "h2"],
                    cost="high",
                ),
            ],
        )
    )
    save_engine(workspace / "files", source.deepagent_thread_id, engine)
    loop = _FakeResearchLoop(store)
    service = ResearchMapService(agent_loop_factory=lambda *_args: loop)

    launched = asyncio.run(
        service.tick_workspace(workspace=workspace, workspace_id="default")
    )

    assert launched == 1
    children = service.related_children(store, source.thread_id)
    assert len(children) == 1
    assert children[0].meta[ACTION_ID_META] == "automatic-check"
    assert children[0].meta[LAUNCH_MODE_META] == "auto"
    assert store.list_messages(source.thread_id) == []
    persisted = load_engine(workspace / "files", source.deepagent_thread_id)
    assert persisted.state.active_action_id == "automatic-check"
    snapshot = service.automation_snapshot(
        store=store,
        source_thread=source,
        engine=persisted,
    )
    assert snapshot["enabled"] is True
    assert snapshot["status"] == "running"

    launched_again = asyncio.run(
        service.tick_workspace(workspace=workspace, workspace_id="default")
    )
    assert launched_again == 0
    assert len(service.related_children(store, source.thread_id)) == 1


def test_human_map_child_keeps_action_active_until_user_evidence_arrives(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    source = store.create_thread(
        title="Human evidence campaign",
        entrypoint="research",
    )
    engine = HypothesisEngine(
        HypothesisEngineState(
            question="What did the user observe?",
            hypotheses=[_hypothesis("h1"), _hypothesis("h2")],
            actions=[
                VerificationAction(
                    id="ask-user",
                    executor=ExecutionLane.HUMAN,
                    question="What did the user observe?",
                    task="Ask for the missing observation.",
                    target_hypotheses=["h1", "h2"],
                    decision_rule="The supplied observation distinguishes the hypotheses.",
                    information_value="high",
                    cost="low",
                )
            ],
        )
    )
    save_engine(workspace / "files", source.deepagent_thread_id, engine)
    loop = _FakeResearchLoop(store)
    service = ResearchMapService(agent_loop_factory=lambda *_args: loop)
    result = asyncio.run(
        service.launch_action(
            workspace=workspace,
            workspace_id="default",
            source_thread_id=source.thread_id,
            action_id="ask-user",
            expected_revision=0,
            launch_mode="manual",
        )
    )

    reconciled = service.reconcile_finished_child(
        workspace=workspace,
        workspace_id="default",
        child_thread_id=result["thread"]["thread_id"],
        terminal_status="done",
    )

    assert reconciled is False
    persisted = load_engine(workspace / "files", source.deepagent_thread_id)
    assert persisted.state.active_action_id == "ask-user"
    prompt = loop.submissions[0][1].text
    assert "Ask the user for the required evidence in this thread" in prompt
