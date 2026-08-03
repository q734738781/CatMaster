from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from catmaster.research.knowledge_graph.models import (
    ExperimentCreateRequest,
    GraphCreateRequest,
)
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.tools.base import ensure_project_space_layout
from catmaster.research.knowledge_graph.store import ResearchGraphStore
from catmaster.webui.agent_loop import ThreadAgentLoopService
from catmaster.webui.server import create_app
from catmaster.webui.thread_models import ThreadMessage
from catmaster.webui.thread_store import ThreadStore


def _register(client: TestClient, username: str) -> None:
    captcha = client.get("/api/auth/captcha").json()
    numbers = [int(value) for value in re.findall(r"\d+", captcha["question"])]
    response = client.post(
        "/api/auth/register",
        json={
            "username": username,
            "password": "correct-password-123",
            "captcha_id": captcha["captcha_id"],
            "captcha_answer": str(sum(numbers)),
        },
    )
    assert response.status_code == 200


def test_workspace_first_graph_api_and_cross_thread_binding(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path / "default", create=True)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    thread_a = client.post(
        "/api/workspaces/default/threads",
        json={"title": "Thread A", "entrypoint": "research"},
    ).json()["thread"]
    thread_b = client.post(
        "/api/workspaces/default/threads",
        json={"title": "Thread B", "entrypoint": "research"},
    ).json()["thread"]
    created = client.post(
        "/api/workspaces/default/research-graphs",
        json={
            "question": "Which mechanism controls selectivity?",
            "title": "Selectivity mechanism",
            "completion_criterion": (
                "A discriminating Result with a source resolves the selectivity mechanism."
            ),
            "initial_hypotheses": [
                {
                    "claim": "Pathway A controls selectivity.",
                    "rationale": "A has the lower barrier.",
                    "predictions": ["The A marker correlates with selectivity."],
                }
            ],
        },
    )
    assert created.status_code == 200
    graph = created.json()
    graph_id = graph["graph"]["graph_id"]
    assert graph["graph"]["completion_criterion"].startswith(
        "A discriminating Result"
    )
    node_id = graph["nodes"][0]["node_id"]

    for thread in (thread_a, thread_b):
        bound = client.put(
            f"/api/threads/{thread['thread_id']}/active-research-graph",
            json={"graph_id": graph_id, "focus_node_id": node_id},
        )
        assert bound.status_code == 200
        assert bound.json()["thread"]["active_research_graph_id"] == graph_id

    ThreadStore(
        workspace=tmp_path / "default",
        workspace_id="default",
    ).update_thread(thread_a["thread_id"], workspace_id="workspace-before-rename")
    catalog = client.get(
        "/api/workspaces/default/research-graphs",
        params={"include_archived": "true", "thread_id": thread_a["thread_id"]},
    )
    assert catalog.status_code == 200
    assert catalog.json()["graphs"][0]["graph_id"] == graph_id

    catalog = client.get(
        "/api/workspaces/default/research-graphs",
        params={"thread_id": thread_b["thread_id"]},
    )
    assert catalog.status_code == 200
    assert catalog.json()["graphs"][0]["bound_thread_count"] == 2
    assert catalog.json()["graphs"][0]["bound_to_current_thread"] is True

    stale_revision = graph["graph"]["revision"]
    added = client.post(
        f"/api/workspaces/default/research-graphs/{graph_id}/hypotheses",
        json={
            "expected_revision": stale_revision,
            "claim": "Pathway B controls selectivity.",
            "rationale": "B has a distinct marker.",
            "predictions": ["The B marker correlates with selectivity."],
        },
    )
    assert added.status_code == 200
    stale = client.post(
        f"/api/workspaces/default/research-graphs/{graph_id}/hypotheses",
        json={
            "expected_revision": stale_revision,
            "claim": "Stale hypothesis.",
            "rationale": "Stale.",
            "predictions": [],
        },
    )
    assert stale.status_code == 409
    assert "changed in another thread" in stale.json()["detail"]["message"]

    assert client.get(
        f"/api/threads/{thread_a['thread_id']}/hypothesis-engine"
    ).status_code == 404


def test_graph_api_rejects_cross_workspace_refs_even_with_known_path(
    tmp_path: Path,
) -> None:
    ensure_project_space_layout(tmp_path / "default", create=True)
    ensure_project_space_layout(tmp_path / "other", create=True)
    (tmp_path / "other" / "files" / "foreign.md").write_text(
        "foreign",
        encoding="utf-8",
    )
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    graph = client.post(
        "/api/workspaces/default/research-graphs",
        json={"question": "Question?", "initial_hypotheses": [{"claim": "Claim"}]},
    ).json()
    response = client.post(
        f"/api/workspaces/default/research-graphs/{graph['graph']['graph_id']}/refs",
        json={
            "expected_revision": graph["graph"]["revision"],
            "node_id": graph["nodes"][0]["node_id"],
            "ref_kind": "note",
            "ref_id": "../other/files/foreign.md",
        },
    )
    assert response.status_code == 409
    assert "workspace" in response.json()["detail"]


def test_graph_api_accepts_a_sourced_observation_without_an_experiment(
    tmp_path: Path,
) -> None:
    ensure_project_space_layout(tmp_path / "default", create=True)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    graph = client.post(
        "/api/workspaces/default/research-graphs",
        json={
            "question": "What explains the new operando feature?",
            "initial_hypotheses": [{"claim": "The feature is interfacial."}],
        },
    ).json()
    hypothesis_id = graph["nodes"][0]["node_id"]

    response = client.post(
        (
            "/api/workspaces/default/research-graphs/"
            f"{graph['graph']['graph_id']}/results"
        ),
        json={
            "expected_revision": graph["graph"]["revision"],
            "title": "Collaboration result",
            "summary": "A reversible band appeared only under the reactant feed.",
            "judgments": [
                {
                    "hypothesis_node_id": hypothesis_id,
                    "relation": "inconclusive",
                }
            ],
            "refs": [
                {
                    "ref_kind": "url",
                    "ref_id": "https://example.org/shared-result",
                }
            ],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    result = next(node for node in payload["nodes"] if node["kind"] == "result")
    assert result["body"]["summary"].startswith("A reversible band")
    assert result["refs"][0]["ref_id"] == "https://example.org/shared-result"
    assert not any(
        edge["relation"] == "produces"
        and edge["target_node_id"] == result["node_id"]
        for edge in payload["edges"]
    )

    replaced = client.put(
        (
            "/api/workspaces/default/research-graphs/"
            f"{graph['graph']['graph_id']}/results/{result['node_id']}"
            f"/judgments/{hypothesis_id}"
        ),
        json={
            "expected_revision": payload["graph"]["revision"],
            "relation": "opposes",
        },
    )
    assert replaced.status_code == 200
    assert {
        edge["relation"]
        for edge in replaced.json()["edges"]
        if edge["source_node_id"] == result["node_id"]
        and edge["target_node_id"] == hypothesis_id
    } == {"opposes"}

    cleared = client.put(
        (
            "/api/workspaces/default/research-graphs/"
            f"{graph['graph']['graph_id']}/results/{result['node_id']}"
            f"/judgments/{hypothesis_id}"
        ),
        json={
            "expected_revision": replaced.json()["graph"]["revision"],
            "relation": "unjudged",
        },
    )
    assert cleared.status_code == 200
    assert not any(
        edge["source_node_id"] == result["node_id"]
        and edge["target_node_id"] == hypothesis_id
        and edge["relation"] in {"supports", "opposes", "inconclusive"}
        for edge in cleared.json()["edges"]
    )


def test_internal_research_planning_threads_are_hidden_from_user_lists(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    visible = client.post(
        "/api/workspaces/default/threads",
        json={"title": "Visible research thread", "entrypoint": "research"},
    ).json()["thread"]
    store = ThreadStore(workspace=workspace, workspace_id="default")
    marked = store.create_thread(
        thread_id="thread_rg_marked",
        title="Plan next step: marked",
        entrypoint="research",
        meta={"internal_kind": "research_graph_planning"},
    )
    legacy = store.create_thread(
        thread_id="thread_rg_legacy",
        title="Plan next step: legacy",
        entrypoint="research",
    )

    listed = client.get("/api/workspaces/default/threads")
    assert listed.status_code == 200
    assert [thread["thread_id"] for thread in listed.json()["threads"]] == [
        visible["thread_id"]
    ]
    assert client.get(f"/api/threads/{marked.thread_id}").status_code == 200

    graph = ResearchGraphService(
        workspace=workspace,
        workspace_id="default",
    ).create_graph(GraphCreateRequest(question="Which route should be tested?"))
    graph_id = graph["graph"]["graph_id"]
    for thread_id in (visible["thread_id"], marked.thread_id, legacy.thread_id):
        store.update_thread(thread_id, active_research_graph_id=graph_id)
    catalog = ResearchGraphService(
        workspace=workspace,
        workspace_id="default",
    ).catalog()
    assert catalog[0]["bound_thread_count"] == 1


def test_graph_plan_response_serializes_its_internal_thread(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ensure_project_space_layout(tmp_path / "default", create=True)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    graph = client.post(
        "/api/workspaces/default/research-graphs",
        json={"question": "Which literature-grounded route should be tested?"},
    ).json()

    async def _fake_plan_next_step(
        service: ResearchGraphService,
        graph_id: str,
        *,
        expected_revision: int,
        focus_node_id: str = "",
    ) -> dict:
        thread = service.thread_store.create_thread(
            thread_id="thread_rg_serialization",
            title="Plan next step",
            entrypoint="research",
            meta={"internal_kind": "research_graph_planning"},
        )
        return {
            "accepted": True,
            "deduplicated": False,
            "thread": thread,
            **service.presentation(graph_id),
        }

    monkeypatch.setattr(
        ResearchGraphService,
        "plan_next_step",
        _fake_plan_next_step,
    )
    response = client.post(
        (
            "/api/workspaces/default/research-graphs/"
            f"{graph['graph']['graph_id']}/plan"
        ),
        json={"expected_revision": graph["graph"]["revision"]},
    )

    assert response.status_code == 200
    assert response.json()["thread"]["thread_id"] == "thread_rg_serialization"


def test_graph_context_api_returns_an_explicit_partial_focus_snippet(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path / "default", create=True)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    graph = client.post(
        "/api/workspaces/default/research-graphs",
        json={"question": "Question?", "initial_hypotheses": [{"claim": "Claim"}]},
    ).json()
    response = client.post(
        f"/api/workspaces/default/research-graphs/{graph['graph']['graph_id']}/context",
        json={"focus_node_id": graph["nodes"][0]["node_id"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["markdown"].startswith("# Active Research Graph")
    assert payload["presentation"]["partial"] is True
    assert payload["presentation"]["total_count"] == 1
    assert "Completion criterion:" in payload["markdown"]
    assert "query_research_graph_sql" in payload["markdown"]
    assert "body_json" not in response.text


def test_identical_graph_ids_are_isolated_between_authenticated_users(
    tmp_path: Path,
) -> None:
    app = create_app(project_space_root=str(tmp_path))
    alice = TestClient(app)
    bob = TestClient(app)
    _register(alice, "alice_graph")
    _register(bob, "bob_graph")
    assert alice.get("/api/bootstrap").status_code == 200
    assert bob.get("/api/bootstrap").status_code == 200

    alice_graph = alice.post(
        "/api/workspaces/default/research-graphs",
        json={"question": "Alice private question?"},
    ).json()
    graph_id = alice_graph["graph"]["graph_id"]
    bob_workspace = tmp_path / "users" / "bob_graph" / "default"
    ResearchGraphStore(bob_workspace).create_graph(
        graph_id=graph_id,
        title="Bob graph",
        question="Bob private question?",
    )

    alice_view = alice.get(
        f"/api/workspaces/default/research-graphs/{graph_id}"
    )
    bob_view = bob.get(f"/api/workspaces/default/research-graphs/{graph_id}")
    assert alice_view.status_code == 200
    assert bob_view.status_code == 200
    assert alice_view.json()["graph"]["question"] == "Alice private question?"
    assert bob_view.json()["graph"]["question"] == "Bob private question?"


def test_bound_research_execution_and_writing_turns_get_graph_context(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    graph = ResearchGraphService(workspace=workspace).create_graph(
        GraphCreateRequest(
            question="Which pathway controls selectivity?",
            initial_hypotheses=[
                {
                    "claim": "Pathway A controls selectivity.",
                    "rationale": "A has the lower barrier.",
                    "predictions": ["The A marker tracks selectivity."],
                }
            ],
        )
    )
    loop = object.__new__(ThreadAgentLoopService)
    loop.workspace = workspace
    research_context = {
        "research_graph_id": graph["graph"]["graph_id"],
        "research_focus_node_id": graph["nodes"][0]["node_id"],
        "research_launch_id": "",
    }

    injected = loop._research_graph_turn_content(
        prompt="What should we test next?",
        turn_content="What should we test next?",
        entrypoint="research",
        research_context=research_context,
    )
    assert isinstance(injected, str)
    assert injected.startswith("# Active Research Graph")
    assert "Which pathway controls selectivity?" in injected
    assert "Pathway A controls selectivity." in injected
    assert injected.endswith("# Current user request\nWhat should we test next?")
    assert not injected.lstrip().startswith("{")

    for entrypoint in ("experiment", "literature_review", "writing"):
        lane_injected = loop._research_graph_turn_content(
            prompt="Use the bound scientific context for this task.",
            turn_content="Use the bound scientific context for this task.",
            entrypoint=entrypoint,
            research_context=research_context,
        )
        assert isinstance(lane_injected, str)
        assert lane_injected.startswith("# Active Research Graph")
        assert graph["graph"]["graph_id"] in lane_injected
        assert graph["nodes"][0]["node_id"] in lane_injected

    unchanged = loop._research_graph_turn_content(
        prompt="Write this up.",
        turn_content="Write this up.",
        entrypoint="writing",
        research_context={
            "research_graph_id": "",
            "research_focus_node_id": "",
            "research_launch_id": "",
        },
    )
    assert unchanged == "Write this up."


def test_turn_snapshot_pins_a_matching_active_launch_and_resume_reuses_it(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    service = ResearchGraphService(workspace=workspace, workspace_id="default")
    graph = service.create_graph(
        GraphCreateRequest(
            question="Which pathway controls selectivity?",
            initial_hypotheses=[{"claim": "Pathway A controls selectivity."}],
        )
    )
    experiment = service.add_experiment(
        graph["graph"]["graph_id"],
        ExperimentCreateRequest(
            expected_revision=graph["graph"]["revision"],
            objective="Test pathway A.",
            plan_summary="Compare the pathway marker under matched conditions.",
            decision_rule="A reproducible marker change supports pathway A.",
            execution_lane="experiment",
            state="ready",
        ),
    )
    thread = service.thread_store.create_thread(
        title="Long-lived experiment thread",
        entrypoint="experiment",
    )
    thread = service.thread_store.update_thread(
        thread.thread_id,
        active_research_graph_id=graph["graph"]["graph_id"],
        research_focus_node_id=experiment["node"]["node_id"],
    )
    launch, claimed = service.store.claim_launch(
        graph["graph"]["graph_id"],
        experiment["node"]["node_id"],
        expected_revision=experiment["graph"]["revision"],
        replicate=False,
        lease_owner="test-worker",
    )
    assert claimed is True
    service.store.update_launch(
        launch["launch_id"],
        status="running",
        thread_id=thread.thread_id,
    )
    presented_experiment = next(
        node
        for node in service.presentation(graph["graph"]["graph_id"])["nodes"]
        if node["node_id"] == experiment["node"]["node_id"]
    )
    assert presented_experiment["active_launch"]["activity"] == (
        "waiting_continue"
    )
    loop = object.__new__(ThreadAgentLoopService)
    loop.workspace = workspace
    loop.store = service.thread_store

    snapshot = loop._research_turn_context(
        thread=thread,
        entrypoint="experiment",
    )
    assert snapshot == {
        "research_graph_id": graph["graph"]["graph_id"],
        "research_focus_node_id": experiment["node"]["node_id"],
        "research_launch_id": launch["launch_id"],
    }
    service.thread_store.append_message(
        ThreadMessage(
            id="msg_interrupted_snapshot",
            thread_id=thread.thread_id,
            role="assistant",
            status="interrupted",
            meta={"entrypoint": "experiment", **snapshot},
        )
    )
    service.thread_store.update_thread(thread.thread_id, entrypoint="writing")
    assert loop._resume_research_context(thread.thread_id) == (
        "experiment",
        snapshot,
    )

    changed_thread = service.thread_store.update_thread(
        thread.thread_id,
        research_focus_node_id=graph["nodes"][0]["node_id"],
    )
    assert loop._research_turn_context(
        thread=changed_thread,
        entrypoint="experiment",
        inherited=snapshot,
    ) == snapshot

    service.store.update_launch(
        launch["launch_id"],
        status="completed",
        thread_id=thread.thread_id,
    )
    terminal_resume = loop._research_turn_context(
        thread=changed_thread,
        entrypoint="experiment",
        inherited=snapshot,
    )
    assert terminal_resume["research_launch_id"] == ""


def test_internal_planning_turn_uses_the_same_partial_focus_contract(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    graph = ResearchGraphService(workspace=workspace).create_graph(
        GraphCreateRequest(
            question="Which mechanism should guide the next experiment?",
            initial_hypotheses=[
                {"claim": f"Mechanism {index:02d} controls the response."}
                for index in range(30)
            ],
        )
    )
    loop = object.__new__(ThreadAgentLoopService)
    loop.workspace = workspace
    focus_node = next(
        node
        for node in graph["nodes"]
        if node["title"] == "Mechanism 00 controls the response."
    )
    research_context = {
        "research_graph_id": graph["graph"]["graph_id"],
        "research_focus_node_id": focus_node["node_id"],
        "research_launch_id": "",
    }
    ordinary = loop._research_graph_turn_content(
        prompt="What should we test next?",
        turn_content="What should we test next?",
        entrypoint="research",
        research_context=research_context,
    )
    planning = loop._research_graph_turn_content(
        prompt="Re-evaluate the runnable routes from current evidence.",
        turn_content="Re-evaluate the runnable routes from current evidence.",
        entrypoint="research",
        research_context=research_context,
    )

    assert isinstance(ordinary, str)
    assert isinstance(planning, str)
    assert "Mechanism 29 controls the response." not in ordinary
    assert "Mechanism 29 controls the response." not in planning
    assert "explicitly partial" in planning
    assert "query_research_graph_sql" in planning
