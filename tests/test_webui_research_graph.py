from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from catmaster.research.knowledge_graph.models import GraphCreateRequest
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.tools.base import ensure_project_space_layout
from catmaster.research.knowledge_graph.store import ResearchGraphStore
from catmaster.webui.agent_loop import ThreadAgentLoopService
from catmaster.webui.server import create_app


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
    node_id = graph["nodes"][0]["node_id"]

    for thread in (thread_a, thread_b):
        bound = client.put(
            f"/api/threads/{thread['thread_id']}/active-research-graph",
            json={"graph_id": graph_id, "focus_node_id": node_id},
        )
        assert bound.status_code == 200
        assert bound.json()["thread"]["active_research_graph_id"] == graph_id

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


def test_graph_context_api_is_human_readable_and_bounded(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path / "default", create=True)
    client = TestClient(create_app(project_space_root=str(tmp_path), no_login=True))
    graph = client.post(
        "/api/workspaces/default/research-graphs",
        json={"question": "Question?", "initial_hypotheses": [{"claim": "Claim"}]},
    ).json()
    response = client.post(
        f"/api/workspaces/default/research-graphs/{graph['graph']['graph_id']}/context",
        json={"query": "Claim", "max_nodes": 4, "max_chars": 2_000},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["markdown"].startswith("# Active Research Graph")
    assert len(payload["markdown"]) <= 2_000
    assert payload["presentation"]["total_count"] == 1
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


def test_bound_research_and_execution_turns_get_readable_bounded_graph_context(
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
    thread = SimpleNamespace(
        active_research_graph_id=graph["graph"]["graph_id"],
        research_focus_node_id=graph["nodes"][0]["node_id"],
    )

    injected = loop._research_graph_turn_content(
        thread=thread,
        prompt="What should we test next?",
        turn_content="What should we test next?",
        entrypoint="research",
    )
    assert isinstance(injected, str)
    assert injected.startswith("# Active Research Graph")
    assert "Which pathway controls selectivity?" in injected
    assert "Pathway A controls selectivity." in injected
    assert injected.endswith("# Current user request\nWhat should we test next?")
    assert not injected.lstrip().startswith("{")

    for entrypoint in ("experiment", "literature_review"):
        execution_injected = loop._research_graph_turn_content(
            thread=thread,
            prompt="Execute the bound proposal and record its outcome.",
            turn_content="Execute the bound proposal and record its outcome.",
            entrypoint=entrypoint,
        )
        assert isinstance(execution_injected, str)
        assert execution_injected.startswith("# Active Research Graph")
        assert graph["graph"]["graph_id"] in execution_injected
        assert graph["nodes"][0]["node_id"] in execution_injected

    unchanged = loop._research_graph_turn_content(
        thread=thread,
        prompt="Write this up.",
        turn_content="Write this up.",
        entrypoint="writing",
    )
    assert unchanged == "Write this up."
