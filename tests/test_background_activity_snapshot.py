"""Activity snapshots use execution state even when saved cards disagree."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from catmaster.webui.thread_models import MessagePart, ThreadMessage
from catmaster.webui.thread_store import ThreadStore
from test_webui_thread_v2 import create_app, _workspace


def _card(store, parent, child, message_id, status, run_id="old-run"):
    store.append_message(ThreadMessage(id=message_id, thread_id=parent, role="assistant",
        status="completed", parts=[MessagePart(id="part_subagent_" + child, type="subagent",
            status=status, text="Previous run progress", meta={"task_id": child,
                "thread_id": child, "source": "writing_specialist", "run_id": run_id,
                "native_status": status, "activity_tool": "read_file"})]))


@pytest.mark.parametrize("entrypoint", ["research", "persistent_research"])
def test_imported_duplicate_cards_and_live_parallel_tasks(tmp_path, monkeypatch, entrypoint):
    workspace = _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    root = client.post("/api/workspaces/default/threads",
        json={"title": "Ceramic", "entrypoint": entrypoint}).json()["thread"]["thread_id"]
    store = ThreadStore(workspace=workspace)
    store.create_thread(thread_id="literature", parent_thread_id=root,
        entrypoint="literature_review", meta={"background_task": True,
            "saved_task_status": "interrupted", "parent_message_id": "lit-latest"})
    store.update_thread("literature", status="stopped")
    for mid, status in [("lit-old", "running"), ("lit-cancelled", "interrupted"), ("lit-latest", "interrupted")]:
        _card(store, root, "literature", mid, status)
    for tid, status in [("writer-a", "PENDING"), ("writer-b", "ENQUEUED")]:
        store.create_thread(thread_id=tid, parent_thread_id=root,
            entrypoint="writing", meta={"background_task": True, "last_run_id": tid + "-run",
                "parent_message_id": tid + "-launch", "agent_name": "writing_specialist",
                "task_description": "Rewrite " + tid, "on_completion": "notify"})
        store.update_thread(tid, status="running")
        _card(store, root, tid, tid + "-launch", "completed", tid + "-run")
        app.state.test_execution.rows[tid + "-run"] = SimpleNamespace(status=status, output=None)
    store.append_message(ThreadMessage(id="later-user", thread_id=root, role="user", parts=[]))

    # Refresh must not load a child result or any old full message to find cards.
    original_get = ThreadStore.get_message
    with monkeypatch.context() as patch:
        patch.setattr(ThreadStore, "get_message", lambda *args, **kwargs: pytest.fail("Full message loaded for Activity"))
        payload = client.get(f"/api/threads/{root}/messages").json()
    cards = {part["id"]: part for part in payload["active_parts"]}
    assert set(cards) == {"part_subagent_writer-a", "part_subagent_writer-b"}
    assert cards["part_subagent_writer-a"]["status"] == "running"
    assert cards["part_subagent_writer-b"]["status"] == "pending"
    assert cards["part_subagent_writer-a"]["task_description"] == "Rewrite writer-a"
    assert cards["part_subagent_writer-a"]["on_completion"] == "notify"
    assert cards["part_subagent_writer-a"]["text"] == "Previous run progress"
    assert cards["part_subagent_writer-a"]["fields"]
    assert client.get(f"/api/threads/{root}/async-subagents/literature").json()["part"]["status"] == "interrupted"
    if entrypoint == "persistent_research":
        thread = client.get(f"/api/threads/{root}").json()["thread"]
        assert not any(part["type"] == "subagent" for part in thread["research_activity"]["active_parts"])

    # Completion removes only the completed branch, even before its saved card
    # catches up. Re-use keeps the other thread's identity without old progress.
    app.state.test_execution.rows["writer-a-run"] = SimpleNamespace(status="SUCCESS", output={"status": "success"})
    child = store.get_thread("writer-b")
    store.update_thread("writer-b", meta={**child.meta, "last_run_id": "writer-b-next"})
    app.state.test_execution.rows["writer-b-next"] = SimpleNamespace(status="ENQUEUED", output=None)
    payload = client.get(f"/api/threads/{root}/messages").json()
    assert [part["id"] for part in payload["active_parts"]] == ["part_subagent_writer-b"]
    assert payload["active_parts"][0]["text"] == ""
    assert payload["active_parts"][0]["fields"] == []
    assert original_get(store, root, "lit-old").parts[0].status == "running"
    assert app.state.test_execution.calls == []


@pytest.mark.parametrize("status, output, expected", [
    ("PENDING", None, "running"),
    ("ENQUEUED", None, "pending"),
    ("DELAYED", None, "pending"),
    ("SUCCESS", {"status": "success"}, "completed"),
    ("SUCCESS", {"status": "interrupted"}, "interrupted"),
    ("CANCELLED", None, "interrupted"),
    ("ERROR", None, "failed"),
    (None, None, "interrupted"),
])
def test_activity_and_detail_share_status_without_a_saved_card(tmp_path, status, output, expected):
    workspace = _workspace(tmp_path)
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    root = client.post("/api/workspaces/default/threads", json={"title": "Root"}).json()["thread"]["thread_id"]
    store = ThreadStore(workspace=workspace)
    store.create_thread(thread_id="child", parent_thread_id=root,
        meta={"background_task": True, "last_run_id": "child-run", "agent_name": "writing_specialist"})
    store.update_thread("child", status="running")
    if status:
        app.state.test_execution.rows["child-run"] = SimpleNamespace(status=status, output=output)
    detail = client.get(f"/api/threads/{root}/async-subagents/child").json()["part"]
    assert detail["status"] == expected
    cards = client.get(f"/api/threads/{root}/messages").json()["active_parts"]
    if expected in {"running", "pending"}:
        assert len(cards) == 1
        assert cards[0]["status"] == expected
        assert cards[0]["detail_ref"] == detail["detail_ref"]
    else:
        assert cards == []
    assert app.state.test_execution.calls == []
