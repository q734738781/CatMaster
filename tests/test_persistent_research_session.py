"""Regression coverage for Persistent entry, shared pause and async result visibility."""
import asyncio

from catmaster.research.knowledge_graph.models import GraphCreateRequest, NodePatchRequest, ResultCreateRequest
from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.storage import connect_workspace_db
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.projections.research_sessions import project_research_session_activity
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import ThreadRole, ThreadStopRequest, ThreadSubmitRequest
from catmaster.webui.thread_store import ThreadStore
from test_local_execution_graph_notifications import Queue


def test_persistent_submit_adopts_legacy_pause_and_stop_overrides_late_evidence(tmp_path):
    async def scenario():
        workspace = tmp_path / "workspace"
        (workspace / "files").mkdir(parents=True)
        store, queue = ThreadStore(workspace=workspace), Queue()
        service = LocalThreadService(workspace=workspace, workspace_id="w", store=store,
            broker=ThreadEventBroker(workspace=workspace),
            artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="w"),
            normalize_entrypoint=lambda x: x or "research", permission_mode_for_thread=lambda *_: "auto",
            execution=queue)
        graph_service = ResearchGraphService(workspace=workspace,
            agent_loop_factory=lambda *_: service)
        root = await service.create_thread(entrypoint="persistent_research")
        await service.submit(thread_id=root.thread_id,
            payload=ThreadSubmitRequest(text="Investigate Cu literature and recommend external experiments."))
        root = store.get_thread(root.thread_id)
        gid = root.active_research_graph_id
        assert gid and len(queue.accepted) == 1

        def evidence():
            with connect_workspace_db(workspace) as conn:
                graph_service.store._write_event(conn, graph_id=gid, revision=1,
                    change="result.recorded", thread_id="independent-source")

        graph = graph_service.store.get_graph(gid)
        graph_service.store.update_graph(gid, expected_revision=graph["revision"],
            changes={"orchestration_mode": "manual"})
        evidence()
        await graph_service.tick()
        assert len(queue.accepted) == 1  # Preserve an existing user's legacy pause.
        await service.submit(thread_id=root.thread_id, payload=ThreadSubmitRequest(text="Continue the existing objective."))
        assert graph_service.store.get_graph(gid)["orchestration_mode"] == "auto"
        evidence()
        await graph_service.tick()
        assert len(queue.accepted) == 3
        await service.stop(thread_id=root.thread_id, payload=ThreadStopRequest())
        root = store.get_thread(root.thread_id)
        activity = project_research_session_activity(root, threads=store.list_threads(),
            graph_service=graph_service, workspace=workspace)
        assert activity.automation_paused and activity.state == "paused"
        evidence()
        await graph_service.tick()
        assert len(queue.accepted) == 3
        await service.submit(thread_id=root.thread_id, payload=ThreadSubmitRequest(text="Resume with the saved evidence."))
        await graph_service.tick()
        assert len(queue.accepted) == 4  # Old events are already in the user turn.
        assert not store.get_thread(root.thread_id).meta["automation_paused"]
    asyncio.run(scenario())


def test_nested_research_activity_follows_descendants_without_chat_cards(tmp_path):
    service = ResearchGraphService(workspace=tmp_path)
    store = service.thread_store
    root = store.create_thread(entrypoint="persistent_research")
    service.create_graph(GraphCreateRequest(question="Cu electrode experiment recommendations"),
        session_root_thread_id=root.thread_id)
    root = store.get_thread(root.thread_id)
    branch = store.create_thread(parent_thread_id=root.thread_id,
        meta={"background_task": True, "research_branch": True, "agent_name": "research_specialist"})
    child = store.create_thread(parent_thread_id=branch.thread_id,
        meta={"background_task": True, "agent_name": "litreview_agent"})
    store.update_thread(child.thread_id, status="running")
    activity = project_research_session_activity(root, threads=store.list_threads(),
        graph_service=service, workspace=tmp_path)
    assert activity.state == "running"
    assert activity.active_child_thread_id == child.thread_id
    assert activity.execution_count == 2
    assert not activity.automation_paused
    store.update_thread(child.thread_id, status="idle")
    activity = project_research_session_activity(root, threads=store.list_threads(),
        graph_service=service, workspace=tmp_path)
    assert activity.state == "waiting_continue"
    assert activity.active_child_thread_id == ""


def test_async_results_supersede_legacy_results_and_amend_the_same_milestone(tmp_path):
    service = ResearchGraphService(workspace=tmp_path)
    store = service.thread_store
    root = store.create_thread(entrypoint="persistent_research")
    graph = service.create_graph(GraphCreateRequest(question="Cu electrode comparisons"),
        session_root_thread_id=root.thread_id)["graph"]
    gid = graph["graph_id"]
    legacy = store.create_thread(parent_thread_id=root.thread_id,
        thread_role=ThreadRole.RESEARCH_EXECUTION)
    service.record_result(gid, ResultCreateRequest(expected_revision=graph["revision"],
        summary="Earlier result", refs=[{"ref_kind": "thread", "ref_id": legacy.thread_id}]))
    result = service.record_result(gid, ResultCreateRequest(
        expected_revision=service.store.get_graph(gid)["revision"], title="Matched electrolyte comparison",
        methods="Compare matched Cu cell conditions in the source studies.",
        summary="The source comparison separates two effects.",
        conclusion="A matched control can discriminate them."))["node"]
    assert service.latest_session_result(gid, session_root_thread_id=root.thread_id)["node_id"] == result["node_id"]
    before = store.list_messages(root.thread_id)
    message_id = service._session_message_id("result", f"{gid}_{result['node_id']}")
    message = store.get_message(root.thread_id, message_id)
    assert result["body"]["methods"] in message.parts[0].text
    assert result["body"]["conclusion"] in message.parts[0].text
    service.update_bound_result(graph_id=gid, experiment_node_id="", result_node_id=result["node_id"],
        title=result["title"], summary="Corrected source comparison.", refs=[],
        conclusion="The interpretation remains conditional on the cell design.")
    after = store.get_message(root.thread_id, message_id)
    assert len(store.list_messages(root.thread_id)) == len(before)
    assert after.created_at == message.created_at
    assert "Corrected source comparison." in after.parts[0].text
    assert "conditional on the cell design" in after.parts[0].text
    assert result["body"]["methods"] in after.parts[0].text
    assert any(event.event == "message.updated" and event.message_id == message_id
        for event in ThreadEventBroker(workspace=tmp_path).replay(root.thread_id))
    # Editing a Result through the Graph UI uses the same visible milestone.
    latest = service.store.get_node(gid, result["node_id"])
    service.update_node(gid, result["node_id"], NodePatchRequest(
        expected_revision=service.store.get_graph(gid)["revision"],
        expected_node_revision=latest["revision"], title="Corrected electrolyte comparison", body=latest["body"]))
    assert "Corrected electrolyte comparison" in store.get_message(root.thread_id, message_id).parts[0].text
    assert len(store.list_messages(root.thread_id)) == len(before)
