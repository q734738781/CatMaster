"""Execute two scientific cycles through the actual Research branch tool surface."""
import asyncio

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from catmaster.research.knowledge_graph.service import ResearchGraphService
from catmaster.research.knowledge_graph.models import GraphCreateRequest
from catmaster.specialists.runtime import SpecialistRunner
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_store import ThreadStore
from test_local_execution import Model


def test_branch_keeps_full_cycles_despite_stale_planning_record(tmp_path, monkeypatch):
    async def scenario():
        workspace = tmp_path / "workspace"
        (workspace / "files").mkdir(parents=True)
        (workspace / "files/evidence.txt").write_text("Fixture observations: A=2, B=1; matched control A=1, B=1.")
        store = ThreadStore(workspace=workspace)
        graph_service = ResearchGraphService(workspace=workspace)
        root = store.create_thread(entrypoint="persistent_research")
        graph = graph_service.create_graph(GraphCreateRequest(question="Explain the fixture comparison."),
            session_root_thread_id=root.thread_id)["graph"]
        gid = graph["graph_id"]
        branch = store.create_thread(entrypoint="research", parent_thread_id=root.thread_id,
            meta={"background_task": True, "research_branch": True, "task_cost": "low"})
        store.update_thread(branch.thread_id, active_research_graph_id=gid)
        # A historical row must not change a current Research branch's role.
        planning, _ = graph_service.store.claim_planning(gid, expected_revision=graph["revision"])
        graph_service.store.update_planning(gid, planning["planning_id"], start_revision=planning["revision"],
            status="attached", thread_id=branch.thread_id)
        requests = []
        bound_surfaces = []

        class ResearchModel(Model):
            def bind_tools(self, tools, **kwargs):
                bound_surfaces.append({getattr(t, "name", "") for t in tools})
                return self

            async def _agenerate(self, messages, **kwargs):
                step = len(requests)
                cycle, phase = divmod(step, 4)
                revision = graph_service.store.get_graph(gid)["revision"]
                base = {"graph_id": gid, "expected_revision": revision}
                nodes = graph_service.store.get_snapshot(gid)["nodes"]
                if step >= 8:
                    answer = AIMessage(content="The matched control removes the apparent difference; branch complete.")
                else:
                    if phase == 0:
                        name, args = "add_research_hypothesis", {**base,
                            "claim": ["The unmatched groups differ.", "The matched groups have equal outcomes."][cycle]}
                    elif phase == 1:
                        hypothesis = next(n for n in nodes if n["kind"] == "hypothesis"
                            and n["body"]["claim"] == ["The unmatched groups differ.", "The matched groups have equal outcomes."][cycle])
                        name, args = "add_research_experiment", {**base,
                            "title": f"Comparison {cycle}", "objective": "Compare the fixture groups.", "state": "ready",
                            "plan_summary": ["Compare A and B.", "Compare the matched A and B control."][cycle],
                            "decision_rule": "The difference is zero if the outcomes are equal.",
                            "tests_hypothesis_ids": [hypothesis["node_id"]]}
                    elif phase == 2:
                        name, args = "task", {"subagent_type": "experiment_specialist",
                            "description": f"Read /evidence.txt and report fixture comparison {cycle}."}
                    else:
                        experiment = next(n for n in nodes if n["kind"] == "experiment" and n["title"] == f"Comparison {cycle}")
                        name, args = "record_research_result", {**base,
                            "experiment_node_id": experiment["node_id"],
                            "methods": ["Unmatched comparison.", "Matched control comparison."][cycle],
                            "summary": ["Difference is 1.", "Difference is 0."][cycle],
                            "conclusion": ["Check a matched control.", "The original difference is absent in the matched control."][cycle],
                            "refs": [{"ref_kind": "note", "ref_id": "evidence.txt"}]}
                    requests.append(name)
                    answer = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"cycle_{step}"}])
                self.responses = [answer]
                self.i = 0
                return await super()._agenerate(messages, **kwargs)

        def model(_self, role, **kwargs):
            if role == "research_lead":
                return ResearchModel(responses=[])
            return Model(responses=[AIMessage(content="Fixture evidence: unmatched difference 1; matched difference 0.")])

        monkeypatch.setattr(SpecialistRunner, "_build_deepagent_chat_model", model)
        service = LocalThreadService(workspace=workspace, workspace_id="w", store=store,
            broker=ThreadEventBroker(workspace=workspace),
            artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="w"),
            normalize_entrypoint=lambda x: x or "research", permission_mode_for_thread=lambda *_: "auto", execution=None)
        packet = {"thread_id": branch.thread_id, "run_id": "two-cycle-branch",
            "assistant_message_id": "branch-answer", "entrypoint": "research", "task_cost": "low",
            "model_config": "", "permission_mode": "auto", "research": {"research_graph_id": gid}, "input_message_id": ""}
        async with service._graph(packet) as agent:
            result = await agent.ainvoke({"messages": [HumanMessage(content="Explain the fixture, following up as needed.")]},
                {"configurable": {"thread_id": branch.thread_id}})
        nodes = graph_service.store.get_snapshot(gid)["nodes"]
        assert {kind: sum(n["kind"] == kind for n in nodes) for kind in ("hypothesis", "experiment", "result")} == {
            "hypothesis": 2, "experiment": 2, "result": 2}
        assert len([m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "task"]) == 2
        assert all("record_research_result" in tools and "task" in tools for tools in bound_surfaces)
        assert all("set_research_graph_completion" not in tools for tools in bound_surfaces)
        assert not graph_service.store.get_graph(gid)["completed"]
        assert not graph_service.store.list_decisions(gid)
        assert not any(message.role == "user" for message in store.list_messages(root.thread_id))

    asyncio.run(scenario())
