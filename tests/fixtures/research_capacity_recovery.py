"""Crash while a researcher is queued at a native cost-change checkpoint."""
import asyncio
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from catmaster.runtime.execution import ExecutionHost
from catmaster.runtime.research_capacity import ResearchPoolConfig
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import ThreadSubmitRequest
from catmaster.webui.thread_store import ThreadStore


class State(TypedDict):
    messages: Annotated[list, add_messages]
    evidence: str


async def main():
    root, mode = Path(sys.argv[1]), sys.argv[2]
    workspace = root / "workspace"
    (workspace / "files").mkdir(parents=True, exist_ok=True)
    store = ThreadStore(workspace=workspace)

    @asynccontextmanager
    async def factory(service, packet):
        name = store.get_thread(packet["thread_id"]).title
        async def prepare(state):
            if name == "changing":
                path = root / "preparations"
                path.write_text(str(int(path.read_text()) + 1) if path.exists() else "1")
            return {"evidence": "prior method evidence"}
        async def run(state):
            if name == "blocker":
                (root / "blocker_started").touch()
                while not (root / "release").exists():
                    await asyncio.sleep(.05)
            else:
                response = interrupt({"kind": "research_capacity", "task_cost": "high", "reason": "formed method"})
                assert response["admitted"] and response["task_cost"] == "high"
                assert (root / "release").exists()
                row = next(r for r in service.execution.capacity.rows() if r["run_id"] == packet["run_id"])
                assert row["tier"] == "high" and row["state"] == "active"
                (root / "validated").write_text(state["evidence"])
            return {"messages": [AIMessage(content="Done: " + state["evidence"])]}
        async with AsyncSqliteSaver.from_conn_string(str(workspace / "metadata/deepagent_threads.sqlite")) as saver:
            graph = StateGraph(State).add_node("prepare", prepare).add_node("run", run)
            graph.add_edge(START, "prepare").add_edge("prepare", "run").add_edge("run", END)
            yield graph.compile(checkpointer=saver)

    class Service(LocalThreadService):
        async def publish_capacity(self, packet, tier, granted):
            await super().publish_capacity(packet, tier, granted)
            if store.get_thread(packet["thread_id"]).title == "changing" and tier == "high" and not granted:
                (root / "waiting").write_text(json.dumps(self.execution.capacity.snapshot()))

    host = ExecutionHost(root / "execution.sqlite", lambda *_: service,
        capacity_config=ResearchPoolConfig(2, {"low": 1, "medium": 1, "high": 1}))
    service = Service(workspace=workspace, workspace_id="w", store=store,
        broker=ThreadEventBroker(workspace=workspace), artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="w"),
        normalize_entrypoint=lambda x: x or "research", permission_mode_for_thread=lambda *_: "auto",
        execution=host, graph_factory=factory)
    await host.start()
    try:
        if mode == "launch":
            parent = await service.create_thread()
            accepted = []
            for name, tier in (("blocker", "high"), ("changing", "low")):
                thread = await service.create_thread(title=name, parent_thread_id=parent.thread_id,
                    metadata={"background_task": True, "task_cost": tier, "on_completion": "notify"})
                packet = await service.submit(thread_id=thread.thread_id, payload=ThreadSubmitRequest(text="Investigate this question"))
                accepted.append({"thread_id": thread.thread_id, "run_id": packet["run_id"]})
                if name == "blocker":
                    while not (root / "blocker_started").exists():
                        await asyncio.sleep(.05)
            (root / "accepted.json").write_text(json.dumps(accepted))
        accepted = json.loads((root / "accepted.json").read_text())
        for packet in accepted:
            handle = await host.client.retrieve_workflow_async(packet["run_id"])
            assert (await handle.get_result(polling_interval_sec=.05))["status"] == "success"
        async with factory(service, accepted[1]) as graph:
            state = await graph.aget_state({"configurable": {"thread_id": accepted[1]["thread_id"]}})
        (root / "result.json").write_text(json.dumps({"preparations": int((root / "preparations").read_text()),
            "evidence": (root / "validated").read_text(), "pool": host.capacity.snapshot(),
            "human_messages": sum(m.type == "human" for m in state.values["messages"])}))
    finally:
        await host.close()


asyncio.run(main())
