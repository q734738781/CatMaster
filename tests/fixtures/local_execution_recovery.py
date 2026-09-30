"""Real DBOS/LangGraph subprocess used to test process-death recovery."""
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from deepagents import create_deep_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from catmaster.runtime.execution import ExecutionHost
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import ThreadSubmitRequest
from catmaster.webui.thread_store import ThreadStore


class State(TypedDict):
    messages: Annotated[list, add_messages]
    result: str


class Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


async def main():
    root, mode, phase = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    workspace = root / "workspace"
    (workspace / "files").mkdir(parents=True, exist_ok=True)
    store = ThreadStore(workspace=workspace)

    @asynccontextmanager
    async def graph_factory(service, packet):
        async def completed_work(state):
            path = root / "completed_count"
            path.write_text(str(int(path.read_text()) + 1) if path.exists() else "1")
            return {"result": "saved evidence"}

        async def final_work(state):
            (root / "waiting").write_text("ready")
            if phase == "inside_graph":
                while not (root / "release").exists():
                    await asyncio.sleep(.02)
            return {"messages": [AIMessage(content="Recovered " + state["result"], id="final-answer")]}

        async with AsyncSqliteSaver.from_conn_string(str(workspace / "metadata/deepagent_threads.sqlite")) as saver:
            if phase == "after_child_acceptance" and not store.get_thread(packet["thread_id"]).parent_thread_id:
                yield create_deep_agent(Model(responses=[AIMessage(content="", tool_calls=[{
                    "id":"fixed-delegation", "name":"start_async_task", "args":{
                        "agent":"writing_specialist", "description":"Recover saved evidence", "on_completion":"notify"}}]),
                    AIMessage(content="Recovered saved evidence")]), tools=service.background_tools(packet), checkpointer=saver)
                return
            graph = StateGraph(State).add_node("completed_work", completed_work).add_node("final_work", final_work)
            graph.add_edge(START, "completed_work").add_edge("completed_work", "final_work").add_edge("final_work", END)
            yield graph.compile(checkpointer=saver)

    class Service(LocalThreadService):
        async def execute_turn(self, packet):
            result = await super().execute_turn(packet)
            if phase == "before_step_commit" and not (root / "crashed").exists():
                (root / "crashed").write_text("native checkpoint committed; DBOS step not committed")
                os._exit(92)
            return result

        async def prepare_completion(self, packet, result):
            if phase == "before_delivery" and not (root / "crashed").exists():
                (root / "crashed").write_text("graph step committed; delivery not yet committed")
                os._exit(91)
            return await super().prepare_completion(packet, result)

    class Host(ExecutionHost):
        async def enqueue(self, packet):
            result = await super().enqueue(packet)
            if phase == "after_child_acceptance" and store.get_thread(packet["thread_id"]).parent_thread_id and not (root / "crashed").exists():
                (root / "crashed").write_text("child SQL accepted; parent tool result not checkpointed")
                os._exit(93)
            return result
    service = None
    host = Host(root / "execution.sqlite", lambda *_: service)
    service = Service(workspace=workspace, workspace_id="workspace", store=store,
        broker=ThreadEventBroker(workspace=workspace),
        artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="workspace"),
        normalize_entrypoint=lambda x: x or "research", permission_mode_for_thread=lambda *_: "auto",
        execution=host, graph_factory=graph_factory)
    await host.start()
    try:
        if mode == "launch":
            thread = await service.create_thread()
            submitted = await service.submit(thread_id=thread.thread_id, payload=ThreadSubmitRequest(text="Recover this exact turn"))
            (root / "accepted.json").write_text(json.dumps({"thread_id": thread.thread_id, "run_id": submitted["run_id"]}))
        accepted = json.loads((root / "accepted.json").read_text())
        handle = await host.client.retrieve_workflow_async(accepted["run_id"])
        result = await asyncio.wait_for(handle.get_result(polling_interval_sec=.02), 40)
        children = [t for t in store.list_threads() if t.parent_thread_id]
        for child in children:
            child_handle = await host.client.retrieve_workflow_async(child.meta["last_run_id"])
            await asyncio.wait_for(child_handle.get_result(polling_interval_sec=.02), 30)
        async with graph_factory(service, {"thread_id": accepted["thread_id"], "run_id": accepted["run_id"]}) as graph:
            state = await graph.aget_state({"configurable": {"thread_id": accepted["thread_id"]}})
        (root / "result.json").write_text(json.dumps({"result": result,
            "human_messages": sum(m.type == "human" for m in state.values["messages"]),
            "final": state.values["messages"][-1].content,
            "completed_work_count": int((root / "completed_count").read_text()),
            "child_threads": len(children)}))
    finally:
        await host.close()


asyncio.run(main())
