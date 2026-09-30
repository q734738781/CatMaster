"""Native delta persistence must cover child graphs and old SQLite histories."""
from __future__ import annotations

import asyncio
import base64
import random
import sqlite3
from types import SimpleNamespace

import aiosqlite
from deepagents.backends import FilesystemBackend
from langchain.agents import AgentState
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.channels.delta import DeltaChannel
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command, interrupt
from PIL import Image
from pydantic import Field

from catmaster.llm.config import AgentRuntimeConfig
from catmaster.runtime.checkpoint_serde import FileSafeCheckpointSerializer
from catmaster.specialists.runtime import SpecialistRunner


class Model(FakeMessagesListChatModel):
    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def runner():
    instance = SpecialistRunner.__new__(SpecialistRunner)
    instance.llm_profile = SimpleNamespace(agent_runtime=AgentRuntimeConfig())
    instance.runtime_context = {"local_execution": True}
    return instance


def call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def image_payload(tmp_path):
    path = tmp_path / "figure.png"
    pixels = random.Random(0).randbytes(128 * 128 * 3)
    Image.frombytes("RGB", (128, 128), pixels).save(path)
    return base64.b64encode(path.read_bytes()).decode()


def image_blocks(messages):
    return [block for message in messages if isinstance(message.content, list)
            for block in message.content if isinstance(block, dict) and block.get("type") == "image"]


def build_graph(tmp_path, saver, root_model, worker_model, tools, *, legacy=False):
    return runner()._create_deep_agent(
        model=root_model,
        backend=FilesystemBackend(root_dir=tmp_path, virtual_mode=True),
        checkpointer=saver,
        # Reproduce the old full-message channel using the upstream schema,
        # without writing or translating checkpoint blobs ourselves.
        **({"state_schema": AgentState} if legacy else {}),
        subagents=[{"name": "reader", "description": "Read a figure",
                    "system_prompt": "Read the supplied figure and report observations.",
                    "model": worker_model, "tools": tools}],
    )


def test_native_schema_reaches_raw_and_compiled_child_graphs(tmp_path, monkeypatch):
    import deepagents.middleware.subagents as subagents

    compiled = []
    original = subagents.create_sub_agent

    def capture(spec, **kwargs):
        graph = original(spec, **kwargs)
        compiled.append(graph)
        return graph

    monkeypatch.setattr(subagents, "create_sub_agent", capture)
    model = Model(responses=[AIMessage(content="Unused")])
    backend = FilesystemBackend(root_dir=tmp_path, virtual_mode=True)
    child = runner()._create_deep_agent(model=model, backend=backend, subagents=[{
        "name": "reader", "description": "Read figures", "system_prompt": "Read figures",
        "model": model, "tools": [],
    }])
    root = runner()._create_deep_agent(model=model, backend=backend, subagents=[{
        "name": "specialist", "description": "Coordinate reading", "runnable": child,
    }])
    assert compiled  # Includes raw workers and upstream general-purpose children.
    assert all(isinstance(graph.channels["messages"], DeltaChannel)
               for graph in [root, child, *compiled])
    assert not model.seen


def test_image_is_preserved_without_a_full_copy_at_each_worker_step(tmp_path):
    payload = image_payload(tmp_path)

    @tool
    def observation(index: int) -> str:
        """Record one observation without returning the figure again."""
        return f"Observation {index} recorded"

    async def scenario(legacy):
        path = tmp_path / ("full.sqlite" if legacy else "delta.sqlite")
        root_model = Model(responses=[call("task", {
            "subagent_type": "reader", "description": "Inspect /figure.png",
        }, "delegate"), AIMessage(content="Reading finished")])
        worker = Model(responses=[
            call("read_file", {"file_path": "/figure.png"}, "read"),
            *[call("observation", {"index": i}, f"observe-{i}") for i in range(8)],
            AIMessage(content="Figure inspected"),
        ])
        async with aiosqlite.connect(path) as conn:
            saver = AsyncSqliteSaver(conn, serde=FileSafeCheckpointSerializer())
            graph = build_graph(tmp_path, saver, root_model, worker, [observation], legacy=legacy)
            result = await graph.ainvoke({"messages": [HumanMessage(content="Inspect the figure")]},
                                        {"configurable": {"thread_id": "reading"}})
            assert result["messages"][-1].content == "Reading finished"
        # Every model call following read_file still sees the original image.
        assert len(worker.seen) == 10
        for messages in worker.seen[1:]:
            assert [block["base64"] for block in image_blocks(messages)] == [payload]
        with sqlite3.connect(path) as conn:
            blobs = [row[0] for row in conn.execute("SELECT checkpoint FROM checkpoints")]
            blobs.extend(row[0] for row in conn.execute("SELECT value FROM writes"))
        return sum(blob.count(payload.encode()) for blob in blobs)

    full_copies = asyncio.run(scenario(True))
    delta_copies = asyncio.run(scenario(False))
    assert full_copies > 10
    assert 1 <= delta_copies <= 2


def test_old_full_child_checkpoint_resumes_as_delta_after_sqlite_reopen(tmp_path):
    payload = image_payload(tmp_path)
    config = {"configurable": {"thread_id": "existing-reading"}}
    path = tmp_path / "existing.sqlite"
    resumed = []

    @tool
    def approve_reading() -> str:
        """Request approval before finishing the reading."""
        decision = interrupt({"question": "Finish reading?"})
        resumed.append(decision)
        return f"Decision: {decision}"

    async def scenario():
        async with aiosqlite.connect(path) as conn:
            saver = AsyncSqliteSaver(conn, serde=FileSafeCheckpointSerializer())
            old_graph = build_graph(tmp_path, saver,
                Model(responses=[call("task", {
                    "subagent_type": "reader", "description": "Inspect /figure.png",
                }, "delegate")]),
                Model(responses=[call("read_file", {"file_path": "/figure.png"}, "read"),
                                 call("approve_reading", {}, "approve")]),
                [approve_reading], legacy=True)
            first = await old_graph.ainvoke({"messages": [HumanMessage(content="Inspect the figure")]}, config)
            assert first["__interrupt__"]
            # Tool-created subagents are dynamic and are not exposed by the
            # parent's static get_subgraphs traversal. Inspect the saved child.
            async for saved in saver.alist(config):
                if saved.config["configurable"]["checkpoint_ns"]:
                    break
            else:
                raise AssertionError("No child checkpoint was saved")
            child_config = saved.config
            # The old child really has a full list, not a delta snapshot.
            assert isinstance(saved.checkpoint["channel_values"]["messages"], list)
            old_messages = saved.checkpoint["channel_values"]["messages"]
            assert image_blocks(old_messages)[0]["base64"] == payload

        # Reopen the native SQLite store and build the corrected graph. No
        # import, history replay input, blob rewrite, or migration reader.
        async with aiosqlite.connect(path) as conn:
            saver = AsyncSqliteSaver(conn, serde=FileSafeCheckpointSerializer())
            worker = Model(responses=[call("approve_reading", {}, "approve-again")])
            graph = build_graph(tmp_path, saver,
                Model(responses=[AIMessage(content="Reading finished")]), worker, [approve_reading])
            result = await graph.ainvoke(Command(resume="approved"), config)
            assert result["__interrupt__"]
            assert resumed == ["approved"]
            assert len(worker.seen) == 1
            assert image_blocks(worker.seen[0])[0]["base64"] == payload
            assert any(isinstance(message, ToolMessage) and message.tool_call_id == "approve"
                       for message in worker.seen[0])
            assert {message.id for message in old_messages} <= {message.id for message in worker.seen[0]}
            latest_child = {"configurable": {key: value for key, value in child_config["configurable"].items()
                                             if key != "checkpoint_id"}}
            saved = await saver.aget_tuple(latest_child)
            assert "messages" not in saved.checkpoint["channel_values"]

        async with aiosqlite.connect(path) as conn:
            saver = AsyncSqliteSaver(conn, serde=FileSafeCheckpointSerializer())
            worker = Model(responses=[AIMessage(content="Figure inspected after approval")])
            graph = build_graph(tmp_path, saver, Model(responses=[AIMessage(content="Reading finished")]),
                                worker, [approve_reading])
            result = await graph.ainvoke(Command(resume="approved again"), config)
            assert result["messages"][-1].content == "Reading finished"
            assert resumed == ["approved", "approved again"]
            assert len(worker.seen) == 1
            assert image_blocks(worker.seen[0])[0]["base64"] == payload
            assert {message.id for message in old_messages} <= {message.id for message in worker.seen[0]}
            state = await graph.aget_state(config)
            assert not state.next
            assert len([message for message in state.values["messages"] if message.type == "human"]) == 1

    asyncio.run(scenario())
