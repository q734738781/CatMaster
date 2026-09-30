"""Native DBOS + LangGraph exercise of task slots, safe switching and UI state."""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from contextlib import asynccontextmanager

from deepagents import create_deep_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from catmaster.runtime.execution import ExecutionHost
from catmaster.runtime.research_capacity import ResearchCapacity, ResearchPoolConfig
from catmaster.specialists.research_capacity import ResearchCapacityBoundary
from catmaster.specialists.runtime import SpecialistRunner
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.local_execution import LocalThreadService
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import ThreadSubmitRequest, ThreadStopRequest, ThreadCheckpointContinueRequest
from catmaster.webui.thread_store import ThreadStore
from test_local_execution import Model


def test_pool_total_and_fifo_skip_full_tier(tmp_path):
    capacity = ResearchCapacity(tmp_path / "execution.sqlite", ResearchPoolConfig(2, {"low": 2, "medium": 1, "high": 1}))
    capacity.setup()
    def packet(rid):
        return {"run_id": rid, "workspace": str(tmp_path), "thread_id": rid}
    assert capacity.acquire(packet("A"), "high", 0)[0]
    assert not capacity.acquire(packet("B"), "high", 0)[0]
    assert capacity.acquire(packet("C"), "low", 0)[0]
    assert not capacity.acquire(packet("D"), "medium", 0)[0]
    # Re-opening admission retains the running computation, not a fresh pool.
    reopened = ResearchCapacity(capacity.path, capacity.config)
    assert reopened.snapshot()["active"] == 2
    assert reopened.acquire(packet("A"), "high", 0)[0]
    # C safely paused at its checkpoint; changing to high releases only C's low slot.
    assert not capacity.acquire(packet("C"), "high", 1)[0]
    assert next(r for r in capacity.rows() if r["run_id"] == "D")["state"] == "active"
    assert capacity.release("A") == ["B"]
    assert capacity.release("B") == ["C"]
    assert capacity.snapshot()["active"] == 2


def test_steer_transfers_reservation_without_starting_waiting_expensive_task(tmp_path):
    capacity = ResearchCapacity(tmp_path / "execution.sqlite", ResearchPoolConfig(2, {"low": 1, "medium": 1, "high": 1}))
    capacity.setup()
    a = {"run_id": "first", "workspace": str(tmp_path), "thread_id": "A"}
    b = {"run_id": "waiting", "workspace": str(tmp_path), "thread_id": "B"}
    assert capacity.acquire(a, "high", 0)[0]
    assert not capacity.acquire(b, "high", 0)[0]
    assert capacity.acquire({**a, "run_id": "steered"}, "high", 0)[0]
    assert not capacity.acquire(b, "high", 0)[0]
    assert {r["run_id"] for r in capacity.rows()} == {"steered", "waiting"}
    assert capacity.release("first") == []
    assert capacity.release("steered") == ["waiting"]


def test_capacity_feedback_coalesces_and_honors_notify_pause_and_completed_scope(tmp_path):
    async def exercise():
        workspace = tmp_path / "workspace"
        (workspace / "files").mkdir(parents=True)
        capacity = ResearchCapacity(tmp_path / "execution.sqlite", ResearchPoolConfig(3, {"low": 2, "medium": 1, "high": 1}))
        capacity.setup()
        host = SimpleNamespace(capacity=capacity, enqueue=AsyncMock())
        store = ThreadStore(workspace=workspace)
        service = LocalThreadService(workspace=workspace, workspace_id="w", store=store,
            broker=ThreadEventBroker(workspace=workspace), artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="w"),
            normalize_entrypoint=lambda x: x or "research", permission_mode_for_thread=lambda *_: "auto", execution=host)
        service._prepare_turn = AsyncMock(return_value={"run_id": "notice"})
        root = await service.create_thread(entrypoint="persistent_research")
        packets = []
        for name in "ABC":
            child = await service.create_thread(parent_thread_id=root.thread_id,
                metadata={"background_task": True, "on_completion": "resume_parent", "task_description": name})
            packet = {"run_id": name, "thread_id": child.thread_id, "workspace": str(workspace)}
            packets.append(packet)
            capacity.acquire(packet, "high", 0)
        await asyncio.gather(*(service.publish_capacity(p, "high", False) for p in packets[1:]))
        assert host.enqueue.await_count == 1
        await service.capacity_changed(packets[0])
        assert host.enqueue.await_count == 1
        notice = service._prepare_turn.call_args.args[1].text
        assert "Waiting questions" in notice and "B" in notice and "C" in notice
        # A fresh episode under each explicit boundary must produce no wakeup.
        for mode in ("notify", "pause", "complete"):
            parent = store.get_thread(root.thread_id)
            store.update_thread(root.thread_id, meta={**parent.meta, "research_capacity_notice_active": False,
                                                      "automation_paused": mode == "pause"})
            child = store.get_thread(packets[1]["thread_id"])
            store.update_thread(child.thread_id, meta={**child.meta, "on_completion": "notify" if mode == "notify" else "resume_parent"})
            service._persistent_automation_enabled = lambda _: mode != "complete"
            await service.capacity_changed(packets[1])
        assert host.enqueue.await_count == 1
    asyncio.run(exercise())


def test_capacity_checkpoint_recovers_after_process_death(tmp_path):
    script = Path(__file__).parent / "fixtures/research_capacity_recovery.py"
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
    with (tmp_path / "launch.log").open("w") as log:
        process = subprocess.Popen([sys.executable, str(script), str(tmp_path), "launch"], env=env, stdout=log, stderr=log)
        try:
            deadline = time.monotonic() + 30
            while not (tmp_path / "waiting").exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(.05)
            assert (tmp_path / "waiting").exists(), (tmp_path / "launch.log").read_text()[-4000:]
            state = json.loads((tmp_path / "waiting").read_text())
            assert state["tiers"]["high"]["active"] == state["tiers"]["high"]["waiting"] == 1
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
    (tmp_path / "release").touch()
    recovered = subprocess.run([sys.executable, str(script), str(tmp_path), "recover"], env=env,
        capture_output=True, text=True, timeout=50)
    assert recovered.returncode == 0, recovered.stderr[-4000:]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["preparations"] == result["human_messages"] == 1
    assert result["evidence"] == "prior method evidence"
    assert result["pool"]["active"] == 0


def test_native_parallel_cost_switch_preserves_context_and_waiting_slots(tmp_path):
    async def exercise():
        workspace = tmp_path / "workspace"
        (workspace / "files").mkdir(parents=True)
        store, broker = ThreadStore(workspace=workspace), ThreadEventBroker(workspace=workspace)
        starts = {name: asyncio.Event() for name in "ABCD"}
        releases = {name: asyncio.Event() for name in "ABD"}
        prereq, cost_waiting, b_waiting = asyncio.Event(), asyncio.Event(), asyncio.Event()
        models, counts = {}, {"prerequisite": 0, "post": 0}

        def call(name, args=None, cid=None):
            return AIMessage(content="", tool_calls=[{"name": name, "args": args or {}, "id": cid or name}])

        @asynccontextmanager
        async def factory(service, packet):
            child = store.get_thread(packet["thread_id"])
            if child.parent_thread_id:
                name = child.meta["task_description"]
                @tool
                async def work() -> str:
                    """Run and wait for this task's entire computation."""
                    starts[name].set()
                    await releases[name].wait()
                    return name + " evidence"
                @tool
                async def prerequisite() -> str:
                    """Read evidence before deciding on an expensive method."""
                    counts["prerequisite"] += 1
                    prereq.set()
                    return "PRIOR_EVIDENCE_PRESERVED"
                @tool
                async def expensive_work() -> str:
                    """Execute the newly formed method, only after high admission."""
                    rows = service.execution.capacity.rows()
                    row = next(r for r in rows if r["run_id"] == packet["run_id"])
                    assert row["state"] == "active" and row["tier"] == "high"
                    counts["post"] += 1
                    starts["C"].set()
                    return "discriminating result"
                tools = [work, prerequisite, expensive_work, *service.research_capacity_tools(packet)]
                responses = [call("work"), AIMessage(content=name + " result")]
                if name == "C":
                    switch = {"name": "set_research_task_cost", "args": {"task_cost": "high", "reason": "formed method requires DFT"}, "id": "switch"}
                    responses = [call("prerequisite"),
                        AIMessage(content="", tool_calls=[switch, {"name": "expensive_work", "args": {}, "id": "unsafe"}]),
                        AIMessage(content="", tool_calls=[{**switch, "id": "switch-safe"}]),
                        call("expensive_work"), AIMessage(content="C result")]
            else:
                tools = service.background_tools(packet)
                message = store.get_message(child.thread_id, packet["input_message_id"])
                text = message.parts[0].text
                responses = [AIMessage(content="foreground still interactive")]
                if text == "launch":
                    @tool
                    async def settled(name: str) -> str:
                        """Wait for the test branch to reach its controlled boundary."""
                        await {"A": starts["A"], "B": b_waiting, "C": cost_waiting}[name].wait()
                        return "ready"
                    tools = [*tools, settled]
                    responses = []
                    for name, tier in [("A", "high"), ("B", "high"), ("C", "low"), ("D", "medium")]:
                        responses.append(call("start_async_task", {"agent": "research_specialist", "description": name,
                            "task_cost": tier, "on_completion": "notify"}, "start-"+name))
                        if name != "D":
                            responses.append(call("settled", {"name": name}, "settled-"+name))
                    responses.append(AIMessage(content="accepted"))
            model = models.setdefault(child.thread_id if child.parent_thread_id else packet["run_id"], Model(responses=responses))
            async with AsyncSqliteSaver.from_conn_string(str(workspace / "metadata/deepagent_threads.sqlite")) as saver:
                yield create_deep_agent(model, tools=tools,
                    middleware=[ResearchCapacityBoundary(), *SpecialistRunner._build_default_middleware()],
                    checkpointer=saver)

        host = ExecutionHost(tmp_path / "execution.sqlite", lambda *_: service,
            capacity_config=ResearchPoolConfig(2, {"low": 1, "medium": 1, "high": 1}))
        service = LocalThreadService(workspace=workspace, workspace_id="w", store=store, broker=broker,
            artifact_registry=ArtifactRegistry(workspace=workspace, workspace_id="w"),
            normalize_entrypoint=lambda x: x or "research", permission_mode_for_thread=lambda *_: "auto",
            execution=host, graph_factory=factory)
        original_publish = service.publish_capacity
        async def publish(packet, tier, granted):
            await original_publish(packet, tier, granted)
            if store.get_thread(packet["thread_id"]).meta.get("task_description") == "C" and tier == "high" and not granted:
                cost_waiting.set()
            if store.get_thread(packet["thread_id"]).meta.get("task_description") == "B" and not granted:
                b_waiting.set()
        service.publish_capacity = publish
        async def finish(rid):
            handle = await host.client.retrieve_workflow_async(rid)
            return await asyncio.wait_for(handle.get_result(polling_interval_sec=.02), 30)
        await host.start()
        try:
            root = await service.create_thread()
            submitted = await service.submit(thread_id=root.thread_id, payload=ThreadSubmitRequest(text="launch"))
            assert (await finish(submitted["run_id"]))["status"] == "success"
            await asyncio.wait_for(asyncio.gather(starts["A"].wait(), prereq.wait(), cost_waiting.wait(), starts["D"].wait()), 25)
            assert not starts["B"].is_set() and not starts["C"].is_set()
            assert counts == {"prerequisite": 1, "post": 0}
            children = {t.meta["task_description"]: t for t in store.list_threads() if t.parent_thread_id == root.thread_id}
            c = await service.task(root.thread_id, children["C"].thread_id)
            assert c["status"] == "pending" and c["task_cost"] == "high"
            assert c["capacity_state"] == "waiting"
            assert store.get_message(children["C"].thread_id, "answer_" + c["run_id"]).status == "streaming"
            assert not any(p.type == "interrupt" for m in store.list_messages(children["C"].thread_id) for p in m.parts)
            # Stopping a capacity wait is an ordinary user stop; continuing it
            # requires no fabricated approval and preserves the same checkpoint.
            await service.stop(thread_id=children["C"].thread_id, payload=ThreadStopRequest())
            paused_message = store.get_message(children["C"].thread_id, "answer_" + c["run_id"])
            assert paused_message.status == "interrupted"
            await service.continue_from_checkpoint(thread_id=children["C"].thread_id,
                payload=ThreadCheckpointContinueRequest(message_id=paused_message.id))
            children["C"] = store.get_thread(children["C"].thread_id)
            foreground = await service.submit(thread_id=root.thread_id, payload=ThreadSubmitRequest(text="status"))
            assert (await finish(foreground["run_id"]))["status"] == "success"
            releases["A"].set()
            await finish(children["A"].meta["last_run_id"])
            await asyncio.wait_for(starts["B"].wait(), 15)
            assert not starts["C"].is_set()
            releases["B"].set()
            await finish(children["B"].meta["last_run_id"])
            assert (await finish(children["C"].meta["last_run_id"]))["status"] == "success"
            assert counts == {"prerequisite": 1, "post": 1}
            releases["D"].set()
            await finish(children["D"].meta["last_run_id"])
            assert host.capacity.snapshot()["active"] == 0
            saved = store.get_message(children["C"].thread_id, "answer_" + children["C"].meta["last_run_id"])
            async with service._graph(saved.structured_sidecar["execution"]) as graph:
                state = await graph.aget_state({"configurable": {"thread_id": children["C"].thread_id}})
                messages = state.values["messages"]
                assert any(getattr(m, "content", "") == "PRIOR_EVIDENCE_PRESERVED" for m in messages)
                switches = [m for m in messages if isinstance(m, ToolMessage) and m.tool_call_id == "switch-safe"]
                assert len(switches) == 1 and switches[0].status == "success"
                assert json.loads(switches[0].content) == {"task_cost": "high", "admitted": True}
        finally:
            for event in releases.values():
                event.set()
            await host.close()
    asyncio.run(exercise())


def test_many_concurrent_admissions_keep_shared_and_tier_caps(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    config = ResearchPoolConfig(7, {'low': 4, 'medium': 3, 'high': 2})
    capacity = ResearchCapacity(tmp_path / 'execution.sqlite', config)
    capacity.setup()
    def accept(i):
        packet = {'run_id': str(i), 'workspace': str(tmp_path / str(i % 3)), 'thread_id': str(i)}
        capacity.acquire(packet, ('low', 'medium', 'high')[i % 3], 0)
        snapshot = capacity.snapshot()
        assert snapshot['active'] <= config.agent_pool_size
        assert all(t['active'] <= t['limit'] for t in snapshot['tiers'].values())
    with ThreadPoolExecutor(max_workers=16) as writers:
        list(writers.map(accept, range(64)))
    assert len(capacity.rows()) == 64
    completed = set()
    while capacity.rows():
        active = [r['run_id'] for r in capacity.rows() if r['state'] == 'active']
        assert active and not (set(active) & completed)
        completed.update(active)
        with ThreadPoolExecutor(max_workers=8) as writers:
            list(writers.map(capacity.release, active))
        assert capacity.snapshot()['active'] <= config.agent_pool_size
    assert len(completed) == 64
