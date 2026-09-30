"""DBOS owns durable turns; LangGraph owns their conversational checkpoints.

Public DBOSClient submission is intentional: agent tools run inside a DBOS step,
where starting an internal child workflow is forbidden (DBOS 2.31).
Only small references cross the workflow boundary, never model/image histories.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable
from uuid import NAMESPACE_URL, uuid5

from dbos import DBOS, DBOSClient, SetEnqueueOptions, SetWorkflowID
from .research_capacity import ResearchCapacity, ResearchPoolConfig

QUEUE = "catmaster-agent-turns"
RESEARCH_QUEUE = "catmaster-research-turns"
WORKFLOW = "catmaster.agent_turn"
_host: ExecutionHost | None = None


def thread_key(workspace: str | Path, thread_id: str) -> str:
    # Internal queue partition identity; legacy readable thread IDs can repeat
    # across workspaces, so thread_id alone cannot serialize the correct graph.
    return str(uuid5(NAMESPACE_URL, f"{Path(workspace).resolve()}#{thread_id}"))


@DBOS.step(name="catmaster.execute_graph", preemptible=True)
async def _execute_graph(packet: dict[str, Any]) -> dict[str, Any]:
    return await _required_host().call("execute_turn", packet)


@DBOS.step(name="catmaster.prepare_completion")
async def _prepare_completion(packet: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    return await _required_host().call("prepare_completion", packet, result)


@DBOS.step(name="catmaster.publish_acceptance")
async def _publish_acceptance(packet: dict[str, Any]) -> None:
    await _required_host().call("publish_acceptance", packet)


@DBOS.step(name="catmaster.research_admission")
async def _admit(packet, tier, epoch):
    host = _required_host()
    granted, awakened = await asyncio.to_thread(host.capacity.acquire, packet, tier, epoch)
    await host.wake_capacity([rid for rid in awakened if rid != packet['run_id']], f"admit:{packet['run_id']}:{epoch}")
    await host.call("publish_capacity", packet, tier, granted)
    return granted


@DBOS.step(name="catmaster.release_research_admission")
async def _release(packet):
    await _required_host().release_capacity(packet["run_id"])
    await _required_host().call("capacity_changed", packet)


@DBOS.workflow(name=WORKFLOW)
async def agent_turn(packet: dict[str, Any]) -> dict[str, Any]:
    tier, epoch = packet.get("task_cost", ""), 0
    while True:
        if tier:
            while not await _admit(packet, tier, epoch):
                # Native durable receive, woken when an eligible slot is granted.
                # Timeout repairs a missed notification without a custom poller.
                await DBOS.recv_async(topic="research_capacity", timeout_seconds=60)
        result = await _execute_graph(packet)
        if result["status"] != "capacity_wait":
            break
        tier, epoch = result["task_cost"], epoch + 1
        packet = {**packet, "task_cost": tier,
                  "capacity_resume": {"checkpoint_id": result["checkpoint_id"],
                                      "decision": {"task_cost": tier, "admitted": True}}}
    if tier:
        await _release(packet)
    completion = await _prepare_completion(packet, result)
    if completion:
        with SetWorkflowID(completion["run_id"]), SetEnqueueOptions(
            queue_partition_key=thread_key(completion["workspace"], completion["thread_id"])
        ):
            await DBOS.enqueue_workflow_async(RESEARCH_QUEUE if completion.get("task_cost") else QUEUE, agent_turn, completion)
        await _publish_acceptance(completion)
    return result


def _required_host() -> ExecutionHost:
    if _host is None:
        raise RuntimeError("Local execution has not been started by the application.")
    return _host


class ExecutionHost:
    """One DBOS host per deployment, with atomic cost admission for agent tasks."""

    def __init__(self, path: Path, service_factory: Callable[..., Any], *, capacity_config=None) -> None:
        self.path = Path(path).resolve()
        self.factory = service_factory
        self.loop: asyncio.AbstractEventLoop | None = None
        self.client: DBOSClient | None = None
        self.capacity_config = capacity_config
        self.capacity: ResearchCapacity | None = None

    async def start(self) -> None:
        global _host
        if _host is not None and _host is not self:
            raise RuntimeError("Only one local execution host is allowed per process.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.capacity_config is None:
            from catmaster.llm.config import LLMProfile
            self.capacity_config = ResearchPoolConfig.from_dict(LLMProfile.from_env_or_file().persistent_research)
        self.capacity = ResearchCapacity(self.path, self.capacity_config)
        await asyncio.to_thread(self.capacity.setup)
        self.loop = asyncio.get_running_loop()
        _host = self
        url = "sqlite:///" + str(self.path)
        DBOS(config={"name": "catmaster", "system_database_url": url,
                     "application_version": "local-sql-1", "executor_id": "local",
                     "run_admin_server": False, "enable_otlp": False})
        self.client = DBOSClient(system_database_url=url, application_name="catmaster")
        await asyncio.to_thread(DBOS.launch)
        await asyncio.to_thread(DBOS.register_queue, QUEUE, partition_concurrency=1,
                                global_concurrency=self.capacity_config.control_concurrency)
        await asyncio.to_thread(DBOS.register_queue, RESEARCH_QUEUE, partition_concurrency=1)
        # Cancellation is durable in DBOS; remove only terminal reservations.
        # Pending recovery keeps its slot, including while a remote job waits.
        for row in await asyncio.to_thread(self.capacity.rows):
            run = await self.run(row["run_id"])
            if run is None or run.status not in {"ENQUEUED", "PENDING", "DELAYED"}:
                successors = await self.runs(Path(row["workspace"]), row["thread_id"], active=True)
                if not successors:
                    await self.release_capacity(row["run_id"])

    async def close(self) -> None:
        global _host
        # DBOS stops its local executor without cancelling durable queued work.
        # A later process recovers accepted turns through the native queue.
        await asyncio.to_thread(DBOS.destroy)
        if self.client:
            await asyncio.to_thread(self.client.destroy)
        self.client = None
        _host = None

    async def call(self, method: str, packet: dict[str, Any], *args: Any) -> Any:
        async def invoke() -> Any:
            service = self.factory(Path(packet["workspace"]), packet["workspace_id"])
            return await getattr(service, method)(packet, *args)

        if asyncio.get_running_loop() is self.loop:
            return await invoke()
        # DBOS may recover workflows on its own event loop. Keep UI callbacks,
        # tool contexts and graph streams on the application's loop. Cancellation
        # propagates through the concurrent future into the actual graph run.
        future = asyncio.run_coroutine_threadsafe(invoke(), self.loop)
        return await asyncio.wrap_future(future)

    async def enqueue(self, packet: dict[str, Any]) -> str:
        if self.client is None:
            raise RuntimeError("Local execution host is not running.")
        handle = await self.client.enqueue_async({
            "workflow_name": WORKFLOW, "queue_name": RESEARCH_QUEUE if packet.get("task_cost") else QUEUE,
            "workflow_id": packet["run_id"],
            "queue_partition_key": thread_key(packet["workspace"], packet["thread_id"]),
        }, packet)
        return handle.get_workflow_id()

    async def wake_capacity(self, run_ids, event):
        if self.client:
            for run_id in run_ids:
                await self.client.send_async(run_id, {"admitted": True}, topic="research_capacity",
                                             idempotency_key=event + ":" + run_id)

    async def release_capacity(self, run_id):
        if self.capacity:
            awakened = await asyncio.to_thread(self.capacity.release, run_id)
            await self.wake_capacity(awakened, "release:" + run_id)

    async def runs(self, workspace: Path, thread_id: str, *, active: bool = False,
                   limit: int = 100, offset: int = 0) -> list[Any]:
        if self.client is None:
            return []
        return await self.client.list_workflows_async(
            workflow_id_prefix=thread_key(workspace, thread_id) + ":",
            status=["ENQUEUED", "PENDING", "DELAYED"] if active else None,
            sort_desc=not active, limit=limit, offset=offset,
            load_input=False, load_output=not active,
        )

    async def run(self, run_id: str) -> Any:
        if self.client is None:
            return None
        rows = await self.client.list_workflows_async(workflow_ids=[run_id], load_input=False)
        return rows[0] if rows else None

    async def cancel(self, run_id: str) -> None:
        if self.client is None:
            raise RuntimeError("Local execution host is not running.")
        await self.client.cancel_workflow_async(run_id)
