"""Workspace binding and UI projection for DBOS-executed native DeepAgents."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import HTTPException
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.types import Command, interrupt
from langgraph.graph import END

from catmaster.llm.config import LLMProfile
from catmaster.runtime.execution import ExecutionHost, thread_key
from catmaster.runtime.multimodal_blocks import build_turn_content, PreparedAttachment, file_to_content_block, text_attachment_block, multimodal_prepare_summary
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.usage_stats import load_usage_summary, summarize_usage_from_observability
from catmaster.specialists.runtime import SpecialistRunner, default_thread_interrupt_on
from catmaster.ui.reporters import NullReporter

from .async_activity import AsyncActivityProjection
from .run_projection import RunProjection, _plain_dict, _content_fragments, _safe_token
from .thread_models import ThreadMessage, MessagePart, ThreadSubmitRequest, ThreadRole
from .thread_support import ThreadServiceSupport, _ENTRYPOINT_TO_MODEL_ROLE

logger = logging.getLogger(__name__)
SPECIALISTS = {"experiment_specialist": "experiment", "writing_specialist": "writing",
               "peer_review_specialist": "peer_review", "litreview_agent": "literature_review",
               "research_specialist": "research", "research_challenger": "research_challenger"}
ACTIVE = {"PENDING", "ENQUEUED", "DELAYED"}


def _text(message: ThreadMessage | None) -> str:
    return "\n".join(part.text for part in message.parts if part.type == "text") if message else ""


def _status(run: Any) -> str:
    if run is None:
        return "interrupted"
    output = run.output if isinstance(run.output, dict) else {}
    return output.get("status") or {"SUCCESS": "success", "ERROR": "error",
        "CANCELLED": "interrupted", "PENDING": "running", "ENQUEUED": "pending",
        "DELAYED": "pending", "MAX_RECOVERY_ATTEMPTS_EXCEEDED": "error"}.get(run.status, "error")


class LocalThreadService(ThreadServiceSupport):
    def __init__(self, *, workspace, workspace_id, store, broker, artifact_registry,
                 normalize_entrypoint, permission_mode_for_thread, execution: ExecutionHost,
                 on_run_finished=None, graph_factory=None):
        self.workspace = Path(workspace).resolve()
        self.workspace_id = workspace_id
        self.store, self.broker, self.artifact_registry = store, broker, artifact_registry
        self.normalize_entrypoint = normalize_entrypoint
        self.permission_mode_for_thread = permission_mode_for_thread
        self.execution = execution
        self.on_run_finished = on_run_finished
        self.graph_factory = graph_factory
        # Only acknowledgement of actual local teardown, never task ownership or
        # a queue. DBOS starts/cancels/recovers all executions.
        self._settled: dict[str, asyncio.Event] = {}
        self._admission: dict[str, asyncio.Lock] = {}
        self._graph_writers: dict[str, asyncio.Lock] = {}

    async def create_thread(self, *, title="", entrypoint="research", permission_mode="",
                            metadata=None, **kwargs):
        tid = kwargs.pop("thread_id", "") or str(uuid.uuid4())
        metadata = {**kwargs.pop("meta", {}), **(metadata or {})}
        return self.store.create_thread(thread_id=tid, deepagent_thread_id=tid,
            title=title, entrypoint=self.normalize_entrypoint(entrypoint),
            meta={**(metadata or {}), "permission_mode": permission_mode or metadata.get("permission_mode") or "auto"}, **kwargs)

    def _thread(self, thread_id):
        thread = self.store.get_thread(thread_id)
        if thread.agent_server_thread_id:
            raise HTTPException(409, "This thread requires the explicitly planned offline import before continuing.")
        if not thread.deepagent_thread_id:
            thread = self.store.update_thread(thread_id, deepagent_thread_id=thread_id)
        return thread

    async def _update_native_thread_metadata(self, thread):
        # Thread descriptors already hold the workspace/native checkpoint binding.
        return None

    async def submit(self, *, thread_id, payload, research_context=None, run_metadata=None):
        async with self._admission.setdefault(thread_id, asyncio.Lock()):
            packet = await self._prepare_turn(thread_id, payload, research_context=research_context,
                                              run_metadata=run_metadata)
            await self.execution.enqueue(packet)
            await self.refresh_thread_status(thread_id)
            return self._submission(packet)

    def _submission(self, packet):
        return {"thread": self.store.get_thread(packet["thread_id"]),
                "run_id": packet["run_id"],
                "message": self.store.get_message(packet["thread_id"], packet["input_message_id"]) if packet["input_message_id"] else None,
                "assistant_message": self.store.get_message(packet["thread_id"], packet["assistant_message_id"]),
                "queued": True}

    async def _prepare_turn(self, thread_id, payload, *, research_context=None,
                            run_metadata=None, identity="", command=None):
        thread = self._thread(thread_id)
        run_id = thread_key(self.workspace, thread_id) + ":" + (identity or str(uuid.uuid4()))
        assistant_id, user_id = "answer_" + run_id, "input_" + run_id
        if self.store.get_message(thread_id, assistant_id):
            return dict(self.store.get_message(thread_id, assistant_id).structured_sidecar["execution"])
        active = await self.execution.runs(self.workspace, thread_id, active=True)
        strategy = payload.strategy
        if active and strategy == "reject":
            raise HTTPException(409, "This conversation already has an active turn.")
        if strategy in {"interrupt", "rollback"}:
            for run in active:
                await self._cancel(run.workflow_id, thread_id=thread_id, preserve_capacity=True)
        rollback_checkpoint = ""
        if strategy == "rollback" and active:
            previous = self.store.get_message(thread_id, "answer_" + active[0].workflow_id)
            rollback_checkpoint = (previous.meta or {}).get("checkpoint_before_turn", "") if previous else ""
            if not rollback_checkpoint:
                raise HTTPException(409, "The interrupted turn has no saved starting checkpoint to replace.")
        # The challenger is an internal async lane, not a user-selectable mode.
        requested_entrypoint = payload.entrypoint or thread.entrypoint
        entrypoint = ("research_challenger" if requested_entrypoint == "research_challenger"
                      and thread.meta.get("agent_name") == "research_challenger"
                      else self.normalize_entrypoint(requested_entrypoint))
        model_config = str(payload.llm_config or thread.meta.get("model_config") or "")
        profile = LLMProfile.from_env_or_file(model_config or None)
        prepared = self.prepare_submit_attachments(thread_id, list(payload.attachments or []),
            capability=self._capability_for_entrypoint(profile=profile, entrypoint=entrypoint))
        prompt = payload.text.strip()
        if not prompt and not prepared and command is None:
            raise HTTPException(400, "Message text is required.")
        mode = self.permission_mode_for_thread(thread, payload.permission_mode or thread.meta.get("permission_mode"))
        thread = self._ensure_default_research_graph_binding(thread=thread, prompt=prompt,
            entrypoint=entrypoint, inherited=research_context if not run_metadata else
            (research_context or self._research_turn_context(thread=thread, entrypoint=entrypoint)))
        research = self._research_turn_context(thread=thread, entrypoint=entrypoint, inherited=research_context)
        packet = {"workspace": str(self.workspace), "workspace_id": self.workspace_id,
                  "thread_id": thread_id, "run_id": run_id, "assistant_message_id": assistant_id,
                  "input_message_id": user_id if command is None else "",
                  "entrypoint": entrypoint, "model_config": model_config, "permission_mode": mode,
                  "research": research, "metadata": dict(run_metadata or {}),
                  "command": command, "strategy": strategy}
        if thread.meta.get("background_task"):
            packet["task_cost"] = thread.meta.get("task_cost", "low")
        if rollback_checkpoint:
            packet["start_checkpoint_id"] = rollback_checkpoint
        if command is None:
            message = ThreadMessage(id=user_id, thread_id=thread_id, role="user", status="completed",
                parts=[MessagePart(id=user_id + "_text", type="text", text=prompt, status="completed"),
                       *[item.history_part for item in prepared if item.history_part]],
                meta={"entrypoint": entrypoint, **research, **({"origin": "runtime",
                    "notification_title": run_metadata.get("catmaster_notification_title", "Background task finished")}
                    if run_metadata else {})},
                structured_sidecar={"attachments": [item.sidecar() for item in prepared]})
            self.store.append_message(message)
            self.broker.emit(thread_id, "message.created", message_id=message.id,
                             data={"message": message.model_dump(mode="json"), "run_id": run_id})
        prior_id, _ = self.store.latest_assistant_run(thread_id)
        prior = self.store.get_message(thread_id, prior_id) if prior_id else None
        # A checkpoint continuation has no new input. Retain the original
        # episode's message reference without copying its conversation history.
        episode_id = user_id if command is None else str(
            (prior.meta.get("episode_id") or prior.meta.get("input_message_id") or prior.id) if prior else run_id)
        assistant = ThreadMessage(id=assistant_id, thread_id=thread_id, role="assistant", status="created",
            parts=[MessagePart(id=assistant_id + "_text", type="text", status="created")],
            meta={"run_id": run_id, "input_message_id": packet["input_message_id"],
                  "episode_id": episode_id, "prior_assistant_message_id": prior_id,
                  "entrypoint": entrypoint, "model_config": model_config, **research,
                  **({"source": thread.meta.get("agent_name", entrypoint)} if thread.parent_thread_id else {})},
            structured_sidecar={"execution": packet})
        self.store.append_message(assistant)
        self.broker.emit(thread_id, "message.created", message_id=assistant.id,
                         data={"message": assistant.model_dump(mode="json"), "run_id": run_id})
        if prepared:
            self.broker.emit(thread_id, "multimodal.prepared", message_id=assistant.id, status="completed",
                data={"run_id": run_id, "message_id": assistant.id, **multimodal_prepare_summary(prepared)})
        cursor = {}
        if not run_metadata and research.get("research_graph_id"):
            from catmaster.research.knowledge_graph.store import ResearchGraphStore
            cursor["research_event_cursor"] = ResearchGraphStore(self.workspace).latest_event_id(research["research_graph_id"])
        self.store.update_thread(thread_id, entrypoint=entrypoint,
            meta={**thread.meta, "model_config": model_config, "permission_mode": mode,
                  "last_run_id": run_id, **cursor, **({"automation_paused": False} if not run_metadata else {})})
        return packet

    def _projection(self, packet):
        return RunProjection(store=self.store, broker=self.broker, artifact_registry=self.artifact_registry,
            thread_id=packet["thread_id"], run_id=packet["run_id"],
            assistant_message_id=packet["assistant_message_id"],
            text_part_id=packet["assistant_message_id"] + "_text", input_message_id=packet["input_message_id"])

    @asynccontextmanager
    async def _graph(self, packet):
        if self.graph_factory:
            async with self.graph_factory(self, packet) as value:
                yield value
            return
        profile = LLMProfile.from_env_or_file(packet["model_config"] or None)
        model = profile.config_for_role(_ENTRYPOINT_TO_MODEL_ROLE[packet["entrypoint"]])
        run_dir = self.workspace / "metadata/runs" / packet["run_id"]
        context = RunContext.load(run_dir) if (run_dir / "meta.json").is_file() else RunContext.create(
            workspace=self.workspace, project_id=self.workspace_id, run_id=packet["run_id"],
            model_name=model.model, provider=model.provider, base_url=model.base_url)
        thread = self._thread(packet["thread_id"])
        root = thread
        while root.parent_thread_id:
            root = self._thread(root.parent_thread_id)
        runner = SpecialistRunner(llm_profile=profile, run_context=context, reporter=NullReporter(),
            run_control=None, interrupt_on=default_thread_interrupt_on() if packet["permission_mode"] == "hitl" else {},
            runtime_context={"thread_id": thread.thread_id, "root_thread_id": root.thread_id,
                             "local_execution": True, "research_branch": thread.meta.get("research_branch", False),
                             **packet["research"]})
        files_root = self.workspace / "files"
        runner._stage_deepagent_assets(files_root, thread_id=thread.thread_id)
        usage = runner._new_usage_callback()
        # Reuse the existing tracker and summary format, including calls from
        # before a process restart. Observability covers pre-fix saved runs.
        previous = load_usage_summary(run_dir) or summarize_usage_from_observability(run_dir)
        usage.usage_metadata = dict(previous.get("raw_usage_metadata", {}))
        usage.call_counts_by_model = dict(previous.get("call_counts_by_model", {}))
        usage.usage_metadata_by_role = dict(previous.get("raw_usage_metadata_by_role", {}))
        usage.call_counts_by_role = dict(previous.get("call_counts_by_role", {}))

        def publish_usage():
            summary = runner._write_usage_summary(usage)
            if summary:
                self.broker.emit(thread.thread_id, "usage.updated", message_id=packet["assistant_message_id"],
                    data={"run_id": packet["run_id"], "usage": summary})

        usage.set_usage_update_callback(publish_usage)
        async with runner._open_agent_runtime(files_root=files_root) as runtime:
            if packet.get("task_cost"):
                runtime["capacity_tools"] = self.research_capacity_tools(packet)
            if packet["entrypoint"] in {"research", "persistent_research"}:
                runtime["background_tools"] = self.background_tools(packet)
                runtime["pending_tasks"] = lambda: self.active_subagent_parts(thread.thread_id)
                runtime["collaboration_middleware"] = self.research_collaboration_middleware(packet)
            agent = await runner._build_entry_agent(entrypoint=packet["entrypoint"], runtime=runtime,
                thread_id=thread.thread_id, tool_thread_id=thread.thread_id)
            yield agent.with_config(callbacks=runner._langchain_callbacks(
                usage_handler=usage, default_agent_name=packet["entrypoint"]))

    async def execute_turn(self, packet):
        # DBOS marks cancellation before a preemptible step has unwound. Its
        # queue can release the partition during that interval. This local
        # writer mutex only covers that teardown boundary; it neither owns
        # task state nor schedules/retries work.
        async with self._graph_writers.setdefault(packet["thread_id"], asyncio.Lock()):
            return await self._execute_turn(packet)

    async def _execute_turn(self, packet):
        tid, rid = packet["thread_id"], packet["run_id"]
        current = self._thread(tid)
        ancestor = current
        while ancestor.parent_thread_id:
            ancestor = self._thread(ancestor.parent_thread_id)
            if ancestor.meta.get("automation_paused") or not self._persistent_automation_enabled(ancestor):
                self._projection(packet).finalize(native_status="interrupted")
                return {"status": "interrupted", "message_id": packet["assistant_message_id"]}
        # Previously accepted discussion-triggered root reviews are no longer
        # executable work. Keep their conversation records without reviving them.
        if packet["metadata"].get("catmaster_discussion_review_graph") or (
                packet["metadata"] and (current.meta.get("automation_paused") or
                not self._persistent_automation_enabled(current))):
            self._projection(packet).finalize(native_status="interrupted")
            return {"status": "interrupted", "message_id": packet["assistant_message_id"]}
        settled = self._settled[rid] = asyncio.Event()
        projection = self._projection(packet)
        thread = self._thread(tid)
        self.store.update_thread(tid, status="running", active_run_id=rid,
                                 active_message_id=packet["assistant_message_id"])
        self.store.update_message(tid, packet["assistant_message_id"], status="streaming")
        self.broker.emit(tid, "thread.status", status="running", data={"run_id": rid, "status": "running"})
        await self._publish_task(tid, "running", rid)
        nested = AsyncActivityProjection(store=self.store, broker=self.broker, thread_id=tid,
            run_id=rid, source=packet["entrypoint"],
            on_update=lambda text, source, tool: self._activity_update(tid, rid, text, source, tool))
        state = None
        capacity_paused = False
        try:
            async with self._graph(packet) as graph:
                config = {"configurable": {"thread_id": thread.deepagent_thread_id,
                                           "project_id": self.workspace_id},
                          "metadata": {"catmaster_run_id": rid, "lc_agent_name": packet["entrypoint"]},
                          "recursion_limit": 1000}
                state = await graph.aget_state(config)
                same_turn = (state.metadata or {}).get("catmaster_run_id") == rid
                if packet.get("start_checkpoint_id") and not same_turn:
                    config["configurable"]["checkpoint_id"] = packet["start_checkpoint_id"]
                    state = await graph.aget_state(config)
                # A durable native anchor also makes Replace work on the first
                # turn. Do not tag the empty anchor as an executed turn.
                if not (state.config or {}).get("configurable", {}).get("checkpoint_id"):
                    anchor = await graph.aupdate_state({"configurable": config["configurable"]},
                                                      None, as_node=END)
                    state = await graph.aget_state(anchor)
                current_message = self.store.get_message(tid, packet["assistant_message_id"])
                if not same_turn and "checkpoint_before_turn" not in current_message.meta:
                    self.store.update_message(tid, current_message.id, meta={**current_message.meta,
                        "checkpoint_before_turn": state.config["configurable"]["checkpoint_id"]})
                if same_turn and "checkpoint_id" in config["configurable"]:
                    config["configurable"].pop("checkpoint_id")
                if not packet["input_message_id"]:
                    # A resume starts after the preceding checkpoint message.
                    messages = state.values.get("messages") or []
                    projection.input_message_id = getattr(messages[-1], "id", "") if messages else ""
                if packet.get("capacity_resume") and any(
                    isinstance(i.value, dict) and i.value.get("kind") == "research_capacity" for i in state.interrupts
                ):
                    run_input = Command(resume=packet["capacity_resume"]["decision"])
                elif same_turn:
                    run_input = None  # Native resume; never append the original input again.
                elif packet["command"] is not None:
                    run_input = Command(**packet["command"]) if packet["command"] else None
                else:
                    message = self.store.get_message(tid, packet["input_message_id"])
                    prepared = self._saved_attachments(message, packet)
                    content = self._research_graph_turn_content(prompt=_text(message),
                        turn_content=build_turn_content(_text(message), prepared), entrypoint=packet["entrypoint"],
                        research_context=packet["research"], internal_kind=thread.meta.get("internal_kind", ""))
                    run_input = {"messages": [HumanMessage(id=message.id, content=content)]}
                if not same_turn or state.next:
                    # Installed LangGraph 1.2.11's public astream(version='v2')
                    # returns typed {type, ns, data}; do not invent server events.
                    async for chunk in graph.astream(run_input, config, version="v2", subgraphs=True,
                        stream_mode=["messages", "updates", "custom"], durability="sync"):
                        projection.process(chunk)
                        if chunk.get("ns"):
                            nested.process(chunk)
                        else:
                            self._top_activity(tid, rid, chunk)
                    config["configurable"].pop("checkpoint_id", None)
                    state = await graph.aget_state(config)
                status = "interrupted" if state.next or state.interrupts else "success"
                capacity_pause = next((i.value for i in state.interrupts if isinstance(i.value, dict)
                                       and i.value.get("kind") == "research_capacity"), None)
                if capacity_pause and packet.get("task_cost"):
                    capacity_paused = True
                    # A scheduling pause is neither a user approval nor a result.
                    return {"status": "capacity_wait", "message_id": packet["assistant_message_id"],
                            "task_cost": capacity_pause["task_cost"],
                            "checkpoint_id": state.config["configurable"]["checkpoint_id"]}
                state_view = {"values": state.values, "interrupts": [_plain_dict(i) for i in state.interrupts]}
                completed = projection.finalize(native_status=status, state=state_view)
                if status == "success" and completed.status != "completed":
                    status = "error"
                result = {"status": status, "message_id": completed.id,
                          "checkpoint_id": (state.config or {}).get("configurable", {}).get("checkpoint_id", "")}
                self.store.update_thread(tid, status="interrupted" if status == "interrupted" else
                    "error" if status == "error" else "idle", active_run_id="", active_message_id="")
                await self._publish_task(tid, status, rid)
                return result
        except asyncio.CancelledError:
            projection.finalize(native_status="interrupted")
            self.store.update_thread(tid, status="stopped", active_run_id="", active_message_id="")
            await self._publish_task(tid, "interrupted", rid)
            raise
        except Exception as exc:
            logger.exception("Local graph failed: %s", rid)
            completed = projection.finalize(native_status="error", error=str(exc))
            # A failed native graph remains resumable; its execution result is
            # an error even though delivery of that result can succeed.
            self.store.update_message(tid, completed.id, meta={**completed.meta, "checkpoint_resume_available": True})
            for part in completed.parts:
                if part.type == "error":
                    updated = self.store.update_part(tid, completed.id, part.id,
                        meta={**part.meta, "checkpoint_resume_available": True})
                    updated_part = next(p for p in updated.parts if p.id == part.id)
                    self.broker.emit(tid, "message.part.updated", message_id=completed.id,
                        data={"part": updated_part.model_dump(mode="json")})
            self.store.update_thread(tid, status="error", active_run_id="", active_message_id="")
            await self._publish_task(tid, "error", rid)
            return {"status": "error", "message_id": completed.id, "error": str(exc)}
        finally:
            if not capacity_paused:
                thread_status = self.store.get_thread(tid).status
                nested.finish("success" if thread_status == "idle" else
                              "interrupted" if thread_status in {"stopped", "interrupted"} else "error")
            settled.set()
            self._settled.pop(rid, None)
            remaining = [r for r in await self.execution.runs(self.workspace, tid, active=True)
                         if r.workflow_id != rid]
            if remaining:
                self.store.update_thread(tid, status="running", active_run_id=remaining[0].workflow_id)
            self.broker.emit(tid, "thread.status", status=self.store.get_thread(tid).status.value,
                data={"run_id": rid, "thread": self.store.get_thread(tid).model_dump(mode="json")})

    async def publish_acceptance(self, packet):
        thread = await self.refresh_thread_status(packet["thread_id"])
        self.broker.emit(thread.thread_id, "thread.status", status=thread.status.value,
                         data={"thread": thread.model_dump(mode="json"), "run_id": packet["run_id"]})

    def _saved_attachments(self, message, packet):
        # The accepted UI record contains artifact references, never a copy of
        # the whole agent state. Re-open media only for this submitted turn.
        attachments = []
        for item in message.structured_sidecar.get("attachments", []):
            path = (self.workspace / item["workspace_path"]).resolve()
            path.relative_to(self.workspace)
            block = None
            if item.get("sent_to_model"):
                block = text_attachment_block(path.read_text(), filename=item["filename"],
                    workspace_path=item["workspace_path"]) if item.get("sent_as") == "text_excerpt" else \
                    file_to_content_block(path, mime_type=item["mime_type"], kind=item["kind"], filename=item["filename"])
            attachments.append(PreparedAttachment(**{k: item[k] for k in ["artifact_id", "workspace_path", "filename",
                "mime_type", "size_bytes", "kind"]}, current_turn_block=block, warnings=item.get("warnings", [])))
        return attachments

    async def prepare_completion(self, packet, result):
        tid = packet["thread_id"]
        thread = self.store.get_thread(tid)
        if self.on_run_finished:
            answer = self.store.get_message(tid, packet["assistant_message_id"])
            await self.on_run_finished(workspace=self.workspace, workspace_id=self.workspace_id,
                thread_id=tid, terminal_status=result["status"], run_id=packet["run_id"],
                assistant_message_id=packet["assistant_message_id"], message_id=packet["input_message_id"],
                episode_id=answer.meta.get("episode_id", ""),
                prior_assistant_message_id=answer.meta.get("prior_assistant_message_id", ""),
                entrypoint=packet["entrypoint"], research_launch_id=packet["research"].get("research_launch_id", ""),
                research_graph_id=packet["research"].get("research_graph_id", ""),
                model_config=packet["model_config"], permission_mode=packet["permission_mode"],
                run_dir=str(self.workspace / "metadata/runs" / packet["run_id"]))
        if (not thread.parent_thread_id or thread.meta.get("on_completion", "resume_parent") == "notify"
                or result["status"] == "interrupted" or thread.meta.get("last_run_id") != packet["run_id"]):
            return {}
        if result["status"] == "success" and await self._waiting_research_children(thread):
            return {}  # An interim turn is not this research branch's handoff.
        parent = self.store.get_thread(thread.parent_thread_id)
        if parent.meta.get("automation_paused") or not self._persistent_automation_enabled(parent):
            return {}
        text = (f"Background task {tid} ({thread.meta.get('agent_name', thread.entrypoint)}) "
                f"finished with status {result['status']}. Its result is saved in message {result['message_id']}.\n\n"
                + _text(self.store.get_message(tid, result["message_id"])))
        return await self._prepare_turn(parent.thread_id, ThreadSubmitRequest(text=text,
            entrypoint=parent.entrypoint, strategy="enqueue"),
            run_metadata={"catmaster_async_completion_run_id": packet["run_id"],
                          "catmaster_notification_title": "Background task finished"},
            identity="completion-" + str(uuid.uuid5(uuid.NAMESPACE_URL, packet["run_id"])))

    async def _cancel(self, run_id, *, thread_id="", preserve_capacity=False):
        settled = self._settled.get(run_id)
        await self.execution.cancel(run_id)
        if settled is not None:
            # DBOS cancellation is a request. Do not allow a new turn to write
            # this checkpoint until the cancelled graph actually unwinds.
            await settled.wait()
        if thread_id:
            message = self.store.get_message(thread_id, "answer_" + run_id)
            if message and message.status in {"created", "streaming"} and message.structured_sidecar.get("execution"):
                # A capacity wait has no executing graph to finalize its UI row.
                self._projection(message.structured_sidecar["execution"]).finalize(native_status="interrupted")
        if not preserve_capacity:
            await self.execution.release_capacity(run_id)

    async def stop(self, *, thread_id, payload):
        async with self._admission.setdefault(thread_id, asyncio.Lock()):
            return await self._stop(thread_id=thread_id, payload=payload)

    async def _stop(self, *, thread_id, payload):
        thread = self._thread(thread_id)
        self.store.update_thread(thread_id, meta={**thread.meta, "automation_paused": True})
        runs = await self.execution.runs(self.workspace, thread_id, active=True)
        selected = [r for r in runs if not payload.run_id or r.workflow_id == payload.run_id]
        if payload.run_id and not selected:
            run = await self.execution.run(payload.run_id)
            if run and not payload.run_id.startswith(thread_key(self.workspace, thread_id) + ":"):
                raise HTTPException(404, "Run does not belong to this conversation.")
        self.store.update_thread(thread_id, status="stopping")
        for run in selected:
            await self._cancel(run.workflow_id, thread_id=thread_id)
        if payload.action == "rollback" and selected:
            previous = self.store.get_message(thread_id, "answer_" + selected[0].workflow_id)
            checkpoint = previous.meta.get("checkpoint_before_turn") if previous else None
            if checkpoint:
                async with self._graph_writers.setdefault(thread_id, asyncio.Lock()):
                    async with self._graph(previous.structured_sidecar["execution"]) as graph:
                        await graph.aupdate_state({"configurable": {
                            "thread_id": thread.deepagent_thread_id, "checkpoint_ns": "", "checkpoint_id": checkpoint}},
                            None, as_node=END)
        self.store.update_thread(thread_id, status="stopped", active_run_id="", active_message_id="")
        self.broker.emit(thread_id, "thread.status", status="stopped", data={"status": "stopped"})
        return {"thread": self.store.get_thread(thread_id), "status": "stopped"}

    async def resume(self, *, thread_id, payload, **kwargs):
        async with self._admission.setdefault(thread_id, asyncio.Lock()):
            if await self.execution.runs(self.workspace, thread_id, active=True):
                raise HTTPException(409, "The conversation already has an active turn.")
            return await self._resume(thread_id=thread_id, payload=payload)

    async def _resume(self, *, thread_id, payload):
        decisions = self._decisions_from_public_actions(thread_id, payload.actions) if payload.actions else payload.decisions
        decisions = self._normalize_native_decisions(thread_id, decisions)
        thread = self._thread(thread_id)
        packet = await self._prepare_turn(thread_id, ThreadSubmitRequest(text=payload.text,
            entrypoint=thread.entrypoint), command={"resume": {"decisions": decisions}})
        await self.execution.enqueue(packet)
        self._resolve_projected_interrupts(thread_id=thread_id, resolution=decisions, resumed_run_id=packet["run_id"])
        return self._submission(packet)

    async def continue_from_checkpoint(self, *, thread_id, payload, **kwargs):
        async with self._admission.setdefault(thread_id, asyncio.Lock()):
            return await self._continue_from_checkpoint(thread_id=thread_id, payload=payload)

    async def _continue_from_checkpoint(self, *, thread_id, payload):
        thread = self._thread(thread_id)
        message = self.store.get_message(thread_id, payload.message_id)
        if not message or message.status not in {"failed", "interrupted"}:
            raise HTTPException(409, "Select the interrupted or failed turn to continue.")
        if await self.execution.runs(self.workspace, thread_id, active=True):
            raise HTTPException(409, "The conversation already has an active turn.")
        if message.meta.get("run_id") != thread.meta.get("last_run_id"):
            raise HTTPException(409, "Only the latest interrupted turn can be continued.")
        previous = message.structured_sidecar.get("execution")
        if not previous:
            raise HTTPException(409, "This saved conversation needs a new user message to continue.")
        async with self._graph(previous) as graph:
            state = await graph.aget_state({"configurable": {"thread_id": thread.deepagent_thread_id}})
            if any(not isinstance(i.value, dict) or i.value.get("kind") != "research_capacity" for i in state.interrupts):
                raise HTTPException(409, "Review the pending approval before continuing.")
            if not state.next:
                raise HTTPException(409, "There is no interrupted graph step; send a new message to continue.")
        packet = await self._prepare_turn(thread_id, ThreadSubmitRequest(text="", entrypoint=thread.entrypoint), command={})
        await self.execution.enqueue(packet)
        return self._submission(packet)

    async def refresh_thread_status(self, thread_id):
        thread = self.store.get_thread(thread_id)
        runs = await self.execution.runs(self.workspace, thread_id, active=True)
        if runs:
            current = next((r for r in runs if r.status == "PENDING"), runs[0])
            thread = self.store.update_thread(thread_id, status="running", active_run_id=current.workflow_id,
                meta={**thread.meta, "queued_runs": [r.workflow_id for r in runs if r.status != "PENDING"]})
        elif not any(rid.startswith(thread_key(self.workspace, thread_id) + ":") for rid in self._settled):
            if thread.meta.get("queued_runs"):
                thread = self.store.update_thread(thread_id, meta={**thread.meta, "queued_runs": []})
            if thread.status in {"running", "stopping"}:
                last = await self.execution.run(thread.meta.get("last_run_id", ""))
                status = _status(last)
                thread = self.store.update_thread(thread_id, status="idle" if status == "success" else
                    "error" if status == "error" else "stopped", active_run_id="", active_message_id="",
                    meta={**thread.meta, "queued_runs": []})
        return thread

    async def reconcile(self):
        # DBOS performs crash recovery. Startup only reconciles small UI rows;
        # it does not load checkpoint histories or manufacture accepted work.
        for thread in self.store.list_threads():
            await self.refresh_thread_status(thread.thread_id)
        return 0

    async def reconcile_async_subagents(self, thread_id=""):
        return 0  # Completion delivery is part of the durable workflow itself.

    def research_collaboration_middleware(self, packet):
        from catmaster.research.discussions import ResearchDiscussions
        from catmaster.specialists.research_collaboration import ResearchCollaborationMiddleware
        thread = self._thread(packet["thread_id"])
        if thread.entrypoint not in {"research", "persistent_research"} or not thread.active_research_graph_id:
            return None
        discussions = ResearchDiscussions(self.workspace)
        if not discussions.persistent_owner(thread.active_research_graph_id, thread.thread_id):
            return None
        return ResearchCollaborationMiddleware(discussions=discussions,
            graph_id=thread.active_research_graph_id, thread_id=thread.thread_id, run_id=packet["run_id"],
            publish_progress=lambda message_id, data: self.publish_research_progress(packet, message_id, data))

    def background_tools(self, packet):
        service = self
        collaboration = self.research_collaboration_middleware(packet)
        @tool
        async def start_async_task(agent: Literal["research_specialist", "research_challenger", "experiment_specialist", "writing_specialist", "peer_review_specialist", "litreview_agent"],
                                   description: str, runtime: ToolRuntime,
                                   on_completion: Literal["resume_parent", "notify"] = "resume_parent",
                                   task_cost: Literal["low", "medium", "high"] = "medium") -> dict:
            """Start a fresh isolated background specialist and return its task_id immediately.

            description is the complete objective, references, constraints and deliverable.
            resume_parent continues this conversation after completion; notify only updates
            task status. Use a new task for independent work, update_async_task for continuity.
            If only background results remain to be awaited, return an interim update
            and end this turn; resume_parent supplies a later completion input. Do not
            keep the foreground turn alive with sleeps or repeated status polling.
            Experiment handles computations; Writing produces reports; Peer Review reviews;
            litreview_agent discovers and synthesizes literature. All share this workspace.
            research_specialist owns an independent hypothesis-method-result research question,
            including interpretation and its next test; use distinct questions to explore in parallel.
            research_challenger independently checks narrowed premises, overlooked literature
            and methods, and premature completion or external waiting. Supply the original
            user objective/constraints, decisive source and Result references, and the open
            decision, rather than a prescribed answer or method. It researches alternatives
            and recommends a bounded next check; it does not execute experiments or own
            graph-wide completion. Use task_cost=low for this evidence consultation.
            Native synchronous delegates work within this task and share its slot.
            An async delegate is a separate independent task and acquires its own slot
            in the same shared pool. End an interim turn when awaiting only such tasks;
            their own reservations cover their work while this researcher is idle.
            task_cost is the highest expected cost of the formed task, not each tool/job:
            low for literature, evidence analysis and small Python/ML baselines;
            medium for MLFF exploration or substantial model training; high for DFT
            campaigns or laboratory work (including their cheaper prerequisites).
            An open question may start low and change cost when its method becomes concrete.
            Specify low for evidence-only work; omission uses medium for an unspecified task.
            Slots cover the entire task including computational waits; limits do not authorize work.
            """
            child_id = str(uuid.uuid5(uuid.NAMESPACE_URL, packet["run_id"] + ":" + runtime.tool_call_id))
            try:
                child = service.store.get_thread(child_id)
            except KeyError:
                child = service.store.create_thread(thread_id=child_id, deepagent_thread_id=child_id,
                    parent_thread_id=packet["thread_id"], entrypoint=SPECIALISTS[agent],
                    title=" ".join(description.split())[:120],
                    meta={"agent_name": agent, "on_completion": on_completion,
                          "parent_message_id": packet["assistant_message_id"],
                          "task_description": description, "model_config": packet["model_config"],
                          "permission_mode": packet["permission_mode"], "background_task": True,
                          "task_cost": task_cost, "research_branch": agent == "research_specialist",
                          "internal_kind": "async_subagent_activity"})
                service.store.update_thread(child_id,
                    active_research_graph_id=packet["research"].get("research_graph_id", ""))
            child_packet = await service._prepare_turn(child_id,
                ThreadSubmitRequest(text=description, entrypoint=child.entrypoint),
                research_context={**packet["research"], "research_launch_id": ""}, identity="initial")
            await service.execution.enqueue(child_packet)
            await service._publish_task(child_id, "pending", child_packet["run_id"])
            return await service.task(packet["thread_id"], child_id)

        @tool
        async def check_async_task(task_id: str) -> dict:
            """Read a child or same-graph task's complete brief, current work, plan reference and saved result.

            Reading a peer does not grant permission to update or stop its execution.
            """
            return await service.read_task(packet["thread_id"], task_id)

        @tool
        async def list_async_tasks(offset: int = 0, limit: int = 20,
                                   scope: Literal["children", "research_graph"] = "children") -> dict:
            """List task goals, status and explicit work updates; paginate with offset.

            research_graph includes peer tasks in the same bound graph so in-progress
            work can be discovered before a Result exists. children is the default.
            The complete saved result remains available through check_async_task.
            """
            return await service.list_research_tasks(packet["thread_id"], scope=scope, offset=offset, limit=limit)

        if collaboration is None:
            @tool
            async def check_async_task(task_id: str) -> dict:
                """Read this conversation's child's complete brief, status and saved result."""
                return await service.read_task(packet["thread_id"], task_id)

            @tool
            async def list_async_tasks(offset: int = 0, limit: int = 20) -> dict:
                """List this conversation's child tasks; paginate with offset."""
                return await service.list_research_tasks(packet["thread_id"], offset=offset, limit=limit)

        @tool
        async def post_research_message(body: str, runtime: ToolRuntime, title: str = "",
                                        reply_to: str = "", target_task_id: str = "",
                                        node_id: str = "", references: list[str] = [],
                                        resolves_message_id: str = "",
                                        review_outcome: Literal["addressed", "deferred", "follow_up"] = "addressed") -> dict:
            """Post a shared scientific question, partial finding or reply in the bound graph.

            A new topic needs title. To reply, pass the returned message_id in reply_to;
            the reply inherits the topic/node and addresses the original author unless
            target_task_id is supplied. Targeting is a notification, not a private mailbox.
            references may contain DOI/URL, workspace paths or scientific node IDs.
            All graph participants can read the full research_discussions SQL table.
            Only research/persistent_research entrypoints accept targeted notices;
            leave target_task_id empty for other shared discussion.
            Messages never start or wake an agent. Persistent branches can exchange
            findings directly; their main researcher reads shared discussion during
            its existing turns and decides which questions merit further investigation.
            The main researcher can optionally record a decision on ANY message by
            replying with reply_to AND resolves_message_id set to that message_id:
            addressed for an answer/correction, deferred for a reason not to pursue
            now, or follow_up after actually accepting a bounded continuation with
            update_async_task. For follow_up, include its returned task_id in body
            and use it as target_task_id when it is a Research branch. For nested
            tasks, continue the owning branch; peer read access does not grant control.
            This records a decision, not execution success. It never itself schedules
            work. Ordinary peer replies omit resolves_message_id.
            Use for consequential questions or coordination, not routine progress.
            """
            from catmaster.research.discussions import DiscussionPostRequest, ResearchDiscussions
            graph_id = service._thread(packet["thread_id"]).active_research_graph_id
            result = await asyncio.to_thread(ResearchDiscussions(service.workspace).post, graph_id,
                DiscussionPostRequest(body=body, title=title, reply_to=reply_to,
                    target_task_id=target_task_id, node_id=node_id, references=references,
                    resolves_message_id=resolves_message_id, review_outcome=review_outcome),
                author_thread_id=packet["thread_id"],
                message_id="discussion_" + str(uuid.uuid5(uuid.NAMESPACE_URL, packet["run_id"] + runtime.tool_call_id)))
            return result

        @tool
        async def research_pool_status() -> dict:
            """Read shared task concurrency and waiting counts by cost, to plan independent research branches.

            A busy expensive tier does not block cheaper evidence work. Choose useful
            independent questions yourself; idle capacity is not a reason to invent work.
            """
            return await asyncio.to_thread(service.execution.capacity.snapshot)

        @tool
        async def update_async_task(task_id: str, message: str, runtime: ToolRuntime,
                                    strategy: Annotated[Literal["enqueue", "interrupt"],
                                        "Choose interrupt to correct the current task's premise, source claim, method, "
                                        "scope or direction before it continues; choose enqueue for additional evidence "
                                        "or later questions that leave the current work valid. Enqueue is not immediate delivery."] = "enqueue") -> dict:
            """Send a follow-up to the same researcher with its prior context.

            Prefer interrupt when retracting a premise or correcting an instruction
            the researcher is currently using, including an unsupported source-access
            claim. State the replacement and which existing results remain usable;
            do not leave the known error in force until the task finishes. Interrupt
            preserves checkpoints and starts a corrected turn after execution unwinds.
            Use enqueue for additional evidence or later questions when current work
            remains valid. A purely cosmetic edit can wait. Enqueue delivers only
            after the current turn returns, including any synchronous nested worker.
            Neither strategy cancels already submitted remote scientific jobs.
            """
            return await service.update_async_subagent(parent_thread_id=packet["thread_id"], task_id=task_id,
                message=message, strategy=strategy,
                identity=str(uuid.uuid5(uuid.NAMESPACE_URL, packet["run_id"]+runtime.tool_call_id)))

        @tool
        async def cancel_async_task(task_id: str) -> dict:
            """Stop this background agent, preserving progress; does not cancel remote scientific jobs."""
            return await service.stop_async_subagent(parent_thread_id=packet["thread_id"], task_id=task_id)
        result = [start_async_task, check_async_task, list_async_tasks, update_async_task, cancel_async_task,
                  research_pool_status]
        if collaboration is not None:
            result.append(post_research_message)
        return result

    async def publish_research_progress(self, packet, message_id, data):
        child = self._thread(packet["thread_id"])
        if child.meta.get("research_progress_message_id") == message_id:
            return
        progress = {"summary": str(data.get("summary", "")), "next_step": str(data.get("next_step", ""))}
        self.store.update_thread(child.thread_id, meta={**child.meta, "research_progress": progress,
            "research_progress_message_id": message_id})
        self._activity_update(child.thread_id, packet["run_id"], "", "", "")
        self._research_task_event(child, "task.progress")

    def _research_task_event(self, child, change):
        if child.active_research_graph_id:
            from catmaster.research.knowledge_graph.store import ResearchGraphStore
            from catmaster.storage import connect_workspace_db
            with connect_workspace_db(self.workspace) as connection:
                graph = connection.execute("SELECT revision FROM research_graphs WHERE graph_id=?",
                    (child.active_research_graph_id,)).fetchone()
                if graph:
                    ResearchGraphStore._write_event(connection, graph_id=child.active_research_graph_id,
                        revision=graph["revision"], change=change, thread_id=child.thread_id)

    async def read_task(self, viewer_id, task_id, *, include_result=True):
        viewer, task = self._thread(viewer_id), self._thread(task_id)
        if not task.meta.get("background_task") or (task.parent_thread_id != viewer_id and
                (not viewer.active_research_graph_id or viewer.active_research_graph_id != task.active_research_graph_id)):
            raise HTTPException(404, "Task does not belong to this conversation or research graph.")
        if task.parent_thread_id != viewer_id:
            from catmaster.research.discussions import ResearchDiscussions
            discussions = ResearchDiscussions(self.workspace)
            owner = discussions.persistent_owner(viewer.active_research_graph_id, viewer_id)
            if not owner or discussions.persistent_owner(task.active_research_graph_id, task_id) != owner:
                raise HTTPException(404, "Peer task discovery belongs to the Persistent Research session.")
        return await self.task(task.parent_thread_id, task_id, include_result=include_result)

    async def list_research_tasks(self, viewer_id, *, scope="children", offset=0, limit=20):
        viewer = self._thread(viewer_id)
        if scope == "research_graph" and not viewer.active_research_graph_id:
            raise HTTPException(400, "Select a Research Graph before querying peer tasks.")
        tasks = [t for t in self.store.list_threads() if t.meta.get("background_task") and
                 (t.active_research_graph_id == viewer.active_research_graph_id if scope == "research_graph"
                  else t.parent_thread_id == viewer_id)]
        if scope == 'research_graph':
            from catmaster.research.discussions import ResearchDiscussions
            discussions = ResearchDiscussions(self.workspace)
            owner = discussions.persistent_owner(viewer.active_research_graph_id, viewer_id)
            if not owner:
                raise HTTPException(400, "Peer task discovery belongs to a Persistent Research session.")
            tasks = [t for t in tasks if discussions.persistent_owner(viewer.active_research_graph_id, t.thread_id) == owner]
        tasks.sort(key=lambda t: (t.created_at, t.thread_id), reverse=True)
        offset, limit = max(0, offset), max(1, min(100, limit))
        return {"tasks": [await self.read_task(viewer_id, t.thread_id, include_result=False)
                          for t in tasks[offset:offset + limit]],
                "next_offset": offset + limit if offset + limit < len(tasks) else None}

    async def graph_tasks(self, graph_id, *, offset=0, limit=20):
        """Workspace-authorized UI projection; no peer execution control is exposed."""
        tasks = [t for t in self.store.list_threads() if t.meta.get("background_task")
                 and t.active_research_graph_id == graph_id]
        tasks.sort(key=lambda t: (t.created_at, t.thread_id), reverse=True)
        offset, limit = max(0, offset), max(1, min(100, limit))
        from catmaster.research.discussions import ResearchDiscussions
        discussions = ResearchDiscussions(self.workspace)
        return {"tasks": [{**await self.task(t.parent_thread_id, t.thread_id, include_result=False),
                           'accepts_discussion': t.entrypoint in {'research', 'persistent_research'}
                           and bool(discussions.persistent_owner(graph_id, t.thread_id))}
                          for t in tasks[offset:offset + limit]],
                "next_offset": offset + limit if offset + limit < len(tasks) else None}

    def research_capacity_tools(self, packet):
        @tool
        def set_research_task_cost(task_cost: Literal["low", "medium", "high"], reason: str) -> dict:
            """Change this whole task's cost before starting work at a newly justified tier.

            Call alone, after prior delegated work has returned. Low: literature, small
            Python/ML baselines. Medium: MLFF or substantial model training. High: DFT
            or external experiments, including their preparation. State the formed method
            and why it changes expected verification cost; do not estimate compute hours.
            Capacity checks, queueing, and resumption are automatic. With
            available capacity, this call completes without waiting for a slot; otherwise,
            it waits in the background and returns after the task is automatically admitted.
            A successful result contains task_cost and admitted=true. Continue work at the
            granted tier when this call returns. Do not downgrade while expensive work
            remains in progress. Cost admission never grants scientific or experimental authorization.
            """
            return interrupt({"kind": "research_capacity", "task_cost": task_cost, "reason": reason})
        return [set_research_task_cost]

    async def publish_capacity(self, packet, tier, granted):
        child = self.store.get_thread(packet["thread_id"])
        state = "active" if granted else "waiting"
        if child.meta.get("capacity_state") != state or child.meta.get("task_cost") != tier:
            self.store.update_thread(child.thread_id, status="running", active_run_id=packet["run_id"],
                meta={**child.meta, "task_cost": tier, "capacity_state": state})
            await self._publish_task(child.thread_id, "running" if granted else "pending", packet["run_id"])
        await self.capacity_changed(packet)

    async def capacity_changed(self, packet):
        child = self.store.get_thread(packet["thread_id"])
        if not child.parent_thread_id or child.meta.get("on_completion") == "notify":
            return
        parent = self.store.get_thread(child.parent_thread_id)
        while parent.parent_thread_id:
            if parent.meta.get("on_completion") == "notify" or parent.meta.get("automation_paused"):
                return
            parent = self.store.get_thread(parent.parent_thread_id)
        async with self._admission.setdefault(parent.thread_id, asyncio.Lock()):
            parent = self.store.get_thread(parent.thread_id)
            if parent.entrypoint != "persistent_research" or parent.meta.get("automation_paused") or not self._persistent_automation_enabled(parent):
                return
            snapshot = await asyncio.to_thread(self.execution.capacity.snapshot)
            children, owned = {}, {parent.thread_id}
            threads = self.store.list_threads()
            while True:
                found = {t.thread_id: t for t in threads if t.parent_thread_id in owned
                         and t.thread_id not in owned and t.meta.get("on_completion") != "notify"
                         and not t.meta.get("automation_paused")}
                if not found:
                    break
                children.update(found)
                owned.update(found)
            waiting = [children[r["thread_id"]] for r in await asyncio.to_thread(self.execution.capacity.rows)
                       if r["workspace"] == str(self.workspace) and r["thread_id"] in children
                       and r["state"] == "waiting" and r["tier"] == "high"]
            low = snapshot["tiers"]["low"]
            pressure = bool(waiting and low["active"] < low["limit"]
                            and snapshot["active"] < snapshot["agent_pool_size"])
            sequence = int(parent.meta.get("research_capacity_notice_seq", 0))
            if pressure and not parent.meta.get("research_capacity_notice_active"):
                sequence += 1
                notice = await self._prepare_turn(parent.thread_id, ThreadSubmitRequest(
                    entrypoint=parent.entrypoint, strategy="enqueue", text=(
                        "Research capacity changed: expensive research tasks are waiting while cheaper evidence capacity is available. "
                        "Review whether an independent evidence question or alternative method can advance the authorized objective. "
                        "Choose scientifically useful work yourself; do not manufacture work to fill slots or repeat an unchanged plan. "
                        "The existing expensive tasks remain queued. Current pool: " + json.dumps(snapshot)
                        + "\nWaiting questions: " + json.dumps([{"task_id": t.thread_id,
                            "goal": t.meta.get("task_description", "")} for t in waiting], ensure_ascii=False))),
                    run_metadata={"catmaster_notification_title": "Research capacity available"},
                    identity=f"capacity-{sequence}")
                await self.execution.enqueue(notice)
            parent = self.store.get_thread(parent.thread_id)
            self.store.update_thread(parent.thread_id, meta={**parent.meta, "research_capacity_notice_active": pressure,
                                                           "research_capacity_notice_seq": sequence})

    def _child(self, parent_id, task_id):
        child = self.store.get_thread(task_id)
        if child.parent_thread_id != parent_id or not child.meta.get("background_task"):
            raise HTTPException(404, "Task does not belong to this conversation.")
        return child

    async def _waiting_research_children(self, thread):
        if not thread.meta.get("research_branch") or thread.meta.get("automation_paused"):
            return False
        for child in self.store.list_threads():
            if (child.parent_thread_id == thread.thread_id and child.meta.get("background_task")
                    and child.meta.get("on_completion", "resume_parent") == "resume_parent"):
                task = await self.task(thread.thread_id, child.thread_id, include_result=False)
                if task["status"] in {"pending", "running"}:
                    return True
        return False

    async def task(self, parent_id, task_id, *, include_result=True):
        child = self._child(parent_id, task_id)
        rid = child.meta.get("last_run_id", "")
        run = await self.execution.run(rid) if rid else None
        result_id = "answer_" + rid if rid else child.meta.get("saved_result_message_id", "")
        result = self.store.get_message(task_id, result_id) if include_result and result_id else None
        capacity_state = child.meta.get("capacity_state", "")
        status = ("pending" if run and run.status in ACTIVE and capacity_state == "waiting"
                  else _status(run) if rid else child.meta.get("saved_task_status", "interrupted"))
        if status == "success" and await self._waiting_research_children(child):
            status, capacity_state = "pending", "waiting_children"
        return {"task_id": task_id, "thread_id": task_id, "run_id": rid,
                "agent_name": child.meta.get("agent_name", child.entrypoint),
                "entrypoint": child.entrypoint,
                "status": status,
                "task_cost": child.meta.get("task_cost", "low"),
                "capacity_state": capacity_state,
                "on_completion": child.meta.get("on_completion", "resume_parent"),
                "task_description": child.meta.get("task_description", ""),
                "progress": child.meta.get("research_progress", {}),
                "focus_node_id": child.research_focus_node_id,
                "instructions": {"original": child.meta.get("task_description", ""),
                                 "followup": child.meta.get("task_followup", ""),
                                 "followup_status": await self._followup_status(child, run)},
                "result": _text(result) if result and result.status != "streaming" else ""}

    async def _followup_status(self, child, run=None):
        # UI projection of the accepted DBOS run, not a read receipt or a queue.
        rid = child.meta.get("task_followup_run_id", "")
        if not rid:
            return ""  # Older briefs have no accepted-run association.
        if run is None or run.workflow_id != rid:
            run = await self.execution.run(rid)
        if run and run.status in ACTIVE and child.meta.get("capacity_state") == "waiting":
            return "pending"
        return _status(run) if run else ""

    async def active_subagent_parts(self, parent_id):
        """Use the same lifecycle source as task details, never historical cards."""
        from .projections.messages import project_part

        parts = []
        for child in sorted(await asyncio.to_thread(self.store.list_threads), key=lambda t: (t.created_at, t.thread_id)):
            if child.parent_thread_id != parent_id or not child.meta.get("background_task"):
                continue
            task = await self.task(parent_id, child.thread_id, include_result=False)
            if task["status"] not in {"pending", "running"}:
                continue
            mid = child.meta.get("parent_message_id", "")
            part_id = "part_subagent_" + _safe_token(child.thread_id, "task")
            saved = await asyncio.to_thread(self.store.get_message_part, parent_id, mid, part_id) if mid else None
            # A re-used thread may already have a new run while its card still
            # describes the previous one. Do not attach that run's old progress.
            same_run = saved is not None and saved.meta.get("run_id") == task["run_id"]
            meta = dict(saved.meta) if same_run else {}
            meta.update(source=task["agent_name"], task_id=child.thread_id,
                thread_id=child.thread_id, run_id=task["run_id"], native_status=task["status"],
                task_description=" ".join(task["task_description"].split())[:180],
                task_followup=task["instructions"]["followup"],
                task_followup_status=task["instructions"]["followup_status"],
                on_completion=task["on_completion"])
            meta.update(task_cost=task["task_cost"], capacity_state=task["capacity_state"])
            meta["research_progress"] = task["progress"]
            parts.append(project_part({"id": part_id, "type": "subagent", "status": task["status"],
                "text": saved.text if same_run else "", "meta": meta},
                workspace=self.workspace, thread_id=parent_id, message_id=mid))
        return parts

    async def open_async_activity(self, parent_thread_id, task_id):
        return self._child(parent_thread_id, task_id), await self.task(parent_thread_id, task_id)

    async def update_async_subagent(self, *, parent_thread_id, task_id, message="", payload=None, identity="", strategy="interrupt", **kwargs):
        message = payload.message if payload is not None else message
        child = self._child(parent_thread_id, task_id)
        packet = await self._prepare_turn(task_id, ThreadSubmitRequest(text=message,
            entrypoint=child.entrypoint, strategy=strategy), identity=identity,
            research_context={"research_graph_id": child.active_research_graph_id,
                              "research_focus_node_id": child.research_focus_node_id,
                              "research_launch_id": ""})
        child = self.store.get_thread(task_id)
        self.store.update_thread(task_id, meta={**child.meta, "task_followup": message,
            "task_followup_run_id": packet["run_id"]})
        await self.execution.enqueue(packet)
        await self._publish_task(task_id, "pending", packet["run_id"])
        return await self.task(parent_thread_id, task_id)

    async def stop_async_subagent(self, *, parent_thread_id, task_id, payload=None, **kwargs):
        from .thread_models import ThreadStopRequest
        self._child(parent_thread_id, task_id)
        await self.stop(thread_id=task_id, payload=payload or ThreadStopRequest())
        return await self.task(parent_thread_id, task_id)

    async def _publish_task(self, tid, status, rid):
        child = self.store.get_thread(tid)
        followup_status = (status if child.meta.get("task_followup_run_id") == rid and status != "pending"
                           else await self._followup_status(child))
        if status in {"success", "error", "interrupted"} and child.meta.get("last_run_id") == rid:
            waiting = status == "success" and await self._waiting_research_children(child)
            child = self.store.update_thread(tid, meta={**child.meta, "capacity_state": "waiting_children" if waiting else ""})
            if waiting:
                status = "pending"
        self._research_task_event(child, "task.updated")
        if not child.parent_thread_id or not child.meta.get("background_task"):
            return
        parent_id = child.parent_thread_id
        mid = child.meta.get("parent_message_id", "")
        if not mid or not self.store.get_message(parent_id, mid):
            return
        projection = RunProjection(store=self.store, broker=self.broker, artifact_registry=self.artifact_registry,
            thread_id=parent_id, run_id=rid, assistant_message_id=mid, text_part_id="unused")
        projection._project_async_tasks({tid: {"thread_id": tid, "run_id": rid, "status": status,
            "agent_name": child.meta.get("agent_name", child.entrypoint)}})
        self._activity_update(tid, rid, "", child.meta.get("agent_name", ""), "",
                              followup_status=followup_status)

    def _activity_update(self, tid, rid, text, source, tool, *, followup_status=None):
        child = self.store.get_thread(tid)
        mid = child.meta.get("parent_message_id", "")
        if not child.parent_thread_id or not mid:
            return
        message = self.store.get_message(child.parent_thread_id, mid)
        part_id = "part_subagent_" + _safe_token(tid, "task")
        part = next((p for p in message.parts if p.id == part_id), None) if message else None
        if part is None or part.meta.get("run_id") != rid:
            return
        meta = {**part.meta, "task_description": " ".join(child.meta.get("task_description", "").split())[:180],
                "research_progress": child.meta.get("research_progress", {}),
                "task_cost": child.meta.get("task_cost", ""), "capacity_state": child.meta.get("capacity_state", ""),
                "task_followup": child.meta.get("task_followup", ""),
                "task_followup_status": (followup_status if followup_status is not None
                    else part.meta.get("task_followup_status", "")),
                "on_completion": child.meta.get("on_completion", "resume_parent"),
                "latest_update": text or part.meta.get("latest_update", ""), "activity_source": source,
                "activity_tool": tool}
        if meta == part.meta:
            return
        updated = self.store.update_part(child.parent_thread_id, mid, part_id,
            text=text or part.text, meta=meta)
        updated_part = next(p for p in updated.parts if p.id == part_id)
        self.broker.emit(child.parent_thread_id, "message.part.updated", message_id=mid,
                         data={"part": updated_part.model_dump(mode="json"), "run_id": rid})

    def _top_activity(self, tid, rid, chunk):
        if chunk.get("type") != "updates":
            return
        for update in _plain_dict(chunk.get("data")).values():
            for raw in _plain_dict(update).get("messages", []):
                message = _plain_dict(raw)
                if message.get("type") != "ai":
                    continue
                calls = message.get("tool_calls") or []
                text, _ = _content_fragments(message.get("content"))
                progress = next((c for c in calls if c.get("name") == "notify_progress"), None)
                if progress:
                    text = str(_plain_dict(progress.get("args")).get("summary") or text)
                self._activity_update(tid, rid, text, "", calls[-1].get("name", "") if calls else "")

    async def reconcile_research_graph_updates(self, graph_id, root_thread_id):
        from catmaster.storage import connect_workspace_db
        parent = self.store.get_thread(root_thread_id)
        if parent.entrypoint != "persistent_research":
            return
        from catmaster.research.knowledge_graph.store import ResearchGraphStore
        graph = ResearchGraphStore(self.workspace).get_graph(graph_id)
        if graph.get("completed") or graph.get("archived"):
            return
        if (parent.meta.get("automation_paused") or not self._persistent_automation_enabled(parent)
                or not parent.meta.get("last_run_id")):
            return
        cursor = int(parent.meta.get("research_event_cursor", 0))
        with connect_workspace_db(self.workspace) as conn:
            rows = conn.execute("SELECT * FROM ui_events WHERE graph_id=? AND event_id>? ORDER BY event_id LIMIT 200",
                                (graph_id, cursor)).fetchall()
        if not rows:
            return
        owned = {root_thread_id, parent.deepagent_thread_id}
        threads = self.store.list_threads()
        # Include the full descendant chain: worker Result writes must not
        # bypass a parent's notify setting via a separate graph event watcher.
        changed = True
        while changed:
            descendants = {t.thread_id for t in threads if t.parent_thread_id in owned}
            changed = not descendants.issubset(owned)
            owned |= descendants
        relevant = []
        for row in rows:
            if row["thread_id"] in owned:
                continue
            payload = json.loads(row["payload_json"])
            if payload.get("change") in {"result.recorded", "result.updated", "claim.revised", "result.judgment_updated"}:
                relevant.append(payload)
            elif (payload.get("change") == "node.updated"
                  and payload.get("details", {}).get("previous_node", {}).get("kind") == "result"):
                relevant.append(payload)
            elif payload.get("change") == "graph.updated":
                changes = payload.get("details", {}).get("changes", {})
                if changes.get("completed") == 0:
                    relevant.append(payload)
        end = int(rows[-1]["event_id"])
        if relevant:
            packet = await self._prepare_turn(root_thread_id, ThreadSubmitRequest(entrypoint=parent.entrypoint,
                text="New evidence or an explicit continuation was recorded in the bound Research Graph. "
                     "Review it within the user's authorized stage; do not repeat completed work on unchanged evidence.\n"
                     + json.dumps(relevant, ensure_ascii=False)),
                run_metadata={"catmaster_research_event_id": end, "catmaster_notification_title": "Research evidence updated"},
                identity=f"graph-event-{end}")
            await self.execution.enqueue(packet)
        current = self.store.get_thread(root_thread_id)
        self.store.update_thread(root_thread_id, meta={**current.meta, "research_event_cursor": end})
