from __future__ import annotations

import json
import re
from dataclasses import asdict, is_dataclass
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from .artifact_registry import ArtifactRegistry
from .thread_events import ThreadEventBroker
from .thread_models import ArtifactPart, MessagePart, ThreadMessage
from .thread_store import ThreadStore, new_id


def _plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "_asdict"):
        dumped = value._asdict()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}


def _safe_token(value: str, fallback: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "").strip())
    return (token or fallback)[:140]


def _content_fragments(content: Any) -> tuple[str, str]:
    if isinstance(content, str):
        return content, ""
    if not isinstance(content, list):
        return "", ""
    text: list[str] = []
    reasoning: list[str] = []
    for item in content:
        if isinstance(item, str):
            text.append(item)
            continue
        block = _plain_dict(item)
        kind = str(block.get("type") or "").strip().lower()
        value = block.get("text")
        if value is None:
            value = block.get("content")
        if value is None and kind in {"reasoning", "thinking"}:
            value = block.get("reasoning") or block.get("thinking")
        if not isinstance(value, str):
            continue
        if kind in {
            "reasoning",
            "reasoning_content",
            "thinking",
            "analysis",
        }:
            reasoning.append(value)
        elif kind in {"text", "output_text", ""}:
            text.append(value)
    return "".join(text), "".join(reasoning)


def _tool_calls(message: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    def append(row: dict[str, Any]) -> None:
        # Provider-native function-call blocks can expose both an item ID
        # (``fc_...``) and the actual call ID (``call_...``). ToolMessage uses
        # the latter, so it is the stable identity for the UI projection.
        call_id = str(
            row.get("tool_call_id") or row.get("call_id") or row.get("id") or ""
        ).strip()
        if call_id:
            row = {**row, "id": call_id}
        key = call_id or json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        if key in seen:
            return
        seen.add(key)
        rows.append(row)

    # AIMessageChunk derives ``tool_calls`` from ``tool_call_chunks``. Reading
    # both surfaces projects every streamed fragment twice. LangChain also
    # documents that only the first chunk normally carries name/id; subsequent
    # argument fragments carry ``index`` with a null id. Prefer the native chunk
    # surface while streaming and the complete call surface after aggregation.
    native_type = str(message.get("type") or "").lower()
    if (
        "chunk" in native_type
        and isinstance(message.get("tool_call_chunks"), list)
        and message.get("tool_call_chunks")
    ):
        structured_keys = ("tool_call_chunks",)
    elif isinstance(message.get("tool_calls"), list):
        structured_keys = ("tool_calls",)
    else:
        structured_keys = ("tool_call_chunks",)
    for key in structured_keys:
        value = message.get(key)
        if not isinstance(value, list):
            continue
        for item in value:
            row = _plain_dict(item)
            if row:
                append(row)
    structured_indices = {row.get("index") for row in rows if row.get("index") is not None}
    # OpenAI custom tools such as native apply_patch are content blocks rather
    # than function-call rows. LangChain exposes these through content_blocks
    # as ``non_standard`` wrappers, while serialized Agent Protocol messages
    # can retain the provider-native block directly. Both are documented
    # message surfaces and belong in the same typed tool projection.
    content = message.get("content_blocks") or message.get("content")
    if isinstance(content, list):
        for item in content:
            block = _plain_dict(item)
            if str(block.get("type") or "") == "non_standard":
                block = _plain_dict(block.get("value"))
            if str(block.get("type") or "") not in {
                "custom_tool_call",
                "function_call",
                "tool_call",
                "tool_call_chunk",
            }:
                continue
            # content_blocks is a normalized view of the same function call,
            # not a second stream. Id-less fragments correlate by index.
            # Custom tools remain independent even alongside function calls.
            if block.get("type") != "custom_tool_call" and block.get("index") in structured_indices:
                continue
            append(
                {
                    "id": str(
                        block.get("tool_call_id")
                        or block.get("call_id")
                        or block.get("id")
                        or ""
                    ),
                    "name": str(block.get("name") or block.get("tool") or ""),
                    "args": block.get("args", block.get("arguments", block.get("input"))),
                    "index": block.get("index"),
                }
            )
    return rows


def _tool_input(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {"partial": value}
    return dict(decoded) if isinstance(decoded, dict) else {"value": decoded}


def _delta_offset(message: ThreadMessage, part_id: str, delta: str) -> int:
    """Position before the appended fragment, in browser UTF-16 code units."""
    text = next(part.text for part in message.parts if part.id == part_id)
    return len(text[:-len(delta)].encode("utf-16-le")) // 2


def _reported_files(text: str) -> list[str]:
    """Collect declared file references, never discover arbitrary workspace files."""

    files: list[str] = []
    seen: set[str] = set()
    for candidate in re.findall(r"(?:\]\((?:<)?)(sandbox:/[^\s)>]+)", text):
        path = unquote(candidate.removeprefix("sandbox:/"))
        if path and path not in seen:
            seen.add(path)
            files.append(path)
    in_files = False
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*$", line)
        if heading:
            title = heading.group(2).strip().rstrip(":").lower()
            in_files = title in {"files", "output files", "outputs", "输出文件", "产出文件", "文件", "报告文件"}
            continue
        if not in_files or not re.match(r"^[-*]\s+", line):
            continue
        code_match = re.search(r"`([^`]+)`", line)
        candidate = (
            code_match.group(1).strip()
            if code_match
            else re.sub(r"^[-*]\s+", "", line).strip()
        )
        if not code_match and ":" in candidate:
            candidate = candidate.split(":", 1)[0].strip()
        if not candidate or candidate.lower() in {
            "none",
            "(none)",
            "none reported",
            "(none reported)",
        }:
            continue
        if candidate not in seen:
            seen.add(candidate)
            files.append(candidate)
    return files


class RunProjection:
    """Project one native run without becoming a second lifecycle authority.

    Only top-level, non-internal model calls are conversational output.
    ``updates`` and ``values`` are used for typed state cards only, so
    middleware message replacement can never become a new assistant answer.
    """

    def __init__(
        self,
        *,
        store: ThreadStore,
        broker: ThreadEventBroker,
        artifact_registry: ArtifactRegistry,
        thread_id: str,
        run_id: str,
        assistant_message_id: str,
        text_part_id: str,
        input_message_id: str = "",
        on_async_task: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.store = store
        self.broker = broker
        self.artifact_registry = artifact_registry
        self.thread_id = str(thread_id)
        self.run_id = str(run_id)
        self.assistant_message_id = str(assistant_message_id)
        self.text_part_id = str(text_part_id)
        self.input_message_id = str(input_message_id)
        self.on_async_task = on_async_task
        self._model_messages: dict[str, dict[str, Any]] = {}
        self._model_order: list[str] = []
        self._tool_parts: dict[str, str] = {}
        self._tool_stream_slots: dict[tuple[str, int], dict[str, Any]] = {}
        self._interrupt_parts: set[str] = set()
        self._async_task_status: dict[str, tuple[str, str]] = {}
        current = self.store.get_message(self.thread_id, self.assistant_message_id)
        self._compaction_part_id = next((
            part.id for part in reversed(current.parts if current else [])
            if part.meta.get("kind") == "context_compaction" and part.status == "running"
        ), "")

    def process(self, chunk: Any) -> None:
        event = _plain_dict(chunk)
        if not event:
            return
        raw_event = str(event.get("event") or "").strip()
        event_type = str(event.get("type") or raw_event).strip()
        namespace = event.get("ns")
        # ``runs.stream(..., version="v2")`` yields typed dictionaries, while
        # ``runs.join_stream`` currently yields the SDK's v1 ``StreamPart``
        # named tuple. Its SSE event encodes a subgraph namespace as
        # ``<mode>|<namespace>``. Normalize both documented surfaces here so a
        # WebUI restart can rejoin a run without exposing child output as root
        # chat content.
        if "type" not in event and raw_event:
            pieces = raw_event.split("|")
            event_type = pieces[0]
            namespace = pieces[1:]
        if namespace not in (None, [], ()):
            return
        data = event.get("data")
        if event_type in {"messages", "messages-tuple"}:
            self._process_message_data(data)
        elif event_type in {"updates", "values"}:
            self._project_state(data)
        interrupts = event.get("interrupts")
        if interrupts is None and event_type == "values" and isinstance(data, dict):
            interrupts = data.get("__interrupt__")
        self._project_interrupts(interrupts)

    def _process_message_data(self, data: Any) -> None:
        if isinstance(data, (list, tuple)) and len(data) == 2:
            message = _plain_dict(data[0])
            metadata = _plain_dict(data[1])
        else:
            message = _plain_dict(data)
            metadata = {}
        if not message:
            return
        # LangChain's internal-call transformer filters v3 run.messages, but
        # Native v2 message streams still include these calls.
        # Summarization runs inside the root model node: namespace/node alone
        # cannot distinguish it from the answer. Use upstream call metadata.
        source = metadata.get("lc_source") or _plain_dict(message.get("additional_kwargs")).get("lc_source")
        if source == "summarization":
            self._project_compaction(message, metadata)
            return
        if metadata.get("lc_internal_call"):
            return
        native_type = str(message.get("type") or "").strip().lower()
        if native_type in {"ai", "aimessage", "aimessagechunk"}:
            self._finish_compaction()
            self._process_ai_message(message, metadata)
        elif native_type in {"tool", "toolmessage", "toolmessagechunk"}:
            self._process_tool_message(message)

    def _project_compaction(self, message: dict[str, Any], metadata: dict[str, Any]) -> None:
        # Responses streaming can change message.id from resp_* to lc_run--*
        # within one model call. LangGraph's model task namespace/step is stable
        # across those chunks. lc_internal_call is a process marker, not an ID.
        invocation = str(metadata.get("langgraph_checkpoint_ns") or "")
        if not invocation and metadata.get("langgraph_step") is not None:
            invocation = f"step_{metadata['langgraph_step']}"
        if invocation:
            part_id = f"part_compaction_{_safe_token(invocation, self.run_id)}"
        else:
            part_id = self._compaction_part_id or f"part_compaction_{_safe_token(str(message.get('id') or ''), self.run_id)}"
        if self._compaction_part_id and self._compaction_part_id != part_id:
            self._finish_compaction()
        self._compaction_part_id = part_id
        self._ensure_part(MessagePart(
            id=part_id, type="trace", status="running", text="正在压缩上下文…",
            meta={"kind": "context_compaction", "run_id": self.run_id},
        ))
        if (message.get("chunk_position") == "last"
                or str(message.get("type") or "").lower() in {"ai", "aimessage"}
                or _plain_dict(message.get("usage_metadata")).get("total_tokens") is not None):
            self._finish_compaction()

    def _finish_compaction(self, status: str = "completed") -> None:
        if not self._compaction_part_id:
            return
        part_id, self._compaction_part_id = self._compaction_part_id, ""
        text = {
            "completed": "上下文已压缩，继续处理请求。",
            "interrupted": "上下文压缩已中断。",
            "failed": "上下文压缩未完成。",
        }[status]
        updated = self.store.update_part(
            self.thread_id, self.assistant_message_id, part_id, status=status, text=text,
        )
        part = next(part for part in updated.parts if part.id == part_id)
        self.broker.emit(
            self.thread_id, "message.part.updated", message_id=self.assistant_message_id,
            status=status, data={"part": part.model_dump(mode="json"), "run_id": self.run_id},
        )

    def _process_ai_message(
        self,
        message: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        # Namespace and internal-call filtering happen before accumulation.
        native_id = _safe_token(
            str(message.get("id") or metadata.get("run_id") or ""),
            f"model_{len(self._model_order) + 1}",
        )
        current = self._model_messages.setdefault(
            native_id,
            {"text": "", "reasoning": "", "has_tools": False, "complete": False},
        )
        if native_id not in self._model_order:
            self._model_order.append(native_id)
        text, reasoning = _content_fragments(
            message.get("content_blocks") or message.get("content")
        )
        previous_text = str(current.get("text") or "")
        is_chunk = "chunk" in str(message.get("type") or "").lower()
        if is_chunk:
            current["text"] = previous_text + text
            text_delta = text
            previous_reasoning = str(current.get("reasoning") or "")
            current["reasoning"] = previous_reasoning + reasoning
            reasoning_delta = reasoning
        else:
            current["text"] = text
            text_delta = (
                text[len(previous_text) :]
                if text.startswith(previous_text)
                else (text if not previous_text else "")
            )
            previous_reasoning = str(current.get("reasoning") or "")
            current["reasoning"] = reasoning or previous_reasoning
            reasoning_delta = (
                reasoning[len(previous_reasoning) :]
                if reasoning.startswith(previous_reasoning)
                else reasoning
            )
            current["complete"] = True
        if text_delta:
            updated = self.store.add_text_delta(
                self.thread_id,
                self.assistant_message_id,
                self.text_part_id,
                text_delta,
                persist=True,
            )
            self.broker.emit(
                self.thread_id,
                "message.delta",
                message_id=self.assistant_message_id,
                status="streaming",
                data={
                    "part_id": self.text_part_id,
                    "delta": text_delta,
                    # Browser string offsets are UTF-16 code units. The cursor
                    # may precede its REST snapshot; offsets make overlap safe.
                    "text_offset": _delta_offset(updated, self.text_part_id, text_delta),
                    "run_id": self.run_id,
                },
            )
        if reasoning_delta:
            part_id = f"part_reasoning_{native_id}"
            self._ensure_part(
                MessagePart(
                    id=part_id,
                    type="reasoning",
                    status="streaming",
                    text="",
                    meta={"run_id": self.run_id, "native_message_id": native_id},
                )
            )
            updated = self.store.add_text_delta(
                self.thread_id,
                self.assistant_message_id,
                part_id,
                reasoning_delta,
                persist=True,
            )
            self.broker.emit(
                self.thread_id,
                "reasoning.delta",
                message_id=self.assistant_message_id,
                status="streaming",
                data={
                    "part_id": part_id,
                    "delta": reasoning_delta,
                    "run_id": self.run_id,
                    "text_offset": _delta_offset(updated, part_id, reasoning_delta),
                },
            )
        calls = _tool_calls(message)
        if calls:
            current["has_tools"] = True
        for call in calls:
            if is_chunk:
                self._project_tool_call_chunk(native_id, call)
            else:
                self._project_tool_call(call)

    def _project_tool_call_chunk(
        self,
        native_message_id: str,
        call: dict[str, Any],
    ) -> None:
        """Coalesce LangChain's id-less argument chunks into one tool call."""

        raw_index = call.get("index")
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            # Some provider adapters omit an index but still expose a stable
            # call ID. Such a chunk can be projected directly; an anonymous
            # unindexed fragment cannot be correlated safely and stays hidden.
            if str(
                call.get("tool_call_id")
                or call.get("call_id")
                or call.get("id")
                or ""
            ).strip():
                self._project_tool_call(call)
            return

        slot = self._tool_stream_slots.setdefault(
            (native_message_id, index),
            {"id": "", "name": "", "args": ""},
        )
        call_id = str(
            call.get("tool_call_id")
            or call.get("call_id")
            or call.get("id")
            or ""
        ).strip()
        if call_id:
            slot["id"] = call_id
        name = str(call.get("name") or call.get("tool") or "").strip()
        if name:
            slot["name"] = name
        args = call.get("args")
        if isinstance(args, str):
            slot["args"] = str(slot.get("args") or "") + args
        elif isinstance(args, dict):
            slot["args"] = dict(args)
        elif args is not None:
            slot["args"] = args
        if not str(slot.get("id") or "").strip():
            return
        self._project_tool_call(
            {
                "id": slot["id"],
                "name": slot.get("name") or "",
                "args": slot.get("args"),
            }
        )

    def _project_tool_call(self, call: dict[str, Any]) -> None:
        raw_call_id = str(
            call.get("tool_call_id")
            or call.get("call_id")
            or call.get("id")
            or ""
        ).strip()
        if not raw_call_id:
            return
        call_id = _safe_token(
            raw_call_id,
            "tool",
        )
        name = str(call.get("name") or call.get("tool") or "").strip()
        part_id = self._tool_parts.get(call_id)
        if part_id is None:
            part_id = f"part_tool_{call_id}"
            self._tool_parts[call_id] = part_id
            part = MessagePart(
                id=part_id,
                type="tool-call",
                status="running",
                text="",
                meta={
                    "tool_call_id": call_id,
                    "tool": name,
                    "input": _tool_input(call.get("args")),
                    "run_id": self.run_id,
                },
            )
            self._ensure_part(part)
            self.broker.emit(
                self.thread_id,
                "tool_call.started",
                message_id=self.assistant_message_id,
                status="running",
                data={
                    "part_id": part_id,
                    "tool_call_id": call_id,
                    "tool": name,
                    "input": _tool_input(call.get("args")),
                    "run_id": self.run_id,
                },
            )
            return
        try:
            existing = self.store.get_message(
                self.thread_id, self.assistant_message_id
            )
            existing_part = next(
                (part for part in (existing.parts if existing else []) if part.id == part_id),
                None,
            )
            meta = dict(existing_part.meta or {}) if existing_part else {}
            if name:
                meta["tool"] = name
            args = call.get("args")
            if args not in (None, "", {}):
                meta["input"] = _tool_input(args)
            self.store.update_part(
                self.thread_id,
                self.assistant_message_id,
                part_id,
                meta=meta,
            )
        except KeyError:
            return

    def _process_tool_message(self, message: dict[str, Any]) -> None:
        call_id = _safe_token(
            str(message.get("tool_call_id") or ""),
            "unknown_tool",
        )
        part_id = self._tool_parts.get(call_id)
        if not part_id:
            self._project_tool_call(
                {
                    "id": call_id,
                    "name": str(message.get("name") or ""),
                    "args": {},
                }
            )
            part_id = self._tool_parts.get(call_id)
        if not part_id:
            return
        output = message.get("content")
        try:
            current_message = self.store.get_message(
                self.thread_id, self.assistant_message_id
            )
            current_part = next(
                (
                    part
                    for part in (current_message.parts if current_message else [])
                    if part.id == part_id
                ),
                None,
            )
            meta = dict(current_part.meta or {}) if current_part else {}
            meta["output"] = output
            self.store.update_part(
                self.thread_id,
                self.assistant_message_id,
                part_id,
                status="completed",
                meta=meta,
            )
        except KeyError:
            return
        self.broker.emit(
            self.thread_id,
            "tool_call.completed",
            message_id=self.assistant_message_id,
            status="completed",
            data={
                "part_id": part_id,
                "tool_call_id": call_id,
                "tool": str(message.get("name") or ""),
                "output": output,
                "run_id": self.run_id,
            },
        )

    def _project_state(self, data: Any) -> None:
        payload = _plain_dict(data)
        if not payload:
            return
        # Updates are node-keyed. Only todos are live UI state here. Retained
        # async_tasks belong to historical checkpoints; local execution publishes
        # child lifecycle events explicitly through _project_async_tasks.
        stack = [payload]
        while stack:
            current = stack.pop()
            if not isinstance(current, dict):
                continue
            todos = current.get("todos")
            if isinstance(todos, list):
                self.broker.emit(
                    self.thread_id,
                    "activity.updated",
                    message_id=self.assistant_message_id,
                    status="running",
                    data={
                        "run_id": self.run_id,
                        "part": {
                            "id": f"part_todo_{_safe_token(self.run_id, 'run')}",
                            "type": "progress",
                            "status": "running",
                            "title": "Research plan",
                            "items": todos,
                        },
                    },
                )
            stack.extend(
                value for value in current.values() if isinstance(value, dict)
            )

    def _project_async_tasks(self, tasks: dict[str, Any]) -> None:
        for task_id, raw in tasks.items():
            task = _plain_dict(raw)
            if not task:
                continue
            status = str(task.get("status") or "running").lower()
            key = str(task_id)
            identity = (status, str(task.get("run_id") or ""))
            if self._async_task_status.get(key) == identity:
                if self.on_async_task is not None:
                    self.on_async_task({"task_id": key, **task})
                continue
            self._async_task_status[key] = identity
            event = (
                "subagent.started"
                if status in {"created", "pending", "running", "cancel_requested"}
                else "subagent.completed"
            )
            part_id = f"part_subagent_{_safe_token(key, 'task')}"
            visible_status = (
                "pending" if status in {"created", "pending"} else "running"
                if status in {"running", "cancel_requested"}
                else "completed"
                if status == "success"
                else "interrupted"
                if status in {"cancelled", "interrupted"}
                else "failed"
            )
            part = MessagePart(
                id=part_id,
                type="subagent",
                status=visible_status,
                text=(
                    "Cancellation requested; waiting for graph teardown."
                    if status == "cancel_requested"
                    else f"Async specialist run is {status}."
                ),
                meta={
                    "source": str(task.get("agent_name") or "Background specialist"),
                    "task_id": key,
                    "thread_id": str(task.get("thread_id") or key),
                    "run_id": str(task.get("run_id") or ""),
                    "native_status": status,
                },
            )
            current_message = self.store.get_message(
                self.thread_id, self.assistant_message_id
            )
            existing = next(
                (
                    candidate
                    for candidate in (current_message.parts if current_message else [])
                    if candidate.id == part_id
                ),
                None,
            )
            if existing is None:
                self._ensure_part(part)
            else:
                part.meta = {**existing.meta, **part.meta}
                if existing.meta.get("run_id") != part.meta.get("run_id"):
                    part.meta.pop("task_followup", None)
                if part.meta.get("latest_update") and status != "cancel_requested":
                    part.text = str(part.meta["latest_update"])
                self.store.update_part(
                    self.thread_id,
                    self.assistant_message_id,
                    part_id,
                    status=visible_status,
                    text=part.text,
                    meta=part.meta,
                )
            self.broker.emit(
                self.thread_id,
                event,
                message_id=self.assistant_message_id,
                status=visible_status,
                data={"part": part.model_dump(mode="json"), "run_id": self.run_id},
            )
            if self.on_async_task is not None:
                self.on_async_task({"task_id": key, **task})

    def _project_interrupts(self, interrupts: Any) -> None:
        if not isinstance(interrupts, list):
            return
        for raw in interrupts:
            interrupt = _plain_dict(raw)
            value = interrupt.get("value")
            value_dict = _plain_dict(value)
            if value_dict.get("kind") == "research_capacity":
                continue  # Scheduling is shown on the task card, not as user HITL.
            interrupt_id = _safe_token(
                str(interrupt.get("id") or value_dict.get("id") or ""),
                f"interrupt_{len(self._interrupt_parts) + 1}",
            )
            if interrupt_id in self._interrupt_parts:
                continue
            self._interrupt_parts.add(interrupt_id)
            part = MessagePart(
                id=f"part_interrupt_{interrupt_id}",
                type="interrupt",
                status="pending",
                text=str(value_dict.get("description") or value or "Approval required."),
                meta={
                    "interrupt_id": interrupt_id,
                    # Preserve Agent Protocol's interruption envelope. The
                    # public projection and resume validator both need the
                    # action requests together with their review configs.
                    "payload": {"interrupts": [interrupt]},
                    "run_id": self.run_id,
                },
            )
            self._ensure_part(part)
            self.broker.emit(
                self.thread_id,
                "interrupt.created",
                message_id=self.assistant_message_id,
                status="pending",
                data={"part": part.model_dump(mode="json"), "run_id": self.run_id},
            )

    def _ensure_part(self, part: MessagePart) -> None:
        current = self.store.get_message(self.thread_id, self.assistant_message_id)
        if current is None or any(existing.id == part.id for existing in current.parts):
            return
        self.store.append_part(
            self.thread_id,
            self.assistant_message_id,
            part,
        )
        self.broker.emit(
            self.thread_id,
            "message.part.created",
            message_id=self.assistant_message_id,
            status=part.status,
            data={"part": part.model_dump(mode="json"), "run_id": self.run_id},
        )

    def final_text_from_state(self, state: Any) -> tuple[str, str]:
        state_dict = _plain_dict(state)
        values = _plain_dict(state_dict.get("values"))
        messages = values.get("messages")
        if not isinstance(messages, list):
            messages = []
        boundary = -1
        if self.input_message_id:
            for index, raw in enumerate(messages):
                if str(_plain_dict(raw).get("id") or "") == self.input_message_id:
                    boundary = index
        candidates: list[dict[str, Any]] = []
        for raw in messages[boundary + 1 :]:
            message = _plain_dict(raw)
            native_type = str(message.get("type") or "").lower()
            # A later HumanMessage belongs to a later queued run. Never let a
            # fast successor's answer become this run's final projection.
            if native_type in {"human", "humanmessage"}:
                if _plain_dict(message.get("additional_kwargs")).get("catmaster_in_turn"):
                    continue  # A checkpointed research notice stays within this execution.
                break
            if native_type not in {"ai", "aimessage"} or _tool_calls(message):
                continue
            native_id = str(message.get("id") or "")
            if boundary < 0 and native_id not in self._model_messages:
                # Resume/recovery without a durable boundary must fail closed
                # instead of selecting an arbitrary historical AIMessage.
                continue
            candidates.append(message)
        if candidates:
            selected = candidates[-1]
            text, _reasoning = _content_fragments(
                selected.get("content_blocks") or selected.get("content")
            )
            return text.strip(), str(selected.get("id") or "")
        for native_id in reversed(self._model_order):
            candidate = self._model_messages[native_id]
            if candidate.get("has_tools"):
                continue
            text = str(candidate.get("text") or "").strip()
            if text:
                return text, native_id
        return "", ""

    def _reported_artifact_parts(
        self,
        text: str,
        *,
        existing_parts: list[MessagePart],
    ) -> list[ArtifactPart]:
        existing_ids = {
            str(getattr(part, "artifact_id", "") or "")
            for part in existing_parts
            if part.type == "artifact"
        }
        parts: list[ArtifactPart] = []
        for raw_path in _reported_files(text):
            path = raw_path
            candidate = Path(raw_path).expanduser()
            if candidate.is_absolute():
                try:
                    path = str(candidate.resolve().relative_to(self.artifact_registry.workspace))
                except (OSError, ValueError):
                    continue
            try:
                artifact = self.artifact_registry.register_path(
                    path,
                    thread_id=self.thread_id,
                    message_id=self.assistant_message_id,
                    run_id=self.run_id,
                    summary="Reported by the completed agent run.",
                    meta={"source": "agent_final_report"},
                )
            except (OSError, ValueError):
                # A textual mention is not an artifact unless it resolves to
                # an existing user-workspace file.
                continue
            if artifact.artifact_id in existing_ids:
                continue
            existing_ids.add(artifact.artifact_id)
            parts.append(
                ArtifactPart(
                    id=f"part_artifact_{_safe_token(artifact.artifact_id, 'reported')}",
                    status="completed",
                    artifact_id=artifact.artifact_id,
                    renderer=artifact.renderer,
                    title=artifact.title,
                    summary=artifact.summary,
                    path=artifact.path,
                    meta=artifact.model_dump(mode="json"),
                )
            )
        return parts

    def finalize(self, *, native_status: str, state: Any = None, error: str = "") -> ThreadMessage:
        self._finish_compaction(
            "completed" if native_status == "success"
            else "interrupted" if native_status == "interrupted" else "failed"
        )
        state_dict = _plain_dict(state)
        state_values = _plain_dict(state_dict.get("values"))
        self._project_interrupts(
            state_dict.get("interrupts") or state_values.get("__interrupt__")
        )
        current = self.store.get_message(self.thread_id, self.assistant_message_id)
        if current is None:
            raise KeyError(f"Message not found: {self.assistant_message_id}")
        status = str(native_status or "error").lower()
        text, native_message_id = self.final_text_from_state(state)
        parts: list[MessagePart] = []
        for part in current.parts:
            payload = part.model_dump(mode="json")
            if part.id == self.text_part_id:
                payload.update(
                    {
                        "text": part.text if status == "interrupted" else text,
                        "status": "interrupted" if status == "interrupted" else "completed" if status == "success" and text else "failed",
                    }
                )
            elif part.type != "subagent" and str(part.status or "").lower() in {"created", "running", "streaming"}:
                payload["status"] = "interrupted" if status == "interrupted" else "completed"
            parts.append(MessagePart.model_validate(payload))
        if status == "success" and text:
            message_status = "completed"
            parts.extend(
                self._reported_artifact_parts(text, existing_parts=parts)
            )
        elif status == "interrupted":
            message_status = "interrupted"
        else:
            message_status = "failed"
            if not error:
                error = (
                    "The native run completed without a current-run final assistant answer."
                    if status == "success"
                    else f"Agent run ended with status {status}."
                )
            parts.append(
                MessagePart(
                    id=new_id("part_error"),
                    type="error",
                    status="failed",
                    text="",
                    meta={"summary": error, "native_status": status, "run_id": self.run_id},
                )
            )
        meta = dict(current.meta or {})
        meta.update(
            {
                "run_id": self.run_id,
                "native_status": status,
                "native_message_id": native_message_id,
            }
        )
        completed = self.store.update_message(
            self.thread_id,
            self.assistant_message_id,
            status=message_status,
            parts=parts,
            meta=meta,
        )
        # Native Steer/Stop is an expected interruption. Publish its saved
        # content/status without synthesizing a failure card in the browser.
        event = ("message.completed" if message_status == "completed" else
                 "message.updated" if message_status == "interrupted" else "message.failed")
        self.broker.emit(
            self.thread_id,
            event,
            message_id=self.assistant_message_id,
            status=message_status,
            data={
                "message": completed.model_dump(mode="json"),
                "run_id": self.run_id,
                "error": error,
            },
        )
        return completed


__all__ = ["RunProjection"]
