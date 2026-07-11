from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable

from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langgraph.types import Command

from catmaster.runtime.observability_store import ObservabilityStore
from catmaster.runtime.usage_stats import summarize_usage_from_observability
from catmaster.tools.base import workspace_root
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import ArtifactPart, InterruptRecord, MessagePart, ThreadMessage, ThreadStatus
from catmaster.webui.thread_store import ThreadStore, new_id

from .runtime import RUN_STATE_FILE, SpecialistRunner
from .schemas import ResearchGoalRecord, SpecialistEntrypoint

logger = logging.getLogger(__name__)

_APPROVAL_DECISIONS = {"approve", "edit", "reject", "respond"}


def _model_dump(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    return dict(value)


def _json_safe(value: Any, *, max_text: int = 12_000) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= max_text else value[: max_text - 20] + "\n...[truncated]"
    if isinstance(value, BaseMessage):
        return {
            "type": value.__class__.__name__,
            "content": _message_text(value),
            "id": str(getattr(value, "id", "") or ""),
            "name": str(getattr(value, "name", "") or ""),
            "tool_call_id": str(getattr(value, "tool_call_id", "") or ""),
        }
    if hasattr(value, "value"):
        payload = {
            "type": value.__class__.__name__,
            "value": _json_safe(getattr(value, "value", None), max_text=max_text),
        }
        object_id = str(getattr(value, "id", "") or "").strip()
        if object_id:
            payload["id"] = object_id
        return payload
    if isinstance(value, dict):
        return {str(key): _json_safe(item, max_text=max_text) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, max_text=max_text) for item in value]
    try:
        return _json_safe(_model_dump(value), max_text=max_text)
    except Exception:
        return str(value)


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").lower()
                if item_type in {"reasoning", "thinking", "reasoning_text", "reasoning_content", "redacted_reasoning"}:
                    continue
                text = item.get("text")
                if text:
                    chunks.append(str(text))
        return "\n".join(chunks)
    return str(content or "")


def _event_method(event: Any) -> str:
    if isinstance(event, dict):
        return str(event.get("method") or event.get("event") or event.get("name") or "").strip()
    return str(getattr(event, "method", "") or getattr(event, "event", "") or getattr(event, "name", "")).strip()


def _event_params(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        params = event.get("params")
        return params if isinstance(params, dict) else {}
    params = getattr(event, "params", None)
    return params if isinstance(params, dict) else {}


def _event_data(event: Any) -> Any:
    params = _event_params(event)
    if "data" in params:
        return params.get("data")
    if isinstance(event, dict):
        return event.get("data")
    return getattr(event, "data", None)


def _event_metadata(event: Any) -> dict[str, Any]:
    if isinstance(event, dict):
        out: dict[str, Any] = {}
        metadata = event.get("metadata")
        if isinstance(metadata, dict):
            out.update(metadata)
        params = event.get("params")
        if isinstance(params, dict) and isinstance(params.get("metadata"), dict):
            out.update(params["metadata"])
        if isinstance(params, dict) and params.get("namespace") not in (None, "", []):
            out["namespace"] = params.get("namespace")
        data = event.get("data")
        if isinstance(data, dict) and isinstance(data.get("metadata"), dict):
            out.update(data["metadata"])
        return out
    metadata = getattr(event, "metadata", None)
    return metadata if isinstance(metadata, dict) else {}


def _coerce_stream_result(value: Any) -> Any:
    if inspect.isawaitable(value):
        return value
    return value


_INTERNAL_AGENT_MARKERS = {
    "materials_worker",
    "ml_worker",
    "dynamics_worker",
    "orca_xtb_worker",
    "litreview_agent",
    "writing_worker_agent",
    "writing_polisher_agent",
    "peer_review_worker_agent",
    "general-purpose",
    "subagent",
}


def _metadata_label(metadata: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _is_internal_stream_source(metadata: dict[str, Any], *, tool_name: str = "") -> bool:
    if not metadata:
        return False
    namespace = metadata.get("namespace")
    if namespace not in (None, "", []):
        return True
    labels = [
        _metadata_label(metadata, "lc_agent_name", "agent_name", "agent", "subagent", "name"),
        _metadata_label(metadata, "langgraph_node", "node"),
        _metadata_label(metadata, "langgraph_path", "path"),
        _metadata_label(metadata, "langgraph_checkpoint_ns", "checkpoint_ns"),
        str(namespace or ""),
        str(tool_name or ""),
    ]
    text = " ".join(label for label in labels if label).lower()
    if not text:
        return False
    if "task:" in text or " task " in f" {text} ":
        return True
    return any(marker in text for marker in _INTERNAL_AGENT_MARKERS)


def _stream_source_name(metadata: dict[str, Any], *, default: str = "subagent") -> str:
    for key in ("lc_agent_name", "agent_name", "agent", "subagent", "langgraph_node", "node", "name"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value
    namespace = metadata.get("namespace")
    if namespace:
        if isinstance(namespace, (list, tuple)):
            parts = [str(item).strip() for item in namespace if str(item).strip()]
            if parts:
                return " / ".join(parts)
        text = str(namespace).strip()
        if text:
            return text
    return default


def _subagent_source_name(metadata: dict[str, Any]) -> str:
    if not _is_internal_stream_source(metadata):
        return ""
    for key in ("lc_agent_name", "agent_name", "agent", "subagent", "name"):
        value = str(metadata.get(key) or "").strip()
        if value and value.lower() not in {"agent", "tools", "tool"}:
            return value
    checkpoint_ns = str(metadata.get("langgraph_checkpoint_ns") or metadata.get("checkpoint_ns") or "").strip()
    match = re.search(r"(?:^|:)task:([^:]+)", checkpoint_ns)
    if match:
        return match.group(1)
    namespace = metadata.get("namespace")
    rows = namespace if isinstance(namespace, (list, tuple)) else [namespace]
    for row in rows:
        text = str(row or "").strip()
        if text.startswith("task:"):
            return text.split(":", 1)[1] or text
    return _stream_source_name(metadata)


def _message_reasoning_text(message: Any) -> str:
    chunks: list[str] = []

    def append_text(value: Any) -> None:
        if isinstance(value, str) and value:
            chunks.append(value)

    def append_summary(value: Any) -> None:
        if isinstance(value, str):
            append_text(value)
            return
        if not isinstance(value, list):
            return
        for item in value:
            if isinstance(item, str):
                append_text(item)
            elif isinstance(item, dict):
                append_text(item.get("text"))

    def append_reasoning_block(block: Any) -> bool:
        if not isinstance(block, dict):
            return False
        block_type = str(block.get("type") or "").lower()
        if block_type == "reasoning":
            append_text(block.get("reasoning"))
            append_summary(block.get("summary"))
            append_text(block.get("text"))
            return True
        if block_type in {"reasoning-delta", "thinking-delta"}:
            append_text(block.get("reasoning"))
            append_text(block.get("text"))
            append_summary(block.get("summary"))
            return True
        return False

    blocks = getattr(message, "content_blocks", None)
    if isinstance(blocks, list):
        for block in blocks:
            append_reasoning_block(block)
    elif isinstance(message, dict):
        content_blocks = message.get("content_blocks")
        if isinstance(content_blocks, list):
            for block in content_blocks:
                append_reasoning_block(block)
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                append_reasoning_block(block)
        append_reasoning_block(message)
    else:
        content = getattr(message, "content", None)
        if isinstance(content, list):
            for block in content:
                append_reasoning_block(block)

    for container_name in ("additional_kwargs", "response_metadata"):
        container = getattr(message, container_name, None)
        if not isinstance(container, dict):
            continue
        # `reasoning_content` is normalized into `content_blocks` by current
        # LangChain, but keep `reasoning_text` for provider bridges that expose
        # this value only as message metadata.
        append_text(container.get("reasoning_text"))

    out: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        text = str(chunk or "")
        if not text.strip() or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return "\n".join(out)


_WORKSPACE_PATH_RE = re.compile(r"^[A-Za-z0-9._/@+\-]+$")


def _looks_like_workspace_path(text: str) -> bool:
    candidate = str(text or "").strip()
    if not candidate or len(candidate) > 500:
        return False
    if candidate.startswith(("-", "$")) or "://" in candidate:
        return False
    if any(char.isspace() for char in candidate):
        return False
    if "=" in candidate:
        return False
    if not _WORKSPACE_PATH_RE.fullmatch(candidate):
        return False
    parts = Path(candidate.replace("\\", "/").lstrip("/")).parts
    if any(part in {"", ".", ".."} for part in parts):
        return False
    return "/" in candidate or bool(Path(candidate).suffix)


def _extract_workspace_paths_from_text(text: str) -> list[str]:
    paths: list[str] = []
    for match in re.finditer(r"`([^`\n]{1,500})`", str(text or "")):
        candidate = match.group(1).strip()
        if not candidate or candidate in {"(none reported)", "none"}:
            continue
        if not _looks_like_workspace_path(candidate):
            continue
        paths.append(candidate)
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        key = path.replace("\\", "/").lstrip("/")
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out[:50]


def _extract_sidecar_artifact_paths(payload: Any) -> list[str]:
    paths: list[str] = []
    sidecar_keys = {"artifacts", "artifact_paths", "artifact_refs", "files"}
    path_keys = {"path", "file", "output_path", "workspace_path", "relpath"}

    def _add_path(value: Any) -> None:
        if not isinstance(value, str):
            return
        text = value.strip()
        if not text or text.startswith("art_") or "://" in text or len(text) > 500:
            return
        if "/" in text or Path(text).suffix:
            paths.append(text)

    def _walk(value: Any, *, in_sidecar: bool = False, key: str = "") -> None:
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                sub_key_s = str(sub_key)
                next_in_sidecar = in_sidecar or sub_key_s in sidecar_keys
                if next_in_sidecar and sub_key_s in path_keys:
                    _add_path(sub_value)
                _walk(sub_value, in_sidecar=next_in_sidecar, key=sub_key_s)
            return
        if isinstance(value, list):
            for item in value:
                _walk(item, in_sidecar=in_sidecar, key=key)
            return
        if in_sidecar:
            _add_path(value)

    _walk(payload)
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = path.replace("\\", "/").lstrip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(path)
    return out[:50]


class CatMasterStreamTranslator:
    """Translate LangGraph/LangChain stream chunks into CatMaster thread events."""

    def __init__(
        self,
        *,
        store: ThreadStore,
        events: ThreadEventBroker,
        artifact_registry: ArtifactRegistry,
        thread_id: str,
        message_id: str,
        text_part_id: str,
        run_id: str,
        observability_store: ObservabilityStore | None = None,
        resume_tool_inputs: list[dict[str, Any]] | None = None,
    ) -> None:
        self.store = store
        self.events = events
        self.artifact_registry = artifact_registry
        self.thread_id = thread_id
        self.message_id = message_id
        self.text_part_id = text_part_id
        self.run_id = run_id
        self.observability_store = observability_store
        self.last_values: Any = None
        self.final_text_from_stream = ""
        self.tool_parts_by_call_id: dict[str, str] = {}
        self.tool_call_id_by_index: dict[str, str] = {}
        self.tool_names_by_call_id: dict[str, str] = {}
        self.tool_inputs_by_call_id: dict[str, Any] = {}
        self.tool_arg_buffers_by_call_id: dict[str, str] = {}
        self.tool_stream_meta_by_call_id: dict[str, dict[str, Any]] = {}
        self.resume_tool_inputs_by_name: dict[str, list[Any]] = {}
        for item in resume_tool_inputs or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("tool") or "").strip()
            args = item.get("args") if "args" in item else item.get("input")
            if name and isinstance(args, dict):
                self.resume_tool_inputs_by_name.setdefault(name, []).append(_json_safe(args))
        self.completed_tool_messages: set[str] = set()
        self.historical_completed_tool_call_ids = self._historical_completed_tool_call_ids()
        self.interrupt_id = ""
        self.reasoning_part_id = ""
        self.reasoning_text_emitted = ""
        self.observed_reasoning_event_id = 0
        self.subagent_parts_by_source: dict[str, str] = {}

    def _historical_completed_tool_call_ids(self) -> set[str]:
        """Tool ids already represented by earlier WebUI messages.

        LangGraph `values` streams are full state snapshots. On a later turn the
        checkpoint state can include ToolMessages from prior turns; those should
        not be projected into the new assistant message as fresh tool calls.
        Pending HITL tool calls are intentionally excluded so a resumed tool can
        still be rendered when it finally completes.
        """
        out: set[str] = set()
        try:
            messages = self.store.list_messages(self.thread_id)
        except Exception:
            return out
        for message in messages:
            if message.id == self.message_id:
                continue
            for part in message.parts:
                if part.type != "tool-call":
                    continue
                meta = dict(part.meta or {})
                call_id = str(meta.get("tool_call_id") or getattr(part, "tool_call_id", "") or "").strip()
                if not call_id:
                    continue
                status = str(part.status or "").strip().lower()
                has_output = "output" in meta and meta.get("output") not in (None, "")
                if status in {"completed", "failed"} or has_output:
                    out.add(call_id)
        return out

    def apply_v3_event(self, event: Any) -> None:
        method = _event_method(event)
        params = _event_params(event)
        data = _event_data(event)
        metadata = _event_metadata(event)
        interrupts = params.get("interrupts") or (data.get("__interrupt__") if isinstance(data, dict) else None)
        if interrupts:
            self._handle_interrupts(interrupts)
        if method in {"on_chat_model_stream", "on_llm_stream"}:
            chunk = data.get("chunk") if isinstance(data, dict) and "chunk" in data else data
            self._handle_message_event((chunk, metadata))
            return
        if method in {"on_chat_model_end", "on_llm_end"}:
            self._handle_model_end_text(data, metadata=metadata)
            self._handle_final_tool_calls(data, metadata=metadata)
            return
        if method == "on_tool_start":
            name = str((event.get("name") if isinstance(event, dict) else getattr(event, "name", "")) or "").strip()
            payload = {"id": str(data.get("id") or name or "tool") if isinstance(data, dict) else name, "name": name, "args": data.get("input") if isinstance(data, dict) else {}}
            self._handle_tool_call_payload(payload, metadata=metadata)
            return
        if method == "on_tool_end":
            output = data.get("output") if isinstance(data, dict) else data
            if isinstance(output, ToolMessage):
                call_id = str(getattr(output, "tool_call_id", "") or "").strip()
                stream_fields = self._tool_stream_fields(metadata)
                if call_id and stream_fields:
                    self.tool_stream_meta_by_call_id[call_id] = {**self.tool_stream_meta_by_call_id.get(call_id, {}), **stream_fields}
                self._handle_tool_message(output)
            else:
                name = str((event.get("name") if isinstance(event, dict) else getattr(event, "name", "")) or "").strip()
                self._handle_tool_end_payload(name=name, output=output, metadata=metadata)
            return
        if method in {"messages", "messages-tuple", "message"}:
            if self._handle_protocol_message_event(data, metadata=metadata):
                return
            self._handle_message_event((data, metadata))
            return
        if method in {"values", "updates"}:
            self.last_values = data
            self._handle_values(data, metadata=metadata)
            return
        if method in {"tool_calls", "tool_call"}:
            self._handle_tool_call_payload(data, metadata=metadata)

    def apply_astream_chunk(self, chunk: Any) -> None:
        if isinstance(chunk, tuple) and len(chunk) == 2:
            mode, payload = chunk
            if mode == "messages":
                self._handle_message_event(payload)
            elif mode in {"updates", "values"}:
                self.last_values = payload
                self._handle_values(payload, metadata={})
            return
        if isinstance(chunk, dict):
            if "messages" in chunk:
                self._handle_message_event(chunk.get("messages"))
            self.last_values = chunk
            self._handle_values(chunk, metadata={})

    def complete(self, final_text: str, *, sidecar: dict[str, Any] | None = None) -> None:
        text = str(final_text or "").strip()
        message = self.store.get_message(self.thread_id, self.message_id)
        current_text = ""
        if message:
            for part in message.parts:
                if part.id == self.text_part_id:
                    current_text = str(part.text or "")
                    break
        if text and text != current_text:
            self.store.update_part(
                self.thread_id,
                self.message_id,
                self.text_part_id,
                text=text,
                status="completed",
            )
        else:
            self.store.update_part(
                self.thread_id,
                self.message_id,
                self.text_part_id,
                status="completed",
            )
        self.store.update_message(
            self.thread_id,
            self.message_id,
            status="completed",
            structured_sidecar=dict(sidecar or {}),
        )
        completed_message = self.store.get_message(self.thread_id, self.message_id)
        self._emit(
            "message.completed",
            status="completed",
            data={
                "message_id": self.message_id,
                "part_id": self.text_part_id,
                "text": text,
                "structured_sidecar": dict(sidecar or {}),
                "message": completed_message.model_dump(mode="json") if completed_message is not None else None,
            },
        )

    def fail(self, error: str) -> None:
        self.store.update_message(self.thread_id, self.message_id, status="failed")
        self._emit(
            "message.failed",
            status="failed",
            data={"message_id": self.message_id, "error": str(error or "")},
        )

    def _emit(self, event: str, *, status: str = "", data: dict[str, Any] | None = None) -> None:
        payload = dict(data or {})
        payload.setdefault("run_id", self.run_id)
        payload.setdefault("thread_id", self.thread_id)
        payload.setdefault("message_id", self.message_id)
        self.events.emit(
            self.thread_id,
            event,
            message_id=self.message_id,
            status=status,
            data=payload,
        )

    def _handle_message_event(self, data: Any) -> None:
        metadata: dict[str, Any] = {}
        message = data
        if isinstance(data, (tuple, list)) and data:
            message = data[0]
            if len(data) > 1 and isinstance(data[1], dict):
                metadata = data[1]
        if isinstance(message, (AIMessageChunk, AIMessage)):
            self._handle_tool_calls(message, metadata=metadata)
            reasoning_delta = _message_reasoning_text(message)
            if reasoning_delta:
                self._append_reasoning_if_new(reasoning_delta, metadata=metadata)
            delta = _message_text(message)
            if delta:
                if _is_internal_stream_source(metadata):
                    self._append_subagent_delta(delta, metadata=metadata)
                else:
                    self._append_text_delta(delta)
            return
        if isinstance(message, ToolMessage):
            self._handle_tool_message(message)
            return
        if isinstance(message, dict) and (
            "content" in message
            or "role" in message
            or "content_blocks" in message
            or str(message.get("type") or "").lower() in {"reasoning", "reasoning-delta", "thinking-delta"}
        ):
            reasoning_delta = _message_reasoning_text(message)
            if reasoning_delta:
                self._append_reasoning_if_new(reasoning_delta, metadata=metadata)
            delta = _message_text(message)
            if delta:
                if _is_internal_stream_source(metadata):
                    self._append_subagent_delta(delta, metadata=metadata)
                else:
                    self._append_text_delta(delta)

    def _handle_protocol_message_event(self, data: Any, *, metadata: dict[str, Any]) -> bool:
        rows = data if isinstance(data, list) else [data]
        handled = False
        for row in rows:
            if not isinstance(row, dict):
                continue
            event_name = str(row.get("event") or row.get("type") or "").strip()
            if event_name != "content-block-delta":
                continue
            delta = row.get("delta")
            if not isinstance(delta, dict):
                continue
            delta_type = str(delta.get("type") or "").strip()
            text = ""
            if delta_type in {"text-delta", "text"}:
                text = str(delta.get("text") or "")
            elif delta_type in {"reasoning-delta", "thinking-delta"}:
                text = _message_reasoning_text(delta)
                if text:
                    self._append_reasoning_if_new(text, metadata=metadata)
                    handled = True
                continue
            if not text:
                continue
            if metadata.get("namespace") or _is_internal_stream_source(metadata):
                self._append_subagent_delta(text, metadata=metadata)
            else:
                self._append_text_delta(text)
            handled = True
        return handled

    def _handle_values(self, data: Any, *, metadata: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            return
        messages = data.get("messages")
        if not isinstance(messages, list):
            return
        for message in messages:
            if isinstance(message, ToolMessage):
                self._handle_tool_message(message)

    def _append_text_delta(self, delta: str) -> None:
        text = str(delta or "")
        if not text:
            return
        self.final_text_from_stream += text
        self.store.add_text_delta(self.thread_id, self.message_id, self.text_part_id, text)
        self._emit(
            "message.delta",
            status="streaming",
            data={
                "thread_id": self.thread_id,
                "message_id": self.message_id,
                "part_id": self.text_part_id,
                "delta": text,
            },
        )

    def _append_text_if_new(self, text: str) -> None:
        value = str(text or "").strip()
        if not value:
            return
        current = self.final_text_from_stream
        if current and value.startswith(current):
            tail = value[len(current):]
            if tail:
                self._append_text_delta(tail)
            return
        if value in current:
            return
        self._append_text_delta(value)

    def _append_reasoning_delta(self, delta: str, *, metadata: dict[str, Any]) -> None:
        text = str(delta or "")
        if not text:
            return
        if not self.reasoning_part_id:
            self.reasoning_part_id = new_id("part_reasoning")
            part = MessagePart(
                id=self.reasoning_part_id,
                type="reasoning",
                text="",
                status="streaming",
                meta={"source": _stream_source_name(metadata, default="model"), "metadata": _json_safe(metadata, max_text=2_000)},
            )
            self.store.append_part(self.thread_id, self.message_id, part)
            self._emit(
                "message.part.created",
                status="streaming",
                data={"message_id": self.message_id, "part": part.model_dump(mode="json")},
            )
        self.store.add_text_delta(self.thread_id, self.message_id, self.reasoning_part_id, text)
        self._emit(
            "reasoning.delta",
            status="streaming",
            data={"message_id": self.message_id, "part_id": self.reasoning_part_id, "delta": text},
        )

    def _append_reasoning_if_new(self, text: str, *, metadata: dict[str, Any]) -> None:
        value = str(text or "")
        if not value:
            return
        current = self.reasoning_text_emitted
        if current and value.startswith(current):
            tail = value[len(current):]
            if tail:
                self.reasoning_text_emitted = value
                self._append_reasoning_delta(tail, metadata=metadata)
            return
        if value in current:
            return
        self.reasoning_text_emitted = f"{current}{value}"
        self._append_reasoning_delta(value, metadata=metadata)

    def flush_observed_reasoning(self) -> None:
        """Bridge callback-observed model-end reasoning into the thread stream.

        LangGraph v3 message events are the primary path. This fallback uses the
        local LangChain callback record after it has already normalized provider
        payloads into the run's canonical `LLM_CALL_END.reasoning_text` field.
        """
        if self.observability_store is None:
            return
        try:
            page = self.observability_store.read_events_page(
                names=["LLM_CALL_END"],
                channel="callback",
                run_id=self.run_id,
                after_id=self.observed_reasoning_event_id,
                limit=200,
            )
        except Exception:
            return
        max_id = self.observed_reasoning_event_id
        for event in list(page.get("events") or []):
            event_id = int(event.get("id") or 0)
            if event_id > max_id:
                max_id = event_id
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            reasoning = str(payload.get("reasoning_text") or "").strip()
            if not reasoning:
                continue
            metadata = {
                "agent_name": str(payload.get("agent_name") or event.get("agent_name") or ""),
                "node": str(payload.get("node") or event.get("node") or ""),
                "model": str(payload.get("model") or event.get("model") or ""),
                "callback_run_id": str(payload.get("callback_run_id") or event.get("callback_run_id") or ""),
                "parent_callback_run_id": str(payload.get("parent_callback_run_id") or event.get("parent_callback_run_id") or ""),
            }
            self._append_reasoning_if_new(reasoning, metadata=metadata)
        self.observed_reasoning_event_id = max_id

    def _append_progress_if_new(self, text: str, *, metadata: dict[str, Any]) -> None:
        value = str(text or "").strip()
        if not value:
            return
        current = self.reasoning_text_emitted
        if value in current:
            return
        prefix = "\n\n" if current else ""
        self.reasoning_text_emitted = f"{current}{prefix}{value}"
        self._append_reasoning_delta(f"{prefix}{value}", metadata=metadata)

    def _append_subagent_delta(self, delta: str, *, metadata: dict[str, Any]) -> None:
        text = str(delta or "")
        if not text:
            return
        source = _stream_source_name(metadata)
        part_id = self.subagent_parts_by_source.get(source)
        if not part_id:
            part_id = new_id("part_subagent")
            self.subagent_parts_by_source[source] = part_id
            part = MessagePart(
                id=part_id,
                type="subagent",
                text="",
                status="running",
                meta={"source": source, "metadata": _json_safe(metadata, max_text=2_000)},
            )
            self.store.append_part(self.thread_id, self.message_id, part)
            self._emit(
                "subagent.started",
                status="running",
                data={"message_id": self.message_id, "part_id": part_id, "source": source, "part": part.model_dump(mode="json")},
            )
        self.store.add_text_delta(self.thread_id, self.message_id, part_id, text)
        self._emit(
            "subagent.delta",
            status="running",
            data={"message_id": self.message_id, "part_id": part_id, "source": source, "delta": text},
        )

    def _handle_tool_calls(self, message: Any, *, metadata: dict[str, Any]) -> None:
        calls = list(getattr(message, "tool_calls", None) or [])
        calls.extend(list(getattr(message, "tool_call_chunks", None) or []))
        for call in calls:
            self._handle_tool_call_payload(call, metadata=metadata)

    def _handle_final_tool_calls(self, payload: Any, metadata: dict[str, Any] | None = None) -> None:
        for call in self._extract_final_tool_calls(payload):
            self._handle_tool_call_payload(call, metadata=metadata or {})

    def _handle_model_end_text(self, payload: Any, *, metadata: dict[str, Any]) -> None:
        for message in self._extract_messages_from_payload(payload):
            reasoning = _message_reasoning_text(message)
            if reasoning:
                self._append_reasoning_if_new(reasoning, metadata=metadata)
            text = _message_text(message).strip()
            if not text:
                continue
            has_tool_calls = bool(self._extract_final_tool_calls(message))
            if has_tool_calls or _is_internal_stream_source(metadata):
                self._append_progress_if_new(text, metadata=metadata)
            else:
                self._append_text_if_new(text)

    def _extract_messages_from_payload(self, payload: Any) -> list[Any]:
        out: list[Any] = []
        seen: set[int] = set()

        def add(value: Any) -> None:
            if value is None:
                return
            marker = id(value)
            if marker in seen:
                return
            seen.add(marker)
            out.append(value)

        def visit(value: Any, depth: int = 0) -> None:
            if value is None or depth > 8:
                return
            if isinstance(value, (AIMessageChunk, AIMessage)):
                add(value)
                return
            if isinstance(value, dict):
                if "content" in value and ("role" in value or "type" in value):
                    add(value)
                for key in ("output", "message", "chunk", "messages", "generations"):
                    if key in value:
                        visit(value.get(key), depth + 1)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, depth + 1)
                return
            for attr in ("output", "message", "chunk", "generations"):
                if hasattr(value, attr):
                    visit(getattr(value, attr), depth + 1)

        visit(payload)
        return out

    def _extract_final_tool_calls(self, payload: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(value: Any) -> None:
            row = value
            if not isinstance(row, dict):
                row = _json_safe(row)
            if not isinstance(row, dict):
                return
            raw = row.get("raw")
            if isinstance(raw, dict) and (raw.get("name") or raw.get("id") or raw.get("args") is not None):
                row = raw
            if not (row.get("name") or row.get("tool") or row.get("id") or row.get("tool_call_id")):
                return
            key = str(row.get("id") or row.get("tool_call_id") or row.get("index") or row.get("name") or row)
            if key in seen:
                return
            seen.add(key)
            out.append(row)

        def visit(value: Any, depth: int = 0) -> None:
            if value is None or depth > 8:
                return
            if isinstance(value, (AIMessageChunk, AIMessage)):
                for call in list(getattr(value, "tool_calls", None) or []):
                    add(call)
                for call in list(getattr(value, "tool_call_chunks", None) or []):
                    add(call)
                extra = getattr(value, "additional_kwargs", None)
                if isinstance(extra, dict):
                    visit(extra.get("tool_calls"), depth + 1)
                return
            if isinstance(value, dict):
                if "raw" in value or value.get("type") in {"tool_call", "function"}:
                    add(value)
                for key in ("parsed_tool_calls", "tool_calls", "tool_call_chunks", "output", "message", "chunk", "generations"):
                    if key in value:
                        visit(value.get(key), depth + 1)
                return
            if isinstance(value, (list, tuple)):
                for item in value:
                    visit(item, depth + 1)
                return
            for attr in ("message", "chunk", "output", "generations"):
                if hasattr(value, attr):
                    visit(getattr(value, attr), depth + 1)

        visit(payload)
        return out

    def _handle_tool_call_payload(self, payload: Any, *, metadata: dict[str, Any] | None = None) -> None:
        if not payload:
            return
        stream_fields = self._tool_stream_fields(metadata or {})
        rows = payload if isinstance(payload, list) else [payload]
        for row in rows:
            if not isinstance(row, dict):
                row = _json_safe(row)
            if not isinstance(row, dict):
                continue
            index = str(row.get("index") if row.get("index") is not None else "").strip()
            explicit_call_id = str(row.get("id") or row.get("tool_call_id") or "").strip()
            if explicit_call_id and index:
                self.tool_call_id_by_index[index] = explicit_call_id
            call_id = explicit_call_id or (self.tool_call_id_by_index.get(index, "") if index else "")
            if not call_id:
                call_id = index
            if not call_id:
                call_id = f"tc_{len(self.tool_parts_by_call_id) + 1}"
            name = str(row.get("name") or row.get("tool") or self.tool_names_by_call_id.get(call_id, "")).strip()
            if name:
                self.tool_names_by_call_id[call_id] = name
            if stream_fields:
                self.tool_stream_meta_by_call_id[call_id] = {**self.tool_stream_meta_by_call_id.get(call_id, {}), **stream_fields}
            input_payload = self._record_tool_input(call_id, row)
            observed_fields = self._observed_tool_stream_fields(tool_name=name, input_payload=input_payload)
            if observed_fields:
                self.tool_stream_meta_by_call_id[call_id] = {**self.tool_stream_meta_by_call_id.get(call_id, {}), **observed_fields}
            part_id = self.tool_parts_by_call_id.get(call_id)
            if not part_id:
                part_id = new_id("part_tool")
                self.tool_parts_by_call_id[call_id] = part_id
                part = MessagePart(
                    id=part_id,
                    type="tool-call",
                    status="running",
                    meta={
                        "tool_call_id": call_id,
                        "tool": name,
                        "input": input_payload,
                        **self.tool_stream_meta_by_call_id.get(call_id, {}),
                    },
                )
                self.store.append_part(self.thread_id, self.message_id, part)
                self._emit(
                    "tool_call.started",
                    status="running",
                    data={"tool_call_id": call_id, "part_id": part_id, "tool": name, "input": input_payload, **self.tool_stream_meta_by_call_id.get(call_id, {})},
                )
            else:
                self._update_tool_part_meta(part_id, tool_call_id=call_id, tool=name, input_payload=input_payload, stream_meta=self.tool_stream_meta_by_call_id.get(call_id, {}))
                self._emit(
                    "tool_call.delta",
                    status="running",
                    data={"tool_call_id": call_id, "part_id": part_id, "tool": name, "input": input_payload, "delta": _json_safe(row), **self.tool_stream_meta_by_call_id.get(call_id, {})},
                )

    def _tool_stream_fields(self, metadata: dict[str, Any]) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        source = _subagent_source_name(metadata) or _stream_source_name(metadata, default="")
        if source:
            fields["agent_name"] = source
            fields["subagent_source"] = source
        namespace = metadata.get("namespace")
        if namespace not in (None, "", []):
            fields["stream_namespace"] = _json_safe(namespace, max_text=2_000)
        return fields

    def _observed_tool_stream_fields(self, *, tool_name: str, input_payload: Any) -> dict[str, Any]:
        name = str(tool_name or "").strip()
        if not name or self.observability_store is None:
            return {}
        try:
            page = self.observability_store.read_events_page(
                names=["TOOL_CALL_START", "TOOL_RAW_INPUT"],
                channel="callback",
                run_id=self.run_id,
                limit=500,
            )
        except Exception:
            return {}
        expected = self._canonical_tool_input(input_payload)
        fallback: dict[str, Any] = {}
        for event in reversed(list(page.get("events") or [])):
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            event_tool = str(payload.get("tool") or payload.get("tool_name") or event.get("tool") or "").strip()
            if event_tool != name:
                continue
            source = str(payload.get("agent_name") or event.get("agent_name") or "").strip()
            if not source:
                continue
            candidate = {
                "agent_name": source,
                "subagent_source": source,
            }
            if not fallback:
                fallback = candidate
            observed_input = self._canonical_tool_input(self._observed_tool_input_from_payload(payload))
            if expected and observed_input and observed_input == expected:
                return candidate
        return fallback

    @staticmethod
    def _canonical_tool_input(value: Any) -> str:
        try:
            return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception:
            return ""

    @staticmethod
    def _observed_tool_input_from_payload(payload: dict[str, Any]) -> Any:
        for key in ("params_full", "raw_params", "input", "args"):
            value = payload.get(key)
            if value not in (None, ""):
                return value
        compact = payload.get("params_compact")
        if isinstance(compact, str) and compact.strip():
            try:
                return json.loads(compact)
            except Exception:
                return compact
        return {}

    def _record_tool_input(self, call_id: str, row: dict[str, Any]) -> Any:
        if "args" not in row and "input" not in row:
            return self._best_tool_input(call_id)
        value = row.get("args") if "args" in row else row.get("input")
        if isinstance(value, str):
            if value:
                current = self.tool_arg_buffers_by_call_id.get(call_id, "")
                combined = current + value
                self.tool_arg_buffers_by_call_id[call_id] = combined
                parsed = self._parse_tool_args(combined)
            else:
                parsed = self.tool_inputs_by_call_id.get(call_id, {})
        else:
            parsed = _json_safe(value or {})
        if parsed == {}:
            observed = self._lookup_observed_tool_input(call_id)
            if observed is not None:
                parsed = observed
        if parsed == {}:
            resumed = self._consume_resume_tool_input(call_id, str(row.get("name") or row.get("tool") or ""))
            if resumed is not None:
                parsed = resumed
        self.tool_inputs_by_call_id[call_id] = parsed
        return parsed

    def _best_tool_input(self, call_id: str) -> Any:
        cached = self.tool_inputs_by_call_id.get(call_id)
        if cached not in (None, {}):
            return cached
        observed = self._lookup_observed_tool_input(call_id)
        if observed is not None:
            self.tool_inputs_by_call_id[call_id] = observed
            return observed
        return cached or {}

    def _consume_resume_tool_input(self, call_id: str, tool_name: str = "") -> Any | None:
        name = str(tool_name or self.tool_names_by_call_id.get(call_id, "")).strip()
        if not name:
            return None
        queued = self.resume_tool_inputs_by_name.get(name)
        if not queued:
            return None
        value = queued.pop(0)
        if not queued:
            self.resume_tool_inputs_by_name.pop(name, None)
        self.tool_inputs_by_call_id[call_id] = value
        return value

    def _lookup_observed_tool_input(self, call_id: str) -> Any | None:
        if not call_id or self.observability_store is None:
            return None
        try:
            page = self.observability_store.read_events_page(names=["LLM_RAW_RESPONSE"], run_id=self.run_id, limit=500)
        except Exception:
            return None
        for event in reversed(page.get("events") or []):
            payload = event.get("payload") if isinstance(event, dict) else None
            found = self._find_tool_input_in_payload(payload, call_id)
            if found is not None:
                return found
        return None

    def _find_tool_input_in_payload(self, payload: Any, call_id: str) -> Any | None:
        if payload is None:
            return None
        if isinstance(payload, dict):
            raw = payload.get("raw")
            if isinstance(raw, dict) and str(raw.get("id") or raw.get("tool_call_id") or "") == call_id:
                return _json_safe(raw.get("args") if "args" in raw else raw.get("input") or {})
            if str(payload.get("id") or payload.get("tool_call_id") or "") == call_id:
                if "args" in payload or "input" in payload:
                    return _json_safe(payload.get("args") if "args" in payload else payload.get("input") or {})
                args_json = payload.get("args_json")
                if isinstance(args_json, str) and args_json.strip():
                    try:
                        return _json_safe(json.loads(args_json))
                    except Exception:
                        return {"_raw": args_json}
            for key in ("parsed_tool_calls", "tool_calls", "raw_tool_calls", "generations"):
                found = self._find_tool_input_in_payload(payload.get(key), call_id)
                if found is not None:
                    return found
            return None
        if isinstance(payload, (list, tuple)):
            for item in payload:
                found = self._find_tool_input_in_payload(item, call_id)
                if found is not None:
                    return found
        return None

    @staticmethod
    def _parse_tool_args(text: str) -> Any:
        raw = str(text or "")
        if not raw.strip():
            return {}
        try:
            return _json_safe(json.loads(raw))
        except Exception:
            return {"_raw": raw}

    def _tool_part_meta(self, part_id: str) -> dict[str, Any]:
        message = self.store.get_message(self.thread_id, self.message_id)
        if not message:
            return {}
        for part in message.parts:
            if part.id == part_id:
                return dict(part.meta or {})
        return {}

    def _update_tool_part_meta(self, part_id: str, *, tool_call_id: str, tool: str = "", input_payload: Any = None, output: Any = None, status: str | None = None, text: str | None = None, stream_meta: dict[str, Any] | None = None) -> None:
        current = self._tool_part_meta(part_id)
        next_meta = {
            **current,
            "tool_call_id": tool_call_id or current.get("tool_call_id", ""),
            "tool": tool or current.get("tool", ""),
            "input": input_payload if input_payload is not None else current.get("input", {}),
            **dict(stream_meta or {}),
        }
        if output is not None:
            next_meta["output"] = output
        updates: dict[str, Any] = {"meta": next_meta}
        if status is not None:
            updates["status"] = status
        if text is not None:
            updates["text"] = text
        self.store.update_part(self.thread_id, self.message_id, part_id, **updates)

    def _handle_tool_message(self, message: ToolMessage) -> None:
        call_id = str(getattr(message, "tool_call_id", "") or "").strip()
        if call_id and call_id in self.historical_completed_tool_call_ids and call_id not in self.tool_parts_by_call_id:
            return
        output_text = _message_text(message)
        message_key = str(getattr(message, "id", "") or "").strip()
        if not message_key:
            message_key = self._canonical_tool_input(
                {
                    "tool_call_id": call_id,
                    "name": str(getattr(message, "name", "") or ""),
                    "output": output_text,
                }
            ) or str(id(message))
        if message_key in self.completed_tool_messages:
            return
        self.completed_tool_messages.add(message_key)
        part_id = self.tool_parts_by_call_id.get(call_id)
        if not part_id:
            tool_name = str(getattr(message, "name", "") or call_id or "tool")
            fallback_call_id = call_id or tool_name
            self._handle_tool_call_payload({"id": fallback_call_id, "name": tool_name, "args": {}})
            call_id = fallback_call_id
            part_id = self.tool_parts_by_call_id.get(call_id, "")
        if part_id:
            current_meta = self._tool_part_meta(part_id)
            tool_name = str(getattr(message, "name", "") or current_meta.get("tool") or call_id or "tool")
            input_payload = self._best_tool_input(call_id) or current_meta.get("input", {})
            observed_fields = self._observed_tool_stream_fields(tool_name=tool_name, input_payload=input_payload)
            if observed_fields:
                self.tool_stream_meta_by_call_id[call_id] = {**self.tool_stream_meta_by_call_id.get(call_id, {}), **observed_fields}
            output_payload = _json_safe(_message_text(message))
            self._update_tool_part_meta(
                part_id,
                tool_call_id=call_id,
                tool=tool_name,
                input_payload=input_payload,
                output=output_payload,
                status="completed",
                text=_message_text(message),
                stream_meta=self.tool_stream_meta_by_call_id.get(call_id, {}),
            )
        self._emit(
            "tool_call.completed",
            status="completed",
            data={
                "tool_call_id": call_id,
                "part_id": part_id or "",
                "tool": self.tool_names_by_call_id.get(call_id, str(getattr(message, "name", "") or "")),
                "input": self._best_tool_input(call_id) or self._tool_part_meta(part_id or "").get("input", {}),
                "output": _json_safe(output_text),
                **self.tool_stream_meta_by_call_id.get(call_id, {}),
            },
        )
        artifact_payload = getattr(message, "artifact", None)
        for record in self._register_tool_artifacts(artifact_payload, tool_call_id=call_id):
            self.store.append_part(
                self.thread_id,
                self.message_id,
                ArtifactPart(
                    id=new_id("part_artifact"),
                    artifact_id=record.artifact_id,
                    renderer=record.renderer,
                    title=record.title,
                    summary=record.summary,
                    path=record.path,
                    status="completed",
                ),
            )
            self._emit(
                "artifact.created",
                status="completed",
                data=record.model_dump(mode="json"),
            )
        for receipt in self._extract_remote_receipts(artifact_payload):
            self._upsert_receipt_part(receipt, tool_call_id=call_id)

    def _handle_tool_end_payload(self, *, name: str, output: Any, metadata: dict[str, Any]) -> None:
        call_id = str(name or _stream_source_name(metadata, default="tool") or "tool")
        stream_fields = self._tool_stream_fields(metadata)
        if stream_fields:
            self.tool_stream_meta_by_call_id[call_id] = {**self.tool_stream_meta_by_call_id.get(call_id, {}), **stream_fields}
        part_id = self.tool_parts_by_call_id.get(call_id)
        if not part_id:
            self._handle_tool_call_payload({"id": call_id, "name": name or call_id, "args": {}}, metadata=metadata)
            part_id = self.tool_parts_by_call_id.get(call_id, "")
        text = _message_text(output)
        if part_id:
            current_meta = self._tool_part_meta(part_id)
            input_payload = self._best_tool_input(call_id) or current_meta.get("input", {})
            observed_fields = self._observed_tool_stream_fields(tool_name=name or str(current_meta.get("tool") or call_id), input_payload=input_payload)
            if observed_fields:
                self.tool_stream_meta_by_call_id[call_id] = {**self.tool_stream_meta_by_call_id.get(call_id, {}), **observed_fields}
            self._update_tool_part_meta(
                part_id,
                tool_call_id=call_id,
                tool=name or str(current_meta.get("tool") or call_id),
                input_payload=input_payload,
                output=_json_safe(output),
                status="completed",
                text=text,
                stream_meta=self.tool_stream_meta_by_call_id.get(call_id, {}),
            )
        self._emit(
            "tool_call.completed",
            status="completed",
            data={
                "tool_call_id": call_id,
                "part_id": part_id,
                "tool": name or call_id,
                "input": self._best_tool_input(call_id) or self._tool_part_meta(part_id).get("input", {}),
                "output": _json_safe(output),
                **self.tool_stream_meta_by_call_id.get(call_id, {}),
            },
        )

    def _register_tool_artifacts(self, artifact_payload: Any, *, tool_call_id: str = "") -> list[Any]:
        paths = self._extract_artifact_paths(artifact_payload)
        out = []
        for path in paths:
            try:
                record = self.artifact_registry.register_path(
                    path,
                    thread_id=self.thread_id,
                    message_id=self.message_id,
                    tool_call_id=tool_call_id,
                    run_id=self.run_id,
                    meta={"source": "tool_artifact"},
                )
            except Exception:
                continue
            out.append(record)
        return out

    def _extract_artifact_paths(self, payload: Any) -> list[str]:
        paths: list[str] = []

        def _walk(value: Any, key: str = "") -> None:
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    _walk(sub_value, str(sub_key))
                return
            if isinstance(value, list):
                for item in value:
                    _walk(item, key)
                return
            if not isinstance(value, str):
                return
            text = value.strip()
            if not text or len(text) > 500:
                return
            key_l = key.lower()
            looks_path_key = (
                "path" in key_l
                or key_l.endswith("_rel")
                or key_l.endswith("_file")
                or key_l in {"file", "files", "href"}
            )
            suffix = Path(text).suffix.lower()
            looks_file = bool(suffix) or "/" in text
            if looks_path_key and looks_file:
                paths.append(text)

        _walk(payload)
        deduped: list[str] = []
        seen: set[str] = set()
        for path in paths:
            if path not in seen:
                seen.add(path)
                deduped.append(path)
        return deduped[:50]

    def _extract_remote_receipts(self, payload: Any) -> list[dict[str, Any]]:
        receipts: list[dict[str, Any]] = []
        receipt_keys = {
            "remote_context_id",
            "submission_hash",
            "receipt_rel",
            "submitted_at",
            "updated_at",
            "jobs",
            "job_status_counts",
            "task_state_counts",
            "submission_dir",
            "work_base",
            "resources",
        }

        def _walk(value: Any) -> None:
            if isinstance(value, dict):
                normalized = _json_safe(value)
                keys = set(normalized.keys()) if isinstance(normalized, dict) else set()
                if (
                    isinstance(normalized, dict)
                    and keys.intersection(receipt_keys)
                    and any(normalized.get(key) not in (None, "", [], {}) for key in ("remote_context_id", "submission_hash", "receipt_rel", "submission_dir"))
                ):
                    receipts.append(normalized)
                for sub_value in value.values():
                    _walk(sub_value)
                return
            if isinstance(value, list):
                for item in value:
                    _walk(item)

        _walk(payload)
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        for receipt in receipts:
            key = "|".join(
                str(receipt.get(item) or "")
                for item in ("remote_context_id", "submission_hash", "receipt_rel", "submission_dir")
            )
            if not key.strip("|"):
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(receipt)
        return out[:20]

    def _upsert_receipt_part(self, receipt: dict[str, Any], *, tool_call_id: str = "") -> None:
        receipt_id = str(receipt.get("remote_context_id") or receipt.get("submission_hash") or receipt.get("receipt_rel") or "").strip()
        part_id = f"part_receipt_{re.sub(r'[^A-Za-z0-9_.:-]+', '_', receipt_id)[:80]}" if receipt_id else new_id("part_receipt")
        message = self.store.get_message(self.thread_id, self.message_id)
        if message and any(part.id == part_id for part in message.parts):
            self.store.update_part(
                self.thread_id,
                self.message_id,
                part_id,
                status=str(receipt.get("status") or "updated"),
                meta={**receipt, "tool_call_id": tool_call_id, "run_id": self.run_id},
            )
        else:
            self.store.append_part(
                self.thread_id,
                self.message_id,
                MessagePart(
                    id=part_id,
                    type="receipt",
                    status=str(receipt.get("status") or "updated"),
                    text=str(receipt.get("receipt_rel") or receipt.get("remote_context_id") or "Remote task receipt"),
                    meta={**receipt, "tool_call_id": tool_call_id, "run_id": self.run_id},
                ),
            )
        self._emit(
            "task_receipt.updated",
            status=str(receipt.get("status") or "updated"),
            data={
                "part_id": part_id,
                "tool_call_id": tool_call_id,
                "receipt": {**receipt, "tool_call_id": tool_call_id, "run_id": self.run_id},
            },
        )

    def _handle_interrupts(self, interrupts: Any) -> None:
        if self.interrupt_id:
            return
        interrupt_id = new_id("interrupt")
        part_id = new_id("part_interrupt")
        self.interrupt_id = interrupt_id
        record = InterruptRecord(
            interrupt_id=interrupt_id,
            thread_id=self.thread_id,
            message_id=self.message_id,
            part_id=part_id,
            status="pending",
            title="Review required",
            body="The agent paused for a human decision.",
            payload={"interrupts": _json_safe(interrupts)},
        )
        self.store.append_part(
            self.thread_id,
            self.message_id,
            MessagePart(
                id=part_id,
                type="interrupt",
                status="pending",
                text=record.body,
                meta=record.model_dump(mode="json"),
            ),
        )
        self.store.update_message(self.thread_id, self.message_id, status="interrupted")
        self.store.update_thread(
            self.thread_id,
            status=ThreadStatus.INTERRUPTED,
            active_message_id=self.message_id,
            active_run_id=self.run_id,
        )
        self._emit(
            "interrupt.created",
            status="interrupted",
            data=record.model_dump(mode="json"),
        )
        self._emit(
            "thread.status",
            status="interrupted",
            data={"status": "interrupted"},
        )


class StreamingSpecialistRunner:
    """Streaming execution adapter for an already constructed SpecialistRunner."""

    def __init__(
        self,
        *,
        runner: SpecialistRunner,
        thread_store: ThreadStore,
        event_broker: ThreadEventBroker,
        artifact_registry: ArtifactRegistry,
        should_stop: Callable[[str], bool] | None = None,
    ) -> None:
        self.runner = runner
        self.thread_store = thread_store
        self.event_broker = event_broker
        self.artifact_registry = artifact_registry
        self.should_stop = should_stop or (lambda _thread_id: False)

    def _emit_thread_event(
        self,
        thread_id: str,
        event: str,
        *,
        message_id: str = "",
        status: str = "",
        data: dict[str, Any] | None = None,
    ) -> None:
        payload = dict(data or {})
        payload.setdefault("run_id", self.runner.run_context.run_id)
        payload.setdefault("thread_id", thread_id)
        if message_id:
            payload.setdefault("message_id", message_id)
        self.event_broker.emit(thread_id, event, message_id=message_id, status=status, data=payload)

    async def arun_turn(
        self,
        *,
        prompt: str,
        content: str | list[dict[str, Any]] | None = None,
        entrypoint: SpecialistEntrypoint,
        thread_id: str,
        message_id: str,
        text_part_id: str,
        deepagent_thread_id: str,
    ) -> dict[str, Any]:
        user_prompt = str(prompt or "").strip()
        message_content: str | list[dict[str, Any]]
        if isinstance(content, list) and content:
            message_content = content
        elif isinstance(content, str) and content.strip():
            message_content = content.strip()
        else:
            message_content = user_prompt
        return await self._run_stream(
            input_payload={"messages": [{"role": "user", "content": message_content}]},
            entrypoint=entrypoint,
            thread_id=thread_id,
            message_id=message_id,
            text_part_id=text_part_id,
            deepagent_thread_id=deepagent_thread_id,
            user_prompt=user_prompt,
            resume=False,
        )

    async def aresume(
        self,
        *,
        decisions: list[dict[str, Any]],
        entrypoint: SpecialistEntrypoint,
        thread_id: str,
        message_id: str,
        text_part_id: str,
        deepagent_thread_id: str,
        resume_tool_inputs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        cleaned = self._validate_decisions(decisions)
        return await self._run_stream(
            input_payload=Command(resume={"decisions": cleaned}),
            entrypoint=entrypoint,
            thread_id=thread_id,
            message_id=message_id,
            text_part_id=text_part_id,
            deepagent_thread_id=deepagent_thread_id,
            user_prompt="",
            resume=True,
            resume_tool_inputs=resume_tool_inputs or [],
        )

    async def _run_stream(
        self,
        *,
        input_payload: Any,
        entrypoint: SpecialistEntrypoint,
        thread_id: str,
        message_id: str,
        text_part_id: str,
        deepagent_thread_id: str,
        user_prompt: str,
        resume: bool,
        resume_tool_inputs: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        runner = self.runner
        files_root = workspace_root(runner.run_context.workspace)
        files_root.mkdir(parents=True, exist_ok=True)
        runner._stage_deepagent_assets(files_root)
        research_kernel_relpath = ""
        research_goal: ResearchGoalRecord | None = None
        research_goal_relpath = ""
        if entrypoint == "research":
            prompt_for_goal = user_prompt or "Resume interrupted thread."
            research_kernel_relpath = runner._ensure_research_kernel_seed(
                files_root=files_root,
                thread_id=deepagent_thread_id,
                prompt=prompt_for_goal,
            )
            research_goal = runner._research_goal_for_run(thread_id=deepagent_thread_id, prompt=prompt_for_goal, resume_feedback=None)
            research_goal_relpath = runner._research_goal_relpath(deepagent_thread_id)

        usage_handler = runner._new_usage_callback()
        translator = CatMasterStreamTranslator(
            store=self.thread_store,
            events=self.event_broker,
            artifact_registry=self.artifact_registry,
            thread_id=thread_id,
            message_id=message_id,
            text_part_id=text_part_id,
            run_id=runner.run_context.run_id,
            observability_store=ObservabilityStore(runner.run_context.run_dir),
            resume_tool_inputs=resume_tool_inputs or [],
        )
        runner._emit("RUN_START", payload={"entrypoint": entrypoint, "status": "running", "thread_id": thread_id})
        self.thread_store.update_thread(thread_id, status=ThreadStatus.RUNNING, active_run_id=runner.run_context.run_id, active_message_id=message_id)
        self._emit_thread_event(thread_id, "thread.status", message_id=message_id, status="running", data={"status": "running"})
        try:
            runner._write_run_state(
                {
                    "schema_version": 1,
                    "entrypoint": entrypoint,
                    "status": "running",
                    "phase": "streaming",
                    "active_specialist": entrypoint,
                    "thread_id": deepagent_thread_id,
                    "webui_thread_id": thread_id,
                    "proposal_review": False,
                    "proposal_revision_count": 0,
                    "pending_human_input": None,
                    "todo_items": [],
                    "artifacts": [],
                    "delegation_log": [],
                    "text_preview": user_prompt[:280],
                    "user_prompt": user_prompt,
                    "final_answer": "",
                    "summary": "",
                    "facts": [],
                    **runner._research_kernel_state_fields(files_root=files_root, thread_id=deepagent_thread_id, relpath=research_kernel_relpath),
                    **runner._research_goal_state_fields(research_goal=research_goal, relpath=research_goal_relpath),
                }
            )
            async with runner._open_agent_runtime(files_root=files_root) as runtime:
                agent = await runner._build_entry_agent(entrypoint=entrypoint, runtime=runtime, thread_id=deepagent_thread_id)
                config = {
                    "configurable": {
                        "thread_id": deepagent_thread_id,
                        "project_id": str(runner.run_context.project_id or "").strip(),
                    },
                    "callbacks": runner._langchain_callbacks(
                        usage_handler=usage_handler,
                        default_agent_name=f"{entrypoint}_specialist",
                    ),
                    "metadata": {"lc_agent_name": f"{entrypoint}_specialist"},
                }
                await self._consume_agent_stream(agent, input_payload=input_payload, config=config, translator=translator)
            if translator.interrupt_id:
                if entrypoint == "research" and research_goal is not None:
                    research_goal = runner._update_research_goal_status(research_goal, status="paused")
                interrupted_state = {
                    "schema_version": 1,
                    "entrypoint": entrypoint,
                    "status": "interrupted",
                    "phase": "hitl_review",
                    "active_specialist": entrypoint,
                    "thread_id": deepagent_thread_id,
                    "webui_thread_id": thread_id,
                    "proposal_review": False,
                    "proposal_revision_count": 0,
                    "pending_human_input": {
                        "kind": "tool_review",
                        "interrupt_id": translator.interrupt_id,
                        "message_id": message_id,
                    },
                    "todo_items": [],
                    "artifacts": [],
                    "delegation_log": [],
                    "text_preview": "Waiting for human review.",
                    "user_prompt": user_prompt,
                    "final_answer": "",
                    "summary": "Waiting for human review.",
                    "facts": [],
                    **runner._research_kernel_state_fields(files_root=files_root, thread_id=deepagent_thread_id, relpath=research_kernel_relpath),
                    **runner._research_goal_state_fields(research_goal=research_goal, relpath=research_goal_relpath),
                }
                runner._write_run_state(interrupted_state)
                runner._write_usage_summary(usage_handler)
                self._emit_usage_updated(thread_id=thread_id, message_id=message_id)
                runner._emit("RUN_PAUSED", payload={"entrypoint": entrypoint, "status": "interrupted", "thread_id": thread_id})
                self.thread_store.update_thread(
                    thread_id,
                    status=ThreadStatus.INTERRUPTED,
                    active_message_id=message_id,
                    active_run_id=runner.run_context.run_id,
                )
                self._emit_thread_event(thread_id, "thread.status", message_id=message_id, status="interrupted", data={"status": "interrupted"})
                return {
                    "run_id": runner.run_context.run_id,
                    "run_dir": str(runner.run_context.run_dir),
                    "status": "interrupted",
                    "summary": "Waiting for human review.",
                    "facts": [],
                    "final_answer": "",
                    "artifacts": [],
                    "interrupt_id": translator.interrupt_id,
                }
            raw_output = translator.last_values
            parsed = runner._finalize_report(runner._coerce_report(raw=raw_output or translator.final_text_from_stream))
            reported_files = [
                *list(parsed["files"]),
                *_extract_workspace_paths_from_text(parsed["text"]),
                *_extract_sidecar_artifact_paths(raw_output),
            ]
            seen_reported_files: set[str] = set()
            deduped_reported_files: list[str] = []
            for path in reported_files:
                key = str(path).replace("\\", "/").lstrip("/")
                if key in seen_reported_files:
                    continue
                seen_reported_files.add(key)
                deduped_reported_files.append(path)
            reported_files = deduped_reported_files
            artifacts = runner._artifact_rows(reported_files)
            artifact_records = self.artifact_registry.register_from_run_state(
                {"artifacts": artifacts, "thread_id": thread_id, "run_id": runner.run_context.run_id},
                thread_id=thread_id,
                message_id=message_id,
                run_id=runner.run_context.run_id,
            )
            for record in artifact_records:
                self.thread_store.append_part(
                    thread_id,
                    message_id,
                    ArtifactPart(
                        id=new_id("part_artifact"),
                        artifact_id=record.artifact_id,
                        renderer=record.renderer,
                        title=record.title,
                        summary=record.summary,
                        path=record.path,
                        status="completed",
                    ),
                )
                self._emit_thread_event(thread_id, "artifact.created", message_id=message_id, status="completed", data=record.model_dump(mode="json"))
            sidecar = {
                "summary": parsed["summary"],
                "facts": list(parsed["facts"]),
                "artifact_ids": [record.artifact_id for record in artifact_records],
                "citations": [],
                "review_target": str(parsed.get("review_target") or "").strip(),
            }
            translator.complete(parsed["text"], sidecar=sidecar)
            if entrypoint == "research" and research_goal is not None:
                research_goal = runner._complete_research_goal(
                    research_goal,
                    completion_audit_md=runner._research_completion_audit_md(objective=research_goal.objective, parsed=parsed),
                )
            run_state = {
                "schema_version": 1,
                "entrypoint": entrypoint,
                "status": "done",
                "phase": "finalized",
                "active_specialist": entrypoint,
                "thread_id": deepagent_thread_id,
                "webui_thread_id": thread_id,
                "proposal_review": False,
                "proposal_revision_count": 0,
                "pending_human_input": None,
                "todo_items": [],
                "artifacts": artifacts,
                "artifact_ids": [record.artifact_id for record in artifact_records],
                "delegation_log": [],
                "text_preview": parsed["text"][:280],
                "user_prompt": user_prompt,
                "final_answer": parsed["text"],
                "summary": parsed["summary"],
                "facts": list(parsed["facts"]),
                "review_target": str(parsed.get("review_target") or "").strip(),
                **runner._research_kernel_state_fields(files_root=files_root, thread_id=deepagent_thread_id, relpath=research_kernel_relpath),
                **runner._research_goal_state_fields(research_goal=research_goal, relpath=research_goal_relpath),
            }
            runner._write_run_state(run_state)
            runner._write_usage_summary(usage_handler)
            self._emit_usage_updated(thread_id=thread_id, message_id=message_id)
            runner._emit("RUN_END", payload={"entrypoint": entrypoint, "status": "done", "thread_id": thread_id})
            self.thread_store.update_thread(thread_id, status=ThreadStatus.IDLE, active_message_id="", active_run_id="")
            self._emit_thread_event(thread_id, "thread.status", message_id=message_id, status="idle", data={"status": "idle"})
            return {
                "run_id": runner.run_context.run_id,
                "run_dir": str(runner.run_context.run_dir),
                "status": "done",
                "summary": parsed["summary"],
                "facts": list(parsed["facts"]),
                "final_answer": parsed["text"],
                "artifacts": artifacts,
                "artifact_ids": [record.artifact_id for record in artifact_records],
            }
        except asyncio.CancelledError:
            self.thread_store.update_thread(thread_id, status=ThreadStatus.STOPPED, active_message_id="", active_run_id="")
            self.thread_store.update_message(thread_id, message_id, status="interrupted")
            runner._write_run_state(
                {
                    "schema_version": 1,
                    "entrypoint": entrypoint,
                    "status": "stopped",
                    "phase": "stopped",
                    "active_specialist": entrypoint,
                    "thread_id": deepagent_thread_id,
                    "webui_thread_id": thread_id,
                    "text_preview": "Thread stopped by user.",
                    "user_prompt": user_prompt,
                    "final_answer": "",
                    "summary": "Thread stopped by user.",
                    "facts": [],
                    "artifacts": [],
                }
            )
            runner._write_usage_summary(usage_handler)
            self._emit_usage_updated(thread_id=thread_id, message_id=message_id)
            runner._emit("RUN_PAUSED", payload={"entrypoint": entrypoint, "status": "stopped", "thread_id": thread_id})
            self._emit_thread_event(thread_id, "thread.status", message_id=message_id, status="stopped", data={"status": "stopped"})
            raise
        except Exception as exc:
            error = str(exc)
            translator.fail(error)
            runner._write_run_state(
                {
                    "schema_version": 1,
                    "entrypoint": entrypoint,
                    "status": "error",
                    "phase": "failed",
                    "active_specialist": entrypoint,
                    "thread_id": deepagent_thread_id,
                    "webui_thread_id": thread_id,
                    "text_preview": error[:280],
                    "user_prompt": user_prompt,
                    "final_answer": "",
                    "summary": error.strip() or "Run failed.",
                    "facts": [],
                    "artifacts": [],
                }
            )
            runner._write_usage_summary(usage_handler)
            self._emit_usage_updated(thread_id=thread_id, message_id=message_id)
            runner._emit("RUN_END", payload={"entrypoint": entrypoint, "status": "error", "error": error, "thread_id": thread_id})
            self.thread_store.update_thread(thread_id, status=ThreadStatus.ERROR, active_message_id="", active_run_id="")
            self._emit_thread_event(thread_id, "thread.status", message_id=message_id, status="error", data={"status": "error", "error": error})
            raise

    def _emit_usage_updated(self, *, thread_id: str, message_id: str) -> None:
        summary = summarize_usage_from_observability(self.runner.run_context.run_dir)
        if not summary:
            return
        self._emit_thread_event(
            thread_id,
            "usage.updated",
            message_id=message_id,
            data={
                "run_id": self.runner.run_context.run_id,
                "run_dir": str(self.runner.run_context.run_dir),
                "usage_summary": summary,
            },
        )

    async def _consume_agent_stream(self, agent: Any, *, input_payload: Any, config: dict[str, Any], translator: CatMasterStreamTranslator) -> None:
        started = False
        try:
            stream_or_awaitable = agent.astream_events(input_payload, config=config, version="v3")
            stream = await stream_or_awaitable if inspect.isawaitable(stream_or_awaitable) else stream_or_awaitable
            async for event in stream:
                started = True
                translator.apply_v3_event(event)
                translator.flush_observed_reasoning()
                if self.should_stop(translator.thread_id):
                    translator.flush_observed_reasoning()
                    raise asyncio.CancelledError("Graceful stop requested.")
            translator.flush_observed_reasoning()
            return
        except Exception as exc:
            if started:
                raise
            logger.info("stream_events(version='v3') failed, falling back to astream: %s", exc)
            if isinstance(input_payload, Command):
                raise
        async for chunk in agent.astream(input_payload, config=config, stream_mode=["messages", "updates"]):
            translator.apply_astream_chunk(chunk)
            translator.flush_observed_reasoning()
            if self.should_stop(translator.thread_id):
                translator.flush_observed_reasoning()
                raise asyncio.CancelledError("Graceful stop requested.")
        translator.flush_observed_reasoning()

    @staticmethod
    def _validate_decisions(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(decisions, list) or not decisions:
            raise ValueError("Resume requires at least one decision.")
        out: list[dict[str, Any]] = []
        for item in decisions:
            if not isinstance(item, dict):
                raise ValueError("Resume decisions must be objects.")
            decision_type = str(item.get("type") or "").strip()
            if decision_type not in _APPROVAL_DECISIONS:
                raise ValueError(f"Invalid resume decision type: {decision_type}")
            cleaned = dict(item)
            if decision_type == "respond" and not str(cleaned.get("message") or "").strip():
                raise ValueError("Respond decisions require a message.")
            if decision_type == "edit":
                edited = cleaned.get("edited_action")
                if not isinstance(edited, dict):
                    raise ValueError("Edit decisions require edited_action.")
                if not str(edited.get("name") or "").strip() or not isinstance(edited.get("args"), dict):
                    raise ValueError("edited_action must include name and args.")
            out.append(cleaned)
        return out


__all__ = ["CatMasterStreamTranslator", "StreamingSpecialistRunner"]
