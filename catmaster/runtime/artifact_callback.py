"""
LangChain callback handlers for artifact persistence, LLM usage tracing,
and UI event emission.

These handlers reproduce the persistence behaviour previously embedded in
LocalToolBackend and ToolCallingTaskStepper, decoupling it from the
orchestration layer so that LangGraph nodes get tracing for free via
``config={"callbacks": [...]}``.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import ToolMessage
from langchain_core.outputs import LLMResult

from catmaster.runtime.artifact_store import ArtifactStore
from catmaster.runtime.tool_observation_projection import project_tool_observation
from catmaster.runtime.trace_store import TraceStore
from catmaster.ui.events import UIEvent, make_event
from catmaster.ui.reporters import Reporter

logger = logging.getLogger(__name__)


def _to_dict_obj(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            return {}
    if hasattr(value, "dict"):
        try:
            dumped = value.dict()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            return {}
    return {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump(mode="json")
            return _json_safe(dumped)
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                elif item.get("type") == "text" and isinstance(item.get("content"), str):
                    parts.append(str(item.get("content")))
        if parts:
            return "\n".join(parts)
    try:
        return json.dumps(_json_safe(content), ensure_ascii=False)
    except Exception:
        return str(content)


def _append_text(out: list[str], value: Any) -> None:
    if not isinstance(value, str):
        return
    text = value.strip()
    if text:
        out.append(text)


def _extract_reasoning_fragments(value: Any, *, in_reasoning: bool = False) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        if in_reasoning:
            _append_text(out, value)
        return out
    if isinstance(value, list):
        for item in value:
            out.extend(_extract_reasoning_fragments(item, in_reasoning=in_reasoning))
        return out
    if not isinstance(value, dict):
        dumped = _to_dict_obj(value)
        if dumped:
            return _extract_reasoning_fragments(dumped, in_reasoning=in_reasoning)
        return out

    block_type = str(value.get("type") or "").strip().lower()
    local_reasoning = in_reasoning or block_type in {"reasoning", "reasoning_content", "thinking"}

    if block_type == "summary_text":
        _append_text(out, value.get("text"))

    if local_reasoning:
        for key in ("text", "reasoning", "reasoning_text", "thinking", "content"):
            field = value.get(key)
            if isinstance(field, str):
                _append_text(out, field)

    for key in ("summary", "content", "items", "parts", "blocks", "reasoning", "reasoning_content"):
        nested = value.get(key)
        if isinstance(nested, (list, dict)):
            out.extend(_extract_reasoning_fragments(nested, in_reasoning=local_reasoning or key.startswith("reasoning") or key == "summary"))
    return out


def _dedupe_preserve_order(items: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in items:
        text = str(raw or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _extract_reasoning_text(message: Any) -> str:
    candidates: list[str] = []
    if message is None:
        return ""

    for value in (
        getattr(message, "content", None),
        getattr(message, "content_blocks", None),
        getattr(message, "additional_kwargs", None),
        getattr(message, "response_metadata", None),
    ):
        if value is not None:
            candidates.extend(_extract_reasoning_fragments(value))

    if not candidates and isinstance(message, dict):
        candidates.extend(_extract_reasoning_fragments(message))

    return "\n".join(_dedupe_preserve_order(candidates))


def _message_to_log_payload(message: Any) -> dict[str, Any]:
    payload = _to_dict_obj(message)
    if not payload:
        payload = {"raw_repr": str(message)}
    payload = dict(payload)
    payload["__class__"] = str(getattr(message, "__class__", type(message)).__name__)
    payload["__type__"] = str(payload.get("type") or "").strip().lower()
    return _json_safe(payload)


def _coerce_tool_projection(output: Any, *, tool_name: str = "") -> dict[str, Any]:
    return project_tool_observation(output, tool_name=tool_name or None)


def _extract_usage_from_llm_result(response: LLMResult) -> Dict[str, Any]:
    def _to_int(value: Any) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        try:
            return int(value)
        except Exception:
            return None

    def _pick_int(mapping: dict[str, Any], keys: list[str]) -> int | None:
        for key in keys:
            if key not in mapping:
                continue
            iv = _to_int(mapping.get(key))
            if iv is not None:
                return iv
        return None

    def _extract_cached_tokens(mapping: dict[str, Any]) -> int | None:
        candidates: list[int] = []

        details_keys = [
            "prompt_tokens_details",
            "input_tokens_details",
            "input_token_details",
        ]
        for details_key in details_keys:
            details = _to_dict_obj(mapping.get(details_key))
            if not details:
                continue
            for key in ("cached_tokens", "cache_read"):
                iv = _to_int(details.get(key))
                if iv is not None:
                    candidates.append(iv)
            for key, value in details.items():
                if "cache_read" in str(key):
                    iv = _to_int(value)
                    if iv is not None:
                        candidates.append(iv)

        for key in ("input_cached_tokens", "cached_tokens", "cache_read"):
            iv = _to_int(mapping.get(key))
            if iv is not None:
                candidates.append(iv)

        if not candidates:
            return None
        return max(candidates)

    def _build_usage(mapping: dict[str, Any]) -> Dict[str, Any]:
        if not mapping:
            return {}
        input_tokens = _pick_int(mapping, ["prompt_tokens", "input_tokens"])
        output_tokens = _pick_int(mapping, ["completion_tokens", "output_tokens"])
        total_tokens = _pick_int(mapping, ["total_tokens"])
        input_cached_tokens = _extract_cached_tokens(mapping)
        input_token_details = (
            _to_dict_obj(mapping.get("input_token_details"))
            or _to_dict_obj(mapping.get("input_tokens_details"))
            or _to_dict_obj(mapping.get("prompt_tokens_details"))
        )
        output_token_details = (
            _to_dict_obj(mapping.get("output_token_details"))
            or _to_dict_obj(mapping.get("output_tokens_details"))
            or _to_dict_obj(mapping.get("completion_tokens_details"))
        )
        usage: Dict[str, Any] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        if input_cached_tokens is not None:
            usage["input_cached_tokens"] = input_cached_tokens
        if input_token_details:
            usage["input_token_details"] = _json_safe(input_token_details)
            usage["prompt_tokens_details"] = _json_safe(input_token_details)
        if output_token_details:
            usage["output_token_details"] = _json_safe(output_token_details)
            usage["completion_tokens_details"] = _json_safe(output_token_details)
        if mapping.get("source") is not None:
            usage["source"] = str(mapping.get("source") or "")
        cost = mapping.get("cost")
        if isinstance(cost, (int, float)):
            usage["cost"] = float(cost)
        cost_details = _to_dict_obj(mapping.get("cost_details"))
        if cost_details:
            usage["cost_details"] = _json_safe(cost_details)
        return usage

    usage: Dict[str, Any] = {}
    if response.llm_output and isinstance(response.llm_output, dict):
        usage = _build_usage(_to_dict_obj(response.llm_output.get("token_usage")))

    if not usage and response.generations:
        for gen_list in response.generations:
            for gen in gen_list:
                gen_info = _to_dict_obj(getattr(gen, "generation_info", None))
                token_usage = _to_dict_obj(gen_info.get("token_usage")) or _to_dict_obj(gen_info.get("usage"))
                usage = _build_usage(token_usage)
                if usage:
                    break

                message = getattr(gen, "message", None)
                usage_meta = _to_dict_obj(getattr(message, "usage_metadata", None))
                if usage_meta:
                    usage = _build_usage({
                        "input_tokens": usage_meta.get("input_tokens"),
                        "output_tokens": usage_meta.get("output_tokens"),
                        "total_tokens": usage_meta.get("total_tokens"),
                        "input_token_details": _to_dict_obj(usage_meta.get("input_token_details")),
                        "output_token_details": _to_dict_obj(usage_meta.get("output_token_details")),
                    })
                if usage:
                    break
            if usage:
                break

    return usage


def _extract_raw_generations(response: LLMResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    generations = list(response.generations or [])
    for i, gen_list in enumerate(generations):
        for j, gen in enumerate(list(gen_list or [])):
            row: dict[str, Any] = {
                "generation_index": i,
                "candidate_index": j,
            }
            message = getattr(gen, "message", None)
            if message is None:
                row["response_text"] = str(getattr(gen, "text", "") or "")
                row["raw_tool_calls"] = []
                row["parsed_tool_calls"] = []
                row["invalid_tool_calls"] = []
                rows.append(row)
                continue

            content = getattr(message, "content", None)
            row["response_text"] = _content_to_text(content)
            row["response_content_raw"] = _json_safe(content)
            reasoning_text = _extract_reasoning_text(message)
            if reasoning_text:
                row["reasoning_text"] = reasoning_text

            additional_kwargs = _to_dict_obj(getattr(message, "additional_kwargs", None))
            response_metadata = _to_dict_obj(getattr(message, "response_metadata", None))
            usage_metadata = _to_dict_obj(getattr(message, "usage_metadata", None))

            raw_tool_calls: list[dict[str, Any]] = []
            for call in list(additional_kwargs.get("tool_calls") or []):
                if not isinstance(call, dict):
                    continue
                fn = call.get("function")
                fn_dict = fn if isinstance(fn, dict) else {}
                raw_tool_calls.append({
                    "id": call.get("id"),
                    "type": call.get("type"),
                    "name": fn_dict.get("name"),
                    "arguments_raw": fn_dict.get("arguments"),
                    "raw": _json_safe(call),
                })

            parsed_tool_calls: list[dict[str, Any]] = []
            for call in list(getattr(message, "tool_calls", None) or []):
                if not isinstance(call, dict):
                    continue
                args = call.get("args")
                if isinstance(args, str):
                    args_json = args
                else:
                    try:
                        args_json = json.dumps(_json_safe(args), ensure_ascii=False)
                    except Exception:
                        args_json = str(args)
                parsed_tool_calls.append({
                    "id": call.get("id"),
                    "type": call.get("type"),
                    "name": call.get("name"),
                    "args_json": args_json,
                    "raw": _json_safe(call),
                })

            invalid_tool_calls = [
                _json_safe(call)
                for call in list(getattr(message, "invalid_tool_calls", None) or [])
            ]

            row["raw_tool_calls"] = raw_tool_calls
            row["parsed_tool_calls"] = parsed_tool_calls
            row["invalid_tool_calls"] = invalid_tool_calls
            row["additional_kwargs"] = _json_safe(additional_kwargs)
            row["response_metadata"] = _json_safe(response_metadata)
            row["usage_metadata"] = _json_safe(usage_metadata)
            rows.append(row)
    return rows


class ArtifactPersistenceHandler(BaseCallbackHandler):
    """Persist every tool invocation's inputs and outputs to disk.

    For each tool call the handler:
    1. Writes ``toolcalls/{key}/input.json`` via ArtifactStore.
    2. Writes ``toolcalls/{key}/output.json`` via ArtifactStore.
    3. Appends a summary record to ``tool_trace.jsonl`` via TraceStore.
    """

    def __init__(
        self,
        artifact_store: ArtifactStore,
        trace_store: TraceStore,
        *,
        run_id: str = "",
    ) -> None:
        super().__init__()
        self.artifact_store = artifact_store
        self.trace_store = trace_store
        self.run_id = run_id
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._call_counter = 0

    def _make_toolcall_key(self, tool_name: str, callback_run_id: Any) -> str:
        self._call_counter += 1
        suffix = str(callback_run_id)[-8:] if callback_run_id else str(self._call_counter)
        safe_name = str(tool_name).replace("/", "_")
        return f"{self.run_id}_{safe_name}_{suffix}" if self.run_id else f"{safe_name}_{suffix}"

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        tool_name = serialized.get("name", "")
        toolcall_key = self._make_toolcall_key(tool_name, run_id)
        self._pending[str(run_id)] = {
            "tool_name": tool_name,
            "toolcall_key": toolcall_key,
            "start_ts": time.time(),
        }

        raw_params: Any
        if inputs is not None:
            raw_params = _json_safe(inputs)
        else:
            try:
                raw_params = json.loads(input_str)
            except Exception:
                raw_params = input_str

        self.artifact_store.write_input(toolcall_key, {
            "tool_name": tool_name,
            "raw_params": raw_params,
            "toolcall_id": toolcall_key,
            "run_id": self.run_id,
        })

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        info = self._pending.pop(str(run_id), {})
        toolcall_key = info.get("toolcall_key", str(run_id))
        tool_name = info.get("tool_name", "")

        projection = _coerce_tool_projection(output, tool_name=tool_name)
        normalized_status = str(projection.get("tool_status") or "success")
        normalized_error = projection.get("error")

        self.artifact_store.write_output(toolcall_key, {
            "raw_output": _json_safe(output),
            "projection": projection,
            "tool_status": normalized_status,
            "tool_name": str(projection.get("tool_name") or tool_name),
        })
        self.trace_store.append_toolcall({
            "role": "langgraph",
            "tool_name": str(projection.get("tool_name") or tool_name),
            "status": normalized_status,
            "error": normalized_error,
            "toolcall_id": toolcall_key,
            "run_id": self.run_id,
        })

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        info = self._pending.pop(str(run_id), {})
        toolcall_key = info.get("toolcall_key", str(run_id))
        tool_name = info.get("tool_name", "")
        projection = _coerce_tool_projection(
            {"status": "error", "tool_name": tool_name, "error": str(error)},
            tool_name=tool_name,
        )

        self.artifact_store.write_output(toolcall_key, {
            "raw_output": {"error": str(error)},
            "projection": projection,
            "tool_status": "error",
            "tool_name": tool_name,
        })
        self.trace_store.append_toolcall({
            "role": "langgraph",
            "tool_name": tool_name,
            "status": "error",
            "error": str(error),
            "toolcall_id": toolcall_key,
            "run_id": self.run_id,
        })


class LLMTracingHandler(BaseCallbackHandler):
    """Record per-call token usage to TraceStore.

    Replaces the manual ``_usage_payload`` bookkeeping that was spread
    across ToolCallingTaskStepper and Orchestrator.
    """

    def __init__(
        self,
        trace_store: TraceStore,
        *,
        run_id: str = "",
        print_raw_tool_calls: bool = False,
        print_llm_context_messages: bool = False,
    ) -> None:
        super().__init__()
        self.trace_store = trace_store
        self.run_id = run_id
        self.print_raw_tool_calls = bool(print_raw_tool_calls)
        self.print_llm_context_messages = bool(print_llm_context_messages)
        self._pending: Dict[str, Dict[str, Any]] = {}

    def _register_start(self, serialized: Dict[str, Any], callback_run_id: uuid.UUID) -> str:
        model_name = str(serialized.get("kwargs", {}).get("model_name", "") or "")
        key = str(callback_run_id)
        current = self._pending.get(key) or {}
        if "start_ts" not in current:
            current["start_ts"] = time.time()
        if model_name:
            current["model"] = model_name
        self._pending[key] = current
        return str(current.get("model") or model_name or "")

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        self._register_start(serialized, run_id)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        model_name = self._register_start(serialized, run_id)
        if not self.print_llm_context_messages:
            return
        callback_run_id = str(run_id)
        for batch_idx, batch_messages in enumerate(list(messages or [])):
            logger.info(
                "[llm.context.start] model=%s callback_run_id=%s batch=%d message_count=%d",
                model_name,
                callback_run_id,
                batch_idx,
                len(list(batch_messages or [])),
            )
            for msg_idx, msg in enumerate(list(batch_messages or [])):
                payload = _message_to_log_payload(msg)
                msg_class = str(payload.get("__class__") or "")
                msg_type = str(payload.get("__type__") or "")
                logger.info(
                    "[llm.context.message] model=%s callback_run_id=%s batch=%d idx=%d class=%s type=%s payload=%s",
                    model_name,
                    callback_run_id,
                    batch_idx,
                    msg_idx,
                    msg_class,
                    msg_type,
                    json.dumps(payload, ensure_ascii=False, default=str),
                )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        info = self._pending.pop(str(run_id), {})
        elapsed_ms = int((time.time() - info.get("start_ts", time.time())) * 1000)

        usage = _extract_usage_from_llm_result(response)
        generations = _extract_raw_generations(response)
        reasoning_text = ""
        for generation in generations:
            reasoning_text = str(generation.get("reasoning_text") or "").strip()
            if reasoning_text:
                break

        self.trace_store.append_event({
            "event": "LLM_USAGE",
            "payload": {
                "role": "langgraph",
                "model": info.get("model", ""),
                "usage": usage,
                "elapsed_ms": elapsed_ms,
                "run_id": self.run_id,
                "reasoning_text": reasoning_text,
            },
        })
        self.trace_store.append_event({
            "event": "LLM_RAW_RESPONSE",
            "payload": {
                "role": "langgraph",
                "model": info.get("model", ""),
                "run_id": self.run_id,
                "callback_run_id": str(run_id),
                "generations": generations,
            },
        })
        if self.print_raw_tool_calls:
            model_name = str(info.get("model") or "")
            callback_run_id = str(run_id)
            for gen in generations:
                gidx = int(gen.get("generation_index") or 0)
                cidx = int(gen.get("candidate_index") or 0)
                raw_tool_calls = list(gen.get("raw_tool_calls") or [])
                parsed_tool_calls = list(gen.get("parsed_tool_calls") or [])
                for tidx, call in enumerate(raw_tool_calls):
                    logger.info(
                        "[llm.raw_tool_call] model=%s callback_run_id=%s gen=%d cand=%d call=%d name=%s arguments_raw=%s",
                        model_name,
                        callback_run_id,
                        gidx,
                        cidx,
                        tidx,
                        str(call.get("name") or ""),
                        str(call.get("arguments_raw") or ""),
                    )
                for tidx, call in enumerate(parsed_tool_calls):
                    logger.info(
                        "[llm.parsed_tool_call] model=%s callback_run_id=%s gen=%d cand=%d call=%d name=%s args_json=%s raw_available=%s",
                        model_name,
                        callback_run_id,
                        gidx,
                        cidx,
                        tidx,
                        str(call.get("name") or ""),
                        str(call.get("args_json") or ""),
                        "yes" if raw_tool_calls else "no",
                    )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        info = self._pending.pop(str(run_id), {})
        self.trace_store.append_event({
            "event": "LLM_ERROR",
            "payload": {
                "role": "langgraph",
                "model": info.get("model", ""),
                "error": str(error),
                "run_id": self.run_id,
            },
        })


class UIEventHandler(BaseCallbackHandler):
    """Bridge LangGraph execution events to the CatMaster Reporter/UIEvent system.

    Emits UIEvent objects for tool starts/ends and LLM calls so that the
    WebUI can display live progress without the graph nodes having to call
    ``_emit()`` manually.
    """

    def __init__(self, reporter: Reporter, *, run_id: str = "") -> None:
        super().__init__()
        self.reporter = reporter
        self.run_id = run_id
        self._tool_pending: Dict[str, Dict[str, Any]] = {}
        self._llm_pending: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _context(metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if isinstance(metadata, dict):
            merged.update(metadata)
        raw_metadata = kwargs.get("metadata")
        if isinstance(raw_metadata, dict):
            merged.update(raw_metadata)
        return merged

    def _emit(
        self,
        name: str,
        *,
        category: str = "tool",
        task_id: Optional[str] = None,
        step_id: Optional[int] = None,
        **extra: Any,
    ) -> None:
        self.reporter.emit(make_event(
            name,
            category=category,
            run_id=self.run_id or None,
            task_id=(task_id or None),
            step_id=step_id,
            payload=extra,
        ))

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        ctx = self._context(metadata, **kwargs)
        task_id = str(ctx.get("task_id") or "")
        step_id = ctx.get("step_id")
        if not isinstance(step_id, int):
            step_id = None
        self._tool_pending[str(run_id)] = {
            "tool": serialized.get("name", ""),
            "task_id": task_id,
            "step_id": step_id,
        }
        self._emit(
            "TOOL_CALL_START",
            category="tool",
            task_id=task_id or None,
            step_id=step_id,
            tool=serialized.get("name", ""),
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        info = self._tool_pending.pop(str(run_id), {})
        parsed = _coerce_tool_projection(output, tool_name=str(info.get("tool") or ""))
        self._emit(
            "TOOL_CALL_END",
            category="tool",
            task_id=str(info.get("task_id") or "") or None,
            step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
            tool=str(parsed.get("tool_name") or info.get("tool") or ""),
            status=str(parsed.get("tool_status") or ""),
            error=parsed.get("error"),
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        info = self._tool_pending.pop(str(run_id), {})
        self._emit(
            "TOOL_CALL_END",
            category="tool",
            task_id=str(info.get("task_id") or "") or None,
            step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
            tool=str(info.get("tool") or ""),
            status="error",
            error=str(error),
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        ctx = self._context(metadata, **kwargs)
        model = str(serialized.get("kwargs", {}).get("model_name") or "")
        self._llm_pending[str(run_id)] = {
            "model": model,
            "start_ts": time.time(),
            "task_id": str(ctx.get("task_id") or ""),
            "step_id": ctx.get("step_id") if isinstance(ctx.get("step_id"), int) else None,
            "node": str(ctx.get("langgraph_node") or ctx.get("node") or ""),
        }
        self._emit(
            "LLM_CALL_START",
            category="llm",
            task_id=str(ctx.get("task_id") or "") or None,
            step_id=ctx.get("step_id") if isinstance(ctx.get("step_id"), int) else None,
            model=model,
            phase="react",
            node=str(ctx.get("langgraph_node") or ctx.get("node") or ""),
        )

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        _ = (messages, parent_run_id, tags)
        self.on_llm_start(serialized, [], run_id=run_id, metadata=metadata, **kwargs)

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        info = self._llm_pending.get(str(run_id), {})
        chunk = kwargs.get("chunk")
        reasoning_text = _extract_reasoning_text(chunk)
        if reasoning_text:
            self._emit(
                "LLM_REASONING_DELTA",
                category="llm",
                task_id=str(info.get("task_id") or "") or None,
                step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
                model=str(info.get("model") or ""),
                phase="react",
                node=str(info.get("node") or ""),
                text=reasoning_text,
            )
        self._emit(
            "LLM_TOKEN_DELTA",
            category="llm",
            task_id=str(info.get("task_id") or "") or None,
            step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
            model=str(info.get("model") or ""),
            phase="react",
            node=str(info.get("node") or ""),
            text=str(token or ""),
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        info = self._llm_pending.pop(str(run_id), {})
        elapsed_ms = int((time.time() - float(info.get("start_ts") or time.time())) * 1000)
        usage = _extract_usage_from_llm_result(response)
        reasoning_text = ""
        generations = _extract_raw_generations(response)
        for generation in generations:
            reasoning_text = str(generation.get("reasoning_text") or "").strip()
            if reasoning_text:
                break
        self._emit(
            "LLM_CALL_END",
            category="llm",
            task_id=str(info.get("task_id") or "") or None,
            step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
            model=str(info.get("model") or ""),
            phase="react",
            node=str(info.get("node") or ""),
            usage=usage,
            elapsed_ms=elapsed_ms,
            reasoning_text=reasoning_text,
        )


def build_callbacks(
    *,
    artifact_store: ArtifactStore,
    trace_store: TraceStore,
    reporter: Reporter,
    run_id: str = "",
    print_raw_tool_calls: bool = False,
    print_llm_context_messages: bool = False,
) -> list[BaseCallbackHandler]:
    """Convenience factory returning the standard callback set."""
    return [
        ArtifactPersistenceHandler(artifact_store, trace_store, run_id=run_id),
        LLMTracingHandler(
            trace_store,
            run_id=run_id,
            print_raw_tool_calls=print_raw_tool_calls,
            print_llm_context_messages=print_llm_context_messages,
        ),
        UIEventHandler(reporter, run_id=run_id),
    ]


__all__ = [
    "ArtifactPersistenceHandler",
    "LLMTracingHandler",
    "UIEventHandler",
    "build_callbacks",
]
