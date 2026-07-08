"""
LangChain callback handlers for artifact persistence, LLM usage tracing,
and UI event emission.

These handlers reproduce the persistence behaviour previously embedded in
LocalToolBackend and ToolCallingTaskStepper, decoupling it from the
orchestration layer so graph-based runners get tracing via
``config={"callbacks": [...]}``.
"""
from __future__ import annotations

import json
import logging
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import ToolMessage
from langchain_core.outputs import LLMResult

from catmaster.runtime import observation_events as obs_events
from catmaster.runtime.artifact_store import ArtifactStore
from catmaster.runtime.observability_store import ObservabilityStore
from catmaster.runtime.tool_observation_projection import project_tool_observation
from catmaster.runtime.trace_store import TraceStore
from catmaster.ui.events import UIEvent, make_event
from catmaster.ui.reporters import Reporter

logger = logging.getLogger(__name__)


def _snippet(text: Any, limit: int = 200) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)] + "..."


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


def _extract_model_name_from_callback(
    serialized: Dict[str, Any] | None,
    *,
    default_model_name: str = "",
    callback_kwargs: Dict[str, Any] | None = None,
) -> str:
    """Best-effort model name extraction from LangChain callback payloads."""
    candidates: list[Any] = []
    data = serialized if isinstance(serialized, dict) else {}
    kwargs = data.get("kwargs")
    if isinstance(kwargs, dict):
        candidates.extend(kwargs.get(key) for key in ("model_name", "model", "model_id", "deployment_name"))
    candidates.extend(data.get(key) for key in ("model_name", "model", "model_id"))
    call_kwargs = callback_kwargs if isinstance(callback_kwargs, dict) else {}
    invocation_params = call_kwargs.get("invocation_params")
    if isinstance(invocation_params, dict):
        candidates.extend(invocation_params.get(key) for key in ("model_name", "model", "model_id", "deployment_name"))
    candidates.extend(call_kwargs.get(key) for key in ("model_name", "model", "model_id", "deployment_name"))
    candidates.append(default_model_name)
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _compact_tool_params(raw_params: Any, *, max_chars: int = 220) -> tuple[str, Any]:
    safe_params = _json_safe(raw_params)
    if isinstance(safe_params, dict):
        filtered = {
            str(key): value
            for key, value in safe_params.items()
            if str(key) not in {"runtime", "callbacks", "config", "backend", "store", "checkpointer"}
        }
        safe_params = filtered
    if isinstance(safe_params, str):
        compact = _snippet(safe_params, max_chars)
        return compact, safe_params
    try:
        text = json.dumps(safe_params, ensure_ascii=False, sort_keys=True)
    except Exception:
        text = str(safe_params)
    return _snippet(text, max_chars), safe_params


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
    local_reasoning = in_reasoning or block_type in {
        "reasoning",
        "reasoning_content",
        "thinking",
        "reasoning.summary",
        "reasoning_summary",
    }

    if block_type == "summary_text":
        _append_text(out, value.get("text"))

    if local_reasoning:
        for key in ("text", "summary", "reasoning", "reasoning_text", "reasoning_content", "thinking", "content"):
            field = value.get(key)
            if isinstance(field, str):
                _append_text(out, field)
    else:
        direct_reasoning = value.get("reasoning_content")
        if isinstance(direct_reasoning, str):
            _append_text(out, direct_reasoning)

    for key in ("summary", "content", "items", "parts", "blocks", "reasoning", "reasoning_content", "reasoning_details"):
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


def _normalize_reasoning_fragment(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _compact_reasoning_fragment(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "", _normalize_reasoning_fragment(value)).lower()


def _filter_reasoning_fragments(items: Sequence[str]) -> list[str]:
    parts = _dedupe_preserve_order(items)
    if not parts:
        return []

    has_sentence_like = any((" " in part) or (len(part) >= 24) for part in parts)
    out: list[str] = []
    for idx, part in enumerate(parts):
        normalized = _normalize_reasoning_fragment(part)
        compact = _compact_reasoning_fragment(normalized)
        if not normalized or not compact:
            continue

        redundant = False
        for other_idx, other in enumerate(parts):
            if other_idx == idx:
                continue
            other_normalized = _normalize_reasoning_fragment(other)
            other_compact = _compact_reasoning_fragment(other_normalized)
            if not other_compact:
                continue
            if other_compact == compact and other_idx < idx:
                redundant = True
                break
            if len(other_compact) <= len(compact):
                continue
            if compact in other_compact:
                if has_sentence_like or len(compact) <= 12:
                    redundant = True
                    break
        if not redundant:
            out.append(normalized)
    return out


def _merge_reasoning_fragments(items: Sequence[str]) -> str:
    parts = _filter_reasoning_fragments(items)
    if not parts:
        return ""
    if any((" " in part) or (len(part) >= 24) for part in parts):
        return " ".join(parts).strip()
    merged = "".join(str(part or "") for part in parts)
    return " ".join(merged.split())


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

    return _merge_reasoning_fragments(candidates)


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
    3. Emits canonical tool observations to ObservabilityStore.
    """

    def __init__(
        self,
        artifact_store: ArtifactStore,
        trace_store: TraceStore | None = None,
        *,
        run_id: str = "",
        write_legacy_trace: bool = False,
    ) -> None:
        super().__init__()
        self.artifact_store = artifact_store
        self.trace_store = trace_store
        self.run_id = run_id
        self.write_legacy_trace = bool(write_legacy_trace)
        self.observability = ObservabilityStore(Path(artifact_store.run_dir))
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
        self._record_tool_event(
            obs_events.TOOL_RAW_INPUT,
            {
                "role": "langgraph",
                "tool_name": tool_name,
                "tool": tool_name,
                "toolcall_id": toolcall_key,
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or ""),
                "run_id": self.run_id,
                "params_full": raw_params,
            },
        )

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
        refs = self.artifact_store.toolcall_refs(toolcall_key)

        self.artifact_store.write_output(toolcall_key, {
            "raw_output": _json_safe(output),
            "projection": projection,
            "tool_status": normalized_status,
            "tool_name": str(projection.get("tool_name") or tool_name),
        })
        record = {
            "role": "langgraph",
            "tool_name": str(projection.get("tool_name") or tool_name),
            "tool": str(projection.get("tool_name") or tool_name),
            "status": normalized_status,
            "tool_status": normalized_status,
            "error": normalized_error,
            "toolcall_id": toolcall_key,
            "callback_run_id": str(run_id),
            "parent_callback_run_id": str(parent_run_id or ""),
            "run_id": self.run_id,
            **refs,
        }
        self._record_tool_event(obs_events.TOOL_RAW_OUTPUT, record)
        if self.write_legacy_trace and self.trace_store is not None:
            self.trace_store.append_toolcall(record)

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
        refs = self.artifact_store.toolcall_refs(toolcall_key)
        record = {
            "role": "langgraph",
            "tool_name": tool_name,
            "tool": tool_name,
            "status": "error",
            "tool_status": "error",
            "error": str(error),
            "toolcall_id": toolcall_key,
            "callback_run_id": str(run_id),
            "parent_callback_run_id": str(parent_run_id or ""),
            "run_id": self.run_id,
            **refs,
        }
        self._record_tool_event(obs_events.TOOL_RAW_OUTPUT, record)
        if self.write_legacy_trace and self.trace_store is not None:
            self.trace_store.append_toolcall(record)

    def _record_tool_event(self, name: str, payload: Dict[str, Any]) -> None:
        try:
            self.observability.record_event(
                source="artifact_callback",
                channel="tool",
                name=name,
                category="tool",
                ts=time.time(),
                seq=None,
                run_id=self.run_id,
                task_id=str(payload.get("task_id") or ""),
                step_id=payload.get("step_id") if isinstance(payload.get("step_id"), int) else None,
                payload=payload,
            )
        except Exception:
            return


class LLMTracingHandler(BaseCallbackHandler):
    """Record compact raw LLM responses and errors to TraceStore."""

    def __init__(
        self,
        trace_store: TraceStore,
        *,
        run_id: str = "",
        default_model_name: str = "",
    ) -> None:
        super().__init__()
        self.trace_store = trace_store
        self.run_id = run_id
        self.default_model_name = str(default_model_name or "").strip()
        self._pending: Dict[str, Dict[str, Any]] = {}

    def _register_start(self, serialized: Dict[str, Any], callback_run_id: uuid.UUID, **kwargs: Any) -> str:
        model_name = _extract_model_name_from_callback(
            serialized,
            default_model_name=self.default_model_name,
            callback_kwargs=kwargs,
        )
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
        model = self._register_start(serialized, run_id, **kwargs)
        if prompts:
            self.trace_store.append_event({
                "event": "LLM_RAW_REQUEST",
                "payload": {
                    "role": "langgraph",
                    "model": model,
                    "run_id": self.run_id,
                    "callback_run_id": str(run_id),
                    "parent_callback_run_id": str(parent_run_id or ""),
                    "prompts": _json_safe(prompts),
                },
            })

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
        model = self._register_start(serialized, run_id, **kwargs)
        self.trace_store.append_event({
            "event": "LLM_RAW_REQUEST",
            "payload": {
                "role": "langgraph",
                "model": model,
                "run_id": self.run_id,
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or ""),
                "messages": _json_safe(messages),
            },
        })

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        info = self._pending.pop(str(run_id), {})
        generations = _extract_raw_generations(response)
        self.trace_store.append_event({
            "event": "LLM_RAW_RESPONSE",
            "payload": {
                "role": "langgraph",
                "model": info.get("model", ""),
                "run_id": self.run_id,
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or ""),
                "generations": generations,
            },
        })

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
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or ""),
            },
        })


class ObservabilityCallbackHandler(BaseCallbackHandler):
    """Record semantic LangChain observations plus raw debug payloads."""

    def __init__(
        self,
        run_dir: Path,
        *,
        run_id: str = "",
        default_agent_name: str = "",
        default_model_name: str = "",
        record_tool_callbacks: bool = True,
    ) -> None:
        super().__init__()
        self.store = ObservabilityStore(Path(run_dir))
        self.run_id = run_id
        self.default_agent_name = str(default_agent_name or "").strip()
        self.default_model_name = str(default_model_name or "").strip()
        self.record_tool_callbacks = bool(record_tool_callbacks)
        self._llm_pending: Dict[str, Dict[str, Any]] = {}
        self._tool_pending: Dict[str, Dict[str, Any]] = {}

    @staticmethod
    def _context(metadata: Optional[Dict[str, Any]] = None, **kwargs: Any) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        if isinstance(metadata, dict):
            merged.update(metadata)
        raw_metadata = kwargs.get("metadata")
        if isinstance(raw_metadata, dict):
            merged.update(raw_metadata)
        return merged

    def _agent_name(self, ctx: Dict[str, Any]) -> str:
        for key in ("lc_agent_name", "agent_name", "agent", "subagent"):
            value = str(ctx.get(key) or "").strip()
            if value:
                return value
        checkpoint_ns = str(ctx.get("langgraph_checkpoint_ns") or ctx.get("checkpoint_ns") or "").strip()
        if checkpoint_ns:
            head = checkpoint_ns.split(":", 1)[0].strip()
            if head and head != "model":
                return head
        return self.default_agent_name

    def _record_raw(self, name: str, *, payload: Dict[str, Any], category: str, task_id: str = "", step_id: Optional[int] = None) -> None:
        try:
            self.store.record_raw_callback(
                name,
                payload=payload,
                category=category,
                run_id=self.run_id,
                task_id=task_id,
                step_id=step_id,
            )
        except Exception:
            return

    def _record_semantic(self, name: str, *, payload: Dict[str, Any], category: str, task_id: str = "", step_id: Optional[int] = None) -> None:
        try:
            self.store.record_event(
                source="langchain_callback",
                channel="callback",
                name=name,
                category=category,
                ts=time.time(),
                seq=None,
                run_id=self.run_id,
                task_id=task_id,
                step_id=step_id,
                thread_id=str(payload.get("thread_id") or ""),
                message_id=str(payload.get("message_id") or ""),
                part_id=str(payload.get("part_id") or ""),
                payload=payload,
            )
        except Exception:
            return

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        ctx = self._context(metadata, **kwargs)
        model = _extract_model_name_from_callback(
            serialized,
            default_model_name=self.default_model_name,
            callback_kwargs=kwargs,
        )
        agent_name = self._agent_name(ctx)
        self._llm_pending[str(run_id)] = {
            "model": model,
            "start_ts": time.time(),
            "agent_name": agent_name,
            "task_id": str(ctx.get("task_id") or ""),
            "step_id": ctx.get("step_id") if isinstance(ctx.get("step_id"), int) else None,
            "parent_callback_run_id": str(parent_run_id or ""),
            "node": str(ctx.get("langgraph_node") or ctx.get("node") or ""),
        }
        if prompts:
            self._record_raw(
                "LLM_RAW_REQUEST",
                category="llm",
                task_id=str(ctx.get("task_id") or ""),
                step_id=ctx.get("step_id") if isinstance(ctx.get("step_id"), int) else None,
                payload={
                    "model": model,
                    "agent_name": agent_name,
                    "callback_run_id": str(run_id),
                    "parent_callback_run_id": str(parent_run_id or ""),
                    "node": str(ctx.get("langgraph_node") or ctx.get("node") or ""),
                    "prompts": _json_safe(prompts),
                },
            )
        self._record_semantic(
            "LLM_CALL_START",
            category="llm",
            task_id=str(ctx.get("task_id") or ""),
            step_id=ctx.get("step_id") if isinstance(ctx.get("step_id"), int) else None,
            payload={
                "model": model,
                "agent_name": agent_name,
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or ""),
                "node": str(ctx.get("langgraph_node") or ctx.get("node") or ""),
            },
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
        _ = tags
        self.on_llm_start(serialized, [], run_id=run_id, parent_run_id=parent_run_id, metadata=metadata, **kwargs)
        info = self._llm_pending.get(str(run_id), {})
        self._record_raw(
            "LLM_RAW_REQUEST",
            category="llm",
            task_id=str(info.get("task_id") or ""),
            step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
            payload={
                "model": str(info.get("model") or ""),
                "agent_name": str(info.get("agent_name") or ""),
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or ""),
                "node": str(info.get("node") or ""),
                "messages": _json_safe(messages),
            },
        )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        info = self._llm_pending.pop(str(run_id), {})
        elapsed_ms = int((time.time() - float(info.get("start_ts") or time.time())) * 1000)
        usage = _extract_usage_from_llm_result(response)
        generations = _extract_raw_generations(response)
        reasoning_text = ""
        text_preview = ""
        tool_names: list[str] = []
        for generation in generations:
            if not text_preview:
                text_preview = _snippet(generation.get("response_text") or "", 320)
            if not reasoning_text:
                reasoning_text = str(generation.get("reasoning_text") or "").strip()
            for tool_call in list(generation.get("parsed_tool_calls") or []):
                name = str(tool_call.get("name") or "").strip()
                if name and name not in tool_names:
                    tool_names.append(name)
        self._record_raw(
            "LLM_RAW_RESPONSE",
            category="llm",
            task_id=str(info.get("task_id") or ""),
            step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
            payload={
                "model": str(info.get("model") or ""),
                "agent_name": str(info.get("agent_name") or ""),
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or info.get("parent_callback_run_id") or ""),
                "node": str(info.get("node") or ""),
                "elapsed_ms": elapsed_ms,
                "usage": usage,
                "generations": generations,
            },
        )
        self._record_semantic(
            "LLM_CALL_END",
            category="llm",
            task_id=str(info.get("task_id") or ""),
            step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
            payload={
                "model": str(info.get("model") or ""),
                "agent_name": str(info.get("agent_name") or ""),
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or info.get("parent_callback_run_id") or ""),
                "node": str(info.get("node") or ""),
                "elapsed_ms": elapsed_ms,
                "usage": usage,
                "reasoning_text": reasoning_text,
                "text_preview": text_preview,
                "tool_calls": tool_names,
            },
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        _ = kwargs
        info = self._llm_pending.pop(str(run_id), {})
        self._record_raw(
            "LLM_ERROR",
            category="llm",
            task_id=str(info.get("task_id") or ""),
            step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
            payload={
                "model": str(info.get("model") or ""),
                "agent_name": str(info.get("agent_name") or ""),
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or info.get("parent_callback_run_id") or ""),
                "node": str(info.get("node") or ""),
                "error": str(error),
            },
        )
        self._record_semantic(
            "LLM_ERROR",
            category="llm",
            task_id=str(info.get("task_id") or ""),
            step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
            payload={
                "model": str(info.get("model") or ""),
                "agent_name": str(info.get("agent_name") or ""),
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or info.get("parent_callback_run_id") or ""),
                "node": str(info.get("node") or ""),
                "error": str(error),
            },
        )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        if not self.record_tool_callbacks:
            return
        ctx = self._context(metadata, **kwargs)
        task_id = str(ctx.get("task_id") or "")
        step_id = ctx.get("step_id") if isinstance(ctx.get("step_id"), int) else None
        agent_name = self._agent_name(ctx)
        tool_name = str(serialized.get("name") or "")
        if inputs is not None:
            raw_params: Any = inputs
        else:
            try:
                raw_params = json.loads(input_str)
            except Exception:
                raw_params = input_str
        params_compact, params_full = _compact_tool_params(raw_params, max_chars=480)
        self._tool_pending[str(run_id)] = {
            "tool": tool_name,
            "start_ts": time.time(),
            "task_id": task_id,
            "step_id": step_id,
            "agent_name": agent_name,
            "parent_callback_run_id": str(parent_run_id or ""),
        }
        self._record_raw(
            "TOOL_RAW_INPUT",
            category="tool",
            task_id=task_id,
            step_id=step_id,
            payload={
                "tool_name": tool_name,
                "tool": tool_name,
                "agent_name": agent_name,
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or ""),
                "params_compact": params_compact,
                "params_full": params_full,
            },
        )
        self._record_semantic(
            "TOOL_CALL_START",
            category="tool",
            task_id=task_id,
            step_id=step_id,
            payload={
                "tool_name": tool_name,
                "tool": tool_name,
                "agent_name": agent_name,
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or ""),
                "params_compact": params_compact,
            },
        )
        if tool_name == "task":
            subagent_type = ""
            if isinstance(raw_params, dict):
                subagent_type = str(raw_params.get("subagent_type") or raw_params.get("agent") or raw_params.get("name") or "").strip()
            self._record_semantic(
                "subagent.started",
                category="subagent",
                task_id=task_id,
                step_id=step_id,
                payload={
                    "agent_name": agent_name,
                    "subagent_type": subagent_type,
                    "description": str(raw_params.get("description") or "") if isinstance(raw_params, dict) else "",
                    "callback_run_id": str(run_id),
                    "parent_callback_run_id": str(parent_run_id or ""),
                    "tool": tool_name,
                    "tool_name": tool_name,
                    "params_compact": params_compact,
                },
            )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        if not self.record_tool_callbacks:
            return
        _ = kwargs
        info = self._tool_pending.pop(str(run_id), {})
        projection = _coerce_tool_projection(output, tool_name=str(info.get("tool") or ""))
        elapsed_ms = int((time.time() - float(info.get("start_ts") or time.time())) * 1000)
        self._record_raw(
            "TOOL_RAW_OUTPUT",
            category="tool",
            task_id=str(info.get("task_id") or ""),
            step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
            payload={
                "tool_name": str(projection.get("tool_name") or info.get("tool") or ""),
                "tool": str(projection.get("tool_name") or info.get("tool") or ""),
                "agent_name": str(info.get("agent_name") or ""),
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or info.get("parent_callback_run_id") or ""),
                "status": str(projection.get("tool_status") or "success"),
                "tool_status": str(projection.get("tool_status") or "success"),
                "error": projection.get("error"),
                "elapsed_ms": elapsed_ms,
                "projection": projection,
                "raw_output": _json_safe(output),
            },
        )
        self._record_semantic(
            "TOOL_CALL_END",
            category="tool",
            task_id=str(info.get("task_id") or ""),
            step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
            payload={
                "tool_name": str(projection.get("tool_name") or info.get("tool") or ""),
                "tool": str(projection.get("tool_name") or info.get("tool") or ""),
                "agent_name": str(info.get("agent_name") or ""),
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or info.get("parent_callback_run_id") or ""),
                "status": str(projection.get("tool_status") or "success"),
                "tool_status": str(projection.get("tool_status") or "success"),
                "error": projection.get("error"),
                "elapsed_ms": elapsed_ms,
                "projection": projection,
            },
        )
        if str(info.get("tool") or "") == "task":
            self._record_semantic(
                "subagent.completed",
                category="subagent",
                task_id=str(info.get("task_id") or ""),
                step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
                payload={
                    "agent_name": str(info.get("agent_name") or ""),
                    "callback_run_id": str(run_id),
                    "parent_callback_run_id": str(parent_run_id or info.get("parent_callback_run_id") or ""),
                    "tool": "task",
                    "tool_name": "task",
                    "status": str(projection.get("tool_status") or "success"),
                    "tool_status": str(projection.get("tool_status") or "success"),
                    "elapsed_ms": elapsed_ms,
                    "projection": projection,
                },
            )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        **kwargs: Any,
    ) -> None:
        if not self.record_tool_callbacks:
            return
        _ = kwargs
        info = self._tool_pending.pop(str(run_id), {})
        self._record_raw(
            "TOOL_RAW_OUTPUT",
            category="tool",
            task_id=str(info.get("task_id") or ""),
            step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
            payload={
                "tool_name": str(info.get("tool") or ""),
                "tool": str(info.get("tool") or ""),
                "agent_name": str(info.get("agent_name") or ""),
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or info.get("parent_callback_run_id") or ""),
                "status": "error",
                "tool_status": "error",
                "error": str(error),
            },
        )
        self._record_semantic(
            "TOOL_CALL_END",
            category="tool",
            task_id=str(info.get("task_id") or ""),
            step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
            payload={
                "tool_name": str(info.get("tool") or ""),
                "tool": str(info.get("tool") or ""),
                "agent_name": str(info.get("agent_name") or ""),
                "callback_run_id": str(run_id),
                "parent_callback_run_id": str(parent_run_id or info.get("parent_callback_run_id") or ""),
                "status": "error",
                "tool_status": "error",
                "error": str(error),
            },
        )
        if str(info.get("tool") or "") == "task":
            self._record_semantic(
                "subagent.error",
                category="subagent",
                task_id=str(info.get("task_id") or ""),
                step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
                payload={
                    "agent_name": str(info.get("agent_name") or ""),
                    "callback_run_id": str(run_id),
                    "parent_callback_run_id": str(parent_run_id or info.get("parent_callback_run_id") or ""),
                    "tool": "task",
                    "tool_name": "task",
                    "status": "error",
                    "tool_status": "error",
                    "error": str(error),
                },
            )


class LangChainStepLogger(BaseCallbackHandler):
    """Compact step-level runtime logging for LangChain callback execution."""

    def __init__(
        self,
        *,
        run_id: str = "",
        logger_name: str = "catmaster.langchain",
        default_model_name: str = "",
    ) -> None:
        super().__init__()
        self.run_id = run_id
        self.logger = logging.getLogger(logger_name)
        self.default_model_name = str(default_model_name or "").strip()
        self._llm_pending: Dict[str, Dict[str, Any]] = {}
        self._tool_pending: Dict[str, Dict[str, Any]] = {}

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[Any]],
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        model_name = _extract_model_name_from_callback(
            serialized,
            default_model_name=self.default_model_name,
            callback_kwargs=kwargs,
        )
        batch_count = len(list(messages or []))
        message_count = sum(len(list(batch or [])) for batch in list(messages or []))
        self._llm_pending[str(run_id)] = {
            "model": model_name,
            "start_ts": time.time(),
        }
        self.logger.info(
            "[langchain.step] phase=llm.start run_id=%s callback_run_id=%s model=%s batches=%d messages=%d",
            self.run_id,
            run_id,
            model_name or "-",
            batch_count,
            message_count,
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
        generations = _extract_raw_generations(response)
        tool_names: list[str] = []
        text_preview = ""
        for generation in generations:
            text_preview = text_preview or _snippet(generation.get("response_text") or "", 160)
            for tool_call in list(generation.get("parsed_tool_calls") or []):
                name = str(tool_call.get("name") or "").strip()
                if name and name not in tool_names:
                    tool_names.append(name)
        self.logger.info(
            "[langchain.step] phase=llm.end run_id=%s callback_run_id=%s model=%s elapsed_ms=%d input_tokens=%s output_tokens=%s tool_calls=%s text_preview=%s",
            self.run_id,
            run_id,
            str(info.get("model") or "") or "-",
            elapsed_ms,
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            ",".join(tool_names[:8]) or "-",
            text_preview or "-",
        )

    def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        info = self._llm_pending.pop(str(run_id), {})
        self.logger.error(
            "[langchain.step] phase=llm.error run_id=%s callback_run_id=%s model=%s error=%s",
            self.run_id,
            run_id,
            str(info.get("model") or "") or "-",
            _snippet(error, 240),
        )

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        tool_name = str(serialized.get("name") or "").strip()
        self._tool_pending[str(run_id)] = {
            "tool": tool_name,
            "start_ts": time.time(),
        }
        self.logger.info(
            "[langchain.step] phase=tool.start run_id=%s callback_run_id=%s tool=%s",
            self.run_id,
            run_id,
            tool_name or "-",
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        info = self._tool_pending.pop(str(run_id), {})
        elapsed_ms = int((time.time() - float(info.get("start_ts") or time.time())) * 1000)
        projection = _coerce_tool_projection(output, tool_name=str(info.get("tool") or ""))
        summary = projection.get("content_preview") or projection.get("error") or ""
        self.logger.info(
            "[langchain.step] phase=tool.end run_id=%s callback_run_id=%s tool=%s status=%s elapsed_ms=%d summary=%s",
            self.run_id,
            run_id,
            str(projection.get("tool_name") or info.get("tool") or "") or "-",
            str(projection.get("tool_status") or "success"),
            elapsed_ms,
            _snippet(summary, 160) or "-",
        )

    def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        info = self._tool_pending.pop(str(run_id), {})
        self.logger.error(
            "[langchain.step] phase=tool.error run_id=%s callback_run_id=%s tool=%s error=%s",
            self.run_id,
            run_id,
            str(info.get("tool") or "") or "-",
            _snippet(error, 240),
        )


class UIEventHandler(BaseCallbackHandler):
    """Bridge LangChain callback events to the CatMaster Reporter/UIEvent system.

    Emits UIEvent objects for tool starts/ends and LLM calls so that the
    WebUI can display live progress without the graph nodes having to call
    ``_emit()`` manually.
    """

    def __init__(
        self,
        reporter: Reporter,
        *,
        run_id: str = "",
        default_agent_name: str = "",
        default_model_name: str = "",
    ) -> None:
        super().__init__()
        self.reporter = reporter
        self.run_id = run_id
        self.default_agent_name = str(default_agent_name or "").strip()
        self.default_model_name = str(default_model_name or "").strip()
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

    def _agent_name(self, ctx: Dict[str, Any]) -> str:
        for key in ("lc_agent_name", "agent_name", "agent", "subagent"):
            value = str(ctx.get(key) or "").strip()
            if value:
                return value
        checkpoint_ns = str(ctx.get("langgraph_checkpoint_ns") or ctx.get("checkpoint_ns") or "").strip()
        if checkpoint_ns:
            head = checkpoint_ns.split(":", 1)[0].strip()
            if head and head != "model":
                return head
        return self.default_agent_name

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        ctx = self._context(metadata, **kwargs)
        task_id = str(ctx.get("task_id") or "")
        step_id = ctx.get("step_id")
        if not isinstance(step_id, int):
            step_id = None
        raw_params: Any
        if inputs is not None:
            raw_params = inputs
        else:
            try:
                raw_params = json.loads(input_str)
            except Exception:
                raw_params = input_str
        params_compact, params_full = _compact_tool_params(raw_params)
        toolcall_id = str(run_id)
        agent_name = self._agent_name(ctx)
        self._tool_pending[str(run_id)] = {
            "tool": serialized.get("name", ""),
            "task_id": task_id,
            "step_id": step_id,
            "toolcall_id": toolcall_id,
            "params_compact": params_compact,
            "params_full": params_full,
            "agent_name": agent_name,
            "parent_callback_run_id": str(parent_run_id or ""),
        }
        self._emit(
            "TOOL_CALL_START",
            category="tool",
            task_id=task_id or None,
            step_id=step_id,
            tool=serialized.get("name", ""),
            toolcall_id=toolcall_id,
            params_compact=params_compact,
            params_full=params_full,
            agent_name=agent_name,
            callback_run_id=toolcall_id,
            parent_callback_run_id=str(parent_run_id or ""),
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
            highlights="\n".join(str(item) for item in list(parsed.get("highlights") or []) if str(item).strip()),
            params_compact=str(info.get("params_compact") or ""),
            toolcall_id=str(info.get("toolcall_id") or ""),
            agent_name=str(info.get("agent_name") or ""),
            callback_run_id=str(info.get("toolcall_id") or ""),
            parent_callback_run_id=str(info.get("parent_callback_run_id") or ""),
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
            params_compact=str(info.get("params_compact") or ""),
            toolcall_id=str(info.get("toolcall_id") or ""),
            agent_name=str(info.get("agent_name") or ""),
            callback_run_id=str(info.get("toolcall_id") or ""),
            parent_callback_run_id=str(info.get("parent_callback_run_id") or ""),
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: uuid.UUID,
        parent_run_id: Optional[uuid.UUID] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        ctx = self._context(metadata, **kwargs)
        model = _extract_model_name_from_callback(
            serialized,
            default_model_name=self.default_model_name,
            callback_kwargs=kwargs,
        )
        agent_name = self._agent_name(ctx)
        self._llm_pending[str(run_id)] = {
            "model": model,
            "start_ts": time.time(),
            "task_id": str(ctx.get("task_id") or ""),
            "step_id": ctx.get("step_id") if isinstance(ctx.get("step_id"), int) else None,
            "node": str(ctx.get("langgraph_node") or ctx.get("node") or ""),
            "agent_name": agent_name,
            "reasoning_emitted": "",
            "parent_callback_run_id": str(parent_run_id or ""),
        }
        self._emit(
            "LLM_CALL_START",
            category="llm",
            task_id=str(ctx.get("task_id") or "") or None,
            step_id=ctx.get("step_id") if isinstance(ctx.get("step_id"), int) else None,
            model=model,
            phase="react",
            node=str(ctx.get("langgraph_node") or ctx.get("node") or ""),
            agent_name=agent_name,
            llm_call_id=str(run_id),
            callback_run_id=str(run_id),
            parent_callback_run_id=str(parent_run_id or ""),
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
        self.on_llm_start(serialized, [], run_id=run_id, parent_run_id=parent_run_id, metadata=metadata, **kwargs)

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: uuid.UUID,
        **kwargs: Any,
    ) -> None:
        info = self._llm_pending.get(str(run_id), {})
        chunk = kwargs.get("chunk")
        reasoning_text = str(_extract_reasoning_text(chunk) or "").strip()
        previous_reasoning = str(info.get("reasoning_emitted") or "")
        if reasoning_text and reasoning_text.startswith(previous_reasoning):
            reasoning_delta = reasoning_text[len(previous_reasoning):]
        else:
            reasoning_delta = ""
        if reasoning_delta:
            info["reasoning_emitted"] = reasoning_text
            self._emit(
                "LLM_REASONING_DELTA",
                category="llm",
                task_id=str(info.get("task_id") or "") or None,
                step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
                model=str(info.get("model") or ""),
                phase="react",
                node=str(info.get("node") or ""),
                text=reasoning_delta,
                agent_name=str(info.get("agent_name") or ""),
                llm_call_id=str(run_id),
                callback_run_id=str(run_id),
                parent_callback_run_id=str(info.get("parent_callback_run_id") or ""),
            )
        token_text = str(token or "")
        if token_text:
            self._emit(
                "LLM_TOKEN_DELTA",
                category="llm",
                task_id=str(info.get("task_id") or "") or None,
                step_id=info.get("step_id") if isinstance(info.get("step_id"), int) else None,
                model=str(info.get("model") or ""),
                phase="react",
                node=str(info.get("node") or ""),
                text=token_text,
                agent_name=str(info.get("agent_name") or ""),
                llm_call_id=str(run_id),
                callback_run_id=str(run_id),
                parent_callback_run_id=str(info.get("parent_callback_run_id") or ""),
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
        text_preview = ""
        tool_names: list[str] = []
        generations = _extract_raw_generations(response)
        for generation in generations:
            if not text_preview:
                text_preview = _snippet(generation.get("response_text") or "", 320)
            reasoning_text = str(generation.get("reasoning_text") or "").strip()
            for tool_call in list(generation.get("parsed_tool_calls") or []):
                name = str(tool_call.get("name") or "").strip()
                if name and name not in tool_names:
                    tool_names.append(name)
            if reasoning_text and text_preview and tool_names:
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
            text_preview=text_preview,
            tool_calls=tool_names,
            agent_name=str(info.get("agent_name") or ""),
            llm_call_id=str(run_id),
            callback_run_id=str(run_id),
            parent_callback_run_id=str(info.get("parent_callback_run_id") or ""),
        )


def build_callbacks(
    *,
    artifact_store: ArtifactStore,
    trace_store: TraceStore | None = None,
    reporter: Reporter,
    run_id: str = "",
    default_model_name: str = "",
    enable_step_logs: bool = False,
) -> list[BaseCallbackHandler]:
    """Convenience factory returning the standard callback set."""
    _ = trace_store
    callbacks: list[BaseCallbackHandler] = [
        ArtifactPersistenceHandler(artifact_store, run_id=run_id),
        ObservabilityCallbackHandler(
            artifact_store.run_dir,
            run_id=run_id,
            default_model_name=default_model_name,
            record_tool_callbacks=False,
        ),
        UIEventHandler(reporter, run_id=run_id, default_model_name=default_model_name),
    ]
    if enable_step_logs:
        callbacks.append(LangChainStepLogger(run_id=run_id, default_model_name=default_model_name))
    return callbacks


__all__ = [
    "ArtifactPersistenceHandler",
    "LangChainStepLogger",
    "LLMTracingHandler",
    "ObservabilityCallbackHandler",
    "UIEventHandler",
    "build_callbacks",
]
