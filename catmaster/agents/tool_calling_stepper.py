from __future__ import annotations

import json
import logging
import threading
from typing import Any, Callable, Dict, Optional

from catmaster.llm.driver import ToolCallingDriver
from catmaster.llm.types import LLMTokenUsage
from catmaster.runtime.checkpoint_store import CheckpointStore
from catmaster.runtime.conversation_state import ConversationState, message_item
from catmaster.runtime.tool_backend import ToolBackend
from catmaster.runtime.trace_store import TraceStore
from catmaster.ui import Reporter, NullReporter, make_event
from catmaster.agents.control_tools import CONTROL_TOOL_NAMES, get_control_tool_schemas


class ToolCallingTaskStepper:
    def __init__(
        self,
        *,
        driver: ToolCallingDriver,
        backend: ToolBackend,
        prompt: Optional[Any] = None,
        control_tools: Optional[list[dict]] = None,
        control_tool_names: Optional[set[str]] = None,
        trace_store: Optional[TraceStore] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
        reporter: Optional[Reporter] = None,
        role: str = "tool_calling_stepper",
        max_steps: int = 20,
        driver_kwargs: Optional[Dict[str, Any]] = None,
        run_id: str = "",
        interrupt_checker: Optional[Callable[[], bool]] = None,
        interrupt_ack: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.driver = driver
        self.backend = backend
        self.prompt = prompt
        self.control_tools = get_control_tool_schemas() if control_tools is None else control_tools
        if control_tool_names is None:
            if control_tools is None:
                self.control_tool_names = CONTROL_TOOL_NAMES
            else:
                self.control_tool_names = {
                    tool.get("name") for tool in self.control_tools if tool.get("name")
                }
        else:
            self.control_tool_names = control_tool_names
        self.trace_store = trace_store
        self.checkpoint_store = checkpoint_store
        self.reporter = reporter or NullReporter()
        self.role = role
        self.max_steps = max_steps
        self.driver_kwargs = driver_kwargs or {}
        self.logger = logging.getLogger(__name__)
        self.run_id = run_id or ""
        self.interrupt_checker = interrupt_checker
        self.interrupt_ack = interrupt_ack

    def _emit(
        self,
        name: str,
        *,
        level: str = "info",
        category: Optional[str] = None,
        task_id: Optional[str] = None,
        step_id: Optional[int] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.reporter.emit(make_event(
            name,
            level=level,
            category=category,
            run_id=self.run_id or None,
            task_id=task_id,
            step_id=step_id,
            payload=payload or {},
        ))

    def run(
        self,
        *,
        task_id: str,
        task_goal: str,
        context_pack: Dict[str, Any],
        initial_instruction: Optional[str] = None,
        function_tools: Optional[list[dict]] = None,
        builtin_tools: Optional[list[dict]] = None,
        tool_descriptions: Optional[str] = None,
        seed_messages: Optional[list[dict]] = None,
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = ConversationState()
        observations: list[dict] = []
        start_step = 0
        if isinstance(resume_state, dict):
            restored_items = resume_state.get("input_items")
            if isinstance(restored_items, list):
                state.input_items.extend(restored_items)
            restored_obs = resume_state.get("observations")
            if isinstance(restored_obs, list):
                observations = list(restored_obs)
            start_step = int(resume_state.get("next_step", 0) or 0)
        elif seed_messages:
            state.input_items.extend(seed_messages)
        else:
            self._seed_state(
                state,
                task_goal=task_goal,
                context_pack=context_pack,
                initial_instruction=initial_instruction,
                tool_descriptions=tool_descriptions,
            )

        for step in range(start_step, self.max_steps):
            if self._interrupt_requested():
                self._ack_interrupt("task_step", {"task_id": task_id, "step_id": step})
                snapshot = self._resume_snapshot(state, observations, next_step=step)
                self._checkpoint(
                    "INTERRUPTED_AT_STEP",
                    task_id=task_id,
                    step_id=step,
                    payload={"resume_snapshot": snapshot},
                )
                return {
                    "status": "interrupted",
                    "finish_reason": "interrupted",
                    "interrupt_phase": "task_step",
                    "local_observations": observations,
                    "resume_state": snapshot,
                }
            tools_schema = list(function_tools or [])
            tools_schema.extend(builtin_tools or [])
            tools_schema.extend(self.control_tools)
            self._checkpoint(
                "STEP_START",
                task_id=task_id,
                step_id=step,
                payload={"n_input_items": len(state.input_items)},
            )
            self._emit("LLM_CALL_START", category="llm", task_id=task_id, step_id=step, payload={
                "kind": "tool_calling",
            })
            driver_payload = dict(self.driver_kwargs)
            # Execute tool calls sequentially; keep parallel_tool_calls False to avoid interleaving.
            turn = self.driver.create_turn(input_items=state.input_items, tools=tools_schema, **driver_payload)
            usage_payload = self._usage_payload(turn.usage)
            self._emit("LLM_CALL_END", category="llm", task_id=task_id, step_id=step, payload={
                "kind": "tool_calling",
                "usage": usage_payload,
                "input_tokens": usage_payload.get("input_tokens"),
                "input_cached_tokens": usage_payload.get("input_cached_tokens"),
                "output_tokens": usage_payload.get("output_tokens"),
                "total_tokens": usage_payload.get("total_tokens"),
            })
            state.append_model_output_items(turn.output_items_raw)
            if self.trace_store is not None:
                self.trace_store.append_event({
                    "event": "LLM_USAGE",
                    "payload": {
                        "task_id": task_id,
                        "step_id": step,
                        "role": self.role,
                        "usage": usage_payload,
                    },
                })
                self.trace_store.append_event({
                    "event": "LLM_OUTPUT_ITEMS",
                    "payload": {
                        "task_id": task_id,
                        "step_id": step,
                        "role": self.role,
                        "output_items": turn.output_items_raw,
                    },
                })
            self._checkpoint(
                "LLM_TURN_DONE",
                task_id=task_id,
                step_id=step,
                payload={
                    "tool_calls": [call.name for call in list(turn.tool_calls or [])],
                    "has_output_text": bool((turn.output_text or "").strip()),
                },
            )

            builtin_calls = self._collect_builtin_calls(turn.output_items_raw)
            if builtin_calls:
                self._record_builtin_calls(turn.output_items_raw, task_id=task_id, step=step)
                observations.append({
                    "step": step,
                    "method": "builtin_calls",
                    "result": builtin_calls,
                })

            if not turn.tool_calls:
                # If we saw builtin calls but no assistant text yet, allow another turn.
                if builtin_calls and not (turn.output_text or "").strip():
                    continue
                self._checkpoint(
                    "STEP_FINISH_MODEL_TEXT",
                    task_id=task_id,
                    step_id=step,
                    payload={"output_text_snippet": self._snippet(turn.output_text or "", 200)},
                )
                return {
                    "status": "done",
                    "finish_reason": "model_text",
                    "output_text": turn.output_text,
                    "local_observations": observations,
                }

            tool_calls = list(turn.tool_calls)
            has_normal_calls = any(
                tool_call.name not in self.control_tool_names for tool_call in tool_calls
            )
            if not has_normal_calls:
                if len(tool_calls) == 1:
                    tool_call = tool_calls[0]
                    raw_params = self._parse_arguments(tool_call.arguments)
                    call_id = tool_call.call_id or f"{task_id}_s{step + 1}_1"
                    self._emit("TOOL_CALL_START", category="tool", task_id=task_id, step_id=step, payload={
                        "tool": tool_call.name,
                        "params_compact": self._compact_params(raw_params),
                        "params_full": self._json_safe(raw_params),
                        "toolcall_id": call_id,
                    })
                    control_payload = raw_params if isinstance(raw_params, dict) else {"raw": raw_params}
                    observations.append({
                        "step": step,
                        "method": tool_call.name,
                        "result": control_payload,
                    })
                    self._emit("TOOL_CALL_END", category="tool", task_id=task_id, step_id=step, payload={
                        "tool": tool_call.name,
                        "status": "control",
                        "toolcall_id": call_id,
                    })
                    self._checkpoint(
                        "CONTROL_TOOL_FINISH",
                        task_id=task_id,
                        step_id=step,
                        payload={"tool": tool_call.name, "toolcall_id": call_id},
                    )
                    return {
                        "status": "done",
                        "finish_reason": tool_call.name,
                        "control_payload": control_payload,
                        "local_observations": observations,
                    }
                reason = (
                    "Only one control tool call is allowed per turn. "
                    "Call task_finish/task_fail alone after reviewing tool outputs."
                )
                for index, tool_call in enumerate(tool_calls):
                    call_id = tool_call.call_id or f"{task_id}_s{step + 1}_{index + 1}"
                    self._skip_tool_call(
                        state=state,
                        observations=observations,
                        step=step,
                        tool_call=tool_call,
                        call_id=call_id,
                        reason=reason,
                    )
                continue

            for idx, tool_call in enumerate(tool_calls):
                call_id = tool_call.call_id or f"{task_id}_s{step + 1}_{idx + 1}"
                if tool_call.name in self.control_tool_names:
                    reason = (
                        "task_finish/task_fail must be called alone after reviewing tool outputs."
                    )
                    self._skip_tool_call(
                        state=state,
                        observations=observations,
                        step=step,
                        tool_call=tool_call,
                        call_id=call_id,
                        reason=reason,
                    )
                    continue

                raw_params = self._parse_arguments(tool_call.arguments)
                toolcall_id = self._toolcall_id(task_id, step, tool_call.name, call_id)
                refs = self._toolcall_refs(toolcall_id)
                self._emit("TOOL_CALL_START", category="tool", task_id=task_id, step_id=step, payload={
                    "tool": tool_call.name,
                    "params_compact": self._compact_params(raw_params),
                    "params_full": self._json_safe(raw_params),
                    "toolcall_id": toolcall_id,
                })
                self._checkpoint(
                    "TOOLCALL_START",
                    task_id=task_id,
                    step_id=step,
                    payload={"tool": tool_call.name, "toolcall_id": toolcall_id},
                )
                done = threading.Event()
                call_error: Exception | None = None
                tool_output: dict = {}

                def _invoke_tool_call() -> None:
                    nonlocal tool_output, call_error
                    try:
                        tool_output = self.backend.call(
                            tool_call.name,
                            tool_call.arguments,
                            toolcall_key=toolcall_id,
                            call_id=call_id,
                        )
                    except Exception as exc:  # defensive fallback
                        call_error = exc
                        tool_output = {
                            "status": "failed",
                            "tool_name": tool_call.name,
                            "data": {},
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    finally:
                        done.set()

                call_thread = threading.Thread(target=_invoke_tool_call, daemon=True)
                call_thread.start()
                interrupted_during_call = False
                cancel_requested = False
                cancel_accepted = False
                while not done.wait(0.2):
                    if not self._interrupt_requested():
                        continue
                    interrupted_during_call = True
                    if not cancel_requested:
                        cancel_requested = True
                        self._ack_interrupt(
                            "toolcall",
                            {
                                "task_id": task_id,
                                "step_id": step,
                                "toolcall_id": toolcall_id,
                                "tool": tool_call.name,
                            },
                        )
                        try:
                            cancel_accepted = bool(self.backend.cancel_active_call(toolcall_id))
                        except Exception:
                            cancel_accepted = False
                        self._emit("INTERRUPT_ACKED", category="run", task_id=task_id, step_id=step, payload={
                            "phase": "toolcall",
                            "toolcall_id": toolcall_id,
                            "tool": tool_call.name,
                            "cancel_accepted": cancel_accepted,
                        })
                        self._checkpoint(
                            "TOOLCALL_CANCEL_REQUESTED",
                            task_id=task_id,
                            step_id=step,
                            payload={
                                "toolcall_id": toolcall_id,
                                "tool": tool_call.name,
                                "cancel_accepted": cancel_accepted,
                            },
                        )
                call_thread.join(timeout=0.01)
                if call_error is not None:
                    self.logger.debug(
                        "tool call raised after fallback output: %s",
                        call_error,
                    )
                if self._is_validation_error(tool_output):
                    reason = tool_output.get("error", "")
                    self._emit("TOOL_VALIDATE_FAILED", level="warning", category="tool", task_id=task_id, step_id=step, payload={
                        "tool": tool_call.name,
                        "reason": self._snippet(reason, 200),
                    })
                event_status = tool_output.get("status", "")
                if self._is_validation_error(tool_output):
                    event_status = "validation_failed"
                self._emit("TOOL_CALL_END", category="tool", task_id=task_id, step_id=step, payload={
                    "tool": tool_call.name,
                    "status": event_status,
                    "highlights": self._tool_highlights(tool_output),
                    "toolcall_id": toolcall_id,
                    "input_ref": refs.get("input_ref", ""),
                    "output_ref": refs.get("output_ref", ""),
                })
                self._checkpoint(
                    "TOOLCALL_END",
                    task_id=task_id,
                    step_id=step,
                    payload={
                        "tool": tool_call.name,
                        "toolcall_id": toolcall_id,
                        "status": event_status,
                        "interrupted_during_call": interrupted_during_call,
                    },
                )

                observations.append({"step": step, "method": tool_call.name, "params": raw_params, "result": tool_output})
                compact_output = self._compact_tool_output_for_llm(tool_call.name, tool_output)
                state.append_function_call_output(call_id, compact_output)

                if interrupted_during_call or self._interrupt_requested():
                    interrupted_payload = {
                        "tool": tool_call.name,
                        "toolcall_id": toolcall_id,
                        "status": tool_output.get("status", ""),
                        "highlights": self._tool_highlights(tool_output),
                        "cancel_accepted": bool(cancel_accepted),
                    }
                    self._emit(
                        "TOOL_CALL_INTERRUPTED",
                        level="warning",
                        category="tool",
                        task_id=task_id,
                        step_id=step,
                        payload=interrupted_payload,
                    )
                    snapshot = self._resume_snapshot(state, observations, next_step=step + 1)
                    self._checkpoint(
                        "INTERRUPTED_DURING_TOOLCALL",
                        task_id=task_id,
                        step_id=step,
                        payload={
                            "toolcall_id": toolcall_id,
                            "tool": tool_call.name,
                            "cancel_accepted": bool(cancel_accepted),
                            "resume_snapshot": snapshot,
                        },
                    )
                    return {
                        "status": "interrupted",
                        "finish_reason": "interrupted",
                        "interrupt_phase": "toolcall",
                        "interrupted_toolcall": interrupted_payload,
                        "local_observations": observations,
                        "resume_state": snapshot,
                    }

                status = str(tool_output.get("status", "")).lower()
                if status != "success":
                    reason = "Skipped due to earlier tool failure; please replan."
                    for offset, remaining in enumerate(tool_calls[idx + 1:], start=idx + 2):
                        remaining_id = remaining.call_id or f"{task_id}_s{step + 1}_{offset}"
                        self._skip_tool_call(
                            state=state,
                            observations=observations,
                            step=step,
                            tool_call=remaining,
                            call_id=remaining_id,
                            reason=reason,
                        )
                    break
            continue

        self._checkpoint(
            "STEP_MAX_STEPS",
            task_id=task_id,
            step_id=self.max_steps,
            payload={"observations_count": len(observations)},
        )
        return {
            "status": "max_steps",
            "finish_reason": "max_steps",
            "local_observations": observations,
        }

    def _interrupt_requested(self) -> bool:
        if self.interrupt_checker is None:
            return False
        try:
            return bool(self.interrupt_checker())
        except Exception:
            return False

    def _ack_interrupt(self, phase: str, details: Dict[str, Any]) -> None:
        if self.interrupt_ack is None:
            return
        try:
            self.interrupt_ack(phase, details)
        except Exception:
            return

    def _checkpoint(
        self,
        event: str,
        *,
        task_id: str,
        step_id: int,
        payload: Dict[str, Any],
    ) -> None:
        if self.checkpoint_store is None:
            return
        body = {
            "run_id": self.run_id or "",
            "task_id": task_id,
            "step_id": step_id,
            "role": self.role,
            "payload": payload or {},
        }
        self.checkpoint_store.append(event, body)
        self.checkpoint_store.write_latest({
            "event": event,
            "task_id": task_id,
            "step_id": step_id,
            "role": self.role,
            "payload": payload or {},
        })

    @staticmethod
    def _resume_snapshot(
        state: ConversationState,
        observations: list[dict],
        *,
        next_step: int,
    ) -> Dict[str, Any]:
        return {
            "input_items": list(state.input_items),
            "observations": list(observations),
            "next_step": max(0, int(next_step)),
        }

    @staticmethod
    def _usage_payload(usage: Any) -> dict[str, Any]:
        normalized = ToolCallingTaskStepper._normalize_usage(usage)
        return normalized.to_dict(include_raw=False)

    @staticmethod
    def _normalize_usage(usage: Any) -> LLMTokenUsage:
        if isinstance(usage, LLMTokenUsage):
            return usage
        if isinstance(usage, dict):
            input_tokens = ToolCallingTaskStepper._to_int(usage.get("input_tokens"))
            output_tokens = ToolCallingTaskStepper._to_int(usage.get("output_tokens"))
            total_tokens = ToolCallingTaskStepper._to_int(usage.get("total_tokens"))
            if total_tokens is None and input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens
            return LLMTokenUsage(
                input_tokens=input_tokens,
                input_cached_tokens=ToolCallingTaskStepper._to_int(usage.get("input_cached_tokens")),
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                source=str(usage.get("source") or "provider"),
                raw=usage.get("raw") if isinstance(usage.get("raw"), dict) else None,
            )
        return LLMTokenUsage(source="missing")

    @staticmethod
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

    @staticmethod
    def _parse_arguments(arguments: Any) -> Any:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            return arguments
        if not arguments.strip():
            return {}
        try:
            return json.loads(arguments)
        except Exception:
            return arguments

    @staticmethod
    def _snippet(text: Any, limit: int = 140) -> str:
        if text is None:
            return ""
        cleaned = " ".join(str(text).split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(0, limit - 3)] + "..."

    @staticmethod
    def _compact_params(params: Any, max_items: int = 4, max_len: int = 140) -> str:
        if not isinstance(params, dict):
            return ToolCallingTaskStepper._snippet(params, max_len)
        parts = []
        for key in list(params.keys())[:max_items]:
            val = params.get(key)
            if isinstance(val, (str, int, float, bool)):
                sval = str(val)
            elif isinstance(val, list):
                sval = f"list[{len(val)}]"
            elif isinstance(val, dict):
                sval = f"dict[{len(val)}]"
            else:
                sval = type(val).__name__
            parts.append(f"{key}={sval}")
        return ToolCallingTaskStepper._snippet(", ".join(parts), max_len)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            json.dumps(value, ensure_ascii=False)
            return value
        except Exception:
            return str(value)

    @staticmethod
    def _tool_highlights(result: Dict[str, Any], max_len: int = 160) -> str:
        if not isinstance(result, dict):
            return ToolCallingTaskStepper._snippet(result, max_len)
        if result.get("error"):
            return ToolCallingTaskStepper._snippet(result.get("error", ""), max_len)
        data = result.get("data")
        if isinstance(data, dict):
            keys = list(data.keys())
            if keys:
                return ToolCallingTaskStepper._snippet("keys: " + ", ".join(keys[:6]), max_len)
            return "data: {}"
        if isinstance(data, list):
            return f"list[{len(data)}]"
        if isinstance(data, str):
            return ToolCallingTaskStepper._snippet(data, max_len)
        return ToolCallingTaskStepper._snippet(str(data), max_len)

    @staticmethod
    def _format_artifact_slice(
        artifact_slice: Any,
        *,
        limit: int = 50,
        max_chars: int = 3000,
    ) -> str:
        if not artifact_slice:
            return "(none)"
        lines: list[str] = []
        if isinstance(artifact_slice, list):
            for entry in artifact_slice[:limit]:
                if isinstance(entry, dict):
                    path = str(entry.get("path", "") or "")
                    kind = str(entry.get("kind", "") or "")
                    desc = str(entry.get("description", "") or "")
                    parts = [path]
                    if kind:
                        parts.append(f"[{kind}]")
                    if desc:
                        parts.append(f"- {desc}")
                    line = " ".join(p for p in parts if p).strip()
                else:
                    line = str(entry).strip()
                if line:
                    lines.append(f"- {line}")
        else:
            lines.append(str(artifact_slice).strip())
        text = "\n".join(lines)
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text

    @staticmethod
    def _is_validation_error(tool_output: dict) -> bool:
        data = tool_output.get("data")
        if isinstance(data, dict) and data.get("error_type") == "validation_error":
            return True
        return False

    @staticmethod
    def _build_initial_text(
        task_goal: str,
        context_pack: Dict[str, Any],
        initial_instruction: Optional[str],
    ) -> str:
        parts = [f"Task goal: {task_goal}"]
        if context_pack:
            parts.append("Context pack:\n" + json.dumps(context_pack, ensure_ascii=False, indent=2))
        return "\n\n".join(parts)

    def _seed_state(
        self,
        state: ConversationState,
        *,
        task_goal: str,
        context_pack: Dict[str, Any],
        initial_instruction: Optional[str],
        tool_descriptions: Optional[str],
    ) -> None:
        if self.prompt is None:
            initial_text = self._build_initial_text(task_goal, context_pack, initial_instruction)
            state.append_input_message("user", initial_text)
            return

        format_kwargs: Dict[str, Any] = {
            "goal": context_pack.get("goal", task_goal),
            "task_detail": context_pack.get("task_detail", ""),
            "expected_outputs": context_pack.get("expected_outputs", ""),
            "suggested_tools": context_pack.get("suggested_tools", ""),
            "reference_hint": context_pack.get("reference_hint", ""),
            "workspace_policy": context_pack.get("workspace_policy", ""),
            "memory_index_excerpt": context_pack.get("memory_index_excerpt", ""),
            # Backward-compatible placeholders for legacy prompts/tests.
            "constraints": context_pack.get("constraints", ""),
            "artifact_slice": self._format_artifact_slice(context_pack.get("artifact_slice", [])),
        }
        input_vars = getattr(self.prompt, "input_variables", None)
        if isinstance(input_vars, list) and input_vars:
            format_kwargs = {
                key: value
                for key, value in format_kwargs.items()
                if key in input_vars
            }
        messages = self.prompt.format_messages(**format_kwargs)
        for msg in messages:
            role = getattr(msg, "role", None) or getattr(msg, "type", "user")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            content = getattr(msg, "content", str(msg))
            state.input_items.append(message_item(role, content))

    @staticmethod
    def _collect_builtin_calls(output_items: list[dict]) -> list[dict]:
        """Extract a compact summary of built-in (non-function) tool calls from output items."""
        calls: list[dict] = []
        for item in output_items:
            item_type = (item.get("type") or "")
            if not item_type.endswith("_call") or item_type == "function_call":
                continue
            call_id = item.get("call_id") or item.get("id") or ""
            status = item.get("status") or ""
            summary: dict[str, Any] = {"type": item_type, "call_id": call_id, "status": status}
            action = item.get("action")
            if isinstance(action, dict):
                if "query" in action:
                    summary["query"] = action.get("query")
                if "url" in action:
                    summary["url"] = action.get("url")
            calls.append(summary)
        return calls

    def _skip_tool_call(
        self,
        *,
        state: ConversationState,
        observations: list[dict],
        step: int,
        tool_call: Any,
        call_id: str,
        reason: str,
    ) -> None:
        tool_output = {
            "status": "failed",
            "tool_name": getattr(tool_call, "name", ""),
            "data": {},
            "error": reason,
        }
        observations.append({"step": step, "method": tool_output["tool_name"], "result": tool_output})
        compact_output = self._compact_tool_output_for_llm(tool_output["tool_name"], tool_output)
        state.append_function_call_output(call_id, compact_output)

    @staticmethod
    def _compact_tool_output_for_llm(
        tool_name: str,
        tool_output: Any,
        *,
        text_limit: int = 800,
    ) -> Dict[str, Any]:
        if not isinstance(tool_output, dict):
            return {
                "status": "failed",
                "tool_name": tool_name or "",
                "data": {},
                "warnings": [],
                "error": ToolCallingTaskStepper._snippet(tool_output, text_limit),
            }

        compact: Dict[str, Any] = {
            "status": str(tool_output.get("status") or ""),
            "tool_name": str(tool_output.get("tool_name") or tool_name or ""),
            "warnings": list(tool_output.get("warnings") or [])[:6],
            "error": ToolCallingTaskStepper._snippet(tool_output.get("error"), text_limit),
        }
        data = tool_output.get("data")
        if not isinstance(data, dict):
            compact["data"] = {}
            return compact

        if tool_name == "bash_exec":
            compact_data: Dict[str, Any] = {}
            for key in (
                "exit_code",
                "timed_out",
                "cancelled",
                "cwd",
                "timeout_s",
                "stdout_path",
                "stderr_path",
                "stdout_tail",
                "stderr_tail",
                "blocked_reason",
            ):
                if key not in data:
                    continue
                value = data.get(key)
                if isinstance(value, str):
                    value = ToolCallingTaskStepper._snippet(value, text_limit)
                compact_data[key] = value
            if "stdout_tail" not in compact_data and data.get("stdout"):
                compact_data["stdout_tail"] = ToolCallingTaskStepper._snippet(data.get("stdout"), text_limit)
            if "stderr_tail" not in compact_data and data.get("stderr"):
                compact_data["stderr_tail"] = ToolCallingTaskStepper._snippet(data.get("stderr"), text_limit)
            compact["data"] = compact_data
            return compact

        data_keys = [str(k) for k in list(data.keys())[:20]]
        scalars: Dict[str, Any] = {}
        paths: Dict[str, str] = {}
        for key, value in data.items():
            if len(scalars) >= 10 and len(paths) >= 10:
                break
            key_s = str(key)
            if isinstance(value, (int, float, bool)) and len(scalars) < 10:
                scalars[key_s] = value
                continue
            if not isinstance(value, str):
                continue
            key_l = key_s.lower()
            if any(token in key_l for token in ("path", "file", "dir", "artifact", "ref")) and len(paths) < 10:
                paths[key_s] = ToolCallingTaskStepper._snippet(value, 300)
                continue
            if len(value) <= 200 and len(scalars) < 10:
                scalars[key_s] = value

        compact_data: Dict[str, Any] = {"data_keys": data_keys}
        if scalars:
            compact_data["scalars"] = scalars
        if paths:
            compact_data["paths"] = paths
        compact["data"] = compact_data
        return compact

    @staticmethod
    def _has_builtin_calls(output_items: list[dict]) -> bool:
        return bool(ToolCallingTaskStepper._collect_builtin_calls(output_items))

    def _record_builtin_calls(self, output_items: list[dict], *, task_id: str, step: int) -> None:
        if self.trace_store is None:
            return
        for item in output_items:
            item_type = item.get("type", "")
            if not item_type.endswith("_call") or item_type == "function_call":
                continue
            record = {
                "task_id": task_id,
                "step_id": step,
                "role": self.role,
                "tool_name": item_type,
                "status": "builtin",
                "call_id": item.get("call_id") or item.get("id"),
            }
            self.trace_store.append_toolcall(record)

    @staticmethod
    def _toolcall_id(task_id: str, step: int, tool_name: str, call_id: str) -> str:
        safe_tool = tool_name.replace("/", "_")
        suffix = str(call_id)[-8:] if call_id else f"s{step + 1}"
        return f"{task_id}_s{step + 1}_{safe_tool}_{suffix}"

    @staticmethod
    def _toolcall_refs(toolcall_id: str) -> Dict[str, str]:
        return {
            "input_ref": f"toolcalls/{toolcall_id}/input.json",
            "output_ref": f"toolcalls/{toolcall_id}/output.json",
        }
