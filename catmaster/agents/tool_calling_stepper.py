from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from catmaster.llm.driver import ToolCallingDriver
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
        reporter: Optional[Reporter] = None,
        role: str = "tool_calling_stepper",
        max_steps: int = 20,
        driver_kwargs: Optional[Dict[str, Any]] = None,
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
        self.reporter = reporter or NullReporter()
        self.role = role
        self.max_steps = max_steps
        self.driver_kwargs = driver_kwargs or {}
        self.logger = logging.getLogger(__name__)

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
    ) -> Dict[str, Any]:
        state = ConversationState()
        observations: list[dict] = []
        if seed_messages:
            state.input_items.extend(seed_messages)
        else:
            self._seed_state(
                state,
                task_goal=task_goal,
                context_pack=context_pack,
                initial_instruction=initial_instruction,
                tool_descriptions=tool_descriptions,
            )

        for step in range(self.max_steps):
            tools_schema = list(function_tools or [])
            tools_schema.extend(builtin_tools or [])
            tools_schema.extend(self.control_tools)
            self._emit("LLM_CALL_START", category="llm", task_id=task_id, step_id=step, payload={
                "kind": "tool_calling",
            })
            driver_payload = dict(self.driver_kwargs)
            # Execute tool calls sequentially; keep parallel_tool_calls False to avoid interleaving.
            turn = self.driver.create_turn(input_items=state.input_items, tools=tools_schema, **driver_payload)
            self._emit("LLM_CALL_END", category="llm", task_id=task_id, step_id=step, payload={
                "kind": "tool_calling",
            })
            state.append_model_output_items(turn.output_items_raw)
            if self.trace_store is not None:
                self.trace_store.append_event({
                    "event": "LLM_OUTPUT_ITEMS",
                    "payload": {
                        "task_id": task_id,
                        "step_id": step,
                        "role": self.role,
                        "output_items": turn.output_items_raw,
                    },
                })

            builtin_calls = self._collect_builtin_calls(turn.output_items_raw)
            if builtin_calls:
                self._record_builtin_calls(turn.output_items_raw, task_id=task_id, step=step)
                observations.append({
                    "step": step,
                    "method": "builtin_calls",
                    "result": builtin_calls,
                })
                state.append_input_message("user", self._format_builtin_calls_summary(builtin_calls))

            if not turn.tool_calls:
                if turn.output_text:
                    observations.append({
                        "step": step,
                        "method": "model_output",
                        "result": {"text": turn.output_text},
                    })
                # If we saw builtin calls but no assistant text yet, allow another turn.
                if builtin_calls and not (turn.output_text or "").strip():
                    continue
                return {
                    "status": "done",
                    "finish_reason": "model_text",
                    "output_text": turn.output_text,
                    "local_observations": observations,
                }

            tool_calls = list(turn.tool_calls)
            pending_summaries: list[str] = []
            has_normal_calls = any(
                tool_call.name not in self.control_tool_names for tool_call in tool_calls
            )
            if not has_normal_calls:
                if len(tool_calls) == 1:
                    tool_call = tool_calls[0]
                    raw_params = self._parse_arguments(tool_call.arguments)
                    self._emit("TOOL_CALL_START", category="tool", task_id=task_id, step_id=step, payload={
                        "tool": tool_call.name,
                        "params_compact": self._compact_params(raw_params),
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
                    })
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
                    summary = self._skip_tool_call(
                        state=state,
                        observations=observations,
                        step=step,
                        tool_call=tool_call,
                        call_id=call_id,
                        reason=reason,
                    )
                    pending_summaries.append(summary)
                if pending_summaries:
                    state.append_input_message(
                        "user",
                        "Tool observations (this turn):\n" + "\n".join(pending_summaries),
                    )
                continue

            for idx, tool_call in enumerate(tool_calls):
                call_id = tool_call.call_id or f"{task_id}_s{step + 1}_{idx + 1}"
                if tool_call.name in self.control_tool_names:
                    reason = (
                        "task_finish/task_fail must be called alone after reviewing tool outputs."
                    )
                    summary = self._skip_tool_call(
                        state=state,
                        observations=observations,
                        step=step,
                        tool_call=tool_call,
                        call_id=call_id,
                        reason=reason,
                    )
                    pending_summaries.append(summary)
                    continue

                raw_params = self._parse_arguments(tool_call.arguments)
                self._emit("TOOL_CALL_START", category="tool", task_id=task_id, step_id=step, payload={
                    "tool": tool_call.name,
                    "params_compact": self._compact_params(raw_params),
                })
                if self.role == "task_runner" and self._is_system_view_request(tool_call.name, raw_params):
                    tool_output = {
                        "status": "failed",
                        "tool_name": tool_call.name,
                        "data": {},
                        "error": (
                            "System view is not available to task_runner. "
                            "Use view='user' and the provided artifact list."
                        ),
                    }
                    reason = tool_output.get("error", "")
                    self._emit("TOOL_VALIDATE_FAILED", level="warning", category="tool", task_id=task_id, step_id=step, payload={
                        "tool": tool_call.name,
                        "reason": self._snippet(reason, 200),
                    })
                else:
                    toolcall_id = self._toolcall_id(task_id, step, tool_call.name, call_id)
                    tool_output = self.backend.call(
                        tool_call.name,
                        tool_call.arguments,
                        toolcall_key=toolcall_id,
                        call_id=call_id,
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
                if self.role == "task_runner" and self._is_system_view_request(tool_call.name, raw_params):
                    event_status = "validation_failed"
                self._emit("TOOL_CALL_END", category="tool", task_id=task_id, step_id=step, payload={
                    "tool": tool_call.name,
                    "status": event_status,
                })

                observations.append({"step": step, "method": tool_call.name, "params": raw_params, "result": tool_output})
                state.append_function_call_output(call_id, tool_output)
                pending_summaries.append(
                    self._format_tool_observation_summary(tool_call.name, call_id, tool_output),
                )

                status = str(tool_output.get("status", "")).lower()
                if status != "success":
                    reason = "Skipped due to earlier tool failure; please replan."
                    for offset, remaining in enumerate(tool_calls[idx + 1:], start=idx + 2):
                        remaining_id = remaining.call_id or f"{task_id}_s{step + 1}_{offset}"
                        summary = self._skip_tool_call(
                            state=state,
                            observations=observations,
                            step=step,
                            tool_call=remaining,
                            call_id=remaining_id,
                            reason=reason,
                        )
                        pending_summaries.append(summary)
                    break
            if pending_summaries:
                state.append_input_message(
                    "user",
                    "Tool observations (this turn):\n" + "\n".join(pending_summaries),
                )
            continue

        return {
            "status": "max_steps",
            "finish_reason": "max_steps",
            "local_observations": observations,
        }

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
    def _is_system_view_request(tool_name: str, params: Any) -> bool:
        if not tool_name.startswith("workspace_"):
            return False
        if not isinstance(params, dict):
            return False
        return params.get("view") == "system"

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
        if initial_instruction:
            parts.append(f"Execution guidance: {initial_instruction}")
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

        messages = self.prompt.format_messages(
            goal=task_goal,
            constraints=context_pack.get("constraints", ""),
            workspace_policy=context_pack.get("workspace_policy", ""),
            whiteboard_excerpt=context_pack.get("whiteboard_excerpt", ""),
            artifact_slice=self._format_artifact_slice(context_pack.get("artifact_slice", [])),
            execution_guidance=initial_instruction or "",
        )
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

    @staticmethod
    def _format_builtin_calls_summary(
        builtin_calls: list[dict],
        *,
        limit: int = 5,
        max_chars: int = 1500,
    ) -> str:
        lines = ["Builtin tool calls observed:"]
        for call in builtin_calls[:limit]:
            parts = [call.get("type", "builtin_call")]
            if call.get("call_id"):
                parts.append(f"id={call.get('call_id')}")
            if call.get("status"):
                parts.append(f"status={call.get('status')}")
            if call.get("query"):
                parts.append(f"query={call.get('query')}")
            if call.get("url"):
                parts.append(f"url={call.get('url')}")
            lines.append("- " + " ".join(parts))
        text = "\n".join(lines)
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text

    @staticmethod
    def _format_tool_observation_summary(
        tool_name: str,
        call_id: str,
        tool_output: dict,
        *,
        max_chars: int = 2000,
    ) -> str:
        status = tool_output.get("status", "")
        error = tool_output.get("error")
        data = tool_output.get("data", {})
        hints: list[str] = []
        if isinstance(data, dict):
            for key in ("path", "paths", "root", "destination", "output_path", "dir", "directory"):
                if key in data:
                    hints.append(f"{key}={data.get(key)}")
            if "entries" in data and isinstance(data.get("entries"), list):
                hints.append(f"entries={len(data.get('entries') or [])}")
            if "next_token" in data:
                hints.append(f"next_token={data.get('next_token')}")
            if "attempt_count" in data:
                hints.append(f"attempt={data.get('attempt_count')}/{data.get('max_attempts')}")
            if "next_step" in data and data.get("next_step"):
                hints.append(f"hint={data.get('next_step')}")
        parts = [f"Tool observation: {tool_name}", f"id={call_id}", f"status={status}"]
        if error:
            parts.append(f"error={error}")
        if hints:
            parts.append(" ".join(hints))
        text = " | ".join(parts)
        if len(text) > max_chars:
            return text[: max_chars - 3] + "..."
        return text

    def _skip_tool_call(
        self,
        *,
        state: ConversationState,
        observations: list[dict],
        step: int,
        tool_call: Any,
        call_id: str,
        reason: str,
    ) -> str:
        tool_output = {
            "status": "failed",
            "tool_name": getattr(tool_call, "name", ""),
            "data": {},
            "error": reason,
        }
        observations.append({"step": step, "method": tool_output["tool_name"], "result": tool_output})
        state.append_function_call_output(call_id, tool_output)
        return self._format_tool_observation_summary(tool_output["tool_name"], call_id, tool_output)

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
