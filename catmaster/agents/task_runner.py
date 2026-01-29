from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from catmaster.runtime import ArtifactStore, TraceStore
from catmaster.tools.registry import ToolRegistry
from catmaster.ui import Reporter, NullReporter, make_event
from catmaster.agents.llm_utils import llm_text


class TaskStepper:
    def __init__(
        self,
        *,
        llm: ChatOpenAI,
        registry: ToolRegistry,
        tool_executor,
        artifact_store: ArtifactStore,
        trace_store: TraceStore,
        prompt,
        role: str = "task_runner",
        log_fn=None,
        log_llm_console: bool = False,
        max_steps: int = 100,
        reporter: Optional[Reporter] = None,
    ):
        self.llm = llm
        self.registry = registry
        self.tool_executor = tool_executor
        self.artifact_store = artifact_store
        self.trace_store = trace_store
        self.prompt = prompt
        self.role = role
        self.log_fn = log_fn
        self.log_llm_console = log_llm_console
        self.max_steps = max_steps
        self.logger = logging.getLogger(__name__)
        self.reporter = reporter or NullReporter()

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

    def _ui_debug(self) -> bool:
        return bool(getattr(self.reporter, "ui_debug", False))

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
            return TaskStepper._snippet(params, max_len)
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
        return TaskStepper._snippet(", ".join(parts), max_len)

    @staticmethod
    def _tool_highlights(result: Dict[str, Any], max_len: int = 160) -> str:
        if not isinstance(result, dict):
            return TaskStepper._snippet(result, max_len)
        if result.get("error"):
            return TaskStepper._snippet(result.get("error", ""), max_len)
        data = result.get("data")
        if isinstance(data, dict):
            keys = list(data.keys())
            if keys:
                return TaskStepper._snippet("keys: " + ", ".join(keys[:6]), max_len)
            return "data: {}"
        if isinstance(data, list):
            return f"list[{len(data)}]"
        if isinstance(data, str):
            return TaskStepper._snippet(data, max_len)
        if data is None:
            return ""
        return TaskStepper._snippet(str(data), max_len)

    def run(
        self,
        *,
        task_id: str,
        task_goal: str,
        tools_schema: str,
        context_pack: Dict[str, Any],
        initial_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        max_parse_failures = 30
        max_decision_failures = 30
        memories: Dict[str, Any] = {
            "todo": [task_goal],
            "observations": [],
            "next_step": initial_instruction or f"Start task: {task_goal}",
            "toolcall_seq": 0,
            "pending_toolcall": None,
            "parse_failures": 0,
            "decision_failures": 0,
        }

        for step in range(self.max_steps):
            last_result = memories["observations"][-1] if memories["observations"] else {}
            instruction = memories.get("next_step", "")
            self._emit("TASK_STEP_START", category="task", task_id=task_id, step_id=step, payload={
                "instruction_snippet": self._snippet(instruction, 200),
            })
            messages = self.prompt.format_messages(
                goal=task_goal,
                observations=json.dumps(memories.get("observations", []), ensure_ascii=False),
                last_result=json.dumps(last_result, ensure_ascii=False),
                tools=tools_schema,
                instruction=instruction,
                whiteboard_excerpt=context_pack.get("whiteboard_excerpt", ""),
                artifact_slice=json.dumps(context_pack.get("artifact_slice", []), ensure_ascii=False),
                constraints=context_pack.get("constraints", ""),
                workspace_policy=context_pack.get("workspace_policy", ""),
            )
            self._emit("LLM_CALL_START", category="llm", task_id=task_id, step_id=step, payload={
                "kind": "task_step",
            })
            t0 = time.perf_counter()
            resp = self.llm.invoke(messages)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            raw = llm_text(resp)
            if self.log_fn:
                self.log_fn("task_step_prompt", task_id=task_id, step=step, messages=self._messages_to_dict(messages))
                self.log_fn("task_step_response", task_id=task_id, step=step, content=raw)
            if self.log_llm_console:
                self.logger.debug("[TASK][LLM RAW][%s][step=%s] %s", task_id, step, raw)
            payload = {"kind": "task_step", "elapsed_ms": elapsed_ms}
            if self._ui_debug():
                payload["raw_snippet"] = self._snippet(raw, 240)
            self._emit("LLM_CALL_END", category="llm", task_id=task_id, step_id=step, payload=payload)

            try:
                decision = _parse_json_block(raw)
            except Exception as exc:
                self.logger.debug("[TASK][JSON PARSE FAILED][%s][step=%s] %s", task_id, step, exc)
                self._emit("TASK_JSON_PARSE_FAILED", level="warning", category="task", task_id=task_id, step_id=step, payload={
                    "error": str(exc),
                })
                memories["parse_failures"] = memories.get("parse_failures", 0) + 1
                memories["observations"].append({
                    "step": step,
                    "method": "task_parse",
                    "result": {"status": "failed", "error": f"{exc}"},
                })
                if memories["parse_failures"] > max_parse_failures:
                    return {
                        "status": "done",
                        "finish_reason": "task_fail",
                        "local_observations": memories["observations"],
                    }
                memories["next_step"] = (
                    "Re-emit ONE valid JSON object conforming to the schema. "
                    "Use double quotes for keys/strings. No extra text."
                )
                continue

            if not isinstance(decision, dict):
                self._emit("TASK_DECISION_INVALID", level="warning", category="task", task_id=task_id, step_id=step, payload={
                    "reason": "Decision must be a JSON object",
                })
                memories["decision_failures"] = memories.get("decision_failures", 0) + 1
                memories["observations"].append({
                    "step": step,
                    "method": "task_decision",
                    "result": {"status": "failed", "error": "Decision must be a JSON object"},
                })
                if memories["decision_failures"] > max_decision_failures:
                    return {
                        "status": "done",
                        "finish_reason": "task_fail",
                        "local_observations": memories["observations"],
                    }
                memories["next_step"] = (
                    "Return exactly one valid JSON object with action=call/task_finish/task_fail."
                )
                continue

            action = (decision.get("action") or "").strip().lower()
            method = decision.get("method")
            params = decision.get("params", {})
            next_step = decision.get("next_step", "")
            reasoning = decision.get("reasoning", "")
            if action in {"call", "task_finish", "task_fail"}:
                self._emit("TASK_DECISION", category="task", task_id=task_id, step_id=step, payload={
                    "action": action,
                    "method": method,
                    "params_compact": self._compact_params(params),
                    "next_step_snippet": self._snippet(next_step, 160),
                    "reasoning_snippet": self._snippet(reasoning, 160),
                })
            if action == "call":
                if not method or not self.registry.get_tool_info(method):
                    self._emit("TASK_DECISION_INVALID", level="warning", category="task", task_id=task_id, step_id=step, payload={
                        "reason": f"Invalid tool name: {method}",
                    })
                    memories["decision_failures"] = memories.get("decision_failures", 0) + 1
                    memories["observations"].append({
                        "step": step,
                        "method": "task_decision",
                        "result": {"status": "failed", "error": f"Invalid tool name: {method}"},
                    })
                    if memories["decision_failures"] > max_decision_failures:
                        return {
                            "status": "done",
                            "finish_reason": "task_fail",
                            "local_observations": memories["observations"],
                        }
                    memories["next_step"] = (
                        "Choose a valid tool name from the available tools and provide valid params."
                    )
                    continue
                toolcall_id = f"{task_id}_step{step + 1}_{method}"
                refs = self.artifact_store.toolcall_refs(toolcall_id)
                if self.role == "task_runner" and isinstance(params, dict) and params.get("view") == "system":
                    tool_output = {
                        "status": "failed",
                        "tool_name": method,
                        "data": {},
                        "error": "System view is not available to task_runner. Use view='user' and the provided artifact list.",
                    }
                    self.artifact_store.write_input(toolcall_id, {
                        "raw_params": params,
                        "validated_params": None,
                        "tool_name": method,
                        "toolcall_id": toolcall_id,
                        "status": "validation_failed",
                    })
                    self.artifact_store.write_output(toolcall_id, {
                        "toolresult": tool_output,
                        "full_output": tool_output,
                    })
                    record = self._toolcall_record(
                        task_id=task_id,
                        step=step,
                        method=method,
                        validated_params=None,
                        tool_output=tool_output,
                        toolcall_id=toolcall_id,
                        refs=refs,
                    )
                    self.trace_store.append_toolcall(record)
                    memories["observations"].append({"step": step, "method": method, "result": tool_output})
                    memories["next_step"] = (
                        "System view is not available. Use view='user' and the provided artifact list."
                    )
                    reason = tool_output.get("error", "")
                    self._emit("TOOL_VALIDATE_FAILED", level="warning", category="tool", task_id=task_id, step_id=step, payload={
                        "tool": method,
                        "reason": self._snippet(reason, 200),
                    })
                    self._emit("TOOL_CALL_END", category="tool", task_id=task_id, step_id=step, payload={
                        "tool": method,
                        "status": "validation_failed",
                        "highlights": self._snippet(reason, 200),
                    })
                    continue
                self._emit("TOOL_CALL_START", category="tool", task_id=task_id, step_id=step, payload={
                    "tool": method,
                    "params_compact": self._compact_params(params),
                })
                pending = memories.get("pending_toolcall") or {}
                if pending.get("method") == method:
                    toolcall_key = pending.get("toolcall_key")
                else:
                    toolcall_key = f"{method}:{memories.get('toolcall_seq', 0)}"
                    memories["pending_toolcall"] = {"method": method, "toolcall_key": toolcall_key}
                validation = self.tool_executor.validate(method, params, toolcall_key=toolcall_key)
                if not validation.get("ok"):
                    tool_output = validation.get("tool_output", {})
                    reason = validation.get("error_digest") or tool_output.get("error", "")
                    self.artifact_store.write_input(toolcall_id, {
                        "raw_params": params,
                        "validated_params": None,
                        "tool_name": method,
                        "toolcall_id": toolcall_id,
                        "status": "validation_failed",
                    })
                    self.artifact_store.write_output(toolcall_id, {
                        "toolresult": tool_output,
                        "full_output": tool_output,
                    })
                    record = self._toolcall_record(
                        task_id=task_id,
                        step=step,
                        method=method,
                        validated_params=None,
                        tool_output=tool_output,
                        toolcall_id=toolcall_id,
                        refs=refs,
                    )
                    self.trace_store.append_toolcall(record)
                    memories["observations"].append({"step": step, "method": method, "result": tool_output})
                    self._emit("TOOL_VALIDATE_FAILED", level="warning", category="tool", task_id=task_id, step_id=step, payload={
                        "tool": method,
                        "reason": self._snippet(reason, 200),
                    })
                    self._emit("TOOL_CALL_END", category="tool", task_id=task_id, step_id=step, payload={
                        "tool": method,
                        "status": "validation_failed",
                        "highlights": self._snippet(reason, 200),
                    })
                    memories["next_step"] = validation.get(
                        "next_step",
                        f"Fix parameters for tool {method} to satisfy the schema.",
                    )
                    continue

                validated_params = validation.get("validated_params", {})
                memories["pending_toolcall"] = None
                self.artifact_store.write_input(toolcall_id, {
                    "raw_params": params,
                    "validated_params": validated_params,
                    "tool_name": method,
                    "toolcall_id": toolcall_id,
                    "status": "validated",
                    "input_ref": refs["input_ref"],
                })
                func = self.registry.get_tool_function(method)
                try:
                    result = func(validated_params)
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "tool_name": method,
                        "data": {},
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                if not isinstance(result, dict):
                    result = {
                        "status": "failed",
                        "tool_name": method,
                        "data": {},
                        "error": f"Tool {method} returned non-dict output",
                    }
                self.logger.debug(
                    "[TASK][TOOL RESULT][%s][step=%s] %s status=%s",
                    task_id,
                    step,
                    method,
                    result.get("status"),
                )
                self.artifact_store.write_output(toolcall_id, {
                    "toolresult": result,
                    "full_output": result,
                })
                record = self._toolcall_record(
                    task_id=task_id,
                    step=step,
                    method=method,
                    validated_params=validated_params,
                    tool_output=result,
                    toolcall_id=toolcall_id,
                    refs=refs,
                )
                self.trace_store.append_toolcall(record)
                self._emit("TOOL_CALL_END", category="tool", task_id=task_id, step_id=step, payload={
                    "tool": method,
                    "status": result.get("status", ""),
                    "highlights": self._tool_highlights(result),
                })
                memories["observations"].append({"step": step, "method": method, "result": result})
                if result.get("status") == "success":
                    memories["toolcall_seq"] = memories.get("toolcall_seq", 0) + 1
                    memories["pending_toolcall"] = None
                    memories["next_step"] = decision.get("next_step", memories.get("next_step", ""))
                else:
                    memories["next_step"] = (
                        f"Tool {method} failed. Inspect inputs/outputs and retry with corrected params or choose a safer diagnostic tool."
                    )
                continue

            if action == "task_finish":
                self.logger.debug("[TASK][DONE][%s][step=%s] task_finish", task_id, step)
                return {
                    "status": "done",
                    "finish_reason": "task_finish",
                    "local_observations": memories["observations"],
                }

            if action == "task_fail":
                self.logger.debug("[TASK][DONE][%s][step=%s] task_fail", task_id, step)
                return {
                    "status": "done",
                    "finish_reason": "task_fail",
                    "local_observations": memories["observations"],
                }

            self._emit("TASK_DECISION_INVALID", level="warning", category="task", task_id=task_id, step_id=step, payload={
                "reason": f"Invalid action: {action}",
            })
            memories["decision_failures"] = memories.get("decision_failures", 0) + 1
            memories["observations"].append({
                "step": step,
                "method": "task_decision",
                "result": {"status": "failed", "error": f"Invalid action: {action}"},
            })
            if memories["decision_failures"] > max_decision_failures:
                return {
                    "status": "done",
                    "finish_reason": "task_fail",
                    "local_observations": memories["observations"],
                }
            memories["next_step"] = (
                "Return exactly one valid JSON object with action=call/task_finish/task_fail."
            )
            continue

        return {
            "status": "max_steps",
            "finish_reason": "max_steps",
            "local_observations": memories["observations"],
        }

    def _toolcall_record(
        self,
        *,
        task_id: str,
        step: int,
        method: str,
        validated_params: Optional[Dict[str, Any]],
        tool_output: Dict[str, Any],
        toolcall_id: str,
        refs: Dict[str, str],
    ) -> Dict[str, Any]:
        return {
            "task_id": task_id,
            "step_id": step,
            "role": self.role,
            "tool_name": method,
            "validated_params": validated_params,
            "status": tool_output.get("status"),
            "error": tool_output.get("error"),
            "toolcall_id": toolcall_id,
            "input_ref": refs.get("input_ref"),
            "output_ref": refs.get("output_ref"),
        }

    @staticmethod
    def _messages_to_dict(messages: List[Any]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for msg in messages:
            formatted.append(
                {
                    "type": getattr(msg, "type", getattr(msg, "role", "unknown")),
                    "content": getattr(msg, "content", str(msg)),
                }
            )
        return formatted


class KeyArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: Optional[str] = None
    description: Optional[str] = None
    kind: Optional[str] = None


class WhiteboardOp(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: str
    section: str
    record_type: Optional[str] = None
    id: Optional[str] = None
    text: Optional[str] = None
    reason: Optional[str] = None
    superseded_by: Optional[str] = None
    path: Optional[str] = None
    kind: Optional[str] = None
    description: Optional[str] = None
    rationale: Optional[str] = None


class TaskSummaryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_outcome: Literal["success", "needs_intervention", "failure"]
    task_summary: str
    key_artifacts: List[KeyArtifact] = Field(default_factory=list)
    whiteboard_ops: List[WhiteboardOp] = Field(default_factory=list)


class TaskSummarizer:
    def __init__(
        self,
        *,
        llm: ChatOpenAI,
        prompt,
        repair_prompt,
        log_fn=None,
        log_llm_console: bool = False,
    ):
        self.llm = llm
        self.prompt = prompt
        self.repair_prompt = repair_prompt
        self.log_fn = log_fn
        self.log_llm_console = log_llm_console
        self.logger = logging.getLogger(__name__)
        self.last_raw: str = ""
        self._structured_llm = llm.with_structured_output(TaskSummaryOutput)

    def run(
        self,
        *,
        task_id: str,
        task_goal: str,
        finish_reason: str,
        local_observations: List[Dict[str, Any]],
        whiteboard_text: str,
        error: Optional[str] = None,
        use_repair_prompt: bool = False,
    ) -> Dict[str, Any]:
        prompt = self.repair_prompt if use_repair_prompt else self.prompt
        messages = prompt.format_messages(
            task_id=task_id,
            task_goal=task_goal,
            finish_reason=finish_reason,
            local_observations=json.dumps(local_observations, ensure_ascii=False),
            whiteboard_text=whiteboard_text,
            error=error or "",
        )
        result = self._structured_llm.invoke(messages)
        parsed = self._normalize_structured_output(result)
        self.last_raw = json.dumps(parsed.model_dump(), ensure_ascii=False)
        if self.log_fn:
            event = "task_summary_prompt_repair" if use_repair_prompt else "task_summary_prompt"
            self.log_fn(event, task_id=task_id, messages=self._messages_to_dict(messages))
            self.log_fn("task_summary_response", task_id=task_id, content=self.last_raw)
        if self.log_llm_console:
            self.logger.debug("[TASK][SUMMARY RAW][%s] %s", task_id, self.last_raw)

        outcome = (parsed.task_outcome or "").strip().lower()
        if outcome not in {"success", "failure", "needs_intervention"}:
            raise ValueError(f"Invalid task_outcome: {outcome}")
        summary = parsed.task_summary
        if not isinstance(summary, str):
            raise ValueError("task_summary must be a string")
        normalized_ops: List[Dict[str, Any]] = []
        for item in parsed.whiteboard_ops:
            op_dict = item.model_dump(exclude_none=True)
            op_type = str(op_dict.get("op", "")).strip().upper()
            if not op_type:
                raise ValueError("whiteboard_ops.op is required")
            record_type = op_dict.get("record_type")
            if record_type is not None:
                record_type = str(record_type).strip().upper()
            normalized_item = dict(op_dict)
            normalized_item["op"] = op_type
            if record_type:
                normalized_item["record_type"] = record_type
            normalized_ops.append(normalized_item)

        normalized_artifacts: List[Dict[str, Any]] = []
        for item in parsed.key_artifacts:
            art = item.model_dump(exclude_none=True)
            path = (art.get("path") or "").strip()
            if not path:
                continue
            normalized_artifacts.append({
                "path": path,
                "description": (art.get("description") or "").strip(),
                "kind": (art.get("kind") or "").strip(),
            })

        return {
            "task_outcome": outcome,
            "task_summary": summary,
            "key_artifacts": normalized_artifacts,
            "whiteboard_ops": normalized_ops,
        }

    @staticmethod
    def _normalize_structured_output(result: Any) -> TaskSummaryOutput:
        if isinstance(result, TaskSummaryOutput):
            return result
        if isinstance(result, dict):
            return TaskSummaryOutput.model_validate(result)
        return TaskSummaryOutput.model_validate(result)

    @staticmethod
    def _messages_to_dict(messages: List[Any]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for msg in messages:
            formatted.append(
                {
                    "type": getattr(msg, "type", getattr(msg, "role", "unknown")),
                    "content": getattr(msg, "content", str(msg)),
                }
            )
        return formatted

def _parse_json_block(raw: str) -> Dict[str, Any]:
    match = re.search(r"```json\s*(.*?)\s*```", raw, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("Expected JSON wrapped in ```json ... ```")
    json_text = match.group(1).strip()
    return json.loads(json_text)


__all__ = ["TaskStepper", "TaskSummarizer"]
