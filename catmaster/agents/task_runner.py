from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI

from catmaster.runtime import ArtifactStore, TraceStore
from catmaster.tools.registry import ToolRegistry


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

    def run(
        self,
        *,
        task_id: str,
        task_goal: str,
        tools_schema: str,
        context_pack: Dict[str, Any],
        initial_instruction: Optional[str] = None,
    ) -> Dict[str, Any]:
        max_parse_failures = 2
        max_decision_failures = 2
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
            messages = self.prompt.format_messages(
                goal=task_goal,
                observations=json.dumps(memories.get("observations", []), ensure_ascii=False),
                last_result=json.dumps(last_result, ensure_ascii=False),
                tools=tools_schema,
                instruction=memories.get("next_step", ""),
                whiteboard_excerpt=context_pack.get("whiteboard_excerpt", ""),
                artifact_slice=json.dumps(context_pack.get("artifact_slice", []), ensure_ascii=False),
                constraints=context_pack.get("constraints", ""),
                workspace_policy=context_pack.get("workspace_policy", ""),
            )
            resp = self.llm.invoke(messages)
            raw = resp.content
            if self.log_fn:
                self.log_fn("task_step_prompt", task_id=task_id, step=step, messages=self._messages_to_dict(messages))
                self.log_fn("task_step_response", task_id=task_id, step=step, content=raw)
            if self.log_llm_console:
                self.logger.debug("[TASK][LLM RAW][%s][step=%s] %s", task_id, step, raw)

            try:
                decision = _parse_json_block(raw)
            except Exception as exc:
                self.logger.warning("[TASK][JSON PARSE FAILED][%s][step=%s] %s", task_id, step, exc)
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
            if action == "call":
                method = decision.get("method")
                params = decision.get("params", {})
                if not method or not self.registry.get_tool_info(method):
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
                    continue
                self.logger.info("[TASK][TOOL CALL][%s][step=%s] %s %s", task_id, step, method, params)
                pending = memories.get("pending_toolcall") or {}
                if pending.get("method") == method:
                    toolcall_key = pending.get("toolcall_key")
                else:
                    toolcall_key = f"{method}:{memories.get('toolcall_seq', 0)}"
                    memories["pending_toolcall"] = {"method": method, "toolcall_key": toolcall_key}
                validation = self.tool_executor.validate(method, params, toolcall_key=toolcall_key)
                if not validation.get("ok"):
                    tool_output = validation.get("tool_output", {})
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
                self.logger.info(
                    "[TASK][TOOL RESULT][%s][step=%s] %s status=%s data=%s",
                    task_id,
                    step,
                    method,
                    result.get("status"),
                    result.get("data"),
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
                self.logger.info("[TASK][DONE][%s][step=%s] task_finish", task_id, step)
                return {
                    "status": "done",
                    "finish_reason": "task_finish",
                    "local_observations": memories["observations"],
                }

            if action == "task_fail":
                self.logger.info("[TASK][DONE][%s][step=%s] task_fail", task_id, step)
                return {
                    "status": "done",
                    "finish_reason": "task_fail",
                    "local_observations": memories["observations"],
                }

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

    def run(
        self,
        *,
        task_id: str,
        task_goal: str,
        finish_reason: str,
        local_observations: List[Dict[str, Any]],
        whiteboard_text: str,
        whiteboard_path: str,
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
            whiteboard_path=whiteboard_path,
            error=error or "",
        )
        resp = self.llm.invoke(messages)
        raw = resp.content
        if self.log_fn:
            event = "task_summary_prompt_repair" if use_repair_prompt else "task_summary_prompt"
            self.log_fn(event, task_id=task_id, messages=self._messages_to_dict(messages))
            self.log_fn("task_summary_response", task_id=task_id, content=raw)
        if self.log_llm_console:
            self.logger.debug("[TASK][SUMMARY RAW][%s] %s", task_id, raw)
        try:
            parsed = _parse_json_block(raw)
        except Exception as exc:
            raise ValueError(f"TaskSummarizer JSON parse failed: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("TaskSummarizer returned non-object JSON")

        outcome = (parsed.get("task_outcome") or "").strip().lower()
        if outcome not in {"success", "failure", "needs_intervention"}:
            raise ValueError(f"Invalid task_outcome: {outcome}")
        summary = parsed.get("task_summary")
        if not isinstance(summary, str):
            raise ValueError("task_summary must be a string")
        ops = parsed.get("whiteboard_ops")
        if not isinstance(ops, list):
            raise ValueError("whiteboard_ops must be a list")
        normalized_ops: List[Dict[str, Any]] = []
        for item in ops:
            if not isinstance(item, dict):
                raise ValueError("whiteboard_ops items must be objects")
            op_type = str(item.get("op", "")).strip().upper()
            if not op_type:
                raise ValueError("whiteboard_ops.op is required")
            record_type = item.get("record_type")
            normalized_ops.append({
                **item,
                "op": op_type,
                "record_type": str(record_type).strip().upper() if record_type else record_type,
            })

        key_artifacts = parsed.get("key_artifacts") if isinstance(parsed.get("key_artifacts"), list) else []
        normalized: List[Dict[str, Any]] = []
        for item in key_artifacts:
            if not isinstance(item, dict):
                continue
            path = (item.get("path") or "").strip()
            if not path:
                continue
            normalized.append({
                "path": path,
                "description": (item.get("description") or "").strip(),
                "kind": (item.get("kind") or "").strip(),
            })

        return {
            "task_outcome": outcome,
            "task_summary": summary,
            "key_artifacts": normalized,
            "whiteboard_ops": normalized_ops,
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

def _parse_json_block(raw: str) -> Dict[str, Any]:
    match = re.search(r"```json\s*(.*?)\s*```", raw, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("Expected JSON wrapped in ```json ... ```")
    json_text = match.group(1).strip()
    return json.loads(json_text)


__all__ = ["TaskStepper", "TaskSummarizer"]
