#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-driven execution loop with three memories:
- ToDo (long-term plan skeleton)
- NextStep (short-term intent for the next tool call)
- Observation (long-term log of executed actions/results)

Flow: user request -> LLM drafts ToDo & initial NextStep -> execute one tool ->
record Observation -> LLM updates NextStep (and optionally trims ToDo) -> repeat
until LLM says finish.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import time
import random
import sys
from langchain_openai import ChatOpenAI

from catmaster.tools.registry import get_tool_registry
from catmaster.runtime import RunContext, EventLog, ToolExecutor, ArtifactStore
from catmaster.agents.logo import logo_str
from catmaster.agents.orchestrator_prompts import (
    build_plan_prompt,
    build_plan_repair_prompt,
    build_plan_feedback_prompt,
    build_step_prompt,
    build_summary_prompt,
)
import logging


class Orchestrator:
    def __init__(
        self,
        llm: Optional[ChatOpenAI] = None,
        summary_llm: Optional[ChatOpenAI] = None,
        max_steps: int = 6,
        *,
        llm_log_path: Optional[str] = None,
        log_llm_console: bool = True,
        run_context: Optional[RunContext] = None,
        event_log: Optional[EventLog] = None,
        run_dir: Optional[str] = None,
        resume_dir: Optional[str] = None,
        project_id: Optional[str] = None,
        run_id: Optional[str] = None,
        tool_executor: Optional[ToolExecutor] = None,
        max_tool_attempts: int = 3,
        max_plan_attempts: int = 3,
    ):
        self.logger = logging.getLogger(__name__)
        print(logo_str)
        self.llm = llm or ChatOpenAI(
            temperature=0,
            model="gpt-5.2",
            response_format={"type": "json_object"},
            reasoning_effort="medium",
        )
        self.summary_llm = summary_llm or ChatOpenAI(
            temperature=0,
            model="gpt-5.2",
        )
        self.max_steps = max_steps
        self.max_plan_attempts = max_plan_attempts
        self.registry = get_tool_registry()
        self.log_llm_console = log_llm_console
        default_log = Path.home() / ".catmaster" / "logs" / "orchestrator_llm.jsonl"
        self.llm_log_file = Path(llm_log_path).expanduser().resolve() if llm_log_path else default_log
        self.llm_log_file.parent.mkdir(parents=True, exist_ok=True)
        if run_dir and resume_dir:
            raise ValueError("run_dir and resume_dir are mutually exclusive")
        resolved_run_dir = Path(run_dir).expanduser().resolve() if run_dir else None
        resolved_resume_dir = Path(resume_dir).expanduser().resolve() if resume_dir else None
        if run_context:
            self.run_context = run_context
        elif resolved_resume_dir:
            self.run_context = RunContext.load(resolved_resume_dir)
        else:
            self.run_context = RunContext.create(
                run_dir=resolved_run_dir,
                project_id=project_id,
                run_id=run_id,
                model_name=self._resolve_model_name(),
            )
        self.event_log = event_log or EventLog(self.run_context.run_dir)
        self.tool_executor = tool_executor or ToolExecutor(self.registry, max_attempts=max_tool_attempts)
        self.artifact_store = ArtifactStore(self.run_context.run_dir)
        self.resume_dir = str(resolved_resume_dir) if resolved_resume_dir else None

        self.plan_prompt = build_plan_prompt()
        self.plan_repair_prompt = build_plan_repair_prompt()
        self.plan_feedback_prompt = build_plan_feedback_prompt()
        self.step_prompt = build_step_prompt()
        self.summary_prompt = build_summary_prompt()

    def _tool_schema(self) -> str:
        return self.registry.get_tool_descriptions_for_llm()

    def _tool_schema_short(self) -> str:
        return self.registry.get_short_tool_descriptions_for_llm()

    def _resolve_model_name(self) -> str:
        for attr in ("model_name", "model"):
            value = getattr(self.llm, attr, None)
            if isinstance(value, str) and value:
                return value
        return "unknown"

    def _log_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        try:
            if self.event_log:
                self.event_log.append(event, payload=payload)
        except Exception as exc:
            self.logger.debug("Event log append failed: %s", exc)

    def _toolcall_id(self, step: int, method: str) -> str:
        return f"step{step}_{method}"

    def _run_state_path(self) -> Path:
        return self.run_context.run_dir / "run_state.json"

    def _save_run_state(
        self,
        *,
        user_request: str,
        memories: Dict[str, Any],
        transcript: List[Dict[str, Any]],
        next_step_index: int,
        finished: bool = False,
        finish_reason: Optional[str] = None,
        last_error: Optional[str] = None,
    ) -> None:
        payload = {
            "user_request": user_request,
            "memories": memories,
            "transcript": transcript,
            "next_step_index": next_step_index,
            "finished": finished,
            "finish_reason": finish_reason,
            "last_error": last_error,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        path = self._run_state_path()
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    def _load_run_state(self) -> Dict[str, Any]:
        path = self._run_state_path()
        if not path.exists():
            raise FileNotFoundError(f"run_state.json not found in {self.run_context.run_dir}")
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data

    def _normalize_run_state(self, data: Dict[str, Any]) -> Dict[str, Any]:
        state = data if isinstance(data, dict) else {}
        memories = state.get("memories") if isinstance(state.get("memories"), dict) else {}
        memories.setdefault("todo", [])
        memories.setdefault("observations", [])
        memories.setdefault("next_step", "")
        memories.setdefault("toolcall_seq", 0)
        memories.setdefault("pending_toolcall", None)
        transcript = state.get("transcript") if isinstance(state.get("transcript"), list) else []
        next_step_index = state.get("next_step_index")
        if not isinstance(next_step_index, int) or next_step_index < 0:
            next_step_index = len(transcript)
        state["memories"] = memories
        state["transcript"] = transcript
        state["next_step_index"] = next_step_index
        return state

    def _normalize_plan(self, data: Dict[str, Any], user_request: str) -> Dict[str, Any]:
        if not isinstance(data, dict):
            data = {}
        normalized = dict(data)
        todo = normalized.get("todo")
        if not isinstance(todo, list) or not todo:
            todo = [user_request]
        next_step = normalized.get("next_step")
        if not isinstance(next_step, str) or not next_step.strip():
            next_step = user_request
        reasoning = normalized.get("reasoning")
        if not isinstance(reasoning, str):
            reasoning = ""
        normalized["todo"] = todo
        normalized["next_step"] = next_step
        normalized["reasoning"] = reasoning
        return normalized

    def plan(self, user_request: str) -> Dict[str, Any]:
        tools = self._tool_schema_short()
        messages = self.plan_prompt.format_messages(
            user_request=user_request,
            tools=tools,
        )
        last_error = None
        for attempt in range(1, self.max_plan_attempts + 1):
            resp = self.llm.invoke(messages)
            raw = resp.content
            self._write_llm_log("plan_prompt", messages=self._messages_to_dict(messages))
            self._write_llm_log("plan_response", content=raw)
            if self.log_llm_console:
                self.logger.info("[PLAN][LLM RAW][attempt=%s] %s", attempt, raw)
            try:
                data = self._parse_json_response(raw)
            except Exception as exc:
                last_error = str(exc)
                self.logger.error("Failed to parse plan response (attempt %s): %s (%s)", attempt, raw, exc)
                self._log_event("PLAN_PARSE_FAILED", {
                    "attempt": attempt,
                    "error": last_error,
                    "raw": raw,
                })
                messages = self.plan_repair_prompt.format_messages(
                    user_request=user_request,
                    tools=tools,
                    error=last_error,
                    raw=raw,
                )
                continue

            normalized = self._normalize_plan(data, user_request)
            self._log_event("PLAN_CREATED", {
                "todo": normalized.get("todo", []),
                "next_step": normalized.get("next_step", ""),
                "reasoning": normalized.get("reasoning", ""),
            })
            return normalized

        raise ValueError(f"Failed to generate a valid plan after {self.max_plan_attempts} attempts: {last_error}")

    def revise_plan(
        self,
        user_request: str,
        plan: Dict[str, Any],
        feedback: str,
        *,
        feedback_history: Optional[List[Dict[str, Any]]] = None,
        log_llm: bool = False,
    ) -> Dict[str, Any]:
        messages = self.plan_feedback_prompt.format_messages(
            user_request=user_request,
            tools=self._tool_schema_short(),
            plan_json=json.dumps(plan, ensure_ascii=False),
            feedback=feedback,
            feedback_history=json.dumps(feedback_history or [], ensure_ascii=False),
        )
        resp = self.llm.invoke(messages)
        raw = resp.content
        self._write_llm_log("plan_feedback_prompt", messages=self._messages_to_dict(messages))
        self._write_llm_log("plan_feedback_response", content=raw)
        if log_llm or self.log_llm_console:
            self.logger.info("[PLAN FEEDBACK][LLM RAW] %s", raw)
        try:
            data = self._parse_json_response(raw)
        except Exception as exc:
            self.logger.error("Failed to parse plan feedback response: %s (%s), keeping prior plan", raw, exc)
            data = plan
        normalized = self._normalize_plan(data, user_request)
        self._log_event("PLAN_REVISION", {
            "feedback": feedback,
            "plan_before": plan,
            "plan_after": normalized,
        })
        return normalized

    def start_plan_review(self, user_request: str) -> Dict[str, Any]:
        plan = self.plan(user_request)
        return {
            "user_request": user_request,
            "plan": plan,
            "feedback_history": [],
            "approved": False,
            "round": 0,
        }

    def apply_plan_feedback(
        self,
        state: Dict[str, Any],
        feedback: str,
        *,
        log_llm: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(state, dict) or "plan" not in state or "user_request" not in state:
            raise ValueError("Invalid plan review state; expected keys: user_request, plan")
        state.setdefault("feedback_history", [])
        state.setdefault("round", 0)
        feedback_text = feedback or ""
        if self._is_plan_approved(feedback_text):
            state["approved"] = True
            state["feedback_history"].append({
                "round": state.get("round", 0),
                "feedback": feedback_text,
                "approved": True,
                "plan": state["plan"],
            })
            self._log_event("PLAN_APPROVED", {
                "feedback": feedback_text,
                "plan": state["plan"],
            })
            return state
        new_plan = self.revise_plan(
            state["user_request"],
            state["plan"],
            feedback_text,
            feedback_history=state.get("feedback_history", []),
            log_llm=log_llm,
        )
        state["feedback_history"].append({
            "round": state.get("round", 0),
            "feedback": feedback_text,
            "approved": False,
            "plan_before": state["plan"],
            "plan_after": new_plan,
        })
        state["plan"] = new_plan
        state["approved"] = False
        state["round"] = state.get("round", 0) + 1
        return state

    def run(
        self,
        user_request: str,
        *,
        log_llm: bool = False,
        initial_plan: Optional[Dict[str, Any]] = None,
        plan_review: bool = True,
        plan_feedback_provider: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> Dict[str, Any]:
        if self.resume_dir:
            state = self._normalize_run_state(self._load_run_state())
            memories = state["memories"]
            transcript = state["transcript"]
            start_step = state["next_step_index"]
            user_request = state.get("user_request", user_request)
            finish_reason = state.get("finish_reason") or "resumed"
            self._log_event("RUN_RESUMED", {
                "run_dir": self.run_context.run_dir.name,
                "next_step_index": start_step,
            })
        else:
            memories = {
                "todo": [],
                "observations": [],
                "next_step": "",
                "toolcall_seq": 0,
                "pending_toolcall": None,
            }
            if initial_plan:
                initial = self._normalize_plan(initial_plan, user_request)
                self._log_event("PLAN_CREATED", {
                    "todo": initial.get("todo", []),
                    "next_step": initial.get("next_step", ""),
                    "reasoning": initial.get("reasoning", ""),
                    "source": "provided",
                })
            else:
                initial = self.plan(user_request)

            if plan_review:
                state = {
                    "user_request": user_request,
                    "plan": initial,
                    "feedback_history": [],
                    "approved": False,
                    "round": 0,
                }
                while not state.get("approved"):
                    if plan_feedback_provider:
                        feedback = plan_feedback_provider(state)
                    else:
                        if not sys.stdin.isatty():
                            raise RuntimeError("plan_review requires a feedback provider when not running in a TTY")
                        print("\n=== Proposed Plan ===")
                        for i, item in enumerate(state["plan"].get("todo", []), start=1):
                            print(f"{i}. {item}")
                        print(f"\nNext step: {state['plan'].get('next_step', '')}")
                        print("\nEnter feedback to revise, or type 'yes' to approve:")
                        feedback = input("> ").strip()
                    if not feedback:
                        if not plan_feedback_provider:
                            print("Empty input. Please enter feedback to revise, or type 'yes' to approve.")
                        continue
                    state = self.apply_plan_feedback(state, feedback, log_llm=log_llm)
                initial = state["plan"]

            memories["todo"] = initial.get("todo", [])
            memories["next_step"] = initial.get("next_step", "")
            transcript = []
            start_step = 0
            finish_reason = "max_steps_reached"
            self._save_run_state(
                user_request=user_request,
                memories=memories,
                transcript=transcript,
                next_step_index=start_step,
                finished=False,
            )

        for step in range(start_step, self.max_steps):
            decision = self._decide_next(
                memories,
                log_llm=log_llm,
                step=step,
            )
            action = decision.get("action")
            memories["next_step"] = decision.get("next_step", "")

            if action == "finish_project":
                self.logger.info("[STEP %s] LLM decided to finish", step)
                self._log_event("FINISH_DECIDED", {
                    "step": step,
                    "decision": decision,
                })
                finish_reason = "finish_project"
                self._save_run_state(
                    user_request=user_request,
                    memories=memories,
                    transcript=transcript,
                    next_step_index=step + 1,
                    finished=True,
                    finish_reason=finish_reason,
                )
                break
            if action == "skip_step":
                self.logger.info("[STEP %s] Due to decision parse failure or invalid tool name, skip the step", step)
                self._save_run_state(
                    user_request=user_request,
                    memories=memories,
                    transcript=transcript,
                    next_step_index=step + 1,
                    finished=False,
                )
                continue

            method = decision.get("method")
            params = decision.get("params", {})
            self.logger.info("[STEP %s] Calling tool %s with params %s", step, method, params)
            toolcall_id = self._toolcall_id(step, method)
            refs = self.artifact_store.toolcall_refs(toolcall_id)
            pending = memories.get("pending_toolcall")
            if not pending or pending.get("method") != method:
                toolcall_key = f"{method}:{memories.get('toolcall_seq', 0)}"
                memories["pending_toolcall"] = {"method": method, "key": toolcall_key}
            else:
                toolcall_key = pending.get("key")
            validation = self.tool_executor.validate(method, params, toolcall_key=toolcall_key)
            if not validation.get("ok"):
                tool_output = validation.get("tool_output", {})
                if isinstance(tool_output, dict):
                    tool_output.setdefault("data", {})
                    if isinstance(tool_output.get("data"), dict):
                        tool_output["data"]["attempt_count"] = validation.get("attempt_count")
                        tool_output["data"]["max_attempts"] = validation.get("max_attempts")
                input_payload = {
                    "raw_params": params,
                    "validated_params": None,
                    "tool_name": method,
                    "toolcall_id": toolcall_id,
                    "status": "validation_failed",
                }
                self.artifact_store.write_input(toolcall_id, input_payload)
                output_payload = {
                    "toolresult": tool_output,
                    "full_output": tool_output,
                }
                self.artifact_store.write_output(toolcall_id, output_payload)
                self._log_event("TOOLCALL_VALIDATION_FAILED", {
                    "step": step,
                    "method": method,
                    "raw_params": params,
                    "error": validation.get("error_digest", ""),
                    "errors": validation.get("errors", []),
                    "attempt_count": validation.get("attempt_count"),
                    "max_attempts": validation.get("max_attempts"),
                    "toolcall_id": toolcall_id,
                    "input_ref": refs["input_ref"],
                    "output_ref": refs["output_ref"],
                })
                memories["observations"].append({"step": step, "method": method, "result": tool_output})
                transcript.append({
                    "step": step,
                    "method": method,
                    "params": params,
                    "result": tool_output,
                    "validation_failed": True,
                    "toolcall_id": toolcall_id,
                    "input_ref": refs["input_ref"],
                    "output_ref": refs["output_ref"],
                })
                memories["next_step"] = validation.get("next_step", memories.get("next_step", ""))
                self._save_run_state(
                    user_request=user_request,
                    memories=memories,
                    transcript=transcript,
                    next_step_index=step + 1,
                    finished=False,
                )
                continue

            validated_params = validation.get("validated_params", {})
            input_payload = {
                "raw_params": params,
                "validated_params": validated_params,
                "tool_name": method,
                "toolcall_id": toolcall_id,
                "status": "validated",
            }
            self.artifact_store.write_input(toolcall_id, input_payload)
            self._log_event("TOOLCALL_STARTED", {
                "step": step,
                "method": method,
                "raw_params": params,
                "validated_params": validated_params,
                "toolcall_id": toolcall_id,
                "input_ref": refs["input_ref"],
            })
            # Random time sleep of 0.1-0.3 seconds
            time.sleep(random.uniform(0.1, 0.3))
            try:
                result = self._call_tool(method, validated_params)
                self.logger.info("[STEP %s] Result: %s", step, result)
                output_payload = {
                    "toolresult": result,
                    "full_output": result,
                }
                self.artifact_store.write_output(toolcall_id, output_payload)
                self._log_event("TOOLCALL_FINISHED", {
                    "step": step,
                    "method": method,
                    "validated_params": validated_params,
                    "toolcall_id": toolcall_id,
                    "output_ref": refs["output_ref"],
                    "result": result,
                })
                memories["observations"].append({"step": step, "method": method, "result": result})
                transcript.append({
                    "step": step,
                    "method": method,
                    "params": params,
                    "validated_params": validated_params,
                    "result": result,
                    "toolcall_id": toolcall_id,
                    "input_ref": refs["input_ref"],
                    "output_ref": refs["output_ref"],
                })
                memories["pending_toolcall"] = None
                memories["toolcall_seq"] = memories.get("toolcall_seq", 0) + 1
                self._save_run_state(
                    user_request=user_request,
                    memories=memories,
                    transcript=transcript,
                    next_step_index=step + 1,
                    finished=False,
                )
            except Exception as exc:
                tool_result = {
                    "toolcall_id": toolcall_id,
                    "tool_name": method,
                    "status": "failed",
                    "error": str(exc),
                    "input_ref": refs["input_ref"],
                    "output_ref": refs["output_ref"],
                }
                output_payload = {
                    "toolresult": tool_result,
                    "full_output": {"error": str(exc)},
                }
                self.artifact_store.write_output(toolcall_id, output_payload)
                self._log_event("TOOLCALL_FAILED", {
                    "step": step,
                    "method": method,
                    "validated_params": validated_params,
                    "error": str(exc),
                    "toolcall_id": toolcall_id,
                    "output_ref": refs["output_ref"],
                })
                memories["observations"].append({"step": step, "method": method, "result": tool_result})
                transcript.append({
                    "step": step,
                    "method": method,
                    "params": params,
                    "validated_params": validated_params,
                    "result": tool_result,
                    "toolcall_id": toolcall_id,
                    "input_ref": refs["input_ref"],
                    "output_ref": refs["output_ref"],
                    "error": str(exc),
                })
                memories["pending_toolcall"] = None
                memories["toolcall_seq"] = memories.get("toolcall_seq", 0) + 1
                self._save_run_state(
                    user_request=user_request,
                    memories=memories,
                    transcript=transcript,
                    next_step_index=step + 1,
                    finished=False,
                    last_error=str(exc),
                )
                continue

        summary = self._summarize(memories, user_request)
        final_answer = self._llm_summary(memories, user_request)
        self._log_event("RUN_FINISHED", {
            "reason": finish_reason,
            "steps": len(transcript),
        })
        self._save_run_state(
            user_request=user_request,
            memories=memories,
            transcript=transcript,
            next_step_index=step + 1 if 'step' in locals() else 0,
            finished=True,
            finish_reason=finish_reason,
        )

        return {
            "todo": memories["todo"],
            "observations": memories["observations"],
            "transcript": transcript,
            "summary": summary,
            "final_answer": final_answer,
        }

    def run_with_plan_state(
        self,
        plan_state: Dict[str, Any],
        *,
        log_llm: bool = False,
    ) -> Dict[str, Any]:
        if not isinstance(plan_state, dict) or "plan" not in plan_state:
            raise ValueError("Invalid plan_state; expected a dict with a 'plan' key")
        if not plan_state.get("approved"):
            return {
                "todo": plan_state.get("plan", {}).get("todo", []),
                "observations": [],
                "transcript": [],
                "summary": "Plan awaiting approval; apply feedback and approve with 'yes' before running steps.",
                "final_answer": "",
                "plan_state": plan_state,
                "status": "awaiting_plan_approval",
            }
        user_request = plan_state.get("user_request", "")
        return self.run(user_request, log_llm=log_llm, initial_plan=plan_state.get("plan"))

    def _decide_next(
        self,
        memories: Dict[str, Any],
        *,
        log_llm: bool = False,
        step: int = 0,
    ) -> Dict[str, Any]:
        messages = self.step_prompt.format_messages(
            todo=json.dumps(memories.get("todo", [])),
            observations=json.dumps(memories.get("observations", [])),
            last_result=json.dumps(memories.get("observations", [])[-1] if memories.get("observations") else {}),
            tools=self._tool_schema(),
            instruction=memories.get("next_step", ""),
        )
        resp = self.llm.invoke(messages)
        raw = resp.content

        self._write_llm_log("step_prompt", step=step, messages=self._messages_to_dict(messages))
        self._write_llm_log("step_response", step=step, content=raw)
        if log_llm or self.log_llm_console:
            self.logger.info("[DECIDE][LLM RAW][step=%s] %s", step, raw)

        def _controller_obs(status: str, **kw):
            memories["observations"].append({
                "step": step,
                "method": "ORCHESTRATOR_OBSERVATION",
                "result": {"status": status, **kw},
            })

        # 1) Try to parse the raw response as JSON
        try:
            decision = self._parse_json_response(raw)
        except Exception as exc:
            _controller_obs(
                "LLM_DECISION_JSON_PARSE_FAILED",
                error=str(exc),
                raw=raw,
            )
            return {
                "action": "skip_step",
                "method": None,
                "params": {},
                "next_step": "Re-emit a valid JSON decision following the schema. Choose ONE tool call with valid params, or finish_project.",
                "note": "json_parse_failed",
                "reasoning": f"unable to parse LLM decision as JSON: {exc}",
            }

        # 3) Normalize the decision to avoid missing fields / params not being an object
        action = decision.get("action")
        method = decision.get("method")
        params = decision.get("params", {})
        next_step = decision.get("next_step", "")
        note = decision.get("note", "")
        reasoning = decision.get("reasoning", "")

        # 4) Minimum validation (if the tool name does not exist, skip the step, write observation, and end or choose to repair again)
        if action == "call":
            if not method or not self.registry.get_tool_info(method):
                _controller_obs(
                    "TOOLCALL_VALIDATION_FAILED",
                    error="invalid or unknown tool name",
                    method=method,
                )
                return {
                    "action": "skip_step",
                    "method": None,
                    "params": {},
                    "next_step": "skip the step due to invalid tool name in decision",
                    "note": "invalid_tool",
                    "reasoning": f"method is not a valid registered tool: {method}",
                }

        if action == "finish_project":
            method = None
            params = {}

        return {
            "action": action,
            "method": method,
            "params": params,
            "next_step": next_step,
            "note": note,
            "reasoning": reasoning,
        }

    def _call_tool(self, method: str, params: Dict[str, Any]) -> Any:
        func = self.registry.get_tool_function(method)
        return func(params)

    def _summarize(self, memories: Dict[str, Any], user_request: str) -> str:
        lines = [f"Request: {user_request}"]
        for obs in memories.get("observations", []):
            lines.append(f"- {obs.get('method')}: status={obs.get('result', {}).get('status', 'unknown')}")
        return "\n".join(lines)

    def _llm_summary(self, memories: Dict[str, Any], user_request: str) -> str:
        obs_text = json.dumps(memories.get("observations", []), ensure_ascii=False)
        try:
            resp = self.summary_llm.invoke(self.summary_prompt.format_messages(
                user_request=user_request,
                observations=obs_text,
            ))
            return resp.content
        except Exception:
            return self._summarize(memories, user_request)

    def _messages_to_dict(self, messages: List[Any]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        for msg in messages:
            formatted.append(
                {
                    "type": getattr(msg, "type", getattr(msg, "role", "unknown")),
                    "content": getattr(msg, "content", str(msg)),
                }
            )
        return formatted

    # ------------------ JSON parsing helpers ------------------
    def _parse_json_response(self, content: str) -> Dict[str, Any]:
        """Strict JSON parse; raises if invalid."""
        return json.loads(content)

    def _is_plan_approved(self, feedback: str) -> bool:
        if not isinstance(feedback, str):
            return False
        normalized = feedback.strip().lower()
        return normalized in {"yes", "y", "approve", "approved", "ok", "okay"}

    # ------------------ LLM log helper ------------------
    def _write_llm_log(self, event: str, *, content: Optional[str] = None, messages: Optional[List[Dict[str, Any]]] = None, step: Optional[int] = None) -> None:
        """Append prompt/response to a private JSONL log outside the workspace."""
        if not self.llm_log_file:
            return
        record = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "event": event,
        }
        if step is not None:
            record["step"] = step
        if messages is not None:
            record["messages"] = messages
        if content is not None:
            record["content"] = content
        try:
            with self.llm_log_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            self.logger.debug("LLM log write failed: %s", exc)

__all__ = ["Orchestrator"]
