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
from typing import Any, Dict, List, Optional
import time
import random
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from catmaster.tools.registry import get_tool_registry
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
    ):
        self.logger = logging.getLogger(__name__)
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
        self.registry = get_tool_registry()
        self.log_llm_console = log_llm_console
        default_log = Path.home() / ".catmaster" / "logs" / "orchestrator_llm.jsonl"
        self.llm_log_file = Path(llm_log_path).expanduser().resolve() if llm_log_path else default_log
        self.llm_log_file.parent.mkdir(parents=True, exist_ok=True)

        self.plan_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert computational workflow planner.

Available tools:
{tools}

Create:
- A ToDo list (long-term), each item a short clause that can be completed with the listed tools, which will be used as the reference plan skeleton for the next step decision.
- A NextStep (single actionable intent) that can be attempted immediately.
- A reasoning field explaining briefly why this plan/next step was chosen.

Rules:
- Use only the available tools; do NOT invent capabilities.
- Always express any file or directory paths as relative paths; they will be resolved relative to workspace root.
- Return exactly one JSON object; no code fences or extra text.
- JSON schema: {{"todo": [...], "next_step": "...", "reasoning": "..."}}"""),
            ("human", "{user_request}")
        ])

        self.plan_feedback_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an expert computational workflow planner.

Available tools:
{tools}

Revise the plan based on human feedback.

Inputs:
- Original user request
- Current plan JSON
- Human feedback
- Feedback history (oldest first)

Rules:
- Use only the available tools; do NOT invent capabilities.
- If feedback is unclear or conflicts with tool limits, make the smallest safe change and note the constraint in reasoning.
- Always express any file or directory paths as relative paths; they will be resolved relative to workspace root.
- Return exactly one JSON object; no code fences or extra text.
- JSON schema: {{"todo": [...], "next_step": "...", "reasoning": "..."}}"""),
            ("human", "User request: {user_request}\nCurrent plan: {plan_json}\nHuman feedback: {feedback}\nFeedback history: {feedback_history}")
        ])

        self.step_prompt = ChatPromptTemplate.from_messages([
            ("system", """
You are an execution controller. Decide ONE tool call or finish_project.
Context:
- Available tools: {tools}
- Reference Plan Skeleton: {todo}
- Observations so far: {observations}
- Last result: {last_result}

Rules:
- Choose at most one tool per turn.
- Return exactly one JSON object; no code fences or extra text.
- JSON schema (all lowercase keys):
{{"action": "call"|"finish_project",
    "method": "tool_name"|null,
    "params": {{...}},
    "next_step": "a concrete, testable intent that can be acted on in the next turn (either a tool call you plan to attempt, or a condition to check via a specific tool)",
    "note": "optional short self-note for memory",
    "reasoning": "brief rationale for this decision"}}
- If action='call', method MUST exactly match one tool name in available tools. Otherwise set action='finish_project' and explain why.
- If you ensure that to finish the project, set action="finish_project" and method=null in that single object.
- Prefer creating/using subfolders under the workspace for each step; reuse paths returned by previous tool calls instead of guessing.
- Always provide file or directory paths as relative paths; they will be resolved relative to workspace root.
- If a needed file might not exist, first list or create it with the appropriate tool.
- Treat user instruction as a suggestion. If observations contradict it or a better action is available, you CAN revise the plan by choosing a different tool call and writing an updated next_step.
- The controller may skip a turn if your JSON is invalid; in that case, you must output a valid JSON decision next turn.
"""),
            ("human", "Suggested next step (may be revised): {instruction}")
        ])

        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a scientific workflow assistant. Summarize the run from observations and respond the results for the user.
Include key numerical results present in observations.
Mention where outputs are stored (use relative paths if provided). Keep the summary concise and informative."""),
            ("human", "User request: {user_request}\nObservations: {observations}")
        ])

    def _tool_schema(self) -> str:
        return self.registry.get_tool_descriptions_for_llm()

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
        messages = self.plan_prompt.format_messages(
            user_request=user_request,
            tools=self._tool_schema(),
        )
        resp = self.llm.invoke(messages)
        raw = resp.content
        self._write_llm_log("plan_prompt", messages=self._messages_to_dict(messages))
        self._write_llm_log("plan_response", content=raw)
        if self.log_llm_console:
            self.logger.info("[PLAN][LLM RAW] %s", raw)
        try:
            data = self._parse_json_response(raw)
        except Exception as exc:
            self.logger.error("Failed to parse plan response: %s (%s), using user request as fallback", raw, exc)
            data = {"todo": [user_request], "next_step": user_request, "reasoning": ""}
        return self._normalize_plan(data, user_request)

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
            tools=self._tool_schema(),
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
        return self._normalize_plan(data, user_request)

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
    ) -> Dict[str, Any]:
        memories = {
            "todo": [],
            "observations": [],
            "next_step": "",
        }

        initial = self._normalize_plan(initial_plan, user_request) if initial_plan else self.plan(user_request)
        memories["todo"] = initial.get("todo", [])
        memories["next_step"] = initial.get("next_step", "")

        transcript: List[Dict[str, Any]] = []

        for step in range(self.max_steps):
            decision = self._decide_next(
                memories,
                log_llm=log_llm,
                step=step,
            )
            action = decision.get("action")
            memories["next_step"] = decision.get("next_step", "")

            if action == "finish_project":
                self.logger.info("[STEP %s] LLM decided to finish", step)
                break
            if action == "skip_step":
                self.logger.info("[STEP %s] Due to decision parse failure or invalid tool name, skip the step", step)
                continue

            method = decision.get("method")
            params = decision.get("params", {})
            self.logger.info("[STEP %s] Calling tool %s with params %s", step, method, params)
            # Random time sleep of 0.1-0.3 seconds
            time.sleep(random.uniform(0.1, 0.3))
            result = self._call_tool(method, params)
            self.logger.info("[STEP %s] Result: %s", step, result)
            memories["observations"].append({"step": step, "method": method, "result": result})
            transcript.append({"step": step, "method": method, "params": params, "result": result})

        summary = self._summarize(memories, user_request)
        final_answer = self._llm_summary(memories, user_request)

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
                "method": "__controller__",
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
        params = decision.get("params") if isinstance(decision.get("params"), dict) else {}
        next_step = decision.get("next_step", "")
        note = decision.get("note", "")
        reasoning = decision.get("reasoning", "")

        # 4) Minimum validation (if the tool name does not exist, skip the step, write observation, and end or choose to repair again)
        if action == "call":
            if not method or not self.registry.get_tool_function(method):
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
