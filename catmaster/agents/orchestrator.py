#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task-based orchestrator with whiteboard patch workflow and unified tracing.
"""
from __future__ import annotations

import json
import re
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import logging
import sys

from langchain_openai import ChatOpenAI

from catmaster.tools.registry import get_tool_registry
from catmaster.tools.base import system_root, workspace_root
from catmaster.runtime import (
    RunContext,
    ToolExecutor,
    ArtifactStore,
    WhiteboardStore,
    TraceStore,
    ContextPackBuilder,
    ContextPackPolicy,
    whiteboard_ops_apply_atomic,
    whiteboard_ops_persist,
    whiteboard_ops_validate,
)
from catmaster.runtime.artifact_log import ArtifactLog
from catmaster.runtime.whiteboard_ops import persist_whiteboard_diff, append_task_journal_entry, _section_bounds
from catmaster.agents.logo import logo_str
from catmaster.agents.task_runner import TaskStepper, TaskSummarizer
from catmaster.ui import Reporter, NullReporter, make_event
from catmaster.agents.orchestrator_prompts import (
    build_plan_prompt,
    build_plan_repair_prompt,
    build_plan_feedback_prompt,
    build_task_step_prompt,
    build_task_summarizer_prompt,
    build_task_summarizer_repair_prompt,
    build_summary_prompt,
)


class Orchestrator:
    def __init__(
        self,
        llm: ChatOpenAI,
        max_steps: int = 100,
        *,
        summary_llm: Optional[ChatOpenAI] = None,
        llm_log_path: Optional[str] = None,
        log_llm_console: bool = True,
        reporter: Optional[Reporter] = None,
        run_context: Optional[RunContext] = None,
        run_dir: Optional[str] = None,
        resume: bool = False,
        resume_dir: Optional[str] = None,
        project_id: Optional[str] = None,
        run_id: Optional[str] = None,
        tool_executor: Optional[ToolExecutor] = None,
        max_tool_attempts: int = 3,
        max_plan_attempts: int = 3,
        patch_repair_attempts: int = 1,
        summary_repair_attempts: int = 1,
    ):
        self.logger = logging.getLogger(__name__)
        self.reporter = reporter or NullReporter()
        self.reporter.emit(make_event(
            "RUN_INIT_START",
            category="run",
            payload={
                "run_dir": str(run_dir) if run_dir else "",
                "resume": bool(resume),
                "resume_dir": str(resume_dir) if resume_dir else "",
            },
            run_id=run_id,
        ))
        if llm is None:
            raise ValueError("llm must be provided (single shared model).")
        self.llm = llm
        self.summary_llm = summary_llm or llm
        self.max_steps = max_steps
        self.max_plan_attempts = max_plan_attempts
        self.patch_repair_attempts = patch_repair_attempts
        self.summary_repair_attempts = summary_repair_attempts
        self.registry = get_tool_registry()
        self.log_llm_console = log_llm_console
        self.resuming = False

        if run_context:
            self.run_context = run_context
        else:
            if run_dir and (resume or resume_dir):
                raise ValueError("run_dir is mutually exclusive with resume/resume_dir")
            resolved_run_dir = Path(run_dir).expanduser().resolve() if run_dir else None
            resolved_resume = self._resolve_resume_run_dir(resume_dir, resume)
            if resolved_resume:
                self.run_context = RunContext.load(resolved_resume)
                self.resuming = True
            else:
                self.run_context = RunContext.create(
                    run_dir=resolved_run_dir,
                    project_id=project_id,
                    run_id=run_id,
                    model_name=self._resolve_model_name(),
                )
                self.resuming = False

        self.trace_store = TraceStore(self.run_context.run_dir)
        self.tool_executor = tool_executor or ToolExecutor(self.registry, max_attempts=max_tool_attempts)
        self.artifact_store = ArtifactStore(self.run_context.run_dir)

        self.whiteboard = WhiteboardStore.create_default()
        self.whiteboard.ensure_exists()
        self.artifact_log = ArtifactLog(system_root() / "artifacts.csv")
        self.artifact_log.ensure_exists()
        self.context_builder = ContextPackBuilder(self.whiteboard)

        default_log = self.run_context.run_dir / "llm.jsonl"
        self.llm_log_file = Path(llm_log_path).expanduser().resolve() if llm_log_path else default_log
        self.llm_log_file.parent.mkdir(parents=True, exist_ok=True)

        self.plan_prompt = build_plan_prompt()
        self.plan_repair_prompt = build_plan_repair_prompt()
        self.plan_feedback_prompt = build_plan_feedback_prompt()
        self.task_step_prompt = build_task_step_prompt()
        self.task_summary_prompt = build_task_summarizer_prompt()
        self.task_summary_repair_prompt = build_task_summarizer_repair_prompt()
        self.summary_prompt = build_summary_prompt()
        self._emit("RUN_INIT_DONE", payload={
            "run_id": self.run_context.run_id,
            "run_dir": str(self.run_context.run_dir),
            "model_name": self._resolve_model_name(),
            "model_label": self._resolve_model_label(),
            "resuming": self.resuming,
            "llm_log_path": str(self.llm_log_file),
            "trace_paths": {
                "event_trace": str(self.run_context.run_dir / "event_trace.jsonl"),
                "tool_trace": str(self.run_context.run_dir / "tool_trace.jsonl"),
                "patch_trace": str(self.run_context.run_dir / "patch_trace.jsonl"),
                "task_state": str(self.run_context.run_dir / "task_state.json"),
                "whiteboard": str(self.whiteboard.path),
            },
        })

    def _resolve_resume_run_dir(self, resume_dir: Optional[str], resume: bool) -> Optional[Path]:
        if not resume and not resume_dir:
            return None
        base = Path(resume_dir).expanduser().resolve() if resume_dir else workspace_root()
        if (base / "meta.json").exists():
            return base
        sys_root = base if base.name == ".catmaster" else (base / ".catmaster")
        runs_root = sys_root / "runs"
        if not runs_root.exists():
            raise FileNotFoundError(f"Resume requested but runs directory not found: {runs_root}")
        run_dirs = [d for d in runs_root.iterdir() if d.is_dir()]
        if not run_dirs:
            raise FileNotFoundError(f"Resume requested but no run directories found in {runs_root}")
        run_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        for candidate in run_dirs:
            if (candidate / "task_state.json").exists():
                return candidate
        for candidate in run_dirs:
            if (candidate / "meta.json").exists():
                return candidate
        raise FileNotFoundError(f"Resume requested but no valid run metadata found in {runs_root}")

    def _task_state_path(self) -> Path:
        return self.run_context.run_dir / "task_state.json"

    def _load_task_state(self) -> Dict[str, Any]:
        path = self._task_state_path()
        if not path.exists():
            raise FileNotFoundError(f"task_state.json not found in {self.run_context.run_dir}")
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("task_state.json must contain a JSON object")
        for key in ("user_request", "plan", "tasks", "observations", "status"):
            if key not in data:
                raise ValueError(f"task_state.json missing required key: {key}")
        if not isinstance(data["tasks"], list):
            raise ValueError("task_state.json tasks must be a list")
        if not isinstance(data["observations"], list):
            raise ValueError("task_state.json observations must be a list")
        for idx, task in enumerate(data["tasks"], start=1):
            if not isinstance(task, dict):
                raise ValueError(f"task_state.json tasks[{idx}] must be an object")
            for key in ("task_id", "goal", "status"):
                if key not in task:
                    raise ValueError(f"task_state.json tasks[{idx}] missing key: {key}")
        return data

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

    def _resolve_model_label(self) -> str:
        name = self._resolve_model_name()
        kwargs = self._collect_model_kwargs()
        if not kwargs:
            return name
        parts: List[str] = []
        for key in sorted(kwargs.keys()):
            val = kwargs.get(key)
            if val is None:
                continue
            parts.append(f"{key}={self._snippet(val, 24)}")
        if not parts:
            return name
        joined = ";".join(parts)
        if len(joined) > 80:
            joined = self._snippet(joined, 80)
        return f"{name}({joined})"

    def _collect_model_kwargs(self) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        raw = getattr(self.llm, "model_kwargs", None)
        if isinstance(raw, dict):
            merged.update(raw)
        for key in (
            "reasoning_effort",
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ):
            value = getattr(self.llm, key, None)
            if value is None or key in merged:
                continue
            merged[key] = value
        return merged

    def _trace_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        record = {"event": event, "payload": payload or {}}
        self.trace_store.append_event(record)

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
        try:
            run_id = self.run_context.run_id
        except Exception:
            run_id = None
        self.reporter.emit(make_event(
            name,
            level=level,
            category=category,
            run_id=run_id,
            task_id=task_id,
            step_id=step_id,
            payload=payload or {},
        ))

    def _ui_debug(self) -> bool:
        return bool(getattr(self.reporter, "ui_debug", False))

    @staticmethod
    def _snippet(text: Any, limit: int = 160) -> str:
        if text is None:
            return ""
        cleaned = " ".join(str(text).split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(0, limit - 3)] + "..."

    def plan(self, user_request: str) -> Dict[str, Any]:
        tools = self._tool_schema() # It is not known to provide full schema or short here is better
        messages = self.plan_prompt.format_messages(
            user_request=user_request,
            tools=tools,
        )
        last_error = None
        self._emit("PLAN_START", category="plan", payload={"attempts": self.max_plan_attempts})
        for attempt in range(1, self.max_plan_attempts + 1):
            self._emit("LLM_CALL_START", category="llm", payload={"kind": "plan", "attempt": attempt})
            t0 = time.perf_counter()
            resp = self.llm.invoke(messages)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            raw = resp.content
            self._write_llm_log("plan_prompt", messages=self._messages_to_dict(messages))
            self._write_llm_log("plan_response", content=raw)
            if self.log_llm_console:
                self.logger.info("[PLAN][LLM RAW][attempt=%s] %s", attempt, raw)
            payload = {"kind": "plan", "attempt": attempt, "elapsed_ms": elapsed_ms}
            if self._ui_debug():
                payload["raw_snippet"] = self._snippet(raw, 240)
            self._emit("LLM_CALL_END", category="llm", payload=payload)
            try:
                data = self._parse_json_response(raw)
                normalized = self._normalize_plan(data, user_request)
            except Exception as exc:
                last_error = str(exc)
                self._emit("PLAN_PARSE_FAILED", level="warning", category="plan", payload={
                    "error": last_error,
                    "attempt": attempt,
                })
                messages = self.plan_repair_prompt.format_messages(
                    user_request=user_request,
                    tools=tools,
                    error=last_error,
                    raw=raw,
                )
                continue
            self._trace_event("PLAN_CREATED", {
                "todo": normalized.get("todo", []),
                "plan_description": normalized.get("plan_description", ""),
            })
            todo = normalized.get("todo", [])
            self._emit("PLAN_CREATED", category="plan", payload={
                "n_items": len(todo),
                "todo": todo,
                "plan_description_snippet": self._snippet(normalized.get("plan_description", ""), 200),
            })
            self._emit("PLAN_DONE", category="plan", payload={"n_items": len(todo)})
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
        self._emit("LLM_CALL_START", category="llm", payload={"kind": "plan_feedback"})
        t0 = time.perf_counter()
        resp = self.llm.invoke(messages)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        raw = resp.content
        self._write_llm_log("plan_feedback_prompt", messages=self._messages_to_dict(messages))
        self._write_llm_log("plan_feedback_response", content=raw)
        if log_llm or self.log_llm_console:
            self.logger.info("[PLAN FEEDBACK][LLM RAW] %s", raw)
        payload = {"kind": "plan_feedback", "elapsed_ms": elapsed_ms}
        if self._ui_debug():
            payload["raw_snippet"] = self._snippet(raw, 240)
        self._emit("LLM_CALL_END", category="llm", payload=payload)
        data = self._parse_json_response(raw)
        normalized = self._normalize_plan(data, user_request)
        self._trace_event("PLAN_REVISION", {
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
        self._emit("PLAN_REVIEW_FEEDBACK_SUBMITTED", category="plan", payload={
            "feedback_snippet": self._snippet(feedback_text, 160),
        })
        if self._is_plan_approved(feedback_text):
            state["approved"] = True
            state["feedback_history"].append({
                "round": state.get("round", 0),
                "feedback": feedback_text,
                "approved": True,
                "plan": state["plan"],
            })
            self._trace_event("PLAN_APPROVED", {
                "feedback": feedback_text,
                "plan": state["plan"],
            })
            self._emit("PLAN_REVIEW_APPROVED", category="plan", payload={
                "feedback_snippet": self._snippet(feedback_text, 160),
            })
            return state
        self._emit("PLAN_REVIEW_REVISING", category="plan", payload={
            "feedback_snippet": self._snippet(feedback_text, 160),
        })
        new_plan = self.revise_plan(
            state["user_request"],
            state["plan"],
            feedback_text,
            feedback_history=state.get("feedback_history", []),
            log_llm=log_llm,
        )
        self._emit("PLAN_REVIEW_REVISED", category="plan", payload={
            "n_items": len(new_plan.get("todo", [])),
            "todo": new_plan.get("todo", []),
        })
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
        defer_ui = bool(
            plan_review
            and plan_feedback_provider is None
            and sys.stdin.isatty()
            and not self.reporter.is_live()
        )
        ui_started = False

        def start_ui(subtitle: str, *, show_splash: bool = True) -> None:
            nonlocal ui_started
            if ui_started:
                return
            self.reporter.start()
            ui_started = True
            if show_splash:
                self._emit("SPLASH_SHOW", category="splash", payload={
                    "logo": logo_str,
                    "tagline": "Initializing...",
                    "subtitle": subtitle,
                })

        if defer_ui:
            print(logo_str)
            print("Initializing run context and preparing initial plan...")
        else:
            start_ui("Initializing run context and preparing initial plan...")
        if self.resuming:
            if initial_plan is not None:
                raise ValueError("Cannot provide initial_plan when resuming")
            if plan_review:
                raise ValueError("plan_review is not supported when resuming")
        try:
            if self.resuming:
                state = self._load_task_state()
                user_request = state["user_request"]
                plan = self._normalize_plan(state["plan"], user_request)
                tasks = state["tasks"]
                observations = state["observations"]
                status = state.get("status", "running")
                if status in {"done", "failure"}:
                    raise ValueError(f"Cannot resume; run already ended with status {status}")
                self._initialize_whiteboard_goal(user_request)
                if ui_started:
                    self._emit("SPLASH_HIDE", category="splash")
            else:
                plan = self._normalize_plan(initial_plan, user_request) if initial_plan else self.plan(user_request)
                self._initialize_whiteboard_goal(user_request)
                if ui_started:
                    self._emit("SPLASH_HIDE", category="splash")
                if plan_review:
                    review_state = {
                        "user_request": user_request,
                        "plan": plan,
                        "feedback_history": [],
                        "approved": False,
                        "round": 0,
                    }
                    if plan_feedback_provider is None and not sys.stdin.isatty():
                        raise ValueError("plan_review requires a feedback provider when not running in a TTY")
                    while not review_state.get("approved"):
                        plan_description = (review_state["plan"].get("plan_description") or "").strip()
                        self._emit("PLAN_REVIEW_SHOW", category="plan", payload={
                            "todo": review_state["plan"].get("todo", []),
                            "plan_description_snippet": self._snippet(plan_description, 240),
                        })
                        self._emit("PLAN_REVIEW_WAIT_INPUT", category="plan")
                        if plan_feedback_provider:
                            feedback = plan_feedback_provider(review_state)
                            used_ui_prompt = False
                        else:
                            if hasattr(self.reporter, "prompt_plan_feedback") and self.reporter.is_live():
                                feedback = self.reporter.prompt_plan_feedback(
                                    todo=review_state["plan"].get("todo", []),
                                    plan_description=plan_description,
                                )
                                used_ui_prompt = True
                            else:
                                print("\n=== Proposed Plan ===")
                                print("Plan Description:")
                                print(plan_description if plan_description else "(none)")
                                print("\nTherefore, Here is a proposed plan:")
                                for i, item in enumerate(review_state["plan"].get("todo", []), start=1):
                                    print(f"{i}. {item}")
                                print("\nEnter feedback to revise, or type 'yes' to approve:")
                                feedback = input("> ").strip()
                                used_ui_prompt = False
                        if not feedback:
                            if plan_feedback_provider:
                                raise ValueError("plan_review feedback cannot be empty")
                            if not used_ui_prompt:
                                print("Empty input. Please enter feedback to revise, or type 'yes' to approve.")
                            self._emit("PLAN_REVIEW_WAIT_INPUT", category="plan", payload={"error": "empty_input"})
                            continue
                        review_state = self.apply_plan_feedback(review_state, feedback, log_llm=log_llm)
                    plan = review_state["plan"]

                tasks = self._compile_tasks(plan.get("todo", []))
                observations = []
                status = "running"
                self._write_task_state({
                    "user_request": user_request,
                    "plan": plan,
                    "tasks": tasks,
                    "observations": observations,
                    "status": status,
                })
                if defer_ui and not ui_started:
                    start_ui("Starting execution...", show_splash=False)
                    self._emit("SPLASH_HIDE", category="splash")

            self._emit("TASKS_COMPILED", category="task", payload={
                "n_tasks": len(tasks),
                "tasks": tasks,
            })

            while True:
                if status == "needs_intervention":
                    hitl = self._hitl_intervene(
                        user_request=user_request,
                        plan=plan,
                        tasks=tasks,
                        observations=observations,
                        log_llm=log_llm,
                        plan_feedback_provider=plan_feedback_provider,
                    )
                    status = hitl.get("status", status)
                    plan = hitl.get("plan", plan)
                    tasks = hitl.get("tasks", tasks)
                    if status == "needs_intervention":
                        return hitl
                    self._emit("TASKS_COMPILED", category="task", payload={
                        "n_tasks": len(tasks),
                        "tasks": tasks,
                    })

                progressed = False
                for task in tasks:
                    if task.get("status") != "pending":
                        continue
                    progressed = True
                    task_id = task["task_id"]
                    task_goal = task["goal"]
                    self._trace_event("TASK_STARTED", {"task_id": task_id, "goal": task_goal})
                    self._emit("TASK_START", category="task", task_id=task_id, payload={"goal": task_goal})
                    context_pack = self.context_builder.build(task_goal, role="task_runner", policy=ContextPackPolicy())
                    self._emit("TASK_CONTEXT_READY", category="task", task_id=task_id, payload={
                        "excerpt_chars": len(context_pack.get("whiteboard_excerpt", "") or ""),
                        "artifact_slice_count": len(context_pack.get("artifact_slice", []) or []),
                    })
                    stepper = TaskStepper(
                        llm=self.llm,
                        registry=self.registry,
                        tool_executor=self.tool_executor,
                        artifact_store=self.artifact_store,
                        trace_store=self.trace_store,
                        prompt=self.task_step_prompt,
                        log_fn=self._write_llm_log,
                        log_llm_console=log_llm,
                        max_steps=self.max_steps,
                        reporter=self.reporter,
                    )
                    step_result = stepper.run(
                        task_id=task_id,
                        task_goal=task_goal,
                        tools_schema=self._tool_schema(),
                        context_pack=context_pack,
                        initial_instruction=f"Start task: {task_goal}",
                    )
                    summarizer = TaskSummarizer(
                        llm=self.summary_llm,
                        prompt=self.task_summary_prompt,
                        repair_prompt=self.task_summary_repair_prompt,
                        log_fn=self._write_llm_log,
                        log_llm_console=log_llm,
                    )
                    task_result = self._summarize_and_update_whiteboard(
                        summarizer,
                        task_id=task_id,
                        task_goal=task_goal,
                        step_result=step_result,
                    )
                    outcome = task_result["task_outcome"]
                    observation_path = self._write_observation(
                        task_id=task_id,
                        outcome=outcome,
                        summary=task_result["task_summary"],
                        key_artifacts=task_result["key_artifacts"],
                    )
                    self._emit("OBSERVATION_WRITTEN", category="task", task_id=task_id, payload={
                        "path": observation_path,
                    })
                    observations.append({
                        "task_id": task_id,
                        "outcome": outcome,
                        "summary": task_result["task_summary"],
                        "observation_path": observation_path,
                        "key_artifacts": task_result["key_artifacts"],
                        "ops_path": task_result.get("ops_path"),
                        "diff_path": task_result.get("diff_path"),
                    })
                    task["status"] = outcome
                    self._emit("TASK_END", category="task", task_id=task_id, payload={
                        "outcome": outcome,
                        "summary_snippet": self._snippet(task_result.get("task_summary", ""), 200),
                    })
                    self._trace_event("TASK_COMPLETED", {"task_id": task_id, "outcome": outcome})
                    if outcome in {"failure", "needs_intervention"}:
                        status = outcome
                        self._write_task_state({
                            "user_request": user_request,
                            "plan": plan,
                            "tasks": tasks,
                            "observations": observations,
                            "status": status,
                        })
                        break
                    self._write_task_state({
                        "user_request": user_request,
                        "plan": plan,
                        "tasks": tasks,
                        "observations": observations,
                        "status": status,
                    })

                if status == "failure":
                    break
                if status == "needs_intervention":
                    continue
                if status == "running" and not progressed:
                    status = "done"
                    break

            if status in ("done", "failure"):
                self._emit("FINAL_SUMMARY_START", category="final")
                summary = self._summarize_tasks(user_request, observations, status)
                self._write_task_state({
                    "user_request": user_request,
                    "plan": plan,
                    "tasks": tasks,
                    "observations": observations,
                    "status": status,
                    "summary": summary,
                })
                report_paths = self._publish_report(user_request, summary)
                preview_lines = [line for line in (summary or "").splitlines() if line.strip()][:8]
                self._emit("FINAL_SUMMARY_DONE", category="final", payload={
                    "preview_lines": preview_lines,
                    "report_path": report_paths.get("final_report", ""),
                    "run_dir": report_paths.get("run_dir", ""),
                })
                result = {
                    "tasks": tasks,
                    "observations": observations,
                    "summary": summary,
                    "final_answer": summary,
                    "status": status,
                }
                self._emit("RUN_END", category="run", payload={
                    "status": status,
                    "run_dir": report_paths.get("run_dir", ""),
                    "final_report": report_paths.get("final_report", ""),
                    "whiteboard_report": report_paths.get("whiteboard_report", ""),
                    "latest_link": report_paths.get("latest_link", ""),
                })
                try:
                    if hasattr(self.reporter, "show_final_summary") and self.reporter.is_live():
                        self.reporter.show_final_summary(summary)
                except Exception:
                    pass
                return result
        except Exception as exc:
            self._emit("RUN_END", level="error", category="run", payload={
                "status": "error",
                "error": str(exc),
                "run_dir": str(self.run_context.run_dir) if hasattr(self, "run_context") else "",
            })
            raise
        finally:
            self.reporter.close()

    def _next_hitl_index(self) -> int:
        hitl_root = self.run_context.run_dir / "hitl"
        if not hitl_root.exists():
            return 1
        indices: List[int] = []
        for entry in hitl_root.iterdir():
            if not entry.is_dir():
                continue
            match = re.match(r"hitl_(\d+)$", entry.name)
            if match:
                indices.append(int(match.group(1)))
        return max(indices, default=0) + 1

    @staticmethod
    def _next_task_index(tasks: List[Dict[str, Any]]) -> int:
        max_id = 0
        for task in tasks:
            task_id = str(task.get("task_id", ""))
            match = re.match(r"task_(\d+)$", task_id)
            if match:
                max_id = max(max_id, int(match.group(1)))
        if max_id:
            return max_id + 1
        return len(tasks) + 1

    def _rel_run_path(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.run_context.run_dir))
        except Exception:
            return str(path)

    def _hitl_intervene(
        self,
        *,
        user_request: str,
        plan: Dict[str, Any],
        tasks: List[Dict[str, Any]],
        observations: List[Dict[str, Any]],
        log_llm: bool = False,
        plan_feedback_provider: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> Dict[str, Any]:
        hitl_index = self._next_hitl_index()
        hitl_tag = f"hitl_{hitl_index:03d}"
        hitl_dir = self.run_context.run_dir / "hitl" / hitl_tag
        hitl_dir.mkdir(parents=True, exist_ok=True)

        summary = self._summarize_tasks(user_request, observations, status="needs_intervention")
        report_paths = self._publish_report(user_request, summary)
        report_path = report_paths.get("final_report", "")
        report_text = ""
        if report_path:
            try:
                report_text = Path(report_path).read_text(encoding="utf-8")
            except Exception:
                report_text = summary or ""
        else:
            report_text = summary or ""
        interrupted_report_path = hitl_dir / "interrupted_report.md"
        interrupted_report_path.write_text(report_text or "", encoding="utf-8")

        feedback_state = {
            "user_request": user_request,
            "plan": plan,
            "tasks": tasks,
            "observations": observations,
            "status": "needs_intervention",
            "report_path": report_path,
            "report_text": report_text,
            "hitl_dir": str(hitl_dir),
            "hitl_id": hitl_tag,
        }
        feedback = ""
        used_ui_prompt = False
        if plan_feedback_provider:
            feedback = plan_feedback_provider({**feedback_state, "stage": "hitl_feedback"}) or ""
        else:
            if sys.stdin.isatty():
                if hasattr(self.reporter, "prompt_hitl_feedback") and self.reporter.is_live():
                    feedback = self.reporter.prompt_hitl_feedback(
                        report_text=report_text,
                        report_path=report_path,
                    )
                    used_ui_prompt = True
                else:
                    print("\n=== Intervention Required ===")
                    if report_path:
                        print(f"Interrupted report: {report_path}")
                    print("\n--- Interrupted Report ---")
                    print(report_text or "(none)")
                    print("\nEnter feedback to continue, or press Enter to keep paused:")
                    feedback = input("> ").strip()
            else:
                feedback = ""

        human_feedback_path = hitl_dir / "human_feedback.txt"
        human_feedback_path.write_text(feedback or "", encoding="utf-8")

        hitl_meta = {
            "hitl_id": hitl_tag,
            "hitl_dir": self._rel_run_path(hitl_dir),
            "interrupted_report": self._rel_run_path(interrupted_report_path),
            "human_feedback": self._rel_run_path(human_feedback_path),
            "report_path": report_path,
        }

        if not feedback:
            self._write_task_state({
                "user_request": user_request,
                "plan": plan,
                "tasks": tasks,
                "observations": observations,
                "status": "needs_intervention",
                "summary": summary,
                "hitl": hitl_meta,
            })
            return {
                "tasks": tasks,
                "observations": observations,
                "summary": summary,
                "final_answer": summary,
                "status": "needs_intervention",
                "plan": plan,
                "report_path": report_path,
                "hitl_dir": str(hitl_dir),
            }

        report_ref = ""
        if report_path:
            try:
                report_ref = str(Path(report_path).resolve().relative_to(workspace_root()))
            except Exception:
                report_ref = os.path.basename(report_path)

        guidance_snippet = self._snippet(feedback, 9999)
        constraint_text = f"HITL guidance ({hitl_tag}): {guidance_snippet}"
        if report_ref:
            constraint_text += f" | report: {report_ref}"

        ops = [{
            "op": "UPSERT",
            "section": "Constraints",
            "record_type": "CONSTRAINT",
            "id": f"HITL_{hitl_index:03d}",
            "text": constraint_text,
            "rationale": f"HITL intervention {hitl_tag}",
        }]
        ops_file = hitl_dir / "whiteboard_ops.json"
        ops_file.write_text(json.dumps(ops, ensure_ascii=False, indent=2), encoding="utf-8")

        validation = whiteboard_ops_validate(ops)
        before_hash = self.whiteboard.get_hash()
        if not validation.get("ok"):
            self._trace_whiteboard_update(
                task_id=hitl_tag,
                ops_path=self._rel_run_path(ops_file),
                diff_path="",
                status="ops_validation_failed",
                errors=validation.get("errors", []),
                warnings=validation.get("warnings", []),
                before_hash=before_hash,
                after_hash=None,
                repair_attempt=0,
            )
        else:
            try:
                apply_result = whiteboard_ops_apply_atomic(self.whiteboard.path, ops, hitl_tag)
            except Exception as exc:
                apply_result = {"ok": False, "errors": [str(exc)], "warnings": []}

            if apply_result.get("ok"):
                diff_path = ""
                after_hash = self.whiteboard.get_hash()
                try:
                    diff_path = persist_whiteboard_diff(
                        apply_result.get("before_text", ""),
                        apply_result.get("after_text", ""),
                        {"run_id": self.run_context.run_id, "task_id": hitl_tag, "attempt": 0},
                        root=system_root(),
                        whiteboard_path=self.whiteboard.path,
                    )["diff_path"]
                except Exception:
                    diff_path = ""
                self._trace_whiteboard_update(
                    task_id=hitl_tag,
                    ops_path=self._rel_run_path(ops_file),
                    diff_path=diff_path,
                    status="applied",
                    errors=[],
                    warnings=apply_result.get("warnings", []),
                    before_hash=apply_result.get("before_hash"),
                    after_hash=after_hash,
                    repair_attempt=0,
                )
            else:
                self._trace_whiteboard_update(
                    task_id=hitl_tag,
                    ops_path=self._rel_run_path(ops_file),
                    diff_path="",
                    status="apply_failed",
                    errors=apply_result.get("errors", []),
                    warnings=apply_result.get("warnings", []),
                    before_hash=before_hash,
                    after_hash=None,
                    repair_attempt=0,
                )

        completed_ids = [t.get("task_id") for t in tasks if t.get("status") == "success"]
        blocked_ids = [t.get("task_id") for t in tasks if t.get("status") == "needs_intervention"]
        pending_ids = [t.get("task_id") for t in tasks if t.get("status") == "pending"]
        packed_feedback = "\n".join([
            "=== Interrupted Report ===",
            report_text or "(none)",
            "",
            "=== Progress Summary ===",
            f"Completed task_ids: {', '.join([tid for tid in completed_ids if tid]) or '(none)'}",
            f"Blocked task_ids: {', '.join([tid for tid in blocked_ids if tid]) or '(none)'}",
            f"Pending task_ids: {', '.join([tid for tid in pending_ids if tid]) or '(none)'}",
            "",
            "=== Human Feedback ===",
            feedback,
            "",
            "Instruction: produce a plan ONLY for remaining work; do not repeat completed tasks unless requested.",
        ])
        packed_feedback_path = hitl_dir / "packed_feedback.txt"
        packed_feedback_path.write_text(packed_feedback, encoding="utf-8")

        new_plan = self.revise_plan(user_request, plan, packed_feedback, log_llm=log_llm)

        review_state = {
            "user_request": user_request,
            "plan": new_plan,
            "feedback_history": [{
                "round": 0,
                "feedback": packed_feedback,
                "approved": False,
                "plan_before": plan,
                "plan_after": new_plan,
                "kind": "hitl_packed",
            }],
            "approved": False,
            "round": 0,
        }

        if plan_feedback_provider is None and not sys.stdin.isatty():
            self._write_task_state({
                "user_request": user_request,
                "plan": plan,
                "tasks": tasks,
                "observations": observations,
                "status": "needs_intervention",
                "summary": summary,
                "hitl": {
                    **hitl_meta,
                    "packed_feedback": self._rel_run_path(packed_feedback_path),
                    "proposed_plan": new_plan,
                },
            })
            return {
                "tasks": tasks,
                "observations": observations,
                "summary": summary,
                "final_answer": summary,
                "status": "needs_intervention",
                "plan": plan,
                "report_path": report_path,
                "hitl_dir": str(hitl_dir),
            }

        while not review_state.get("approved"):
            plan_description = (review_state["plan"].get("plan_description") or "").strip()
            self._emit("PLAN_REVIEW_SHOW", category="plan", payload={
                "todo": review_state["plan"].get("todo", []),
                "plan_description_snippet": self._snippet(plan_description, 240),
            })
            self._emit("PLAN_REVIEW_WAIT_INPUT", category="plan")
            if plan_feedback_provider:
                feedback = plan_feedback_provider({
                    **review_state,
                    "stage": "hitl_plan_review",
                    "hitl": hitl_meta,
                })
                used_ui_prompt = False
            else:
                if hasattr(self.reporter, "prompt_plan_feedback") and self.reporter.is_live():
                    feedback = self.reporter.prompt_plan_feedback(
                        todo=review_state["plan"].get("todo", []),
                        plan_description=plan_description,
                    )
                    used_ui_prompt = True
                else:
                    print("\n=== Proposed Plan (HITL) ===")
                    print("Plan Description:")
                    print(plan_description if plan_description else "(none)")
                    print("\nTherefore, Here is a proposed plan:")
                    for i, item in enumerate(review_state["plan"].get("todo", []), start=1):
                        print(f"{i}. {item}")
                    print("\nEnter feedback to revise, or type 'yes' to approve:")
                    feedback = input("> ").strip()
                    used_ui_prompt = False
            if not feedback:
                if plan_feedback_provider:
                    raise ValueError("HITL plan_review feedback cannot be empty")
                if not used_ui_prompt:
                    print("Empty input. Please enter feedback to revise, or type 'yes' to approve.")
                self._emit("PLAN_REVIEW_WAIT_INPUT", category="plan", payload={"error": "empty_input"})
                continue
            review_state = self.apply_plan_feedback(review_state, feedback, log_llm=log_llm)

        new_plan = review_state["plan"]
        revised_plan_path = hitl_dir / "revised_plan.json"
        revised_plan_path.write_text(json.dumps(new_plan, ensure_ascii=False, indent=2), encoding="utf-8")

        for task in tasks:
            if task.get("status") in {"pending", "running"}:
                task["status"] = "skipped_deprecated"

        next_index = self._next_task_index(tasks)
        for item in new_plan.get("todo", []) or []:
            tasks.append({
                "task_id": f"task_{next_index:02d}",
                "goal": str(item),
                "status": "pending",
            })
            next_index += 1

        status = "running"
        self._write_task_state({
            "user_request": user_request,
            "plan": new_plan,
            "tasks": tasks,
            "observations": observations,
            "status": status,
            "hitl": {
                **hitl_meta,
                "packed_feedback": self._rel_run_path(packed_feedback_path),
                "revised_plan": self._rel_run_path(revised_plan_path),
                "whiteboard_ops": self._rel_run_path(ops_file),
            },
        })

        return {
            "status": status,
            "plan": new_plan,
            "tasks": tasks,
            "observations": observations,
            "summary": summary,
            "report_path": report_path,
            "hitl_dir": str(hitl_dir),
        }

    def _summarize_and_update_whiteboard(
        self,
        summarizer: TaskSummarizer,
        *,
        task_id: str,
        task_goal: str,
        step_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        finish_reason = step_result.get("finish_reason", "")
        local_observations = step_result.get("local_observations", [])
        whiteboard_text = self.whiteboard.read()
        error = None
        self._emit("TASK_SUMMARIZE_START", category="summary", task_id=task_id, payload={
            "task_goal": self._snippet(task_goal, 200),
        })
        result = self._run_task_summarizer(
            summarizer,
            task_id=task_id,
            task_goal=task_goal,
            finish_reason=finish_reason,
            local_observations=local_observations,
            whiteboard_text=whiteboard_text,
            error=error,
            use_repair_prompt=False,
        )
        if result is None:
            fallback_before = self.whiteboard.get_hash()
            fallback_summary = "Task summarizer failed; journal entry added for manual review."
            self._append_journal_entry(
                task_id=task_id,
                outcome="needs_intervention",
                summary=fallback_summary,
                key_artifacts=[],
            )
            fallback_after = self.whiteboard.get_hash()
            self._trace_whiteboard_update(
                task_id=task_id,
                ops_path="",
                diff_path="",
                status="summary_failed",
                errors=["summary_failed"],
                warnings=[],
                before_hash=fallback_before,
                after_hash=fallback_after,
                repair_attempt=0,
            )
            self._emit("TASK_SUMMARY", category="summary", task_id=task_id, payload={
                "outcome": "needs_intervention",
                "summary_snippet": self._snippet(fallback_summary, 200),
                "artifacts": [],
                "ops_counts": self._count_ops([]),
            })
            self._emit("WHITEBOARD_APPLY_FAIL", level="warning", category="summary", task_id=task_id, payload={
                "error": "summary_failed",
            })
            return {
                "task_outcome": "needs_intervention",
                "task_summary": fallback_summary,
                "key_artifacts": [],
                "whiteboard_ops": [],
                "ops_failed": True,
                "summary_failed": True,
            }

        attempt = 0
        self._emit("TASK_SUMMARY", category="summary", task_id=task_id, payload={
            "outcome": result.get("task_outcome", ""),
            "summary_snippet": self._snippet(result.get("task_summary", ""), 200),
            "artifacts": [item.get("path", "") for item in result.get("key_artifacts", [])][:5],
            "ops_counts": self._count_ops(result.get("whiteboard_ops", [])),
        })
        while attempt <= self.patch_repair_attempts:
            ops = [op for op in result.get("whiteboard_ops", []) if op.get("section") != "Goal"]
            dropped_goal_ops = len(result.get("whiteboard_ops", [])) - len(ops)
            ops_path = whiteboard_ops_persist(
                ops,
                {"run_id": self.run_context.run_id, "task_id": task_id, "attempt": attempt},
                root=system_root(),
            )["ops_path"]
            before_hash = self.whiteboard.get_hash()
            self._emit("WHITEBOARD_APPLY_START", category="summary", task_id=task_id, payload={
                "ops_count": len(ops),
                "attempt": attempt,
            })
            validation = whiteboard_ops_validate(ops)
            if not validation.get("ok"):
                error = "; ".join(validation.get("errors", []))
                self._trace_whiteboard_update(
                    task_id=task_id,
                    ops_path=ops_path,
                    diff_path="",
                    status="ops_validation_failed",
                    errors=validation.get("errors", []),
                    warnings=validation.get("warnings", []) + (["Dropped Goal ops"] if dropped_goal_ops else []),
                    before_hash=before_hash,
                    after_hash=None,
                    repair_attempt=attempt,
                )
                self._emit("WHITEBOARD_APPLY_FAIL", level="warning", category="summary", task_id=task_id, payload={
                    "error": error,
                    "warnings": validation.get("warnings", []) + (["Dropped Goal ops"] if dropped_goal_ops else []),
                    "attempt": attempt,
                })
                attempt += 1
                whiteboard_text = self.whiteboard.read()
                result = self._run_task_summarizer(
                    summarizer,
                    task_id=task_id,
                    task_goal=task_goal,
                    finish_reason=finish_reason,
                    local_observations=local_observations,
                    whiteboard_text=whiteboard_text,
                    error=error,
                    use_repair_prompt=True,
                ) or result
                continue

            try:
                apply_result = whiteboard_ops_apply_atomic(self.whiteboard.path, ops, task_id)
            except Exception as exc:
                apply_result = {"ok": False, "errors": [str(exc)], "warnings": []}

            if apply_result.get("ok"):
                diff_path = ""
                final_text = apply_result.get("after_text", "")
                journal_text = append_task_journal_entry(
                    final_text,
                    task_id=task_id,
                    outcome=result.get("task_outcome", ""),
                    summary=result.get("task_summary", ""),
                    artifacts=[item.get("path", "") for item in result.get("key_artifacts", [])],
                )
                if journal_text != final_text:
                    self.whiteboard.path.write_text(journal_text, encoding="utf-8")
                after_hash = self.whiteboard.get_hash()
                try:
                    diff_path = persist_whiteboard_diff(
                        apply_result.get("before_text", ""),
                        journal_text,
                        {"run_id": self.run_context.run_id, "task_id": task_id, "attempt": attempt},
                        root=system_root(),
                        whiteboard_path=self.whiteboard.path,
                    )["diff_path"]
                except Exception:
                    diff_path = ""
                self._trace_whiteboard_update(
                    task_id=task_id,
                    ops_path=ops_path,
                    diff_path=diff_path,
                    status="applied",
                    errors=[],
                    warnings=apply_result.get("warnings", []) + (["Dropped Goal ops"] if dropped_goal_ops else []),
                    before_hash=apply_result.get("before_hash"),
                    after_hash=after_hash,
                    repair_attempt=attempt,
                )
                self._emit("WHITEBOARD_APPLY_OK", category="summary", task_id=task_id, payload={
                    "counts": self._count_ops(ops),
                    "warnings": apply_result.get("warnings", []) + (["Dropped Goal ops"] if dropped_goal_ops else []),
                    "attempt": attempt,
                })
                result["ops_path"] = ops_path
                result["diff_path"] = diff_path
                self._update_artifact_log(result.get("key_artifacts", []))
                return result

            error = "; ".join(apply_result.get("errors", []))
            self._trace_whiteboard_update(
                task_id=task_id,
                ops_path=ops_path,
                diff_path="",
                status="apply_failed",
                errors=apply_result.get("errors", []),
                warnings=apply_result.get("warnings", []) + (["Dropped Goal ops"] if dropped_goal_ops else []),
                before_hash=before_hash,
                after_hash=None,
                repair_attempt=attempt,
            )
            self._emit("WHITEBOARD_APPLY_FAIL", level="warning", category="summary", task_id=task_id, payload={
                "error": error,
                "warnings": apply_result.get("warnings", []) + (["Dropped Goal ops"] if dropped_goal_ops else []),
                "attempt": attempt,
            })
            attempt += 1
            whiteboard_text = self.whiteboard.read()
            result = self._run_task_summarizer(
                summarizer,
                task_id=task_id,
                task_goal=task_goal,
                finish_reason=finish_reason,
                local_observations=local_observations,
                whiteboard_text=whiteboard_text,
                error=error,
                use_repair_prompt=True,
            ) or result

        fallback_before = self.whiteboard.get_hash()
        self._append_journal_entry(
            task_id=task_id,
            outcome=result["task_outcome"],
            summary=result["task_summary"],
            key_artifacts=result["key_artifacts"],
        )
        fallback_after = self.whiteboard.get_hash()
        self._trace_whiteboard_update(
            task_id=task_id,
            ops_path="",
            diff_path="",
            status="ops_failed",
            errors=[error or "ops_failed"],
            warnings=[],
            before_hash=fallback_before,
            after_hash=fallback_after,
            repair_attempt=attempt,
        )
        self._emit("WHITEBOARD_APPLY_FAIL", level="warning", category="summary", task_id=task_id, payload={
            "error": error or "ops_failed",
            "attempt": attempt,
        })
        result["ops_failed"] = True
        self._update_artifact_log(result.get("key_artifacts", []))
        return result

    def _initialize_whiteboard_goal(self, user_request: str) -> None:
        try:
            text = self.whiteboard.read()
        except Exception:
            return
        lines = text.splitlines()
        bounds = None
        try:
            bounds = _section_bounds(lines)
        except Exception:
            bounds = None
        if not bounds or "Goal" not in bounds:
            return
        start, end = bounds["Goal"]
        body = [line for line in lines[start + 1:end] if line.strip()]
        if any(line.strip() not in {"- (empty)", "- (none)"} for line in body):
            return
        goal_text = " ".join(user_request.strip().split())
        if not goal_text:
            return
        lines[start + 1:end] = [f"- {goal_text}"]
        updated = "\n".join(lines) + ("\n" if text.endswith("\n") else "")
        self.whiteboard.path.write_text(updated, encoding="utf-8")

    def _run_task_summarizer(
        self,
        summarizer: TaskSummarizer,
        *,
        task_id: str,
        task_goal: str,
        finish_reason: str,
        local_observations: List[Dict[str, Any]],
        whiteboard_text: str,
        error: Optional[str],
        use_repair_prompt: bool,
    ) -> Optional[Dict[str, Any]]:
        last_error = error
        for attempt in range(self.summary_repair_attempts + 1):
            self._emit("LLM_CALL_START", category="llm", task_id=task_id, payload={
                "kind": "task_summary",
                "attempt": attempt,
            })
            t0 = time.perf_counter()
            try:
                result = summarizer.run(
                    task_id=task_id,
                    task_goal=task_goal,
                    finish_reason=finish_reason,
                    local_observations=local_observations,
                    whiteboard_text=whiteboard_text,
                    whiteboard_path=str(self.whiteboard.path.relative_to(system_root().parent)),
                    error=last_error,
                    use_repair_prompt=use_repair_prompt or attempt > 0,
                )
            except Exception as exc:
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                payload = {
                    "kind": "task_summary",
                    "attempt": attempt,
                    "elapsed_ms": elapsed_ms,
                    "error": str(exc),
                }
                if self._ui_debug():
                    payload["raw_snippet"] = self._snippet(getattr(summarizer, "last_raw", ""), 240)
                self._emit("LLM_CALL_END", level="warning", category="llm", task_id=task_id, payload=payload)
                last_error = str(exc)
                continue
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            payload = {
                "kind": "task_summary",
                "attempt": attempt,
                "elapsed_ms": elapsed_ms,
            }
            if self._ui_debug():
                payload["raw_snippet"] = self._snippet(getattr(summarizer, "last_raw", ""), 240)
            self._emit("LLM_CALL_END", category="llm", task_id=task_id, payload=payload)
            return result
        return None

    def _update_artifact_log(self, key_artifacts: List[Dict[str, Any]]) -> None:
        if not key_artifacts:
            return
        entries: List[Dict[str, str]] = []
        for item in key_artifacts:
            path = (item.get("path") or "").strip()
            if not path:
                continue
            path_parts = Path(path).parts
            if ".catmaster" in path_parts:
                continue
            if path.startswith("/"):
                try:
                    resolved = Path(path).resolve()
                except Exception:
                    continue
                try:
                    resolved.relative_to(workspace_root())
                except Exception:
                    continue
                try:
                    resolved.relative_to(system_root())
                    continue
                except Exception:
                    pass
            entries.append({
                "path": path,
                "description": (item.get("description") or "").strip(),
                "type": ArtifactLog.infer_type(path),
            })
        if entries:
            self.artifact_log.update(entries)

    def _trace_whiteboard_update(
        self,
        *,
        task_id: str,
        ops_path: str,
        diff_path: str,
        status: str,
        errors: List[str],
        warnings: List[str],
        before_hash: Optional[str],
        after_hash: Optional[str],
        repair_attempt: int,
    ) -> None:
        self.trace_store.append_patch({
            "task_id": task_id,
            "role": "manager",
            "whiteboard_path": str(self.whiteboard.path),
            "ops_path": ops_path,
            "diff_path": diff_path,
            "status": status,
            "errors": errors,
            "warnings": warnings,
            "before_hash": before_hash,
            "after_hash": after_hash,
            "repair_attempt": repair_attempt,
        })

    def _append_journal_entry(
        self,
        *,
        task_id: str,
        outcome: str,
        summary: str,
        key_artifacts: List[Dict[str, Any]],
    ) -> None:
        text = self.whiteboard.read()
        updated = append_task_journal_entry(
            text,
            task_id=task_id,
            outcome=outcome,
            summary=summary,
            artifacts=[item.get("path", "") for item in key_artifacts if item.get("path")],
        )
        self.whiteboard.path.write_text(updated, encoding="utf-8")

    def _compile_tasks(self, todo: List[str]) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        for idx, item in enumerate(todo or [], start=1):
            tasks.append({
                "task_id": f"task_{idx:02d}",
                "goal": str(item),
                "status": "pending",
            })
        return tasks

    def _write_observation(
        self,
        *,
        task_id: str,
        outcome: str,
        summary: str,
        key_artifacts: List[Dict[str, Any]],
    ) -> str:
        obs_dir = self.run_context.run_dir / "observations"
        obs_dir.mkdir(parents=True, exist_ok=True)
        index = len(list(obs_dir.glob("obs_*.md"))) + 1
        fname = f"obs_{index:03d}_{task_id}.md"
        path = obs_dir / fname
        lines = [
            f"# Observation {index}",
            f"- Task: {task_id}",
            f"- Outcome: {outcome}",
            "",
            "## Summary",
            summary or "",
            "",
            "## Key Artifacts",
        ]
        if key_artifacts:
            for item in key_artifacts:
                kpath = item.get("path", "")
                desc = item.get("description", "")
                kind = item.get("kind", "")
                lines.append(f"- {kpath} ({kind}): {desc}")
        else:
            lines.append("- (none)")
        path.write_text("\n".join(lines), encoding="utf-8")
        return str(path.relative_to(self.run_context.run_dir))

    def _write_task_state(self, payload: Dict[str, Any]) -> None:
        path = self.run_context.run_dir / "task_state.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _summarize_tasks(self, user_request: str, observations: List[Dict[str, Any]], status: str) -> str:
        fallback = self._summarize_tasks_fallback(user_request, observations)
        try:
            whiteboard_excerpt = self.whiteboard.read_sections(
                ["Goal", "Key Facts", "Key Files", "Constraints"],
                max_chars=4000,
            )
        except Exception:
            whiteboard_excerpt = ""
        artifacts = self._artifact_log_excerpt(limit=200)
        try:
            messages = self.summary_prompt.format_messages(
                user_request=user_request,
                observations=json.dumps(observations, ensure_ascii=False),
                whiteboard_excerpt=whiteboard_excerpt,
                artifacts=json.dumps(artifacts, ensure_ascii=False),
                status=status,
            )
            self._emit("LLM_CALL_START", category="llm", payload={"kind": "final_summary"})
            t0 = time.perf_counter()
            resp = self.summary_llm.invoke(messages)
            raw = (resp.content or "").strip()
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            self._write_llm_log("final_summary_prompt", messages=self._messages_to_dict(messages))
            self._write_llm_log("final_summary_response", content=raw)
            payload = {"kind": "final_summary", "elapsed_ms": elapsed_ms}
            if self._ui_debug():
                payload["raw_snippet"] = self._snippet(raw, 240)
            self._emit("LLM_CALL_END", category="llm", payload=payload)
            return raw if raw else fallback
        except Exception:
            self._emit("LLM_CALL_END", level="warning", category="llm", payload={
                "kind": "final_summary",
                "error": "final_summary_failed",
            })
            return fallback

    @staticmethod
    def _summarize_tasks_fallback(user_request: str, observations: List[Dict[str, Any]]) -> str:
        lines = [f"Request: {user_request}"]
        for obs in observations:
            lines.append(f"- {obs.get('task_id')}: outcome={obs.get('outcome')} summary={obs.get('summary')}")
        return "\n".join(lines)

    def _artifact_log_excerpt(self, limit: int = 200) -> List[Dict[str, Any]]:
        entries = self.artifact_log.load()
        entries.sort(key=lambda e: e.get("updated_time", ""), reverse=True)
        return entries[:limit]

    @staticmethod
    def _count_ops(ops: List[Dict[str, Any]]) -> Dict[str, int]:
        counts = {
            "FACT": 0,
            "FILE": 0,
            "CONSTRAINT": 0,
            "QUESTION": 0,
            "DEPRECATE": 0,
        }
        for op in ops or []:
            if not isinstance(op, dict):
                continue
            op_type = str(op.get("op", "")).upper()
            if op_type == "DEPRECATE":
                counts["DEPRECATE"] += 1
                continue
            record_type = str(op.get("record_type", "")).upper()
            if record_type in {"FACT", "FILE", "CONSTRAINT"}:
                counts[record_type] += 1
                continue
            section = str(op.get("section", "")).strip().lower()
            if section in {"open questions", "open question", "questions", "question"}:
                counts["QUESTION"] += 1
        return counts

    def _publish_report(self, user_request: str, final_answer: str) -> Dict[str, str]:
        reports_dir = workspace_root() / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        latest_link = reports_dir / "latest_run"
        target = self.run_context.run_dir
        try:
            if latest_link.is_symlink() or latest_link.exists():
                latest_link.unlink()
        except Exception:
            pass
        rel_target = os.path.relpath(target, reports_dir)
        try:
            latest_link.symlink_to(rel_target)
        except Exception:
            # Fallback: create a text pointer if symlink fails
            latest_link.write_text(str(target), encoding="utf-8")

        final_report = reports_dir / "FINAL_REPORT.md"
        final_report.write_text(
            "\n".join([
                "# Final Report",
                "",
                "## User Query",
                user_request,
                "",
                "## Final Answer",
                final_answer,
                "",
            ]),
            encoding="utf-8",
        )

        whiteboard_src = self.whiteboard.path
        whiteboard_dst = reports_dir / "WHITEBOARD.md"
        try:
            shutil.copy2(whiteboard_src, whiteboard_dst)
        except Exception:
            # Best-effort copy
            if whiteboard_src.exists():
                whiteboard_dst.write_text(whiteboard_src.read_text(encoding="utf-8"), encoding="utf-8")
        return {
            "run_dir": str(self.run_context.run_dir),
            "final_report": str(final_report),
            "whiteboard_report": str(whiteboard_dst),
            "latest_link": str(latest_link),
        }

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

    @staticmethod
    def _parse_json_response(content: str) -> Dict[str, Any]:
        match = re.search(r"```json\s*(.*?)\s*```", content, re.IGNORECASE | re.DOTALL)
        if not match:
            raise ValueError("Expected JSON wrapped in ```json ... ```")
        json_text = match.group(1).strip()
        return json.loads(json_text)

    @staticmethod
    def _is_plan_approved(feedback: str) -> bool:
        if not isinstance(feedback, str):
            return False
        normalized = feedback.strip().lower()
        return normalized in {"yes", "y", "approve", "approved", "ok", "okay"}

    def _normalize_plan(self, data: Dict[str, Any], user_request: str) -> Dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("Plan must be a JSON object")
        normalized = dict(data)
        todo = normalized.get("todo")
        if not isinstance(todo, list) or not todo:
            raise ValueError("Plan.todo must be a non-empty list")
        plan_description = normalized.get("plan_description")
        if plan_description is None:
            plan_description = ""
        if not isinstance(plan_description, str):
            raise ValueError("Plan.plan_description must be a string")
        normalized["todo"] = todo
        normalized.pop("next_step", None)
        normalized["plan_description"] = plan_description
        return normalized

    def _write_llm_log(self, event: str, *, content: Optional[str] = None, messages: Optional[List[Dict[str, Any]]] = None, step: Optional[int] = None, **extra: Any) -> None:
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
        if extra:
            record.update(extra)
        with self.llm_log_file.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


__all__ = ["Orchestrator"]
