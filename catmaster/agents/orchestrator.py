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
        print(logo_str)
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

    def _trace_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        record = {"event": event, "payload": payload or {}}
        self.trace_store.append_event(record)

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
                normalized = self._normalize_plan(data, user_request)
            except Exception as exc:
                last_error = str(exc)
                messages = self.plan_repair_prompt.format_messages(
                    user_request=user_request,
                    tools=tools,
                    error=last_error,
                    raw=raw,
                )
                continue
            self._trace_event("PLAN_CREATED", {
                "todo": normalized.get("todo", []),
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
        if self.resuming:
            if initial_plan is not None:
                raise ValueError("Cannot provide initial_plan when resuming")
            if plan_review:
                raise ValueError("plan_review is not supported when resuming")
        if self.resuming:
            state = self._load_task_state()
            user_request = state["user_request"]
            plan = self._normalize_plan(state["plan"], user_request)
            tasks = state["tasks"]
            observations = state["observations"]
            status = state.get("status", "running")
            if status in {"done", "failure", "needs_intervention"}:
                raise ValueError(f"Cannot resume; run already ended with status {status}")
            self._initialize_whiteboard_goal(user_request)
        else:
            plan = self._normalize_plan(initial_plan, user_request) if initial_plan else self.plan(user_request)
            self._initialize_whiteboard_goal(user_request)
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
                    if plan_feedback_provider:
                        feedback = plan_feedback_provider(review_state)
                    else:
                        print("\n=== Proposed Plan ===")
                        reasoning = (review_state["plan"].get("reasoning") or "").strip()
                        print("Plan Description:")
                        print(reasoning if reasoning else "(none)")
                        print("\nTherefore, Here is a proposed plan:")
                        for i, item in enumerate(review_state["plan"].get("todo", []), start=1):
                            print(f"{i}. {item}")
                        print("\nEnter feedback to revise, or type 'yes' to approve:")
                        feedback = input("> ").strip()
                    if not feedback:
                        if plan_feedback_provider:
                            raise ValueError("plan_review feedback cannot be empty")
                        print("Empty input. Please enter feedback to revise, or type 'yes' to approve.")
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

        for task in tasks:
            if task.get("status") != "pending":
                continue
            task_id = task["task_id"]
            task_goal = task["goal"]
            self._trace_event("TASK_STARTED", {"task_id": task_id, "goal": task_goal})
            context_pack = self.context_builder.build(task_goal, role="task_runner", policy=ContextPackPolicy())
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

        if status == "running":
            status = "done"

        summary = self._summarize_tasks(user_request, observations, status)
        self._write_task_state({
            "user_request": user_request,
            "plan": plan,
            "tasks": tasks,
            "observations": observations,
            "status": status,
            "summary": summary,
        })
        self._publish_report(user_request, summary)
        return {
            "tasks": tasks,
            "observations": observations,
            "summary": summary,
            "final_answer": summary,
            "status": status,
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
            return {
                "task_outcome": "needs_intervention",
                "task_summary": fallback_summary,
                "key_artifacts": [],
                "whiteboard_ops": [],
                "ops_failed": True,
                "summary_failed": True,
            }

        attempt = 0
        while attempt <= self.patch_repair_attempts:
            ops = [op for op in result.get("whiteboard_ops", []) if op.get("section") != "Goal"]
            dropped_goal_ops = len(result.get("whiteboard_ops", [])) - len(ops)
            ops_path = whiteboard_ops_persist(
                ops,
                {"run_id": self.run_context.run_id, "task_id": task_id, "attempt": attempt},
                root=system_root(),
            )["ops_path"]
            before_hash = self.whiteboard.get_hash()
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
            try:
                return summarizer.run(
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
                last_error = str(exc)
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
            resp = self.summary_llm.invoke(messages)
            raw = (resp.content or "").strip()
            self._write_llm_log("final_summary_prompt", messages=self._messages_to_dict(messages))
            self._write_llm_log("final_summary_response", content=raw)
            return raw if raw else fallback
        except Exception:
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

    def _publish_report(self, user_request: str, final_answer: str) -> None:
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
        reasoning = normalized.get("reasoning")
        if reasoning is None:
            reasoning = ""
        if not isinstance(reasoning, str):
            raise ValueError("Plan.reasoning must be a string")
        normalized["todo"] = todo
        normalized.pop("next_step", None)
        normalized["reasoning"] = reasoning
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
