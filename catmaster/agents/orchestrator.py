#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task-based orchestrator with file-based memory and unified tracing.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import logging

from catmaster.tools.registry import get_tool_registry
from catmaster.tools.base import project_space_root, system_root, workspace_root
from catmaster.runtime import (
    RunContext,
    ToolExecutor,
    ArtifactStore,
    MemoryStore,
    TraceStore,
    CheckpointStore,
    RunControl,
    ContextPackBuilder,
    ContextPackPolicy,
    write_usage_summary,
)
from catmaster.agents.logo import logo_str
from catmaster.agents.tool_calling_stepper import ToolCallingTaskStepper
from catmaster.agents.plan_control_tools import PLAN_CONTROL_TOOL_NAMES, get_plan_control_tool_schemas
from catmaster.agents.proposal_control_tools import (
    PROPOSAL_CONTROL_TOOL_NAMES,
    get_proposal_control_tool_schemas,
)
from catmaster.agents.director_control_tools import (
    DIRECTOR_CONTROL_TOOL_NAMES,
    get_director_control_tool_schemas,
)
from catmaster.agents.llm_utils import llm_text
from catmaster.llm.driver import ToolCallingDriver
from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_llm_bundle
from catmaster.runtime.conversation_state import message_item
from catmaster.runtime.tool_policy import ToolPolicy
from catmaster.runtime.tool_backend import ToolBackend
from catmaster.runtime.local_tool_backend import LocalToolBackend
from catmaster.ui import Reporter, NullReporter, make_event
from catmaster.agents.orchestrator_prompts import (
    build_plan_prompt,
    build_plan_feedback_prompt,
    build_task_step_prompt,
    build_task_step_repair_prompt,
    build_summary_prompt,
    build_proposal_prompt,
    build_proposal_feedback_prompt,
    build_director_prompt,
    build_memory_patch_prompt,
    build_memory_patch_repair_prompt,
)

_PLANNER_TOOL_ALLOWLIST = [
    "bash_exec",
]
_PROPOSAL_TOOL_ALLOWLIST = [
    "bash_exec",
]
SUPPORTED_LANES = {"fast", "standard"}


class MemoryPatchApplyError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        event_path: str = "",
        patch_path: str = "",
        check_error: str = "",
    ) -> None:
        super().__init__(message)
        self.event_path = event_path
        self.patch_path = patch_path
        self.check_error = check_error


class Orchestrator:
    def __init__(
        self,
        llm: Optional[Any] = None,
        max_steps: int = 200,
        *,
        summary_llm: Optional[Any] = None,
        llm_profile: Optional[LLMProfile] = None,
        llm_log_path: Optional[str] = None,
        log_llm_console: bool = True,
        reporter: Optional[Reporter] = None,
        run_context: Optional[RunContext] = None,
        workspace: Optional[str] = None,
        run_dir: Optional[str] = None,
        resume: bool = False,
        resume_dir: Optional[str] = None,
        project_id: Optional[str] = None,
        run_id: Optional[str] = None,
        tool_executor: Optional[ToolExecutor] = None,
        max_tool_attempts: int = 3,
        max_plan_steps : int = 50,
        patch_repair_attempts: int = 1,
        summary_repair_attempts: int = 1,
        tool_driver: Optional[ToolCallingDriver] = None,
        tool_policy: Optional[ToolPolicy] = None,
        tool_policy_path: Optional[str] = None,
        tool_backend: Optional[ToolBackend] = None,
        run_control: Optional[RunControl] = None,
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
        self.llm_profile = llm_profile
        self._llm_provider: Optional[str] = None
        self._llm_base_url: Optional[str] = None
        self._tool_driver_kind: Optional[str] = None
        self._supports_builtin_tools = True
        if llm is None:
            profile = llm_profile or LLMProfile.from_env_or_file()
            bundle = build_llm_bundle(profile)
            llm = bundle.llm
            summary_llm = summary_llm or bundle.summary_llm
            if tool_driver is None:
                tool_driver = bundle.tool_driver
            self.llm_profile = profile
            self._llm_provider = bundle.provider
            self._llm_base_url = profile.main.base_url
            self._tool_driver_kind = profile.main.tool_calling.driver
            self._supports_builtin_tools = bool(profile.main.tool_calling.supports_builtin_tools)
        elif llm_profile is not None:
            self._llm_provider = llm_profile.main.provider
            self._llm_base_url = llm_profile.main.base_url
            self._tool_driver_kind = llm_profile.main.tool_calling.driver
            self._supports_builtin_tools = bool(llm_profile.main.tool_calling.supports_builtin_tools)
        if self._tool_driver_kind is None and tool_driver is not None:
            self._tool_driver_kind = tool_driver.__class__.__name__
        if llm is None:
            raise ValueError("llm must be provided or resolvable from llm_profile/env")
        self.llm = llm
        self.summary_llm = summary_llm or llm
        self.max_steps = max_steps
        self.max_plan_steps = max_plan_steps
        self.patch_repair_attempts = patch_repair_attempts
        self.summary_repair_attempts = summary_repair_attempts
        self.registry = get_tool_registry()
        self.log_llm_console = log_llm_console
        self.resuming = False
        self.workspace = Path(workspace).expanduser().resolve() if workspace else None

        if run_context:
            self.run_context = run_context
            self.workspace = run_context.workspace
        else:
            if run_dir and (resume or resume_dir):
                raise ValueError("run_dir is mutually exclusive with resume/resume_dir")
            resolved_run_dir = Path(run_dir).expanduser().resolve() if run_dir else None
            resolved_resume = self._resolve_resume_run_dir(resume_dir, resume)
            if resolved_resume:
                self.run_context = RunContext.load(resolved_resume)
                self.workspace = self.run_context.workspace
                self.resuming = True
            else:
                self.run_context = RunContext.create(
                    workspace=self.workspace,
                    run_dir=resolved_run_dir,
                    project_id=project_id,
                    run_id=run_id,
                    model_name=self._resolve_model_name(),
                    provider=self._llm_provider,
                    base_url=self._llm_base_url,
                    driver_kind=self._tool_driver_kind,
                )
                self.resuming = False

        self.trace_store = TraceStore(self.run_context.run_dir)
        self.checkpoint_store = CheckpointStore(self.run_context.run_dir)
        self.run_control = run_control or RunControl(run_id=self.run_context.run_id)
        if not self.run_control.run_id:
            self.run_control.run_id = self.run_context.run_id
        self.tool_executor = tool_executor or ToolExecutor(self.registry, max_attempts=max_tool_attempts)
        self.artifact_store = ArtifactStore(self.run_context.run_dir)
        policy_path = self._resolve_tool_policy_path(tool_policy_path)
        self.tool_policy = tool_policy or ToolPolicy.from_file(policy_path)
        self.tool_backend = tool_backend or LocalToolBackend(
            registry=self.registry,
            tool_executor=self.tool_executor,
            artifact_store=self.artifact_store,
            trace_store=self.trace_store,
            workspace=self.run_context.workspace,
        )

        self.memory_store = MemoryStore.create_default(workspace=self.run_context.workspace)
        self.memory_store.ensure_exists()
        self.context_builder = ContextPackBuilder(self.memory_store)

        default_log = self.run_context.run_dir / "llm.jsonl"
        self.llm_log_file = Path(llm_log_path).expanduser().resolve() if llm_log_path else default_log
        self.llm_log_file.parent.mkdir(parents=True, exist_ok=True)

        self.plan_prompt = build_plan_prompt()
        self.plan_feedback_prompt = build_plan_feedback_prompt()
        self.proposal_prompt = build_proposal_prompt()
        self.proposal_feedback_prompt = build_proposal_feedback_prompt()
        self.director_prompt = build_director_prompt()
        self.task_step_prompt = build_task_step_prompt()
        self.task_step_repair_prompt = build_task_step_repair_prompt()
        self.memory_patch_prompt = build_memory_patch_prompt()
        self.memory_patch_repair_prompt = build_memory_patch_repair_prompt()
        self.summary_prompt = build_summary_prompt()
        self.tool_driver = tool_driver
        if self.tool_driver is None:
            try:
                from catmaster.llm.openai_responses_driver import OpenAIResponsesDriver
            except Exception as exc:
                raise ImportError(
                    "Tool calling is enabled but OpenAI driver is unavailable. Install `openai` or provide tool_driver."
                ) from exc
            self.tool_driver = OpenAIResponsesDriver(model=self._resolve_model_name())
            if self._tool_driver_kind is None:
                self._tool_driver_kind = "openai_responses"
                if self._llm_provider is None:
                    self._llm_provider = "openai"
        self._emit("RUN_INIT_DONE", payload={
            "run_id": self.run_context.run_id,
            "run_dir": str(self.run_context.run_dir),
            "model_name": self._resolve_model_name(),
            "model_label": self._resolve_model_label(),
            "provider": self._llm_provider or "",
            "driver_kind": self._tool_driver_kind or "",
            "base_url": self._llm_base_url or "",
            "prompt_cache_retention": self._prompt_cache_retention() or "",
            "proposal_browse_tools_enabled": self._proposal_browse_tools_enabled(),
            "resuming": self.resuming,
            "llm_log_path": str(self.llm_log_file),
            "trace_paths": {
                "event_trace": str(self.run_context.run_dir / "event_trace.jsonl"),
                "tool_trace": str(self.run_context.run_dir / "tool_trace.jsonl"),
                "patch_trace": str(self.run_context.run_dir / "patch_trace.jsonl"),
                "task_state": str(self.run_context.run_dir / "task_state.json"),
                "memory_index": str(self.memory_store.index_path),
                "memory_events": str(self.memory_store.events_path),
            },
        })
        self._interrupt_context_note = ""

    def _resolve_resume_run_dir(self, resume_dir: Optional[str], resume: bool) -> Optional[Path]:
        if not resume and not resume_dir:
            return None
        base = Path(resume_dir).expanduser().resolve() if resume_dir else project_space_root(self.workspace)
        if (base / "meta.json").exists():
            return base
        sys_root = base if base.name == "metadata" else (base / "metadata")
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
        if "lane" not in data:
            # Legacy runs without lane default to standard.
            data["lane"] = "standard"
        for key in ("user_request", "tasks", "observations", "status"):
            if key not in data:
                raise ValueError(f"task_state.json missing required key: {key}")
        lane = data.get("lane")
        if lane not in SUPPORTED_LANES:
            raise ValueError(f"task_state.json has invalid lane: {lane}")
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
        return self.registry.get_tool_descriptions_for_llm(
            allowlist=self._visible_function_tool_names()
        )

    def _tool_schema_short(self) -> str:
        return self.registry.get_short_tool_descriptions_for_llm(
            allowlist=self._visible_function_tool_names()
        )

    def _planner_tool_schema(self) -> str:
        visible = set(self._visible_function_tool_names())
        allowlist = [name for name in _PLANNER_TOOL_ALLOWLIST if name in visible]
        return self.registry.get_tool_descriptions_for_llm(allowlist=allowlist)

    def _planner_function_tools(self) -> list[dict]:
        tools = self._filtered_function_tools()
        return [
            tool for tool in tools
            if tool.get("name") in _PLANNER_TOOL_ALLOWLIST
        ]

    def _proposal_function_tools(self) -> list[dict]:
        if not self._proposal_browse_tools_enabled():
            return []
        tools = self._filtered_function_tools()
        return [
            tool for tool in tools
            if tool.get("name") in _PROPOSAL_TOOL_ALLOWLIST
        ]

    @staticmethod
    def _resolve_tool_policy_path(tool_policy_path: Optional[str]) -> Path:
        if tool_policy_path:
            return Path(tool_policy_path).expanduser()
        cwd_default = Path("configs/tool_policy.yaml")
        if cwd_default.exists():
            return cwd_default
        repo_default = Path(__file__).resolve().parents[2] / "configs" / "tool_policy.yaml"
        return repo_default

    def _filtered_function_tools(self) -> list[dict]:
        function_tools = list(self.tool_backend.list_function_tools() or [])
        policy_filter = getattr(self.tool_policy, "filter_function_tools", None)
        if callable(policy_filter):
            try:
                return list(policy_filter(function_tools))
            except Exception:
                pass

        allowed = getattr(self.tool_policy, "allowed_tools", None)
        denied = set(getattr(self.tool_policy, "denied_tools", None) or [])
        filtered: list[dict] = []
        for tool in function_tools:
            name = str(tool.get("name") or "").strip()
            if not name:
                continue
            if allowed is not None and name not in allowed:
                continue
            if name in denied:
                continue
            filtered.append(tool)
        return filtered

    def _visible_function_tool_names(self) -> list[str]:
        names: list[str] = []
        seen: set[str] = set()
        for tool in self._filtered_function_tools():
            name = str(tool.get("name") or "").strip()
            if not name or name in seen:
                continue
            names.append(name)
            seen.add(name)
        return names

    @staticmethod
    def _tool_descriptions_from_tools(
        function_tools: list[dict],
        builtin_tools: list[dict],
        control_tools: list[dict],
    ) -> str:
        descriptions: list[str] = []
        for tool in function_tools:
            name = tool.get("name", "")
            if not name:
                continue
            desc = (tool.get("description") or "").strip()
            descriptions.append(f"{name} : {desc}".strip())
        for tool in builtin_tools:
            tool_type = tool.get("type")
            if not tool_type:
                continue
            descriptions.append(f"{tool_type} : builtin tool")
        for tool in control_tools:
            name = tool.get("name")
            if not name:
                continue
            desc = (tool.get("description") or "").strip()
            descriptions.append(f"{name} : {desc}".strip())
        return "\n\n".join(descriptions)

    def _resolve_model_name(self) -> str:
        if self.llm_profile is not None:
            model = getattr(self.llm_profile.main, "model", None)
            if isinstance(model, str) and model:
                return model
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
        if self.llm_profile is not None:
            cfg = self.llm_profile.main
            for key in (
                "reasoning_effort",
                "temperature",
                "max_tokens",
                "max_output_tokens",
                "top_p",
                "frequency_penalty",
                "presence_penalty",
            ):
                value = getattr(cfg, key, None)
                if value is not None:
                    merged[key] = value
        raw = getattr(self.llm, "model_kwargs", None)
        if isinstance(raw, dict):
            merged.update(raw)
        for key in (
            "reasoning_effort",
            "temperature",
            "max_tokens",
            "max_output_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ):
            value = getattr(self.llm, key, None)
            if value is None or key in merged:
                continue
            merged[key] = value
        return merged

    def _tool_driver_kwargs(self) -> Dict[str, Any]:
        kwargs = self._collect_model_kwargs()
        driver_kwargs: Dict[str, Any] = {}
        for key in (
            "reasoning_effort",
            "temperature",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
        ):
            value = kwargs.get(key)
            if value is not None:
                driver_kwargs[key] = value
        if "max_output_tokens" in kwargs and kwargs.get("max_output_tokens") is not None:
            driver_kwargs["max_output_tokens"] = kwargs["max_output_tokens"]
        elif "max_tokens" in kwargs and kwargs.get("max_tokens") is not None:
            driver_kwargs["max_output_tokens"] = kwargs["max_tokens"]
        prompt_cache_retention = self._prompt_cache_retention()
        if prompt_cache_retention:
            driver_kwargs["prompt_cache_retention"] = prompt_cache_retention
        return driver_kwargs

    def _prompt_cache_retention(self) -> Optional[str]:
        if self.llm_profile is None:
            return None
        tool_calling = getattr(self.llm_profile.main, "tool_calling", None)
        if tool_calling is None:
            return None
        value = getattr(tool_calling, "prompt_cache_retention", None)
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return None

    def _proposal_browse_tools_enabled(self) -> bool:
        if self.llm_profile is None:
            return True
        tool_calling = getattr(self.llm_profile.main, "tool_calling", None)
        if tool_calling is None:
            return True
        value = getattr(tool_calling, "proposal_browse_tools_enabled", None)
        return True if value is None else bool(value)

    def _trace_event(self, event: str, payload: Optional[Dict[str, Any]] = None) -> None:
        record = {"event": event, "payload": payload or {}}
        self.trace_store.append_event(record)

    def _write_usage_summary(self) -> None:
        try:
            summary = write_usage_summary(self.run_context.run_dir)
        except Exception as exc:
            self.logger.debug("usage summary write failed: %s", exc)
            return
        self._emit("USAGE_SUMMARY_WRITTEN", category="llm", payload={
            "path": str(self.run_context.run_dir / "usage_summary.json"),
            "calls": summary.get("calls", 0),
            "input_tokens": summary.get("input_tokens", 0),
            "input_cached_tokens": summary.get("input_cached_tokens", 0),
            "output_tokens": summary.get("output_tokens", 0),
            "total_tokens": summary.get("total_tokens", 0),
            "missing_usage_calls": summary.get("missing_usage_calls", 0),
        })

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

    def _interrupt_requested(self) -> bool:
        try:
            return bool(self.run_control.is_interrupt_requested())
        except Exception:
            return False

    def _ack_interrupt(self, phase: str, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        info: Dict[str, Any]
        try:
            info = self.run_control.ack_interrupt(phase=phase, details=details or {})
        except Exception:
            info = {
                "requested": True,
                "acked": True,
                "phase": phase,
                "details": details or {},
            }
        self._emit("INTERRUPT_ACKED", category="run", payload={
            "phase": phase,
            "details": details or {},
            "requested": bool(info.get("requested")),
            "acked": bool(info.get("acked")),
        })
        self.checkpoint_store.append("INTERRUPT_ACKED", {
            "run_id": self.run_context.run_id,
            "phase": phase,
            "details": details or {},
        })
        return info

    def _interrupt_context_text(self, feedback: str, *, phase: str) -> str:
        text = (feedback or "").strip()
        if text:
            return f"Interrupted once at {phase}. User guidance: {text}"
        return f"Interrupted once at {phase}. User provided no feedback."

    def _prompt_interrupt_feedback(self, *, phase: str) -> str:
        guidance = (
            "Execution was interrupted. Provide optional guidance for resume. "
            "Leave empty to continue with no feedback."
        )
        if hasattr(self.reporter, "prompt_interrupt_feedback") and self.reporter.is_live():
            try:
                return str(self.reporter.prompt_interrupt_feedback(
                    guidance=guidance,
                    run_id=self.run_context.run_id,
                    phase=phase,
                ) or "")
            except Exception:
                return ""
        return ""

    @staticmethod
    def _snippet(text: Any, limit: int = 160) -> str:
        if text is None:
            return ""
        cleaned = " ".join(str(text).split())
        if len(cleaned) <= limit:
            return cleaned
        return cleaned[: max(0, limit - 3)] + "..."

    def plan(self, user_request: str) -> Dict[str, Any]:
        tools = self._tool_schema()
        messages = self.plan_prompt.format_messages(
            user_request=user_request,
            tools=tools,
            planner_tools=self._planner_tool_schema(),
        )
        self._emit("PLAN_START", category="plan", payload={"attempts": self.max_plan_steps})
        input_items = self._messages_to_input_items(messages)
        stepper = ToolCallingTaskStepper(
            driver=self.tool_driver,
            backend=self.tool_backend,
            prompt=None,
            control_tools=get_plan_control_tool_schemas(),
            control_tool_names=PLAN_CONTROL_TOOL_NAMES,
            trace_store=self.trace_store,
            checkpoint_store=self.checkpoint_store,
            reporter=self.reporter,
            max_steps=self.max_plan_steps,
            driver_kwargs={
                **self._tool_driver_kwargs(),
                "parallel_tool_calls": self.tool_policy.parallel_tool_calls,
            },
            role="planner",
            run_id=self.run_context.run_id,
        )
        step_result = stepper.run(
            task_id="plan",
            task_goal="Plan tasks",
            context_pack={},
            seed_messages=input_items,
            function_tools=self._planner_function_tools(),
            builtin_tools=[],
        )
        finish_reason = step_result.get("finish_reason", "")
        if finish_reason != "plan_finish":
            raise ValueError(f"Planner did not finish with plan_finish (got {finish_reason})")
        payload = step_result.get("control_payload") or {}
        normalized = self._normalize_plan(payload, user_request)
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
            planner_tools=self._planner_tool_schema(),
            plan_json=json.dumps(plan, ensure_ascii=False),
            feedback=feedback,
            feedback_history=json.dumps(feedback_history or [], ensure_ascii=False),
        )
        input_items = self._messages_to_input_items(messages)
        stepper = ToolCallingTaskStepper(
            driver=self.tool_driver,
            backend=self.tool_backend,
            prompt=None,
            control_tools=get_plan_control_tool_schemas(),
            control_tool_names=PLAN_CONTROL_TOOL_NAMES,
            trace_store=self.trace_store,
            checkpoint_store=self.checkpoint_store,
            reporter=self.reporter,
            max_steps=self.max_plan_steps,
            driver_kwargs={
                **self._tool_driver_kwargs(),
                "parallel_tool_calls": self.tool_policy.parallel_tool_calls,
            },
            role="planner",
            run_id=self.run_context.run_id,
        )
        step_result = stepper.run(
            task_id="plan_feedback",
            task_goal="Revise plan",
            context_pack={},
            seed_messages=input_items,
            function_tools=self._planner_function_tools(),
            builtin_tools=[],
        )
        finish_reason = step_result.get("finish_reason", "")
        if finish_reason != "plan_finish":
            raise ValueError(f"Plan feedback did not finish with plan_finish (got {finish_reason})")
        payload = step_result.get("control_payload") or {}
        normalized = self._normalize_plan(payload, user_request)
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

    def _artifact_index(self) -> List[Dict[str, Any]]:
        try:
            return self.memory_store.artifact_index(limit=500)
        except Exception:
            return []

    def _proposal_path(self) -> Path:
        return self.run_context.run_dir / "proposal.md"

    def _write_proposal(self, proposal_md: str) -> str:
        path = self._proposal_path()
        path.write_text(proposal_md or "", encoding="utf-8")
        return str(path.relative_to(self.run_context.run_dir))

    def _create_proposal(self, user_request: str, *, log_llm: bool = False) -> Dict[str, Any]:
        tools = self._tool_schema()
        proposal_function_tools = self._proposal_function_tools()
        memory_index_excerpt = self.memory_store.read_index(max_lines=200, max_chars=12000)
        artifacts_index = self._artifact_index()
        messages = self.proposal_prompt.format_messages(
            user_request=user_request,
            memory_index_excerpt=memory_index_excerpt,
            artifacts_index=json.dumps(artifacts_index, ensure_ascii=False),
            tools=tools,
        )
        self._emit("PROPOSAL_START", category="plan", payload={"attempts": self.max_plan_steps})
        input_items = self._messages_to_input_items(messages)
        stepper = ToolCallingTaskStepper(
            driver=self.tool_driver,
            backend=self.tool_backend,
            prompt=None,
            control_tools=get_proposal_control_tool_schemas(),
            control_tool_names=PROPOSAL_CONTROL_TOOL_NAMES,
            trace_store=self.trace_store,
            checkpoint_store=self.checkpoint_store,
            reporter=self.reporter,
            max_steps=self.max_plan_steps,
            driver_kwargs={
                **self._tool_driver_kwargs(),
                "parallel_tool_calls": self.tool_policy.parallel_tool_calls,
            },
            role="proposal",
            run_id=self.run_context.run_id,
        )
        step_result = stepper.run(
            task_id="proposal",
            task_goal="Create proposal",
            context_pack={},
            seed_messages=input_items,
            function_tools=proposal_function_tools,
            builtin_tools=[],
        )
        finish_reason = step_result.get("finish_reason", "")
        if finish_reason != "proposal_finish":
            raise ValueError(f"Proposal did not finish with proposal_finish (got {finish_reason})")
        payload = step_result.get("control_payload") or {}
        proposal_md = payload.get("proposal_md")
        work_packages = payload.get("work_packages")
        if not isinstance(proposal_md, str):
            raise ValueError("Proposal must include proposal_md string")
        if not isinstance(work_packages, list):
            raise ValueError("Proposal must include work_packages list")
        self._trace_event("PROPOSAL_CREATED", {
            "n_work_packages": len(work_packages),
        })
        self._emit("PROPOSAL_DONE", category="plan", payload={
            "n_work_packages": len(work_packages),
        })
        return {
            "proposal_md": proposal_md,
            "work_packages": work_packages,
        }

    def _revise_proposal(
        self,
        user_request: str,
        *,
        proposal_md: str,
        work_packages: List[str],
        feedback: str,
        log_llm: bool = False,
    ) -> Dict[str, Any]:
        tools = self._tool_schema()
        proposal_function_tools = self._proposal_function_tools()
        memory_index_excerpt = self.memory_store.read_index(max_lines=200, max_chars=12000)
        artifacts_index = self._artifact_index()
        messages = self.proposal_feedback_prompt.format_messages(
            user_request=user_request,
            proposal_md=proposal_md,
            work_packages_json=json.dumps(work_packages, ensure_ascii=False),
            memory_index_excerpt=memory_index_excerpt,
            artifacts_index=json.dumps(artifacts_index, ensure_ascii=False),
            tools=tools,
            feedback=feedback,
        )
        input_items = self._messages_to_input_items(messages)
        stepper = ToolCallingTaskStepper(
            driver=self.tool_driver,
            backend=self.tool_backend,
            prompt=None,
            control_tools=get_proposal_control_tool_schemas(),
            control_tool_names=PROPOSAL_CONTROL_TOOL_NAMES,
            trace_store=self.trace_store,
            checkpoint_store=self.checkpoint_store,
            reporter=self.reporter,
            max_steps=self.max_plan_steps,
            driver_kwargs={
                **self._tool_driver_kwargs(),
                "parallel_tool_calls": self.tool_policy.parallel_tool_calls,
            },
            role="proposal",
            run_id=self.run_context.run_id,
        )
        step_result = stepper.run(
            task_id="proposal_feedback",
            task_goal="Revise proposal",
            context_pack={},
            seed_messages=input_items,
            function_tools=proposal_function_tools,
            builtin_tools=[],
        )
        finish_reason = step_result.get("finish_reason", "")
        if finish_reason != "proposal_finish":
            raise ValueError(f"Proposal feedback did not finish with proposal_finish (got {finish_reason})")
        payload = step_result.get("control_payload") or {}
        revised_md = payload.get("proposal_md")
        revised_packages = payload.get("work_packages")
        if not isinstance(revised_md, str):
            raise ValueError("Revised proposal must include proposal_md string")
        if not isinstance(revised_packages, list):
            raise ValueError("Revised proposal must include work_packages list")
        self._trace_event("PROPOSAL_REVISION", {
            "feedback": feedback,
            "work_packages_before": work_packages,
            "work_packages_after": revised_packages,
        })
        return {
            "proposal_md": revised_md,
            "work_packages": revised_packages,
        }

    def _director_decide(
        self,
        *,
        user_request: str,
        proposal_md: str,
        work_packages: List[str],
        observations: List[Dict[str, Any]],
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        memory_index_excerpt = self.memory_store.read_index(max_lines=200, max_chars=12000)
        artifacts_index = self._artifact_index()
        director_observations = self._director_observations_view(observations)
        function_tools = self._filtered_function_tools()
        builtin_tools = self.tool_policy.builtin_tools if self._supports_builtin_tools else []
        tools_for_director = self._tool_descriptions_from_tools(function_tools, builtin_tools, [])
        messages = self.director_prompt.format_messages(
            user_request=user_request,
            proposal_md=proposal_md,
            work_packages_json=json.dumps(work_packages, ensure_ascii=False),
            memory_index_excerpt=memory_index_excerpt,
            artifacts_index=json.dumps(artifacts_index, ensure_ascii=False),
            already_done_json=json.dumps(director_observations, ensure_ascii=False),
            tools=tools_for_director,
        )
        input_items = self._messages_to_input_items(messages)
        stepper = ToolCallingTaskStepper(
            driver=self.tool_driver,
            backend=self.tool_backend,
            prompt=None,
            control_tools=get_director_control_tool_schemas(),
            control_tool_names=DIRECTOR_CONTROL_TOOL_NAMES,
            trace_store=self.trace_store,
            checkpoint_store=self.checkpoint_store,
            reporter=self.reporter,
            max_steps=self.max_plan_steps,
            driver_kwargs={
                **self._tool_driver_kwargs(),
                "parallel_tool_calls": self.tool_policy.parallel_tool_calls,
            },
            role="director",
            run_id=self.run_context.run_id,
            interrupt_checker=self._interrupt_requested,
            interrupt_ack=self._ack_interrupt,
        )
        step_result = stepper.run(
            task_id="director",
            task_goal="Decide next action",
            context_pack={},
            seed_messages=input_items,
            function_tools=[],
            builtin_tools=[],
            resume_state=resume_state,
        )
        finish_reason = step_result.get("finish_reason", "")
        if finish_reason == "interrupted":
            return {
                "state": "Interrupted",
                "interrupt_phase": step_result.get("interrupt_phase", "director"),
                "resume_state": step_result.get("resume_state"),
            }
        if finish_reason != "director_decide":
            raise ValueError(f"Director did not finish with director_decide (got {finish_reason})")
        payload = step_result.get("control_payload") or {}
        return payload

    @staticmethod
    def _director_observations_view(observations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sanitized: List[Dict[str, Any]] = []
        for item in observations or []:
            if not isinstance(item, dict):
                continue
            row: Dict[str, Any] = {}
            for key in ("task_id", "outcome", "summary"):
                value = item.get(key)
                if value is None:
                    continue
                text = " ".join(str(value).split())
                if text:
                    row[key] = text

            raw_artifacts = item.get("key_artifacts")
            key_artifacts: List[Dict[str, str]] = []
            if isinstance(raw_artifacts, list):
                for artifact in raw_artifacts:
                    if not isinstance(artifact, dict):
                        continue
                    path = str(artifact.get("path") or "").strip()
                    if not path:
                        continue
                    entry: Dict[str, str] = {"path": path}
                    desc = str(artifact.get("description") or "").strip()
                    kind = str(artifact.get("kind") or "").strip()
                    if desc:
                        entry["description"] = desc
                    if kind:
                        entry["kind"] = kind
                    key_artifacts.append(entry)
            row["key_artifacts"] = key_artifacts

            interrupted = item.get("interrupted_toolcall")
            if isinstance(interrupted, dict):
                safe_interrupt: Dict[str, Any] = {}
                for key in ("tool", "status", "highlights", "cancel_accepted"):
                    if key in interrupted:
                        safe_interrupt[key] = interrupted.get(key)
                if safe_interrupt:
                    row["interrupted_toolcall"] = safe_interrupt

            if row:
                sanitized.append(row)
        return sanitized

    def _review_proposal(
        self,
        *,
        user_request: str,
        proposal_md: str,
        work_packages: List[str],
        log_llm: bool,
        plan_feedback_provider: Optional[Callable[[Dict[str, Any]], str]],
        allow_revise: bool = True,
        persist_fn: Optional[Callable[[str, List[str]], None]] = None,
    ) -> tuple[str, List[str], bool, str]:
        review_state = {
            "user_request": user_request,
            "proposal_md": proposal_md,
            "work_packages": work_packages,
            "plan": {"todo": list(work_packages), "plan_description": proposal_md},
            "feedback_history": [],
            "approved": False,
            "round": 0,
        }
        if plan_feedback_provider is None and not self.reporter.is_live():
            raise ValueError("plan_review requires a live reporter (WebUI). Start WebUI or disable plan_review.")

        last_feedback = ""
        while not review_state.get("approved"):
            plan_description = (review_state["proposal_md"] or "").strip()
            self._emit("PLAN_REVIEW_SHOW", category="plan", payload={
                "todo": review_state["work_packages"],
                "plan_description_snippet": self._snippet(plan_description, 240),
            })
            self._emit("PLAN_REVIEW_WAIT_INPUT", category="plan")
            if plan_feedback_provider:
                feedback = plan_feedback_provider({
                    **review_state,
                    "stage": "proposal_review",
                })
            else:
                if hasattr(self.reporter, "prompt_plan_feedback") and self.reporter.is_live():
                    feedback = self.reporter.prompt_plan_feedback(
                        todo=review_state["work_packages"],
                        plan_description=plan_description,
                    )
                else:
                    raise ValueError("plan_review requires a live reporter (WebUI). Start WebUI or disable plan_review.")
            if not feedback:
                if plan_feedback_provider:
                    raise ValueError("plan_review feedback cannot be empty")
                self._emit("PLAN_REVIEW_WAIT_INPUT", category="plan", payload={"error": "empty_input"})
                continue

            last_feedback = feedback
            if self._is_plan_approved(feedback):
                review_state["approved"] = True
                review_state["feedback_history"].append({
                    "round": review_state.get("round", 0),
                    "feedback": feedback,
                    "approved": True,
                })
                self._emit("PLAN_REVIEW_APPROVED", category="plan", payload={
                    "feedback_snippet": self._snippet(feedback, 160),
                })
                break
            if not allow_revise:
                return proposal_md, work_packages, False, feedback

            self._emit("PLAN_REVIEW_REVISING", category="plan", payload={
                "feedback_snippet": self._snippet(feedback, 160),
            })
            revised = self._revise_proposal(
                user_request,
                proposal_md=proposal_md,
                work_packages=work_packages,
                feedback=feedback,
                log_llm=log_llm,
            )
            proposal_md = revised["proposal_md"]
            work_packages = revised["work_packages"]
            if persist_fn:
                persist_fn(proposal_md, work_packages)
            review_state["proposal_md"] = proposal_md
            review_state["work_packages"] = work_packages
            review_state["plan"] = {"todo": list(work_packages), "plan_description": proposal_md}
            review_state["feedback_history"].append({
                "round": review_state.get("round", 0),
                "feedback": feedback,
                "approved": False,
            })
            review_state["round"] = review_state.get("round", 0) + 1
        return proposal_md, work_packages, True, last_feedback

    def _execute_task(
        self,
        *,
        task_id: str,
        task_goal: str,
        task_goal_short: Optional[str] = None,
        log_llm: bool,
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context_pack = self.context_builder.build(
            task_goal,
            role="task_runner",
            policy=ContextPackPolicy(
                memory_head_lines=200,
                max_memory_chars=12000,
                max_artifacts=300,
                inject_goal_for_worker=False,
            ),
        )
        if self._interrupt_context_note:
            base_constraints = str(context_pack.get("constraints", "") or "").strip()
            if base_constraints:
                context_pack["constraints"] = f"{base_constraints}\n\n{self._interrupt_context_note}"
            else:
                context_pack["constraints"] = self._interrupt_context_note
        self._emit("TASK_CONTEXT_READY", category="task", task_id=task_id, payload={
            "excerpt_chars": len(context_pack.get("memory_index_excerpt", "") or ""),
            "artifact_slice_count": len(context_pack.get("artifact_slice", []) or []),
        })
        filtered_tools = self._filtered_function_tools()
        builtin_tools = self.tool_policy.builtin_tools if self._supports_builtin_tools else []
        stepper = ToolCallingTaskStepper(
            driver=self.tool_driver,
            backend=self.tool_backend,
            prompt=self.task_step_prompt,
            reporter=self.reporter,
            max_steps=min(self.max_steps, self.tool_policy.max_tool_calls_per_task),
            driver_kwargs={
                **self._tool_driver_kwargs(),
                "parallel_tool_calls": self.tool_policy.parallel_tool_calls,
            },
            trace_store=self.trace_store,
            checkpoint_store=self.checkpoint_store,
            role="task_runner",
            run_id=self.run_context.run_id,
            interrupt_checker=self._interrupt_requested,
            interrupt_ack=self._ack_interrupt,
        )
        step_result = stepper.run(
            task_id=task_id,
            task_goal=task_goal,
            context_pack=context_pack,
            initial_instruction=None,
            function_tools=filtered_tools,
            builtin_tools=builtin_tools,
            resume_state=resume_state,
        )
        if step_result.get("finish_reason") == "interrupted":
            return {
                "task_id": task_id,
                "outcome": "interrupted",
                "summary": "Execution interrupted by user.",
                "resume_state": step_result.get("resume_state"),
                "interrupt_phase": step_result.get("interrupt_phase", "toolcall"),
                "interrupted_toolcall": step_result.get("interrupted_toolcall"),
            }
        finish_reason = str(step_result.get("finish_reason") or "")
        control_payload = step_result.get("control_payload")
        task_result = self._normalize_task_result_payload(
            task_goal=task_goal,
            finish_reason=finish_reason,
            payload=control_payload,
            output_text=str(step_result.get("output_text") or ""),
        )
        outcome = task_result["task_outcome"]
        merge_info: Dict[str, Any] = {}
        patch_merge_failed = False
        merge_error = ""
        try:
            merge_info = self._merge_memory_via_git_apply(
                run_id=self.run_context.run_id,
                task_id=task_id,
                outcome=outcome,
                task_goal_short=task_goal_short or task_goal,
                structured_result=task_result["structured_result"],
            )
            self._emit("MEMORY_MERGE_DONE", category="summary", task_id=task_id, payload={
                "event_path": merge_info.get("event_path", ""),
                "memory_index": merge_info.get("memory_index", ""),
                "patch_path": merge_info.get("patch_path", ""),
                "attempts": merge_info.get("attempts", 0),
            })
        except Exception as exc:
            patch_merge_failed = True
            merge_error = str(exc)
            merge_info = {
                "event_path": getattr(exc, "event_path", ""),
                "memory_index": str(self.memory_store.index_path.relative_to(workspace_root(self.run_context.workspace))),
                "patch_path": getattr(exc, "patch_path", ""),
                "check_error": getattr(exc, "check_error", ""),
            }
            self._emit("MEMORY_MERGE_FAILED", level="warning", category="summary", task_id=task_id, payload={
                "event_path": merge_info.get("event_path", ""),
                "memory_index": merge_info.get("memory_index", ""),
                "patch_path": merge_info.get("patch_path", ""),
                "error": merge_error,
            })
            task_result["task_summary"] = (
                f"{task_result['task_summary']} | memory_merge=failed: {self._snippet(merge_error, 280)}"
            ).strip()
            task_result["structured_result"]["summary"] = task_result["task_summary"]
            outcome = "needs_intervention"
            task_result["task_outcome"] = outcome
            patch_path = str(merge_info.get("patch_path") or "").strip()
            if patch_path:
                task_result["key_artifacts"].append({
                    "path": patch_path,
                    "description": "memory patch failed to apply",
                    "kind": "log",
                })

        observation_path = self._write_observation(
            task_id=task_id,
            outcome=outcome,
            summary=task_result["task_summary"],
            key_artifacts=task_result["key_artifacts"],
        )
        self._emit("OBSERVATION_WRITTEN", category="task", task_id=task_id, payload={
            "path": observation_path,
        })
        return {
            "task_id": task_id,
            "outcome": outcome,
            "summary": task_result["task_summary"],
            "observation_path": observation_path,
            "key_artifacts": task_result["key_artifacts"],
            "event_path": merge_info.get("event_path"),
            "memory_merge_failed": patch_merge_failed,
            "memory_merge_error": merge_error if patch_merge_failed else "",
        }

    def _normalize_task_result_payload(
        self,
        *,
        task_goal: str,
        finish_reason: str,
        payload: Any,
        output_text: str,
    ) -> Dict[str, Any]:
        body = payload if isinstance(payload, dict) else {}
        if finish_reason == "task_finish":
            summary = str(body.get("summary") or "").strip() or "Task finished."
            structured = {
                "summary": summary,
                "facts": self._clean_text_list(body.get("facts")),
                "files": self._clean_file_records(body.get("files")),
                "constraints": self._clean_text_list(body.get("constraints")),
                "open_questions": self._clean_text_list(body.get("open_questions")),
                "decisions": self._clean_decisions(body.get("decisions")),
                "next_steps": self._clean_text_list(body.get("next_steps")),
                "artifacts": self._clean_text_list(body.get("artifacts")),
            }
            key_artifacts = [
                {
                    "path": item["path"],
                    "description": item.get("description", ""),
                    "kind": item.get("kind", "output"),
                }
                for item in structured["files"]
            ]
            for path in structured["artifacts"]:
                key_artifacts.append({"path": path, "description": "", "kind": "artifact"})
            return {
                "task_outcome": "success",
                "task_summary": summary,
                "key_artifacts": key_artifacts,
                "structured_result": structured,
            }

        if finish_reason == "task_fail":
            partial = body.get("partial_result") if isinstance(body.get("partial_result"), dict) else {}
            err = str(body.get("error") or output_text or "Task failed").strip()
            needs_human = bool(body.get("needs_human", True))
            structured = {
                "summary": err,
                "facts": self._clean_text_list(partial.get("facts")),
                "files": self._clean_file_records(partial.get("files")),
                "constraints": self._clean_text_list(partial.get("constraints")),
                "open_questions": self._clean_text_list(partial.get("open_questions")),
                "decisions": self._clean_decisions(partial.get("decisions")),
                "next_steps": self._clean_text_list(partial.get("next_steps")),
                "artifacts": self._clean_text_list(partial.get("artifacts")),
            }
            if needs_human and not structured["open_questions"]:
                structured["open_questions"] = [f"Human intervention required for task: {task_goal}"]
            key_artifacts = [
                {
                    "path": item["path"],
                    "description": item.get("description", ""),
                    "kind": item.get("kind", "output"),
                }
                for item in structured["files"]
            ]
            for path in structured["artifacts"]:
                key_artifacts.append({"path": path, "description": "", "kind": "artifact"})
            return {
                "task_outcome": "needs_intervention" if needs_human else "failure",
                "task_summary": err,
                "key_artifacts": key_artifacts,
                "structured_result": structured,
            }

        text = " ".join(output_text.split()).strip()
        summary = text or f"Task ended with unexpected finish_reason={finish_reason or 'unknown'}"
        structured = {
            "summary": summary,
            "facts": [],
            "files": [],
            "constraints": [],
            "open_questions": [],
            "decisions": [],
            "next_steps": [],
            "artifacts": [],
        }
        return {
            "task_outcome": "failure",
            "task_summary": summary,
            "key_artifacts": [],
            "structured_result": structured,
        }

    def _merge_memory_via_git_apply(
        self,
        *,
        run_id: str,
        task_id: str,
        outcome: str,
        task_goal_short: str,
        structured_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        self.memory_store.ensure_exists()
        event = {
            "ts": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "run_id": run_id,
            "task_id": task_id,
            "goal": " ".join(str(task_goal_short or "").split()),
            "outcome": str(outcome or "").strip(),
            "summary": str(structured_result.get("summary") or "").strip(),
            "facts": self._clean_text_list(structured_result.get("facts")),
            "constraints": self._clean_text_list(structured_result.get("constraints")),
            "open_questions": self._clean_text_list(structured_result.get("open_questions")),
            "next_steps": self._clean_text_list(structured_result.get("next_steps")),
            "files": self._clean_file_records(structured_result.get("files")),
            "decisions": self._clean_decisions(structured_result.get("decisions")),
            "artifacts": self._clean_text_list(structured_result.get("artifacts")),
        }
        event_rel = self.memory_store.append_event(event)
        memory_index_text = self.memory_store.read_index(max_lines=2000, max_chars=200000)
        topic_tldrs = self._read_memory_topic_tldrs()
        files_root = workspace_root(self.run_context.workspace)
        patch_dir = files_root / ".logs" / "memory_patches"
        patch_dir.mkdir(parents=True, exist_ok=True)

        max_attempts = max(1, int(self.patch_repair_attempts or 0) + 1)
        last_error = ""
        last_error_context: Dict[str, Any] = {}
        patch_rel = ""
        previous_edit_text = ""

        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                messages = self.memory_patch_prompt.format_messages(
                    run_id=run_id,
                    task_id=task_id,
                    task_goal=task_goal_short,
                    outcome=outcome,
                    event_path=event_rel,
                    structured_result_json=json.dumps(structured_result, ensure_ascii=False),
                    memory_index_text=memory_index_text,
                    topic_tldrs_json=json.dumps(topic_tldrs, ensure_ascii=False),
                )
                llm_kind = "memory_patch"
            else:
                messages = self.memory_patch_repair_prompt.format_messages(
                    previous_edit_text=previous_edit_text,
                    apply_error=last_error or "(none)",
                    apply_error_context_json=json.dumps(last_error_context, ensure_ascii=False),
                    run_id=run_id,
                    task_id=task_id,
                    task_goal=task_goal_short,
                    outcome=outcome,
                    structured_result_json=json.dumps(structured_result, ensure_ascii=False),
                    memory_index_text=memory_index_text,
                    topic_tldrs_json=json.dumps(topic_tldrs, ensure_ascii=False),
                )
                llm_kind = "memory_patch_repair"

            self._emit("LLM_CALL_START", category="llm", payload={"kind": llm_kind, "task_id": task_id, "attempt": attempt})
            t0 = time.perf_counter()
            resp = self.llm.invoke(messages)
            patch_raw = llm_text(resp).strip()
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            self._emit("LLM_CALL_END", category="llm", payload={
                "kind": llm_kind,
                "task_id": task_id,
                "attempt": attempt,
                "elapsed_ms": elapsed_ms,
            })
            self._write_llm_log(f"{llm_kind}_prompt", messages=self._messages_to_dict(messages), task_id=task_id, attempt=attempt)
            self._write_llm_log(f"{llm_kind}_response", content=patch_raw, task_id=task_id, attempt=attempt)

            edit_text = self._normalize_patch_text(patch_raw)
            previous_edit_text = edit_text
            edits_rel = f".logs/memory_patches/memory_{run_id}_{task_id}_a{attempt}.aider"
            edits_abs = files_root / edits_rel
            edits_abs.write_text(edit_text if edit_text.endswith("\n") else f"{edit_text}\n", encoding="utf-8")

            tool_out = self.tool_backend.call(
                "memory_apply_aider_edits",
                json.dumps({
                    "edits_text": edit_text,
                    "allowed_paths": ["MEMORY/", "notes/"],
                    "emit_diff": True,
                }, ensure_ascii=False),
                toolcall_key=f"{task_id}_memory_patch_a{attempt}",
            )
            status = str(tool_out.get("status") or "").strip().lower()
            data = tool_out.get("data") if isinstance(tool_out.get("data"), dict) else {}
            patch_rel = f".logs/memory_patches/memory_{run_id}_{task_id}_a{attempt}.diff"
            patch_abs = files_root / patch_rel
            diff_text = str(data.get("diff_text") or "")
            patch_abs.write_text(diff_text if diff_text.endswith("\n") else f"{diff_text}\n", encoding="utf-8")

            if status == "success":
                self.trace_store.append_patch({
                    "event": "MEMORY_PATCH_ATTEMPT",
                    "payload": {
                        "task_id": task_id,
                        "attempt": attempt,
                        "status": "applied",
                        "event_path": event_rel,
                        "edits_path": edits_rel,
                        "patch_path": patch_rel,
                    },
                })
                return {
                    "event_path": event_rel,
                    "memory_index": str(self.memory_store.index_path.relative_to(workspace_root(self.run_context.workspace))),
                    "patch_path": patch_rel,
                    "attempts": attempt,
                }

            error_code = str(data.get("error_code") or "").strip()
            error_detail = str(data.get("error_detail") or "").strip()
            failed_path = str(data.get("failed_path") or "").strip()
            failed_block_index_raw = data.get("failed_block_index")
            try:
                failed_block_index = int(failed_block_index_raw or 0)
            except Exception:
                failed_block_index = 0
            last_error_context = {
                "error_code": error_code,
                "error_detail": error_detail,
                "failed_path": failed_path,
                "failed_block_index": failed_block_index,
            }
            last_error = " | ".join(
                [
                    part
                    for part in [
                        f"[{error_code}]" if error_code else "",
                        str(tool_out.get("error") or "").strip(),
                        error_detail,
                    ]
                    if part
                ]
            ).strip()
            self.trace_store.append_patch({
                "event": "MEMORY_PATCH_ATTEMPT",
                "payload": {
                    "task_id": task_id,
                    "attempt": attempt,
                    "status": "failed",
                    "event_path": event_rel,
                    "edits_path": edits_rel,
                    "patch_path": patch_rel,
                    "error": last_error,
                    "error_context": last_error_context,
                },
            })

        raise MemoryPatchApplyError(
            f"Memory patch apply failed after {max_attempts} attempts: {last_error or 'unknown error'}",
            event_path=event_rel,
            patch_path=patch_rel,
            check_error=last_error,
        )

    def _read_memory_topic_tldrs(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for topic_path in sorted(self.memory_store.topics_dir.glob("*.md")):
            try:
                text = topic_path.read_text(encoding="utf-8")
            except Exception:
                continue
            out[topic_path.name] = self._topic_tldr_excerpt(text)
        return out

    @staticmethod
    def _topic_tldr_excerpt(text: str, *, fallback_lines: int = 40, max_chars: int = 4000) -> str:
        lines = text.splitlines()
        start_idx = -1
        for i, raw in enumerate(lines):
            if raw.strip().lower() == "## tl;dr":
                start_idx = i
                break
        if start_idx >= 0:
            excerpt_lines: List[str] = [lines[start_idx]]
            for raw in lines[start_idx + 1:]:
                if raw.startswith("## "):
                    break
                excerpt_lines.append(raw)
            excerpt = "\n".join(excerpt_lines).strip()
        else:
            excerpt = "\n".join(lines[:fallback_lines]).strip()
        if len(excerpt) > max_chars:
            return excerpt[:max_chars]
        return excerpt

    @staticmethod
    def _normalize_patch_text(raw: str) -> str:
        text = str(raw or "").strip()
        if text.startswith("```") and text.endswith("```"):
            m = re.match(r"^```[^\n]*\n(.*?)\n```$", text, re.DOTALL)
            if m:
                text = m.group(1).strip()
        return text

    @staticmethod
    def _validate_memory_patch_paths(patch_text: str) -> None:
        text = str(patch_text or "").strip()
        if not text:
            raise ValueError("empty patch")
        touched: set[str] = set()
        for raw in text.splitlines():
            line = raw.strip()
            if line.startswith("diff --git "):
                m = re.match(r"^diff --git a/(.+?) b/(.+?)$", line)
                if m:
                    touched.add(m.group(1).strip())
                    touched.add(m.group(2).strip())
            elif line.startswith("--- "):
                token = line[4:].strip()
                if token != "/dev/null":
                    touched.add(token[2:] if token.startswith("a/") else token)
            elif line.startswith("+++ "):
                token = line[4:].strip()
                if token != "/dev/null":
                    touched.add(token[2:] if token.startswith("b/") else token)

        bad: List[str] = []
        for path in sorted(touched):
            norm = path.strip()
            if not norm:
                continue
            pure = Path(norm)
            if pure.is_absolute() or ".." in pure.parts:
                bad.append(norm)
                continue
            if not (norm.startswith("MEMORY/") or norm.startswith("notes/")):
                bad.append(norm)
        if bad:
            raise ValueError(f"patch touches forbidden paths: {', '.join(bad)}")

    @staticmethod
    def _clean_text_list(raw: Any) -> List[str]:
        if not isinstance(raw, list):
            return []
        out: List[str] = []
        seen: set[str] = set()
        for item in raw:
            text = " ".join(str(item or "").split())
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

    @staticmethod
    def _clean_file_records(raw: Any) -> List[Dict[str, str]]:
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, str]] = []
        seen: set[str] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path") or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            out.append({
                "path": path,
                "description": str(item.get("description") or "").strip(),
                "kind": str(item.get("kind") or "output").strip() or "output",
            })
        return out

    @staticmethod
    def _clean_decisions(raw: Any) -> List[Dict[str, str]]:
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            decision = " ".join(str(item.get("decision") or "").split())
            rationale = " ".join(str(item.get("rationale") or "").split())
            if not decision:
                continue
            out.append({"decision": decision, "rationale": rationale})
        return out

    def _run_fast(
        self,
        user_request: str,
        *,
        log_llm: bool,
        resume_feedback: str,
        defer_ui: bool,
        start_ui: Callable[[str, bool], None],
    ) -> Dict[str, Any]:
        self._interrupt_context_note = ""
        if self.resuming:
            state = self._load_task_state()
            if state.get("lane") != "fast":
                raise ValueError(f"Cannot resume; lane mismatch ({state.get('lane')})")
            user_request = state["user_request"]
            tasks = state["tasks"]
            observations = state["observations"]
            status = state.get("status", "running")
            hitl_history = state.get("hitl_history") or []
            task_resume_checkpoint = state.get("task_resume_checkpoint")
            last_interrupt = state.get("last_interrupt") if isinstance(state.get("last_interrupt"), dict) else {}
            if status in {"done", "failure"}:
                raise ValueError(f"Cannot resume; run already ended with status {status}")
            if status == "interrupted_paused":
                phase = str(last_interrupt.get("phase") or "task_step")
                feedback = (resume_feedback or "").strip() or str(last_interrupt.get("feedback") or "").strip()
                feedback_empty = not bool(feedback)
                self._interrupt_context_note = self._interrupt_context_text(feedback, phase=phase)
                last_interrupt = {
                    **last_interrupt,
                    "feedback": feedback,
                    "feedback_empty": feedback_empty,
                    "resumed_at": datetime.utcnow().isoformat() + "Z",
                }
                status = "running"
                self.run_control.clear_interrupt()
            self._initialize_memory_goal(user_request)
        else:
            self._initialize_memory_goal(user_request)
            tasks = [{
                "task_id": "task_01",
                "goal": user_request,
                "status": "pending",
            }]
            observations = []
            status = "running"
            hitl_history = []
            task_resume_checkpoint = None
            last_interrupt = {}
            self._write_task_state({
                "schema_version": 2,
                "lane": "fast",
                "user_request": user_request,
                "tasks": tasks,
                "observations": observations,
                "status": status,
                "hitl_history": hitl_history,
                "task_resume_checkpoint": task_resume_checkpoint,
                "last_interrupt": last_interrupt,
            })

        self._emit("TASKS_COMPILED", category="task", payload={
            "n_tasks": len(tasks),
            "tasks": tasks,
        })

        if not defer_ui:
            self._emit("SPLASH_HIDE", category="splash")
        if defer_ui:
            start_ui("Starting execution...", False)
            self._emit("SPLASH_HIDE", category="splash")

        while True:
            next_task = None
            for task in tasks:
                if task.get("status") == "pending":
                    next_task = task
                    break
            if next_task is None:
                if status == "running":
                    status = "done"
                break

            task_id = next_task["task_id"]
            task_goal = next_task["goal"]
            self._trace_event("TASK_STARTED", {"task_id": task_id, "goal": task_goal})
            self._emit("TASK_START", category="task", task_id=task_id, payload={"goal": task_goal})
            resume_state = None
            if isinstance(task_resume_checkpoint, dict) and str(task_resume_checkpoint.get("task_id") or "") == task_id:
                resume_state = task_resume_checkpoint.get("resume_state")
            obs = self._execute_task(
                task_id=task_id,
                task_goal=task_goal,
                task_goal_short=(next_task.get("task_packet") or {}).get("goal") if isinstance(next_task.get("task_packet"), dict) else task_goal,
                log_llm=log_llm,
                resume_state=resume_state if isinstance(resume_state, dict) else None,
            )
            outcome = obs["outcome"]
            if outcome == "interrupted":
                phase = str(obs.get("interrupt_phase") or "toolcall")
                interrupt_req = self.run_control.snapshot()
                self._ack_interrupt(phase, {
                    "task_id": task_id,
                    "task_goal": task_goal,
                    "interrupted_toolcall": obs.get("interrupted_toolcall"),
                })
                feedback = self._prompt_interrupt_feedback(phase=phase).strip()
                last_interrupt = {
                    "phase": phase,
                    "source": interrupt_req.get("source", ""),
                    "note": interrupt_req.get("note", ""),
                    "requested_at": interrupt_req.get("request_ts"),
                    "feedback": feedback,
                    "feedback_empty": not bool(feedback),
                }
                task_resume_checkpoint = {
                    "task_id": task_id,
                    "task_goal": task_goal,
                    "resume_state": obs.get("resume_state") if isinstance(obs.get("resume_state"), dict) else {},
                }
                status = "interrupted_paused"
                self._emit("RUN_PAUSED", category="run", task_id=task_id, payload={
                    "phase": phase,
                    "status": status,
                })
                self._write_task_state({
                    "schema_version": 2,
                    "lane": "fast",
                    "user_request": user_request,
                    "tasks": tasks,
                    "observations": observations,
                    "status": status,
                    "hitl_history": hitl_history,
                    "task_resume_checkpoint": task_resume_checkpoint,
                    "last_interrupt": last_interrupt,
                })
                break
            observations.append(obs)
            task_resume_checkpoint = None
            next_task["status"] = obs["outcome"]
            self._emit("TASK_END", category="task", task_id=task_id, payload={
                "outcome": outcome,
                "summary_snippet": self._snippet(obs.get("summary", ""), 200),
            })
            self._trace_event("TASK_COMPLETED", {"task_id": task_id, "outcome": outcome})

            if outcome == "needs_intervention":
                hitl_meta = self._hitl_collect_guidance(
                    user_request=user_request,
                    observations=observations,
                    tasks=tasks,
                    reason=f"task:{task_id}",
                    log_llm=log_llm,
                )
                hitl_history.append(hitl_meta)
                if not hitl_meta.get("feedback"):
                    status = "needs_intervention"
                    self._write_task_state({
                        "schema_version": 2,
                        "lane": "fast",
                        "user_request": user_request,
                        "tasks": tasks,
                        "observations": observations,
                        "status": status,
                        "hitl": hitl_meta,
                        "hitl_history": hitl_history,
                        "task_resume_checkpoint": task_resume_checkpoint,
                        "last_interrupt": last_interrupt,
                    })
                    break
                follow_goal = (
                    "Continue original request with HITL guidance: "
                    f"{self._snippet(hitl_meta.get('feedback', ''), 240)}"
                )
                tasks.append({
                    "task_id": f"task_{self._next_task_index(tasks):02d}",
                    "goal": follow_goal,
                    "status": "pending",
                })
                status = "running"
                self._write_task_state({
                    "schema_version": 2,
                    "lane": "fast",
                    "user_request": user_request,
                    "tasks": tasks,
                    "observations": observations,
                    "status": status,
                    "hitl": hitl_meta,
                    "hitl_history": hitl_history,
                    "task_resume_checkpoint": task_resume_checkpoint,
                    "last_interrupt": last_interrupt,
                })
                continue

            if outcome == "failure":
                status = "failure"
                self._write_task_state({
                    "schema_version": 2,
                    "lane": "fast",
                    "user_request": user_request,
                    "tasks": tasks,
                    "observations": observations,
                    "status": status,
                    "hitl_history": hitl_history,
                    "task_resume_checkpoint": task_resume_checkpoint,
                    "last_interrupt": last_interrupt,
                })
                break

            status = "running"
            self._write_task_state({
                "schema_version": 2,
                "lane": "fast",
                "user_request": user_request,
                "tasks": tasks,
                "observations": observations,
                "status": status,
                "hitl_history": hitl_history,
                "task_resume_checkpoint": task_resume_checkpoint,
                "last_interrupt": last_interrupt,
            })

        state_payload = {
            "schema_version": 2,
            "lane": "fast",
            "user_request": user_request,
            "tasks": tasks,
            "observations": observations,
            "status": status,
            "hitl_history": hitl_history,
            "task_resume_checkpoint": task_resume_checkpoint,
            "last_interrupt": last_interrupt,
        }
        self._write_task_state(state_payload)
        final_answer = observations[-1]["summary"] if observations else ""
        report_paths: Dict[str, str] = {}
        report_summary = final_answer

        if status in {"done", "failure", "needs_intervention"}:
            # Fast lane skips project-level summarization; publish a report using task summaries.
            self._emit("FINAL_SUMMARY_START", category="final")
            report_summary = self._summarize_tasks_fallback(user_request, observations)
            state_payload["summary"] = report_summary
            self._write_task_state(state_payload)
            report_paths = self._publish_report(user_request, report_summary)
            preview_lines = [line for line in (report_summary or "").splitlines() if line.strip()][:8]
            self._emit("FINAL_SUMMARY_DONE", category="final", payload={
                "preview_lines": preview_lines,
                "report_path": report_paths.get("final_report", ""),
                "run_dir": report_paths.get("run_dir", ""),
            })

        result = {
            "tasks": tasks,
            "observations": observations,
            "summary": final_answer,
            "final_answer": final_answer,
            "status": status,
        }
        run_end_payload: Dict[str, str] = {
            "status": status,
            "run_dir": report_paths.get("run_dir", str(self.run_context.run_dir)),
        }
        if report_paths:
            run_end_payload.update({
                "final_report": report_paths.get("final_report", ""),
                "memory_report": report_paths.get("memory_report", ""),
                "latest_link": report_paths.get("latest_link", ""),
            })
        self._emit("RUN_END", category="run", payload=run_end_payload)
        try:
            if hasattr(self.reporter, "show_final_summary") and self.reporter.is_live():
                self.reporter.show_final_summary(final_answer)
        except Exception:
            pass
        return result

    def _run_standard(
        self,
        user_request: str,
        *,
        log_llm: bool,
        resume_feedback: str,
        plan_review: bool,
        plan_feedback_provider: Optional[Callable[[Dict[str, Any]], str]],
        full_auto_major: bool,
        defer_ui: bool,
        start_ui: Callable[[str, bool], None],
    ) -> Dict[str, Any]:
        self._interrupt_context_note = ""
        if self.resuming:
            if plan_review:
                self.logger.warning("plan_review requested while resuming; ignoring and continuing with stored proposal.")
                plan_review = False
            state = self._load_task_state()
            if state.get("lane") != "standard":
                raise ValueError(f"Cannot resume; lane mismatch ({state.get('lane')})")
            user_request = state["user_request"]
            tasks = state["tasks"]
            observations = state["observations"]
            status = state.get("status", "running")
            hitl_history = state.get("hitl_history") or []
            task_resume_checkpoint = state.get("task_resume_checkpoint")
            last_interrupt = state.get("last_interrupt") if isinstance(state.get("last_interrupt"), dict) else {}
            proposal_info = state.get("proposal") or {}
            proposal_path = proposal_info.get("proposal_path") or "proposal.md"
            work_packages = proposal_info.get("work_packages") or []
            proposal_file = self.run_context.run_dir / proposal_path
            if not proposal_file.exists():
                raise FileNotFoundError(f"proposal.md not found: {proposal_file}")
            proposal_md = proposal_file.read_text(encoding="utf-8")
            if status in {"done", "failure"}:
                raise ValueError(f"Cannot resume; run already ended with status {status}")
            if status == "interrupted_paused":
                phase = str(last_interrupt.get("phase") or "task_step")
                feedback = (resume_feedback or "").strip() or str(last_interrupt.get("feedback") or "").strip()
                feedback_empty = not bool(feedback)
                self._interrupt_context_note = self._interrupt_context_text(feedback, phase=phase)
                interrupt_obs = {
                    "task_id": f"interrupt_{datetime.utcnow().strftime('%H%M%S')}",
                    "outcome": "interrupted",
                    "summary": self._interrupt_context_note,
                    "phase": phase,
                    "feedback": feedback,
                    "feedback_empty": feedback_empty,
                }
                observations.append(interrupt_obs)
                last_interrupt = {
                    **last_interrupt,
                    "phase": phase,
                    "feedback": feedback,
                    "feedback_empty": feedback_empty,
                    "resumed_at": datetime.utcnow().isoformat() + "Z",
                }
                status = "running"
                self.run_control.clear_interrupt()
            self._initialize_memory_goal(user_request)
        else:
            self._initialize_memory_goal(user_request)
            proposal = self._create_proposal(user_request, log_llm=log_llm)
            proposal_md = proposal["proposal_md"]
            work_packages = proposal["work_packages"]
            proposal_relpath = self._write_proposal(proposal_md)
            tasks = []
            observations = []
            status = "running"
            hitl_history = []
            task_resume_checkpoint = None
            last_interrupt = {}
            self._write_task_state({
                "schema_version": 2,
                "lane": "standard",
                "user_request": user_request,
                "proposal": {
                    "proposal_path": proposal_relpath,
                    "work_packages": work_packages,
                },
                "tasks": tasks,
                "observations": observations,
                "status": status,
                "hitl_history": hitl_history,
                "task_resume_checkpoint": task_resume_checkpoint,
                "last_interrupt": last_interrupt,
            })

            if plan_review:
                def persist(proposal_text: str, packages: List[str]) -> None:
                    relpath = self._write_proposal(proposal_text)
                    self._write_task_state({
                        "schema_version": 2,
                        "lane": "standard",
                        "user_request": user_request,
                        "proposal": {
                            "proposal_path": relpath,
                            "work_packages": packages,
                        },
                        "tasks": tasks,
                        "observations": observations,
                        "status": status,
                        "hitl_history": hitl_history,
                        "task_resume_checkpoint": task_resume_checkpoint,
                        "last_interrupt": last_interrupt,
                    })

                proposal_md, work_packages, approved, _ = self._review_proposal(
                    user_request=user_request,
                    proposal_md=proposal_md,
                    work_packages=work_packages,
                    log_llm=log_llm,
                    plan_feedback_provider=plan_feedback_provider,
                    allow_revise=True,
                    persist_fn=persist,
                )
                if approved:
                    proposal_relpath = self._write_proposal(proposal_md)
                    self._write_task_state({
                        "schema_version": 2,
                        "lane": "standard",
                        "user_request": user_request,
                        "proposal": {
                            "proposal_path": proposal_relpath,
                            "work_packages": work_packages,
                        },
                        "tasks": tasks,
                        "observations": observations,
                        "status": status,
                        "hitl_history": hitl_history,
                        "task_resume_checkpoint": task_resume_checkpoint,
                        "last_interrupt": last_interrupt,
                    })

        self._emit("TASKS_COMPILED", category="task", payload={
            "n_tasks": len(tasks),
            "tasks": tasks,
        })

        if not defer_ui:
            self._emit("SPLASH_HIDE", category="splash")
        if defer_ui:
            start_ui("Starting execution...", False)
            self._emit("SPLASH_HIDE", category="splash")

        if status == "running":
            for task in tasks:
                if task.get("status") != "pending":
                    continue
                task_id = task["task_id"]
                task_goal = task["goal"]
                self._trace_event("TASK_STARTED", {"task_id": task_id, "goal": task_goal})
                self._emit("TASK_START", category="task", task_id=task_id, payload={"goal": task_goal})
                resume_state = None
                if isinstance(task_resume_checkpoint, dict) and str(task_resume_checkpoint.get("task_id") or "") == task_id:
                    resume_state = task_resume_checkpoint.get("resume_state")
                obs = self._execute_task(
                    task_id=task_id,
                    task_goal=task_goal,
                    task_goal_short=(task.get("task_packet") or {}).get("goal") if isinstance(task.get("task_packet"), dict) else task_goal,
                    log_llm=log_llm,
                    resume_state=resume_state if isinstance(resume_state, dict) else None,
                )
                outcome = obs["outcome"]
                if outcome == "interrupted":
                    phase = str(obs.get("interrupt_phase") or "toolcall")
                    interrupt_req = self.run_control.snapshot()
                    self._ack_interrupt(phase, {
                        "task_id": task_id,
                        "task_goal": task_goal,
                        "interrupted_toolcall": obs.get("interrupted_toolcall"),
                    })
                    feedback = self._prompt_interrupt_feedback(phase=phase).strip()
                    last_interrupt = {
                        "phase": phase,
                        "source": interrupt_req.get("source", ""),
                        "note": interrupt_req.get("note", ""),
                        "requested_at": interrupt_req.get("request_ts"),
                        "feedback": feedback,
                        "feedback_empty": not bool(feedback),
                    }
                    task_resume_checkpoint = {
                        "task_id": task_id,
                        "task_goal": task_goal,
                        "resume_state": obs.get("resume_state") if isinstance(obs.get("resume_state"), dict) else {},
                    }
                    status = "interrupted_paused"
                    self._emit("RUN_PAUSED", category="run", task_id=task_id, payload={
                        "phase": phase,
                        "status": status,
                    })
                    self._write_task_state({
                        "schema_version": 2,
                        "lane": "standard",
                        "user_request": user_request,
                        "proposal": {
                            "proposal_path": self._proposal_path().name,
                            "work_packages": work_packages,
                        },
                        "tasks": tasks,
                        "observations": observations,
                        "status": status,
                        "hitl_history": hitl_history,
                        "task_resume_checkpoint": task_resume_checkpoint,
                        "last_interrupt": last_interrupt,
                    })
                    break
                observations.append(obs)
                task_resume_checkpoint = None
                task["status"] = obs["outcome"]
                self._emit("TASK_END", category="task", task_id=task_id, payload={
                    "outcome": outcome,
                    "summary_snippet": self._snippet(obs.get("summary", ""), 200),
                })
                self._trace_event("TASK_COMPLETED", {"task_id": task_id, "outcome": outcome})
                if outcome == "needs_intervention":
                    hitl_meta = self._hitl_collect_guidance(
                        user_request=user_request,
                        observations=observations,
                        tasks=tasks,
                        reason=f"task:{task_id}",
                        log_llm=log_llm,
                        plan_feedback_provider=plan_feedback_provider,
                    )
                    hitl_history.append(hitl_meta)
                    if not hitl_meta.get("feedback"):
                        status = "needs_intervention"
                    else:
                        status = "running"
                    self._write_task_state({
                        "schema_version": 2,
                        "lane": "standard",
                        "user_request": user_request,
                        "proposal": {
                            "proposal_path": self._proposal_path().name,
                            "work_packages": work_packages,
                        },
                        "tasks": tasks,
                        "observations": observations,
                        "status": status,
                        "hitl": hitl_meta,
                        "hitl_history": hitl_history,
                        "task_resume_checkpoint": task_resume_checkpoint,
                        "last_interrupt": last_interrupt,
                    })
                    break
                if outcome == "failure":
                    status = "failure"
                    self._write_task_state({
                        "schema_version": 2,
                        "lane": "standard",
                        "user_request": user_request,
                        "proposal": {
                            "proposal_path": self._proposal_path().name,
                            "work_packages": work_packages,
                        },
                        "tasks": tasks,
                        "observations": observations,
                        "status": status,
                        "hitl_history": hitl_history,
                        "task_resume_checkpoint": task_resume_checkpoint,
                        "last_interrupt": last_interrupt,
                    })
                    break
                self._write_task_state({
                    "schema_version": 2,
                    "lane": "standard",
                    "user_request": user_request,
                    "proposal": {
                        "proposal_path": self._proposal_path().name,
                        "work_packages": work_packages,
                    },
                    "tasks": tasks,
                    "observations": observations,
                    "status": status,
                    "hitl_history": hitl_history,
                    "task_resume_checkpoint": task_resume_checkpoint,
                    "last_interrupt": last_interrupt,
                })

        if status == "needs_intervention":
            self._emit("RUN_END", category="run", payload={
                "status": status,
                "run_dir": str(self.run_context.run_dir),
            })
            final_answer = observations[-1]["summary"] if observations else ""
            return {
                "tasks": tasks,
                "observations": observations,
                "summary": final_answer,
                "final_answer": final_answer,
                "status": status,
            }

        if status == "interrupted_paused":
            self._emit("RUN_END", category="run", payload={
                "status": status,
                "run_dir": str(self.run_context.run_dir),
            })
            final_answer = observations[-1]["summary"] if observations else ""
            return {
                "tasks": tasks,
                "observations": observations,
                "summary": final_answer,
                "final_answer": final_answer,
                "status": status,
            }

        if status != "failure":
            while True:
                director_resume = None
                if isinstance(task_resume_checkpoint, dict) and str(task_resume_checkpoint.get("task_id") or "") == "director":
                    candidate = task_resume_checkpoint.get("resume_state")
                    if isinstance(candidate, dict):
                        director_resume = candidate
                decision = self._director_decide(
                    user_request=user_request,
                    proposal_md=proposal_md,
                    work_packages=work_packages,
                    observations=observations,
                    resume_state=director_resume,
                )
                state = decision.get("state")
                if state == "Interrupted":
                    phase = str(decision.get("interrupt_phase") or "director")
                    interrupt_req = self.run_control.snapshot()
                    self._ack_interrupt(phase, {"state": "director"})
                    feedback = self._prompt_interrupt_feedback(phase=phase).strip()
                    last_interrupt = {
                        "phase": phase,
                        "source": interrupt_req.get("source", ""),
                        "note": interrupt_req.get("note", ""),
                        "requested_at": interrupt_req.get("request_ts"),
                        "feedback": feedback,
                        "feedback_empty": not bool(feedback),
                    }
                    task_resume_checkpoint = {
                        "task_id": "director",
                        "task_goal": "Decide next action",
                        "resume_state": decision.get("resume_state") if isinstance(decision.get("resume_state"), dict) else {},
                    }
                    status = "interrupted_paused"
                    self._emit("RUN_PAUSED", category="run", payload={
                        "phase": phase,
                        "status": status,
                    })
                    self._write_task_state({
                        "schema_version": 2,
                        "lane": "standard",
                        "user_request": user_request,
                        "proposal": {
                            "proposal_path": self._proposal_path().name,
                            "work_packages": work_packages,
                        },
                        "tasks": tasks,
                        "observations": observations,
                        "status": status,
                        "hitl_history": hitl_history,
                        "task_resume_checkpoint": task_resume_checkpoint,
                        "last_interrupt": last_interrupt,
                    })
                    break
                task_resume_checkpoint = None
                if state == "PerformNextTask":
                    task_goal, task_packet = self._resolve_task_goal_from_decision(decision)
                    task_id = f"task_{self._next_task_index(tasks):02d}"
                    tasks.append({
                        "task_id": task_id,
                        "goal": task_goal,
                        "task_packet": task_packet,
                        "status": "pending",
                    })
                    self._write_task_state({
                        "schema_version": 2,
                        "lane": "standard",
                        "user_request": user_request,
                        "proposal": {
                            "proposal_path": self._proposal_path().name,
                            "work_packages": work_packages,
                        },
                        "tasks": tasks,
                        "observations": observations,
                        "status": status,
                        "hitl_history": hitl_history,
                        "task_resume_checkpoint": task_resume_checkpoint,
                        "last_interrupt": last_interrupt,
                    })
                    self._trace_event("TASK_STARTED", {"task_id": task_id, "goal": task_goal})
                    self._emit("TASK_START", category="task", task_id=task_id, payload={"goal": task_goal})
                    resume_state = None
                    if isinstance(task_resume_checkpoint, dict) and str(task_resume_checkpoint.get("task_id") or "") == task_id:
                        resume_state = task_resume_checkpoint.get("resume_state")
                    obs = self._execute_task(
                        task_id=task_id,
                        task_goal=task_goal,
                        task_goal_short=task_packet.get("goal") if isinstance(task_packet, dict) else task_goal,
                        log_llm=log_llm,
                        resume_state=resume_state if isinstance(resume_state, dict) else None,
                    )
                    outcome = obs["outcome"]
                    if outcome == "interrupted":
                        phase = str(obs.get("interrupt_phase") or "toolcall")
                        interrupt_req = self.run_control.snapshot()
                        self._ack_interrupt(phase, {
                            "task_id": task_id,
                            "task_goal": task_goal,
                            "interrupted_toolcall": obs.get("interrupted_toolcall"),
                        })
                        feedback = self._prompt_interrupt_feedback(phase=phase).strip()
                        last_interrupt = {
                            "phase": phase,
                            "source": interrupt_req.get("source", ""),
                            "note": interrupt_req.get("note", ""),
                            "requested_at": interrupt_req.get("request_ts"),
                            "feedback": feedback,
                            "feedback_empty": not bool(feedback),
                        }
                        task_resume_checkpoint = {
                            "task_id": task_id,
                            "task_goal": task_goal,
                            "resume_state": obs.get("resume_state") if isinstance(obs.get("resume_state"), dict) else {},
                        }
                        status = "interrupted_paused"
                        self._emit("RUN_PAUSED", category="run", task_id=task_id, payload={
                            "phase": phase,
                            "status": status,
                        })
                        self._write_task_state({
                            "schema_version": 2,
                            "lane": "standard",
                            "user_request": user_request,
                            "proposal": {
                                "proposal_path": self._proposal_path().name,
                                "work_packages": work_packages,
                            },
                            "tasks": tasks,
                            "observations": observations,
                            "status": status,
                            "hitl_history": hitl_history,
                            "task_resume_checkpoint": task_resume_checkpoint,
                            "last_interrupt": last_interrupt,
                        })
                        break
                    observations.append(obs)
                    task_resume_checkpoint = None
                    tasks[-1]["status"] = obs["outcome"]
                    self._emit("TASK_END", category="task", task_id=task_id, payload={
                        "outcome": outcome,
                        "summary_snippet": self._snippet(obs.get("summary", ""), 200),
                    })
                    self._trace_event("TASK_COMPLETED", {"task_id": task_id, "outcome": outcome})
                    if outcome == "needs_intervention":
                        hitl_meta = self._hitl_collect_guidance(
                            user_request=user_request,
                            observations=observations,
                            tasks=tasks,
                            reason=f"task:{task_id}",
                            log_llm=log_llm,
                            plan_feedback_provider=plan_feedback_provider,
                        )
                        hitl_history.append(hitl_meta)
                        if not hitl_meta.get("feedback"):
                            status = "needs_intervention"
                        else:
                            status = "running"
                        self._write_task_state({
                            "schema_version": 2,
                            "lane": "standard",
                            "user_request": user_request,
                            "proposal": {
                                "proposal_path": self._proposal_path().name,
                                "work_packages": work_packages,
                            },
                            "tasks": tasks,
                            "observations": observations,
                            "status": status,
                            "hitl": hitl_meta,
                            "hitl_history": hitl_history,
                        })
                        if status == "needs_intervention":
                            break
                        continue
                    if outcome == "failure":
                        status = "failure"
                        self._write_task_state({
                            "schema_version": 2,
                            "lane": "standard",
                            "user_request": user_request,
                            "proposal": {
                                "proposal_path": self._proposal_path().name,
                                "work_packages": work_packages,
                            },
                            "tasks": tasks,
                            "observations": observations,
                            "status": status,
                            "hitl_history": hitl_history,
                        })
                        break
                    status = "running"
                    self._write_task_state({
                        "schema_version": 2,
                        "lane": "standard",
                        "user_request": user_request,
                        "proposal": {
                            "proposal_path": self._proposal_path().name,
                            "work_packages": work_packages,
                        },
                        "tasks": tasks,
                        "observations": observations,
                        "status": status,
                        "hitl_history": hitl_history,
                    })
                    continue

                if state == "MinorReviseProposal":
                    updated_md = decision.get("updated_proposal_md")
                    updated_packages = decision.get("updated_work_packages")
                    if not isinstance(updated_md, str) or not isinstance(updated_packages, list):
                        raise ValueError("Director MinorReviseProposal missing updated proposal fields")
                    proposal_md = updated_md
                    work_packages = updated_packages
                    proposal_relpath = self._write_proposal(proposal_md)
                    self._write_task_state({
                        "schema_version": 2,
                        "lane": "standard",
                        "user_request": user_request,
                        "proposal": {
                            "proposal_path": proposal_relpath,
                            "work_packages": work_packages,
                        },
                        "tasks": tasks,
                        "observations": observations,
                        "status": status,
                        "hitl_history": hitl_history,
                    })
                    continue

                if state == "MajorReviseProposal":
                    updated_md = decision.get("updated_proposal_md")
                    updated_packages = decision.get("updated_work_packages")
                    if not isinstance(updated_md, str) or not isinstance(updated_packages, list):
                        raise ValueError("Director MajorReviseProposal missing updated proposal fields")
                    if full_auto_major:
                        proposal_md = updated_md
                        work_packages = updated_packages
                        proposal_relpath = self._write_proposal(proposal_md)
                        self._write_task_state({
                            "schema_version": 2,
                            "lane": "standard",
                            "user_request": user_request,
                            "proposal": {
                                "proposal_path": proposal_relpath,
                                "work_packages": work_packages,
                            },
                            "tasks": tasks,
                            "observations": observations,
                            "status": status,
                            "hitl_history": hitl_history,
                        })
                        continue

                    proposal_md, work_packages, approved, feedback = self._review_proposal(
                        user_request=user_request,
                        proposal_md=updated_md,
                        work_packages=updated_packages,
                        log_llm=log_llm,
                        plan_feedback_provider=plan_feedback_provider,
                        allow_revise=False,
                        persist_fn=None,
                    )
                    if not approved:
                        status = "needs_intervention"
                        self._write_task_state({
                            "schema_version": 2,
                            "lane": "standard",
                            "user_request": user_request,
                            "proposal": {
                                "proposal_path": self._proposal_path().name,
                                "work_packages": work_packages,
                            },
                            "tasks": tasks,
                            "observations": observations,
                            "status": status,
                            "needs_human": {
                                "reason": "major_revision_rejected",
                                "feedback": feedback,
                                "questions": decision.get("questions_for_human"),
                            },
                            "hitl_history": hitl_history,
                        })
                        break
                    proposal_relpath = self._write_proposal(proposal_md)
                    self._write_task_state({
                        "schema_version": 2,
                        "lane": "standard",
                        "user_request": user_request,
                        "proposal": {
                            "proposal_path": proposal_relpath,
                            "work_packages": work_packages,
                        },
                        "tasks": tasks,
                        "observations": observations,
                        "status": status,
                        "hitl_history": hitl_history,
                    })
                    continue

                if state == "StopAndSynthesize":
                    status = "done"
                    break

                raise ValueError(f"Director returned unknown state: {state}")

        if status in ("done", "failure"):
            self._emit("FINAL_SUMMARY_START", category="final")
            summary = self._summarize_tasks(user_request, observations, status)
            self._write_task_state({
                "schema_version": 2,
                "lane": "standard",
                "user_request": user_request,
                "proposal": {
                    "proposal_path": self._proposal_path().name,
                    "work_packages": work_packages,
                },
                "tasks": tasks,
                "observations": observations,
                "status": status,
                "summary": summary,
                "hitl_history": hitl_history,
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
                "memory_report": report_paths.get("memory_report", ""),
                "latest_link": report_paths.get("latest_link", ""),
            })
            try:
                if hasattr(self.reporter, "show_final_summary") and self.reporter.is_live():
                    self.reporter.show_final_summary(summary)
            except Exception:
                pass
            return result

        self._emit("RUN_END", category="run", payload={
            "status": status,
            "run_dir": str(self.run_context.run_dir),
        })
        final_answer = observations[-1]["summary"] if observations else ""
        return {
            "tasks": tasks,
            "observations": observations,
            "summary": final_answer,
            "final_answer": final_answer,
            "status": status,
        }

    def run(
        self,
        user_request: str,
        *,
        log_llm: bool = False,
        initial_plan: Optional[Dict[str, Any]] = None,
        plan_review: bool = True,
        plan_feedback_provider: Optional[Callable[[Dict[str, Any]], str]] = None,
        lane: str = "standard",
        full_auto_major: bool = False,
        resume_feedback: str = "",
    ) -> Dict[str, Any]:
        if initial_plan is not None:
            raise ValueError("initial_plan is no longer supported; use lane=standard proposal flow instead")
        if lane not in SUPPORTED_LANES:
            raise ValueError(f"Invalid lane: {lane}")

        ui_started = False

        def start_ui(subtitle: str, show_splash: bool = True) -> None:
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

        subtitle = (
            "Initializing run context and preparing initial proposal..."
            if lane == "standard"
            else "Initializing run context..."
        )
        start_ui(subtitle, True)

        try:
            if lane == "fast":
                result = self._run_fast(
                    user_request,
                    log_llm=log_llm,
                    resume_feedback=resume_feedback,
                    defer_ui=False,
                    start_ui=start_ui,
                )
            else:
                result = self._run_standard(
                    user_request,
                    log_llm=log_llm,
                    resume_feedback=resume_feedback,
                    plan_review=plan_review,
                    plan_feedback_provider=plan_feedback_provider,
                    full_auto_major=full_auto_major,
                    defer_ui=False,
                    start_ui=start_ui,
                )
            return result
        except Exception as exc:
            self._emit("RUN_END", level="error", category="run", payload={
                "status": "error",
                "error": str(exc),
                "run_dir": str(self.run_context.run_dir) if hasattr(self, "run_context") else "",
            })
            raise
        finally:
            self._write_usage_summary()
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

    def _hitl_collect_guidance(
        self,
        *,
        user_request: str,
        observations: List[Dict[str, Any]],
        tasks: List[Dict[str, Any]],
        reason: str = "",
        log_llm: bool = False,
        plan_feedback_provider: Optional[Callable[[Dict[str, Any]], str]] = None,
    ) -> Dict[str, Any]:
        hitl_index = self._next_hitl_index()
        hitl_tag = f"hitl_{hitl_index:03d}"
        hitl_dir = self.run_context.run_dir / "hitl" / hitl_tag
        hitl_dir.mkdir(parents=True, exist_ok=True)

        summary = self._summarize_tasks(user_request, observations, status="needs_intervention")
        report_text = summary or ""
        interrupted_report_path = hitl_dir / "interrupted_report.md"
        interrupted_report_path.write_text(report_text or "", encoding="utf-8")

        feedback_state = {
            "user_request": user_request,
            "tasks": tasks,
            "observations": observations,
            "status": "needs_intervention",
            "report_text": report_text,
            "report_path": self._rel_run_path(interrupted_report_path),
            "hitl_dir": str(hitl_dir),
            "hitl_id": hitl_tag,
            "reason": reason,
        }
        feedback = ""
        report_ref = ""
        try:
            report_ref = str(
                interrupted_report_path.resolve().relative_to(workspace_root(self.run_context.workspace))
            )
        except Exception:
            report_ref = self._rel_run_path(interrupted_report_path)

        if plan_feedback_provider:
            feedback = plan_feedback_provider({**feedback_state, "stage": "hitl_feedback"}) or ""
        else:
            if hasattr(self.reporter, "prompt_hitl_feedback") and self.reporter.is_live():
                feedback = self.reporter.prompt_hitl_feedback(
                    report_text=report_text,
                    report_path=report_ref,
                )
            else:
                raise ValueError("HITL feedback requires a live reporter (WebUI) or a feedback provider.")

        human_feedback_path = hitl_dir / "human_feedback.txt"
        human_feedback_path.write_text(feedback or "", encoding="utf-8")

        hitl_meta = {
            "hitl_id": hitl_tag,
            "hitl_dir": self._rel_run_path(hitl_dir),
            "interrupted_report": self._rel_run_path(interrupted_report_path),
            "human_feedback": self._rel_run_path(human_feedback_path),
            "report_path": report_ref,
            "feedback": feedback or "",
        }

        if not feedback:
            return hitl_meta

        guidance_snippet = self._snippet(feedback, 9999)
        constraint_text = f"HITL guidance ({hitl_tag}): {guidance_snippet}"
        if report_ref:
            constraint_text += f" | report: {report_ref}"
        try:
            merge_info = self._merge_memory_via_git_apply(
                run_id=self.run_context.run_id,
                task_id=hitl_tag,
                outcome="needs_intervention",
                task_goal_short=f"HITL guidance ({hitl_tag})",
                structured_result={
                    "summary": f"HITL guidance captured from human feedback ({hitl_tag}).",
                    "facts": [],
                    "files": [],
                    "constraints": [constraint_text],
                    "open_questions": [],
                    "decisions": [],
                    "next_steps": [],
                    "artifacts": [report_ref] if report_ref else [],
                },
            )
            hitl_meta["memory_update_status"] = "applied"
            hitl_meta["memory_event"] = merge_info.get("event_path", "")
        except Exception as exc:
            hitl_meta["memory_update_status"] = "failed"
            hitl_meta["memory_update_error"] = str(exc)
        return hitl_meta

    def _initialize_memory_goal(self, user_request: str) -> None:
        goal = " ".join((user_request or "").split()).strip()
        if not goal:
            return
        self.memory_store.ensure_exists()
        goal_path = self.memory_store.topics_dir / "GOAL.md"
        original = goal_path.read_text(encoding="utf-8") if goal_path.exists() else ""
        updated = original

        primary_match = re.search(r"(?m)^- Primary objective:\s*(.*)$", updated)
        current_primary = (primary_match.group(1) if primary_match else "").strip()
        primary_changed = current_primary != goal

        if primary_match:
            updated = re.sub(
                r"(?m)^- Primary objective:.*$",
                f"- Primary objective: {goal}",
                updated,
                count=1,
            )
        else:
            line = f"- Primary objective: {goal}\n"
            marker = "## TL;DR\n"
            if marker in updated:
                updated = updated.replace(marker, marker + line, 1)
            else:
                updated = updated.rstrip()
                if updated:
                    updated += "\n\n"
                updated += "## TL;DR\n" + line
            primary_changed = True

        lines = updated.splitlines()
        compacted: List[str] = []
        seen_change_log = False
        for raw in lines:
            if raw.strip() == "## Change log":
                if seen_change_log:
                    continue
                seen_change_log = True
            compacted.append(raw)
        updated = "\n".join(compacted)

        if primary_changed:
            ts = datetime.utcnow().strftime("%Y-%m-%d")
            entry = f"- [{ts}] {goal}"
            if "## Change log" not in updated:
                updated = updated.rstrip()
                if updated:
                    updated += "\n\n"
                updated += "## Change log\n"
            if entry not in updated:
                updated = updated.rstrip() + "\n" + entry + "\n"

        normalized = updated.rstrip() + "\n"
        if normalized != (original.rstrip() + "\n"):
            goal_path.write_text(normalized, encoding="utf-8")

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
        body = dict(payload or {})
        existing: Dict[str, Any] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                existing = {}
        status = str(body.get("status") or "")
        if "interrupt_history" not in body and "interrupt_history" in existing:
            body["interrupt_history"] = existing.get("interrupt_history")
        if status == "interrupted_paused":
            for key in ("task_resume_checkpoint", "last_interrupt"):
                if key not in body and key in existing:
                    body[key] = existing.get(key)
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")

    def _summarize_tasks(self, user_request: str, observations: List[Dict[str, Any]], status: str) -> str:
        fallback = self._summarize_tasks_fallback(user_request, observations)
        try:
            memory_index_excerpt = self.memory_store.read_index(max_lines=200, max_chars=12000)
        except Exception:
            memory_index_excerpt = ""
        artifacts = self._artifact_log_excerpt(limit=200)
        try:
            messages = self.summary_prompt.format_messages(
                user_request=user_request,
                observations=json.dumps(observations, ensure_ascii=False),
                memory_index_excerpt=memory_index_excerpt,
                artifacts=json.dumps(artifacts, ensure_ascii=False),
                status=status,
            )
            self._emit("LLM_CALL_START", category="llm", payload={"kind": "final_summary"})
            t0 = time.perf_counter()
            resp = self.summary_llm.invoke(messages)
            raw = llm_text(resp).strip()
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
        return self.memory_store.artifact_index(limit=limit)

    def _resolve_task_goal_from_decision(self, decision: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
        packet = decision.get("task_packet")
        if isinstance(packet, dict):
            goal = str(packet.get("goal") or "").strip()
            if not goal:
                raise ValueError("Director PerformNextTask missing task_packet.goal")
            success = str(packet.get("success_criteria") or "").strip()
            outputs = self._clean_text_list(packet.get("expected_outputs"))
            hints = self._clean_text_list(packet.get("memory_hints"))
            paths = self._clean_text_list(packet.get("path_hints"))
            suggested_tools = self._normalize_suggested_tools(packet.get("suggested_tools"))
            rendered = [goal]
            if success:
                rendered.append(f"Success criteria: {success}")
            if outputs:
                rendered.append("Expected outputs: " + "; ".join(outputs))
            if hints:
                rendered.append("Memory hints: " + ", ".join(hints))
            if paths:
                rendered.append("Path hints: " + ", ".join(paths))
            task_goal = self._with_suggested_tools_hint(" ".join(rendered), suggested_tools)
            packet_norm = {
                "goal": goal,
                "success_criteria": success,
                "expected_outputs": outputs,
                "memory_hints": hints,
                "path_hints": paths,
                "suggested_tools": suggested_tools,
            }
            return task_goal, packet_norm

        task_goal_raw = decision.get("next_task_goal")
        if not task_goal_raw:
            raise ValueError("Director PerformNextTask missing next_task_goal")
        task_goal = self._with_suggested_tools_hint(
            task_goal_raw,
            decision.get("suggested_tools"),
        )
        packet_norm = {
            "goal": str(task_goal_raw).strip(),
            "success_criteria": str(decision.get("success_criteria") or "").strip(),
            "expected_outputs": self._clean_text_list(decision.get("expected_outputs")),
            "memory_hints": [],
            "path_hints": [],
            "suggested_tools": self._normalize_suggested_tools(decision.get("suggested_tools")),
        }
        return task_goal, packet_norm

    @staticmethod
    def _with_suggested_tools_hint(task_goal: Any, suggested_tools: Any) -> str:
        base = str(task_goal or "").strip()
        if not base:
            return base
        lowered = base.lower()
        if "suggested tools:" in lowered or "tool-call hints:" in lowered or "建议工具" in lowered:
            return base
        tools = Orchestrator._normalize_suggested_tools(suggested_tools)
        if not tools:
            return base
        hint = f"(Tool-call hints: {', '.join(tools)}; optional, function-tool names not shell commands)"
        return f"{base} {hint}"

    @staticmethod
    def _normalize_suggested_tools(value: Any, *, limit: int = 5) -> List[str]:
        if not isinstance(value, list):
            return []
        cleaned: List[str] = []
        seen: set[str] = set()
        for item in value:
            tool = str(item or "").strip()
            if not tool:
                continue
            key = tool.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(tool)
            if len(cleaned) >= limit:
                break
        return cleaned

    def _publish_report(self, user_request: str, final_answer: str) -> Dict[str, str]:
        reports_dir = workspace_root(self.run_context.workspace) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        run_reports_dir = self.run_context.run_dir / "reports"
        run_reports_dir.mkdir(parents=True, exist_ok=True)

        run_final_report = run_reports_dir / "FINAL_REPORT.md"
        run_final_report.write_text(
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

        memory_src = self.memory_store.index_path
        run_memory_dst = run_reports_dir / "MEMORY.md"
        try:
            shutil.copy2(memory_src, run_memory_dst)
        except Exception:
            # Best-effort copy
            if memory_src.exists():
                run_memory_dst.write_text(memory_src.read_text(encoding="utf-8"), encoding="utf-8")

        latest_run_copy = reports_dir / "latest_run"
        try:
            if latest_run_copy.is_symlink() or latest_run_copy.is_file():
                latest_run_copy.unlink()
            elif latest_run_copy.is_dir():
                shutil.rmtree(latest_run_copy)
        except Exception:
            pass

        try:
            shutil.copytree(self.run_context.run_dir, latest_run_copy, symlinks=False)
        except Exception:
            # Fallback: keep latest_run as a lightweight report snapshot.
            try:
                if latest_run_copy.exists():
                    if latest_run_copy.is_dir():
                        shutil.rmtree(latest_run_copy)
                    else:
                        latest_run_copy.unlink()
            except Exception:
                pass
            (latest_run_copy / "reports").mkdir(parents=True, exist_ok=True)
            shutil.copy2(run_final_report, latest_run_copy / "reports" / "FINAL_REPORT.md")
            if run_memory_dst.exists():
                shutil.copy2(run_memory_dst, latest_run_copy / "reports" / "MEMORY.md")
            (latest_run_copy / "SOURCE_RUN_DIR.txt").write_text(str(self.run_context.run_dir), encoding="utf-8")

        self._write_latest_run_readme(latest_run_copy)

        return {
            "run_dir": str(self.run_context.run_dir),
            "final_report": str(run_final_report),
            "memory_report": str(run_memory_dst),
            # Explicit run-scoped report paths.
            "run_final_report": str(run_final_report),
            "run_memory_report": str(run_memory_dst),
            # Workspace-level latest run snapshot (copy, not symlink).
            "latest_link": str(latest_run_copy),
            "workspace_latest_link": str(latest_run_copy),
        }

    def _write_latest_run_readme(self, latest_run_copy: Path) -> None:
        readme = latest_run_copy / "README.md"
        text = "\n".join([
            "# latest_run snapshot",
            "",
            "This directory is an audit/debug snapshot of the most recent run.",
            "It is not canonical memory for planning.",
            "",
            "Preferred sources for agent context:",
            "- files/MEMORY/**",
            "- reports/FINAL_REPORT.md",
            "",
            "Open files here only when debugging or when a specific evidence pointer is missing.",
            "",
        ])
        try:
            readme.parent.mkdir(parents=True, exist_ok=True)
            readme.write_text(text, encoding="utf-8")
        except Exception:
            return

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
    def _messages_to_input_items(messages: List[Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for msg in messages:
            role = getattr(msg, "role", None) or getattr(msg, "type", "user")
            if role == "human":
                role = "user"
            elif role == "ai":
                role = "assistant"
            content = getattr(msg, "content", str(msg))
            items.append(message_item(role, content))
        return items

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
