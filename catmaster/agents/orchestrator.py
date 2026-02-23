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
from catmaster.llm.config import LLMProfile, TOOL_CALLING_AGENT_ROLES, LLMConfig
from catmaster.llm.factory import build_chat_model, build_tool_driver
from catmaster.runtime.conversation_state import message_item
from catmaster.runtime.tool_policy import ToolPolicy
from catmaster.runtime.tool_backend import ToolBackend
from catmaster.runtime.local_tool_backend import LocalToolBackend
from catmaster.ui import Reporter, NullReporter, make_event
from catmaster.agents.orchestrator_prompts import (
    build_task_step_prompt,
    build_task_step_repair_prompt,
    build_summary_prompt,
    build_proposal_prompt,
    build_proposal_feedback_prompt,
    build_director_prompt,
    build_memory_patch_prompt,
    build_memory_patch_repair_prompt,
)

_PROPOSAL_TOOL_ALLOWLIST = [
    "bash_exec",
]
_DIRECTOR_TOOL_ALLOWLIST = [
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
        self._tool_drivers_by_role: Dict[str, ToolCallingDriver] = {}
        profile = llm_profile
        if llm is None:
            profile = profile or LLMProfile.from_env_or_file()
            self.llm_profile = profile
        if profile is not None:
            task_cfg = profile.config_for_role("task_runner")
            self._llm_provider = task_cfg.provider
            self._llm_base_url = task_cfg.base_url
            self._tool_driver_kind = task_cfg.tool_calling.driver
            self._supports_builtin_tools = bool(task_cfg.tool_calling.supports_builtin_tools)
            if llm is None:
                llm = build_chat_model(profile.config_for_role("memory_patch"))
                summary_llm = summary_llm or build_chat_model(profile.config_for_role("summary"))
            if tool_driver is None:
                for role in TOOL_CALLING_AGENT_ROLES:
                    self._tool_drivers_by_role[role] = build_tool_driver(profile.config_for_role(role))
            else:
                for role in TOOL_CALLING_AGENT_ROLES:
                    self._tool_drivers_by_role[role] = tool_driver
        elif tool_driver is not None:
            for role in TOOL_CALLING_AGENT_ROLES:
                self._tool_drivers_by_role[role] = tool_driver
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

        self.proposal_prompt = build_proposal_prompt()
        self.proposal_feedback_prompt = build_proposal_feedback_prompt()
        self.director_prompt = build_director_prompt()
        self.task_step_prompt = build_task_step_prompt()
        self.task_step_repair_prompt = build_task_step_repair_prompt()
        self.memory_patch_prompt = build_memory_patch_prompt()
        self.memory_patch_repair_prompt = build_memory_patch_repair_prompt()
        self.summary_prompt = build_summary_prompt()
        self.tool_driver = self._tool_drivers_by_role.get("task_runner") or tool_driver
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
        for role in TOOL_CALLING_AGENT_ROLES:
            self._tool_drivers_by_role.setdefault(role, self.tool_driver)
        self._emit("RUN_INIT_DONE", payload={
            "run_id": self.run_context.run_id,
            "run_dir": str(self.run_context.run_dir),
            "model_name": self._resolve_model_name(),
            "model_label": self._resolve_model_label(),
            "provider": self._llm_provider or "",
            "driver_kind": self._tool_driver_kind or "",
            "base_url": self._llm_base_url or "",
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

    def _proposal_function_tools(self) -> list[dict]:
        if not self._proposal_browse_tools_enabled():
            return []
        tools = self._filtered_function_tools()
        return [
            tool for tool in tools
            if tool.get("name") in _PROPOSAL_TOOL_ALLOWLIST
        ]

    def _director_function_tools(self) -> list[dict]:
        tools = self._filtered_function_tools()
        return [
            tool for tool in tools
            if tool.get("name") in _DIRECTOR_TOOL_ALLOWLIST
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

    def _role_llm_config(self, role: str) -> Optional[LLMConfig]:
        if self.llm_profile is None:
            return None
        try:
            return self.llm_profile.config_for_role(role)
        except Exception:
            return None

    def _role_tool_driver(self, role: str) -> ToolCallingDriver:
        driver = self._tool_drivers_by_role.get(role)
        if driver is not None:
            return driver
        if self.tool_driver is None:
            raise ValueError(f"No tool driver available for role: {role}")
        return self.tool_driver

    def _resolve_model_name(self) -> str:
        cfg = self._role_llm_config("task_runner")
        if cfg is not None:
            model = getattr(cfg, "model", None)
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

    def _collect_model_kwargs(self, role: str = "task_runner") -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        cfg = self._role_llm_config(role)
        if cfg is not None:
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
        if cfg is None:
            model_obj = self.summary_llm if role == "summary" else self.llm
            raw = getattr(model_obj, "model_kwargs", None)
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
                value = getattr(model_obj, key, None)
                if value is None or key in merged:
                    continue
                merged[key] = value
        return merged

    def _tool_driver_kwargs(self, role: str = "task_runner") -> Dict[str, Any]:
        kwargs = self._collect_model_kwargs(role)
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
        cfg = self._role_llm_config(role)
        if cfg is not None:
            tool_calling = getattr(cfg, "tool_calling", None)
            if tool_calling is not None:
                request_options = getattr(tool_calling, "request_options", None)
                if isinstance(request_options, dict) and request_options:
                    driver_kwargs.update(dict(request_options))
                extra_body = getattr(tool_calling, "extra_body", None)
                driver_kind = str(getattr(tool_calling, "driver", "") or "")
                if (
                    driver_kind == "openai_chat_completions"
                    and isinstance(extra_body, dict)
                    and extra_body
                ):
                    existing = driver_kwargs.get("extra_body")
                    merged = dict(existing) if isinstance(existing, dict) else {}
                    merged.update(extra_body)
                    driver_kwargs["extra_body"] = merged
        return driver_kwargs

    def _supports_builtin_tools_for(self, role: str) -> bool:
        cfg = self._role_llm_config(role)
        if cfg is None:
            return bool(self._supports_builtin_tools)
        tool_calling = getattr(cfg, "tool_calling", None)
        if tool_calling is None:
            return False
        return bool(getattr(tool_calling, "supports_builtin_tools", False))

    def _proposal_browse_tools_enabled(self) -> bool:
        if self.llm_profile is None:
            return True
        policies = getattr(self.llm_profile, "agent_policies", None)
        if policies is None:
            return True
        proposal_policy = getattr(policies, "proposal", None)
        if proposal_policy is None:
            return True
        value = getattr(proposal_policy, "browse_tools_enabled", None)
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

    @staticmethod
    def _json_roundtrip(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return str(value)

    @staticmethod
    def _compact_params_for_memory(params: Any, max_items: int = 6, max_len: int = 220) -> str:
        if not isinstance(params, dict):
            return Orchestrator._snippet(params, max_len)
        parts: List[str] = []
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
        return Orchestrator._snippet(", ".join(parts), max_len)

    @staticmethod
    def _compact_tool_output_for_memory(
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
                "error": Orchestrator._snippet(tool_output, text_limit),
            }

        compact: Dict[str, Any] = {
            "status": str(tool_output.get("status") or ""),
            "tool_name": str(tool_output.get("tool_name") or tool_name or ""),
            "warnings": list(tool_output.get("warnings") or [])[:6],
            "error": Orchestrator._snippet(tool_output.get("error"), text_limit),
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
                "stdout",
                "stderr",
                "blocked_reason",
            ):
                if key not in data:
                    continue
                value = data.get(key)
                if isinstance(value, str):
                    value = Orchestrator._snippet(value, text_limit)
                compact_data[key] = value
            if "stdout" not in compact_data and data.get("stdout"):
                compact_data["stdout"] = Orchestrator._snippet(data.get("stdout"), text_limit)
            if "stderr" not in compact_data and data.get("stderr"):
                compact_data["stderr"] = Orchestrator._snippet(data.get("stderr"), text_limit)
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
                paths[key_s] = Orchestrator._snippet(value, 300)
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
    def _compact_local_observations_for_memory(value: Any) -> List[Dict[str, Any]]:
        if not isinstance(value, list):
            return []
        compact: List[Dict[str, Any]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            row: Dict[str, Any] = {}
            step = item.get("step")
            if isinstance(step, (int, float, str)):
                row["step"] = step
            method = str(item.get("method") or "").strip()
            if method:
                row["method"] = method
            if "params" in item:
                row["params_compact"] = Orchestrator._compact_params_for_memory(item.get("params"))
            result = item.get("result")
            if isinstance(result, dict):
                row["result"] = Orchestrator._compact_tool_output_for_memory(
                    method,
                    result,
                )
            elif isinstance(result, list):
                row["result"] = {
                    "list_len": len(result),
                    "list_preview": result[:6],
                }
            elif result is not None:
                row["result"] = Orchestrator._snippet(result, 400)
            if row:
                compact.append(row)
        return compact

    def _write_toolcall_context_log(
        self,
        *,
        task_id: str,
        suffix: str,
        content: Dict[str, Any],
    ) -> str:
        run_root = self.run_context.run_dir
        out_dir = run_root / "audit" / "toolcall_context"
        out_dir.mkdir(parents=True, exist_ok=True)
        name = f"{self.run_context.run_id}_{task_id}_{suffix}.json"
        out_path = out_dir / name
        payload = self._json_roundtrip(content)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            return str(out_path.relative_to(run_root))
        except Exception:
            return str(out_path)

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
            driver=self._role_tool_driver("proposal"),
            backend=self.tool_backend,
            prompt=None,
            control_tools=get_proposal_control_tool_schemas(),
            control_tool_names=PROPOSAL_CONTROL_TOOL_NAMES,
            trace_store=self.trace_store,
            checkpoint_store=self.checkpoint_store,
            reporter=self.reporter,
            max_steps=self.max_plan_steps,
            driver_kwargs={
                **self._tool_driver_kwargs("proposal"),
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
            driver=self._role_tool_driver("proposal"),
            backend=self.tool_backend,
            prompt=None,
            control_tools=get_proposal_control_tool_schemas(),
            control_tool_names=PROPOSAL_CONTROL_TOOL_NAMES,
            trace_store=self.trace_store,
            checkpoint_store=self.checkpoint_store,
            reporter=self.reporter,
            max_steps=self.max_plan_steps,
            driver_kwargs={
                **self._tool_driver_kwargs("proposal"),
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
        director_observations = self._director_observations_view(observations)
        function_tools = self._filtered_function_tools()
        director_function_tools = self._director_function_tools()
        builtin_tools = self.tool_policy.builtin_tools if self._supports_builtin_tools_for("director") else []
        tools_for_director = self._tool_descriptions_from_tools(function_tools, builtin_tools, [])
        messages = self.director_prompt.format_messages(
            user_request=user_request,
            proposal_md=proposal_md,
            work_packages_json=json.dumps(work_packages, ensure_ascii=False),
            memory_index_excerpt=memory_index_excerpt,
            already_done_json=json.dumps(director_observations, ensure_ascii=False),
            tools=tools_for_director,
        )
        input_items = self._messages_to_input_items(messages)
        stepper = ToolCallingTaskStepper(
            driver=self._role_tool_driver("director"),
            backend=self.tool_backend,
            prompt=None,
            control_tools=get_director_control_tool_schemas(),
            control_tool_names=DIRECTOR_CONTROL_TOOL_NAMES,
            trace_store=self.trace_store,
            checkpoint_store=self.checkpoint_store,
            reporter=self.reporter,
            max_steps=self.max_plan_steps,
            driver_kwargs={
                **self._tool_driver_kwargs("director"),
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
            function_tools=director_function_tools,
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
            for key in ("task_id", "outcome", "summary", "failure_kind"):
                value = item.get(key)
                if value is None:
                    continue
                text = " ".join(str(value).split())
                if text:
                    row[key] = text
            if bool(item.get("auto_replan", False)):
                row["auto_replan"] = True

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
        proposal_feedback_provider: Optional[Callable[[Dict[str, Any]], str]],
        allow_revise: bool = True,
        persist_fn: Optional[Callable[[str, List[str]], None]] = None,
    ) -> tuple[str, List[str], bool, str]:
        review_state = {
            "user_request": user_request,
            "proposal_md": proposal_md,
            "work_packages": work_packages,
            "proposal": {"work_packages": list(work_packages), "proposal_md": proposal_md},
            "feedback_history": [],
            "approved": False,
            "round": 0,
        }
        if proposal_feedback_provider is None and not self.reporter.is_live():
            raise ValueError("proposal_review requires a live reporter (WebUI). Start WebUI or disable proposal_review.")

        last_feedback = ""
        while not review_state.get("approved"):
            proposal_description = (review_state["proposal_md"] or "").strip()
            self._emit("PROPOSAL_REVIEW_SHOW", category="plan", payload={
                "todo": review_state["work_packages"],
                "proposal_description_snippet": self._snippet(proposal_description, 240),
            })
            self._emit("PROPOSAL_REVIEW_WAIT_INPUT", category="plan")
            if proposal_feedback_provider:
                feedback = proposal_feedback_provider({
                    **review_state,
                    "stage": "proposal_review",
                })
            else:
                if hasattr(self.reporter, "prompt_proposal_feedback") and self.reporter.is_live():
                    feedback = self.reporter.prompt_proposal_feedback(
                        todo=review_state["work_packages"],
                        proposal_description=proposal_description,
                    )
                else:
                    raise ValueError("proposal_review requires a live reporter (WebUI). Start WebUI or disable proposal_review.")
            if not feedback:
                if proposal_feedback_provider:
                    raise ValueError("proposal_review feedback cannot be empty")
                self._emit("PROPOSAL_REVIEW_WAIT_INPUT", category="plan", payload={"error": "empty_input"})
                continue

            last_feedback = feedback
            if self._is_proposal_review_approved(feedback):
                review_state["approved"] = True
                review_state["feedback_history"].append({
                    "round": review_state.get("round", 0),
                    "feedback": feedback,
                    "approved": True,
                })
                self._emit("PROPOSAL_REVIEW_APPROVED", category="plan", payload={
                    "feedback_snippet": self._snippet(feedback, 160),
                })
                break
            if not allow_revise:
                return proposal_md, work_packages, False, feedback

            self._emit("PROPOSAL_REVIEW_REVISING", category="plan", payload={
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
            review_state["proposal"] = {"work_packages": list(work_packages), "proposal_md": proposal_md}
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
        task_packet: Optional[Dict[str, Any]] = None,
        log_llm: bool,
        resume_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        context_pack = self.context_builder.build(
            task_goal,
            role="task_runner",
            policy=ContextPackPolicy(
                memory_head_lines=200,
                max_memory_chars=12000,
                inject_goal_for_worker=False,
            ),
        )
        packet = task_packet if isinstance(task_packet, dict) else {}
        goal_text = str(packet.get("goal") or task_goal_short or task_goal).strip() or task_goal
        task_detail = str(packet.get("task_detail") or "").strip()
        expected_outputs = self._clean_text_list(packet.get("expected_outputs"))
        suggested_tools = self._normalize_suggested_tools(packet.get("suggested_tools"))
        reference_hint = self._clean_text_list(packet.get("reference_hint"))
        context_pack["goal"] = goal_text
        context_pack["task_detail"] = task_detail or "(none)"
        context_pack["expected_outputs"] = "\n".join(f"- {item}" for item in expected_outputs) if expected_outputs else "(none)"
        context_pack["suggested_tools"] = ", ".join(suggested_tools) if suggested_tools else "(none)"
        context_pack["reference_hint"] = "\n".join(f"- {item}" for item in reference_hint) if reference_hint else "(none)"
        if self._interrupt_context_note:
            base_detail = str(context_pack.get("task_detail", "") or "").strip()
            if base_detail and base_detail != "(none)":
                context_pack["task_detail"] = f"{base_detail}\n\nInterrupt guidance:\n- {self._interrupt_context_note}"
            else:
                context_pack["task_detail"] = f"Interrupt guidance:\n- {self._interrupt_context_note}"
        self._emit("TASK_CONTEXT_READY", category="task", task_id=task_id, payload={
            "excerpt_chars": len(context_pack.get("memory_index_excerpt", "") or ""),
            "reference_hint_count": len(reference_hint),
            "suggested_tools_count": len(suggested_tools),
        })
        filtered_tools = self._filtered_function_tools()
        builtin_tools = self.tool_policy.builtin_tools if self._supports_builtin_tools_for("task_runner") else []
        stepper = ToolCallingTaskStepper(
            driver=self._role_tool_driver("task_runner"),
            backend=self.tool_backend,
            prompt=self.task_step_prompt,
            reporter=self.reporter,
            max_steps=min(self.max_steps, self.tool_policy.max_tool_calls_per_task),
            driver_kwargs={
                **self._tool_driver_kwargs("task_runner"),
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
            task_goal=goal_text,
            context_pack=context_pack,
            initial_instruction=None,
            function_tools=filtered_tools,
            builtin_tools=builtin_tools,
            resume_state=resume_state,
        )
        local_observations_raw = step_result.get("local_observations")
        local_observations = self._json_roundtrip(
            local_observations_raw if isinstance(local_observations_raw, list) else []
        )
        local_observations_for_memory = self._compact_local_observations_for_memory(local_observations)
        finish_reason_raw = str(step_result.get("finish_reason") or "")
        if finish_reason_raw == "interrupted":
            return {
                "task_id": task_id,
                "outcome": "interrupted",
                "summary": "Execution interrupted by user.",
                "resume_state": step_result.get("resume_state"),
                "interrupt_phase": step_result.get("interrupt_phase", "toolcall"),
                "interrupted_toolcall": step_result.get("interrupted_toolcall"),
            }
        finish_reason = finish_reason_raw
        control_payload = step_result.get("control_payload")
        max_steps_context_rel = ""
        if finish_reason == "max_steps":
            max_limit = min(self.max_steps, self.tool_policy.max_tool_calls_per_task)
            try:
                max_steps_context_rel = self._write_toolcall_context_log(
                    task_id=task_id,
                    suffix="max_steps",
                    content={
                        "task_id": task_id,
                        "finish_reason": "max_steps",
                        "max_tool_calls_limit": int(max_limit),
                        "toolcall_context": local_observations_for_memory,
                    },
                )
            except Exception as exc:
                max_steps_context_rel = ""
                self._emit("TOOLCALL_CONTEXT_LOG_FAILED", level="warning", category="task", task_id=task_id, payload={
                    "error": self._snippet(str(exc), 240),
                })
            facts = [
                f"failure_kind=max_steps",
                f"max_tool_calls_limit={int(max_limit)}",
                f"toolcall_observations={len(local_observations_for_memory)}",
            ]
            files: List[Dict[str, str]] = []
            artifacts: List[str] = []
            control_payload = {
                "error": (
                    f"Tool-call limit reached ({int(max_limit)}) before completion "
                    "without task_finish/task_fail."
                ),
                "needs_human": False,
                "hint": "Auto-replan with smaller scope and continue from existing outputs.",
                "auto_replan": True,
                "failure_kind": "max_steps",
                "partial_result": {
                    "summary": "Task hit tool-call limit; prepared for director auto-replan.",
                    "facts": facts,
                    "files": files,
                    "constraints": [],
                    "open_questions": [],
                    "decisions": [
                        {
                            "decision": "Stop current task at tool-call limit.",
                            "rationale": "Prevent open-ended loop and hand off to director for route adjustment.",
                        }
                    ],
                    "next_steps": [
                        "Director should split or narrow the task and re-dispatch.",
                        "Reuse existing outputs and rerun only unfinished subset.",
                    ],
                    "artifacts": artifacts,
                },
            }
            finish_reason = "task_fail"
        task_result = self._normalize_task_result_payload(
            task_goal=task_goal,
            finish_reason=finish_reason,
            payload=control_payload,
            output_text=str(step_result.get("output_text") or ""),
        )
        if str(task_result.get("failure_kind") or "") == "max_steps":
            task_result["structured_result"]["toolcall_context"] = local_observations_for_memory
            task_result["structured_result"]["toolcall_context_count"] = len(local_observations_for_memory)
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
            "auto_replan": bool(task_result.get("auto_replan", False)),
            "failure_kind": str(task_result.get("failure_kind") or ""),
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
                "auto_replan": False,
                "failure_kind": "",
            }

        if finish_reason == "task_fail":
            partial = body.get("partial_result") if isinstance(body.get("partial_result"), dict) else {}
            err = str(body.get("error") or output_text or "Task failed").strip()
            needs_human = bool(body.get("needs_human", True))
            auto_replan = bool(body.get("auto_replan", False))
            failure_kind = str(body.get("failure_kind") or "").strip()
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
                "auto_replan": auto_replan,
                "failure_kind": failure_kind,
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
            "auto_replan": False,
            "failure_kind": str(finish_reason or "unknown").strip(),
        }

    def _commit_director_memory(
        self,
        *,
        commit_reason: str,
        proposal_md: str,
        work_packages: List[str],
        proposal_path: str,
        decision_state: str = "",
        rationale: str = "",
        change_log: str = "",
    ) -> None:
        commit_tag = re.sub(r"[^a-z0-9]+", "_", commit_reason.lower()).strip("_") or "director_commit"
        commit_id = f"director_{commit_tag}_{datetime.utcnow().strftime('%H%M%S%f')}"
        proposal_focus = self._snippet(" ".join(str(proposal_md or "").split()), 600)
        clean_packages = self._clean_text_list(work_packages)
        summary = self._snippet(" ".join(commit_reason.split()), 500) or "Director committed planning decision."

        facts: List[str] = []
        if proposal_focus:
            facts.append(f"Run focus: {proposal_focus}")
        if clean_packages:
            facts.append(f"Work package count: {len(clean_packages)}")
            facts.append("Work packages: " + " | ".join(clean_packages[:8]))

        decisions: List[Dict[str, str]] = []
        clean_rationale = self._snippet(" ".join(str(rationale or "").split()), 600)
        clean_change_log = self._snippet(" ".join(str(change_log or "").split()), 600)
        if decision_state:
            decisions.append({
                "topic": "director_decision",
                "decision": decision_state,
                "rationale": clean_rationale,
            })
        if clean_change_log:
            decisions.append({
                "topic": "proposal_change_log",
                "decision": clean_change_log,
                "rationale": "Major route revision accepted.",
            })

        structured_result = {
            "summary": summary,
            "facts": facts,
            "files": [{"path": proposal_path, "description": "Current approved proposal", "kind": "plan"}] if proposal_path else [],
            "constraints": [],
            "open_questions": [],
            "decisions": decisions,
            "next_steps": clean_packages[:5],
            "artifacts": [proposal_path] if proposal_path else [],
        }

        self._emit("MEMORY_MERGE_START", category="summary", task_id=commit_id, payload={
            "source": "director",
            "reason": summary,
        })
        try:
            merge_info = self._merge_memory_via_git_apply(
                run_id=self.run_context.run_id,
                task_id=commit_id,
                outcome="success",
                task_goal_short=summary,
                structured_result=structured_result,
            )
            self._emit("MEMORY_MERGE_DONE", category="summary", task_id=commit_id, payload={
                "source": "director",
                "event_path": merge_info.get("event_path", ""),
                "memory_index": merge_info.get("memory_index", ""),
                "patch_path": merge_info.get("patch_path", ""),
                "attempts": merge_info.get("attempts", 0),
            })
        except Exception as exc:
            memory_index = ""
            try:
                memory_index = str(self.memory_store.index_path.relative_to(workspace_root(self.run_context.workspace)))
            except Exception:
                memory_index = ""
            self._emit("MEMORY_MERGE_FAILED", level="warning", category="summary", task_id=commit_id, payload={
                "source": "director",
                "memory_index": memory_index,
                "error": str(exc),
            })

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
        topic_texts = self._read_memory_topic_snapshots()
        patch_dir = self.run_context.run_dir / "audit" / "memory_patches"
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
                    structured_result_json=json.dumps(structured_result, ensure_ascii=False),
                    memory_index_text=memory_index_text,
                    topic_goal_text=topic_texts.get("GOAL.md", ""),
                    topic_facts_text=topic_texts.get("FACTS.md", ""),
                    topic_files_text=topic_texts.get("FILES.md", ""),
                    topic_constraints_text=topic_texts.get("CONSTRAINTS.md", ""),
                    topic_questions_text=topic_texts.get("QUESTIONS.md", ""),
                    topic_runbook_text=topic_texts.get("RUNBOOK.md", ""),
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
                    topic_goal_text=topic_texts.get("GOAL.md", ""),
                    topic_facts_text=topic_texts.get("FACTS.md", ""),
                    topic_files_text=topic_texts.get("FILES.md", ""),
                    topic_constraints_text=topic_texts.get("CONSTRAINTS.md", ""),
                    topic_questions_text=topic_texts.get("QUESTIONS.md", ""),
                    topic_runbook_text=topic_texts.get("RUNBOOK.md", ""),
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
            edits_rel = f"audit/memory_patches/memory_{run_id}_{task_id}_a{attempt}.aider"
            edits_abs = self.run_context.run_dir / edits_rel
            edits_abs.write_text(edit_text if edit_text.endswith("\n") else f"{edit_text}\n", encoding="utf-8")

            tool_out = self.tool_backend.call(
                "memory_apply_aider_edits",
                json.dumps({
                    "edits_text": edit_text,
                    "allowed_paths": ["MEMORY/"],
                    "emit_diff": True,
                }, ensure_ascii=False),
                toolcall_key=f"{task_id}_memory_patch_a{attempt}",
            )
            status = str(tool_out.get("status") or "").strip().lower()
            data = tool_out.get("data") if isinstance(tool_out.get("data"), dict) else {}
            patch_rel = f"audit/memory_patches/memory_{run_id}_{task_id}_a{attempt}.diff"
            patch_abs = self.run_context.run_dir / patch_rel
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

    def _read_memory_topic_snapshots(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for name in [
            "GOAL.md",
            "FACTS.md",
            "FILES.md",
            "CONSTRAINTS.md",
            "QUESTIONS.md",
            "RUNBOOK.md",
        ]:
            topic_path = self.memory_store.topics_dir / name
            try:
                out[name] = topic_path.read_text(encoding="utf-8")
            except Exception:
                out[name] = ""
        return out

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
            if not norm.startswith("MEMORY/"):
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
            lowered = path.replace("\\", "/").lower()
            normalized = lowered.lstrip("./")
            if (
                lowered == ".logs"
                or lowered.startswith(".logs/")
                or normalized == "metadata"
                or normalized.startswith("metadata/")
                or normalized == "audit"
                or normalized.startswith("audit/")
                or "/metadata/" in lowered
            ):
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
                task_packet=(next_task.get("task_packet") if isinstance(next_task.get("task_packet"), dict) else None),
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
        proposal_review: bool,
        proposal_feedback_provider: Optional[Callable[[Dict[str, Any]], str]],
        full_auto_major: bool,
        defer_ui: bool,
        start_ui: Callable[[str, bool], None],
    ) -> Dict[str, Any]:
        self._interrupt_context_note = ""
        if self.resuming:
            if proposal_review:
                self.logger.warning("proposal_review requested while resuming; ignoring and continuing with stored proposal.")
                proposal_review = False
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

            if proposal_review:
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
                    proposal_feedback_provider=proposal_feedback_provider,
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

            self._commit_director_memory(
                commit_reason="Initial plan committed for execution.",
                proposal_md=proposal_md,
                work_packages=work_packages,
                proposal_path=proposal_relpath,
                decision_state="InitialPlanCommitted",
            )

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
                    task_packet=(task.get("task_packet") if isinstance(task.get("task_packet"), dict) else None),
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
                        proposal_feedback_provider=proposal_feedback_provider,
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
                        task_packet=(task_packet if isinstance(task_packet, dict) else None),
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
                            proposal_feedback_provider=proposal_feedback_provider,
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
                        if bool(obs.get("auto_replan", False)):
                            status = "running"
                            self._emit("TASK_AUTO_REPLAN", level="warning", category="task", task_id=task_id, payload={
                                "failure_kind": str(obs.get("failure_kind") or "unknown"),
                                "summary_snippet": self._snippet(obs.get("summary", ""), 240),
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
                            })
                            continue
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
                        self._commit_director_memory(
                            commit_reason="Major proposal revision committed automatically.",
                            proposal_md=proposal_md,
                            work_packages=work_packages,
                            proposal_path=proposal_relpath,
                            decision_state="MajorReviseProposal",
                            rationale=str(decision.get("rationale") or ""),
                            change_log=str(decision.get("change_log") or ""),
                        )
                        continue

                    proposal_md, work_packages, approved, feedback = self._review_proposal(
                        user_request=user_request,
                        proposal_md=updated_md,
                        work_packages=updated_packages,
                        log_llm=log_llm,
                        proposal_feedback_provider=proposal_feedback_provider,
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
                    self._commit_director_memory(
                        commit_reason="Major proposal revision approved and committed.",
                        proposal_md=proposal_md,
                        work_packages=work_packages,
                        proposal_path=proposal_relpath,
                        decision_state="MajorReviseProposal",
                        rationale=str(decision.get("rationale") or ""),
                        change_log=str(decision.get("change_log") or ""),
                    )
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
        proposal_review: bool = True,
        proposal_feedback_provider: Optional[Callable[[Dict[str, Any]], str]] = None,
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
                    proposal_review=proposal_review,
                    proposal_feedback_provider=proposal_feedback_provider,
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
        proposal_feedback_provider: Optional[Callable[[Dict[str, Any]], str]] = None,
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

        if proposal_feedback_provider:
            feedback = proposal_feedback_provider({**feedback_state, "stage": "hitl_feedback"}) or ""
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
        if not isinstance(packet, dict):
            raise ValueError("Director PerformNextTask missing task_packet")
        goal = str(packet.get("goal") or "").strip()
        if not goal:
            raise ValueError("Director PerformNextTask missing task_packet.goal")
        task_detail = str(packet.get("task_detail") or "").strip()
        if not task_detail:
            raise ValueError("Director PerformNextTask missing task_packet.task_detail")
        outputs = self._clean_text_list(packet.get("expected_outputs"))
        reference_hint = self._clean_text_list(packet.get("reference_hint"))
        suggested_tools = self._normalize_suggested_tools(packet.get("suggested_tools"))
        rendered = [f"Goal: {goal}", f"Task detail: {task_detail}"]
        if outputs:
            rendered.append("Expected outputs: " + "; ".join(outputs))
        if reference_hint:
            rendered.append("Reference hint: " + "; ".join(reference_hint))
        task_goal = self._with_suggested_tools_hint(" ".join(rendered), suggested_tools)
        packet_norm = {
            "goal": goal,
            "task_detail": task_detail,
            "expected_outputs": outputs,
            "reference_hint": reference_hint,
            "suggested_tools": suggested_tools,
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
    def _is_proposal_review_approved(feedback: str) -> bool:
        if not isinstance(feedback, str):
            return False
        normalized = feedback.strip().lower()
        return normalized in {"yes", "y", "approve", "approved", "ok", "okay"}

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
