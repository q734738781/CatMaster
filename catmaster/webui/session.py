from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import shutil
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from catmaster.tools.base import ensure_project_space_layout, system_root, workspace_root
from catmaster.runtime import RunControl
from catmaster.runtime.usage_stats import load_usage_summary
from catmaster.ui import make_event
from catmaster.specialists import RUN_STATE_FILE

from . import io
from .constants import (
    SIDEBAR_POLL_INTERVAL,
    LIVE_SUMMARY_MAX_EVENTS,
    LIVE_SUMMARY_MAX_JOURNAL_ITEMS,
    LIVE_SUMMARY_MAX_PARAMS_CHARS,
    LIVE_SUMMARY_MIN_INTERVAL_SEC,
    LIVE_SUMMARY_TIMEOUT_SEC,
    LIVE_SUMMARY_TOOL_EVENT_BATCH,
    MAX_TEXT_PREVIEW_CHARS,
    MAX_TRACE_LINES,
)
from .chat_sessions import ChatSessionStore
from .live_state import apply_events, new_live_state, should_refresh_live_summary
from .live_summary_service import summarize_live_state
from .summary_service import snapshot_summary, summarize_run
from .web_reporter import WebReporter

RUN_MODE_NEW = "new_run"
RUN_MODE_RESUME_SELECTED = "resume_selected_run"
SUPPORTED_LANES = {"research", "experiment", "writing", "peer_review", "literature_review"}
LANE_ALIASES = {
    "litreview": "literature_review",
    "literature": "literature_review",
}

_CHAT_HISTORY_MAX_MESSAGES = 60


@dataclass
class WorkspaceRuntime:
    workspace: Optional[Path] = None
    reporter: Optional[WebReporter] = None
    run_thread: Optional[threading.Thread] = None
    run_control: Optional[RunControl] = None
    run_loop: Optional[asyncio.AbstractEventLoop] = None
    run_task: Optional[asyncio.Task[Any]] = None
    run_status: str = "idle"
    run_error: str = ""
    run_info: Dict[str, Any] = field(default_factory=dict)
    selected_run_dir: Optional[Path] = None
    last_event_seq: int = 0
    event_lines: List[str] = field(default_factory=list)
    live_state_by_run: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_chat_session_id: str = ""
    sidebar_cache: Dict[str, Any] = field(
        default_factory=lambda: {
            "workspace": "",
            "runs": [],
            "cards": [],
            "generated_at": 0.0,
        }
    )
    sidebar_cache_dirty: bool = True


def _estimate_tokens(text: str) -> int:
    compact = str(text or "").strip()
    if not compact:
        return 0
    return max(1, len(compact) // 4)


def _format_exception_for_ui(exc: BaseException, *, max_items: int = 6, max_depth: int = 6) -> str:
    lines: list[str] = []

    def _walk(err: BaseException, depth: int) -> None:
        if len(lines) >= max_items:
            return
        indent = "  " * depth
        label = err.__class__.__name__
        message = str(err).strip()
        lines.append(f"{indent}{label}: {message or '(no message)'}")
        if depth >= max_depth:
            return
        nested = getattr(err, "exceptions", None)
        if nested:
            for child in nested:
                if isinstance(child, BaseException):
                    _walk(child, depth + 1)
                    if len(lines) >= max_items:
                        return

    _walk(exc, 0)
    if not lines:
        return str(exc)
    if len(lines) >= max_items:
        lines.append("  ...")
    return "\n".join(lines)


def _entry_system_prompt(lane: str) -> str:
    raw_target = str(lane or "research").strip() or "research"
    target = LANE_ALIASES.get(raw_target, raw_target)
    if target == "research":
        return "ResearchSpecialist entry context."
    if target == "writing":
        return "WritingSpecialist entry context."
    if target == "peer_review":
        return "PeerReviewSpecialist entry context."
    if target == "literature_review":
        return "LitReviewAgent entry context."
    return "ExperimentSpecialist entry context."


class WebSession:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.workspace_root: Optional[Path] = None
        self.workspace: Optional[Path] = None
        self._fallback_runtime = WorkspaceRuntime()
        self._workspace_runtimes: Dict[str, WorkspaceRuntime] = {}
        self.last_event_seq: int = 0
        self.current_prompt_id: str = ""
        self._last_submitted_prompt_ts: float = 0.0
        self._bg_refresh_thread: Optional[threading.Thread] = None
        self._bg_refresh_stop = threading.Event()

    def _runtime_key(self, workspace: Path) -> str:
        try:
            return str(workspace.expanduser().resolve())
        except Exception:
            return str(workspace)

    def _runtime_for_workspace_unlocked(self, workspace: Optional[Path], *, create: bool = True) -> WorkspaceRuntime:
        if workspace is None:
            return self._fallback_runtime
        resolved = workspace.expanduser().resolve()
        key = self._runtime_key(resolved)
        runtime = self._workspace_runtimes.get(key)
        if runtime is None:
            if not create:
                return self._fallback_runtime
            runtime = WorkspaceRuntime(workspace=resolved)
            self._workspace_runtimes[key] = runtime
        elif runtime.workspace is None:
            runtime.workspace = resolved
        return runtime

    def _runtime_for_workspace(self, workspace: Optional[Path] = None, *, create: bool = True) -> WorkspaceRuntime:
        with self._lock:
            target = workspace if workspace is not None else self.workspace
            return self._runtime_for_workspace_unlocked(target, create=create)

    def _current_runtime_unlocked(self) -> WorkspaceRuntime:
        return self._runtime_for_workspace_unlocked(self.workspace, create=True)

    @property
    def reporter(self) -> Optional[WebReporter]:
        with self._lock:
            return self._current_runtime_unlocked().reporter

    @reporter.setter
    def reporter(self, value: Optional[WebReporter]) -> None:
        with self._lock:
            self._current_runtime_unlocked().reporter = value

    @property
    def run_thread(self) -> Optional[threading.Thread]:
        with self._lock:
            return self._current_runtime_unlocked().run_thread

    @run_thread.setter
    def run_thread(self, value: Optional[threading.Thread]) -> None:
        with self._lock:
            self._current_runtime_unlocked().run_thread = value

    @property
    def run_control(self) -> Optional[RunControl]:
        with self._lock:
            return self._current_runtime_unlocked().run_control

    @run_control.setter
    def run_control(self, value: Optional[RunControl]) -> None:
        with self._lock:
            self._current_runtime_unlocked().run_control = value

    @property
    def _run_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        with self._lock:
            return self._current_runtime_unlocked().run_loop

    @_run_loop.setter
    def _run_loop(self, value: Optional[asyncio.AbstractEventLoop]) -> None:
        with self._lock:
            self._current_runtime_unlocked().run_loop = value

    @property
    def _run_task(self) -> Optional[asyncio.Task[Any]]:
        with self._lock:
            return self._current_runtime_unlocked().run_task

    @_run_task.setter
    def _run_task(self, value: Optional[asyncio.Task[Any]]) -> None:
        with self._lock:
            self._current_runtime_unlocked().run_task = value

    @property
    def run_status(self) -> str:
        with self._lock:
            return self._current_runtime_unlocked().run_status

    @run_status.setter
    def run_status(self, value: str) -> None:
        with self._lock:
            self._current_runtime_unlocked().run_status = str(value or "")

    @property
    def run_error(self) -> str:
        with self._lock:
            return self._current_runtime_unlocked().run_error

    @run_error.setter
    def run_error(self, value: str) -> None:
        with self._lock:
            self._current_runtime_unlocked().run_error = str(value or "")

    @property
    def run_info(self) -> Dict[str, Any]:
        with self._lock:
            return self._current_runtime_unlocked().run_info

    @run_info.setter
    def run_info(self, value: Dict[str, Any]) -> None:
        with self._lock:
            self._current_runtime_unlocked().run_info = dict(value or {})

    @property
    def selected_run_dir(self) -> Optional[Path]:
        with self._lock:
            return self._current_runtime_unlocked().selected_run_dir

    @selected_run_dir.setter
    def selected_run_dir(self, value: Optional[Path]) -> None:
        with self._lock:
            self._current_runtime_unlocked().selected_run_dir = value

    @property
    def last_event_seq(self) -> int:
        with self._lock:
            return int(self._current_runtime_unlocked().last_event_seq or 0)

    @last_event_seq.setter
    def last_event_seq(self, value: int) -> None:
        with self._lock:
            try:
                self._current_runtime_unlocked().last_event_seq = int(value or 0)
            except Exception:
                self._current_runtime_unlocked().last_event_seq = 0

    @property
    def event_lines(self) -> List[str]:
        with self._lock:
            return self._current_runtime_unlocked().event_lines

    @event_lines.setter
    def event_lines(self, value: List[str]) -> None:
        with self._lock:
            self._current_runtime_unlocked().event_lines = list(value or [])

    @property
    def live_state_by_run(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return self._current_runtime_unlocked().live_state_by_run

    @live_state_by_run.setter
    def live_state_by_run(self, value: Dict[str, Dict[str, Any]]) -> None:
        with self._lock:
            self._current_runtime_unlocked().live_state_by_run = dict(value or {})

    @property
    def active_chat_session_id(self) -> str:
        with self._lock:
            return self._current_runtime_unlocked().active_chat_session_id

    @active_chat_session_id.setter
    def active_chat_session_id(self, value: str) -> None:
        with self._lock:
            self._current_runtime_unlocked().active_chat_session_id = str(value or "")

    @property
    def _sidebar_cache(self) -> Dict[str, Any]:
        with self._lock:
            return self._current_runtime_unlocked().sidebar_cache

    @_sidebar_cache.setter
    def _sidebar_cache(self, value: Dict[str, Any]) -> None:
        with self._lock:
            self._current_runtime_unlocked().sidebar_cache = dict(value or {})

    @property
    def _sidebar_cache_dirty(self) -> bool:
        with self._lock:
            return bool(self._current_runtime_unlocked().sidebar_cache_dirty)

    @_sidebar_cache_dirty.setter
    def _sidebar_cache_dirty(self, value: bool) -> None:
        with self._lock:
            self._current_runtime_unlocked().sidebar_cache_dirty = bool(value)

    def set_workspace_root(self, path: str) -> Tuple[bool, str, List[Tuple[str, str]]]:
        try:
            root = Path(path).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return False, f"Failed to open project-space root: {exc}", []
        with self._lock:
            self.workspace_root = root
        return True, f"Project-space root: {root}", self._list_workspace_choices(root)

    def _mark_sidebar_cache_dirty(self, *, workspace: Optional[Path] = None) -> None:
        with self._lock:
            self._runtime_for_workspace_unlocked(workspace or self.workspace, create=True).sidebar_cache_dirty = True

    def _compute_sidebar_snapshot(self, workspace: Path) -> Dict[str, Any]:
        runs_root = system_root(workspace=workspace) / "runs"
        runs: List[Tuple[str, str]] = []
        cards: List[Dict[str, Any]] = []
        if runs_root.exists():
            run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
            for run_dir in sorted(run_dirs, key=lambda p: p.name, reverse=True):
                display = run_dir.name
                meta = {}
                meta_path = run_dir / "meta.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        meta = {}
                model = str(meta.get("model_name") or "")
                start = str(meta.get("start_time") or "")
                if model or start:
                    display = f"{run_dir.name} | {model} | {start}"
                runs.append((display, run_dir.name))

                summary = snapshot_summary(run_dir)
                cards.append(
                    {
                        "run_name": run_dir.name,
                        "headline": str(summary.get("headline") or run_dir.name),
                        "summary": str(summary.get("summary") or ""),
                        "next_actions": summary.get("next_actions") if isinstance(summary.get("next_actions"), list) else [],
                        "status": str(summary.get("status") or ""),
                        "source": str(summary.get("source") or "rule"),
                        "model_name": model,
                        "start_time": start,
                        "project_space": str(meta.get("workspace") or ""),
                    }
                )
        return {
            "workspace": str(workspace),
            "runs": runs,
            "cards": cards,
            "generated_at": time.time(),
        }

    def _refresh_sidebar_cache_if_needed(self, *, force: bool = False, workspace: Optional[Path] = None) -> None:
        ws = self._workspace_path(workspace)
        if ws is None:
            with self._lock:
                runtime = self._runtime_for_workspace_unlocked(workspace or self.workspace, create=True)
                runtime.sidebar_cache = {"workspace": "", "runs": [], "cards": [], "generated_at": time.time()}
                runtime.sidebar_cache_dirty = False
            return
        with self._lock:
            runtime = self._runtime_for_workspace_unlocked(ws, create=True)
            stale = (time.time() - float(runtime.sidebar_cache.get("generated_at") or 0.0)) >= max(2, SIDEBAR_POLL_INTERVAL)
            dirty = runtime.sidebar_cache_dirty
            cached_ws = str(runtime.sidebar_cache.get("workspace") or "")
        if not force and not dirty and not stale and cached_ws == str(ws):
            return
        snapshot = self._compute_sidebar_snapshot(ws)
        with self._lock:
            runtime = self._runtime_for_workspace_unlocked(ws, create=True)
            runtime.sidebar_cache = snapshot
            runtime.sidebar_cache_dirty = False

    def _background_refresh_loop(self) -> None:
        while not self._bg_refresh_stop.wait(timeout=1.0):
            try:
                with self._lock:
                    workspaces = [
                        runtime.workspace
                        for runtime in self._workspace_runtimes.values()
                        if isinstance(runtime.workspace, Path)
                    ]
                    if isinstance(self.workspace, Path):
                        workspaces.append(self.workspace)
                for workspace in {self._runtime_key(path): path for path in workspaces}.values():
                    self._refresh_sidebar_cache_if_needed(force=False, workspace=workspace)
            except Exception:
                continue

    def _ensure_background_refresh_thread(self) -> None:
        with self._lock:
            thread = self._bg_refresh_thread
            if thread is not None and thread.is_alive():
                return
            self._bg_refresh_stop.clear()
            thread = threading.Thread(target=self._background_refresh_loop, daemon=True)
            self._bg_refresh_thread = thread
        thread.start()

    def get_sidebar_snapshot(self, *, workspace: Optional[Path] = None) -> Dict[str, Any]:
        ws = self._workspace_path(workspace)
        self._refresh_sidebar_cache_if_needed(force=False, workspace=ws)
        with self._lock:
            runtime = self._runtime_for_workspace_unlocked(ws, create=True)
            snapshot = dict(runtime.sidebar_cache)
            snapshot["runs"] = list(snapshot.get("runs") or [])
            snapshot["cards"] = list(snapshot.get("cards") or [])
        return snapshot

    def open_workspace(self, path: str, *, create: bool = True, set_current: bool = True) -> Tuple[bool, str]:
        try:
            ws = Path(path).expanduser().resolve()
            if ws.exists():
                if not ws.is_dir():
                    return False, f"Project space is not a directory: {ws}"
            else:
                if not create:
                    return False, f"Project space does not exist: {ws}"
                ws.mkdir(parents=True, exist_ok=True)
            ensure_project_space_layout(ws, create=True)
        except Exception as exc:
            return False, f"Failed to open project space: {exc}"
        with self._lock:
            if set_current:
                self.workspace = ws
            runtime = self._runtime_for_workspace_unlocked(ws, create=True)
            if runtime.run_status != "running" and not (runtime.run_thread and runtime.run_thread.is_alive()):
                runtime.run_status = runtime.run_status or "idle"
            runtime.sidebar_cache_dirty = True
        self.ensure_active_chat_session(workspace=ws)
        self._ensure_background_refresh_thread()
        return True, f"Project space: {ws}"

    def resolve_workspace_by_name(self, name: str) -> Optional[Path]:
        root = self.workspace_root
        if root is None:
            return None
        if not name:
            return None
        candidate = str(name or "").strip()
        if candidate in {".", root.name} and self._looks_like_project_space(root):
            target = root.resolve()
        else:
            raw = Path(candidate).expanduser()
            if raw.is_absolute():
                target = raw.resolve()
            else:
                target = (root / candidate).resolve()
        return target

    def open_workspace_by_name(self, name: str, *, set_current: bool = True) -> Tuple[bool, str]:
        root = self.workspace_root
        if root is None:
            return False, "Project-space root not set."
        if not name:
            return False, "Select a project space first."
        target = self.resolve_workspace_by_name(name)
        if target is None:
            return False, "Select a project space first."
        return self.open_workspace(str(target), create=False, set_current=set_current)

    def create_workspace(self, name: str, *, set_current: bool = True) -> Tuple[bool, str]:
        root = self.workspace_root
        if root is None:
            return False, "Project-space root not set."
        if not name:
            return False, "Project-space name is required."
        target = (root / name).resolve()
        return self.open_workspace(str(target), create=True, set_current=set_current)

    def clear_workspace(self) -> Tuple[bool, str]:
        with self._lock:
            ws = self.workspace
            root = self.workspace_root
            running = self.run_thread and self.run_thread.is_alive()
        if running:
            return False, "Run in progress; stop it before clearing the project space."
        if ws is None:
            return False, "Open a project space first."
        try:
            ws = ws.resolve()
            if root is not None:
                root = root.resolve()
                try:
                    ws.relative_to(root)
                except ValueError:
                    return False, f"Project-space path is outside project-space root: {ws}"
            if not ws.exists() or not ws.is_dir():
                return False, f"Project space does not exist: {ws}"
            for entry in ws.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
        except Exception as exc:
            return False, f"Failed to clear project space: {exc}"
        with self._lock:
            self.selected_run_dir = None
            self.last_event_seq = 0
            self.event_lines = []
            self.live_state_by_run = {}
            self.run_control = None
            self.active_chat_session_id = ""
            if self.run_status != "running":
                self.run_status = "idle"
            self._sidebar_cache_dirty = True
        return True, f"Project space cleared: {ws}"

    def list_workspaces(self) -> List[Tuple[str, str]]:
        root = self.workspace_root
        if root is None:
            return []
        return self._list_workspace_choices(root)

    def list_runs(self, *, workspace: Optional[Path] = None) -> List[Tuple[str, str]]:
        snapshot = self.get_sidebar_snapshot(workspace=workspace)
        runs = snapshot.get("runs")
        return list(runs) if isinstance(runs, list) else []

    def select_run(self, run_name: str, *, workspace: Optional[Path] = None) -> str:
        if not run_name:
            return ""
        ws = self._workspace_path(workspace)
        if ws is None:
            return "Open a project space first."
        runs_root = system_root(workspace=ws) / "runs"
        candidate = (runs_root / run_name).resolve()
        sys_root = system_root(workspace=ws).resolve()
        try:
            candidate.relative_to(sys_root)
        except ValueError:
            return "Invalid run selection"
        if not candidate.exists():
            return "Invalid run selection"
        with self._lock:
            runtime = self._runtime_for_workspace_unlocked(ws, create=True)
            if runtime.selected_run_dir is not None:
                try:
                    current = runtime.selected_run_dir.resolve()
                except Exception:
                    current = runtime.selected_run_dir
                if current == candidate:
                    return f"Selected run: {candidate.name}"
            runtime.selected_run_dir = candidate
            runtime.last_event_seq = 0
            runtime.event_lines = []
            runtime.sidebar_cache_dirty = True
        return f"Selected run: {candidate.name}"

    def start_run(
        self,
        *,
        prompt: str,
        lane: str,
        run_mode: str,
        resume_run_name: str,
        proposal_review: bool,
        log_llm: bool,
        full_auto_major: bool,
        llm_config: Optional[str] = None,
        seed_hypotheses: str = "",
        exploration_policy: str = "anchored",
        writing_mode: str = "none",
        output_format: str = "tex",
        target_section: str = "",
        source_campaign_id: str = "",
        title_hint: str = "",
        max_cycles: int = 6,
        max_literature_queries: int = 4,
        max_fast_runs: int = 3,
        max_standard_runs: int = 2,
        allow_deep_report: bool = False,
        workspace: Optional[Path] = None,
    ) -> str:
        mode = str(run_mode or RUN_MODE_NEW).strip()
        if mode not in {RUN_MODE_NEW, RUN_MODE_RESUME_SELECTED}:
            return f"Invalid run mode: {mode}"
        raw_lane = str(lane or "research").strip() or "research"
        requested_lane = LANE_ALIASES.get(raw_lane, raw_lane)
        if requested_lane not in SUPPORTED_LANES:
            requested_lane = "research"
        is_resume = mode == RUN_MODE_RESUME_SELECTED
        session_user_prompt = str(prompt or "")
        message_session_id = ""
        conversation_messages: list[dict[str, str]] = []
        resume_guidance = ""
        resume_payload: Dict[str, Any] = {}

        with self._lock:
            ws = self._workspace_path(workspace)
            runtime = self._runtime_for_workspace_unlocked(ws, create=True)
            if ws is None:
                return "Open a project space first."
            if runtime.run_thread and runtime.run_thread.is_alive():
                return "Run already in progress for this project space."

        resume_dir: Optional[str] = None
        effective_lane = requested_lane
        resume_target: Optional[Path] = None
        if is_resume:
            resume_target, resume_lane, err = self._resolve_resume_target(
                resume_run_name=resume_run_name,
                workspace=ws,
            )
            if err:
                return err
            resume_dir = str(resume_target) if resume_target else None
            effective_lane = resume_lane or "research"
            if resume_target is not None:
                resume_payload = self._read_run_state_payload(resume_target)
                resume_status = str(resume_payload.get("status") or "").strip().lower()
                if resume_status in {"done", "failure"}:
                    return "Selected run is already finished."
                message_session_id = str(resume_payload.get("chat_session_id") or "").strip()
                resume_guidance = session_user_prompt.strip() or "Continue the previous interrupted request."
        else:
            message_session_id = self.ensure_active_chat_session(workspace=ws)
            conversation_messages = self._conversation_messages_for_session(
                session_id=message_session_id,
                max_messages=_CHAT_HISTORY_MAX_MESSAGES,
                workspace=ws,
            )
        if not message_session_id:
            message_session_id = self.ensure_active_chat_session(workspace=ws)
        if is_resume and message_session_id:
            self.select_chat_session(message_session_id, workspace=ws)

        with self._lock:
            runtime.run_status = "starting"
            runtime.run_error = ""
            runtime.last_event_seq = 0
            runtime.event_lines = []
            runtime.live_state_by_run = {}
            runtime.reporter = WebReporter(max_events=2000)
            runtime.run_control = RunControl()
            if resume_target is not None:
                runtime.selected_run_dir = resume_target
        def _run() -> None:
            run_dir: Optional[Path] = None
            run_error = ""
            skip_summarize = False
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            with self._lock:
                runtime.run_loop = loop

            async def _execute() -> Dict[str, Any]:
                nonlocal run_dir
                llm_cfg_mod = sys.modules.get("catmaster.llm.config")
                if llm_cfg_mod is None:
                    llm_cfg_mod = importlib.import_module("catmaster.llm.config")
                LLMProfile = getattr(llm_cfg_mod, "LLMProfile")
                llm_profile = LLMProfile.from_env_or_file(llm_config)
                project_id = self._project_id_for_workspace(ws)
                from catmaster.specialists import build_specialist_runner

                built = build_specialist_runner(
                    workspace=ws,
                    llm_profile=llm_profile,
                    reporter=runtime.reporter,
                    run_control=runtime.run_control,
                    project_id=project_id,
                    run_dir=Path(resume_dir) if resume_dir else None,
                    preferred_entrypoint=effective_lane,
                )
                runner = built.runner
                run_ctx = built.run_context
                run_dir = run_ctx.run_dir
                with self._lock:
                    runtime.run_status = "running"
                    runtime.run_info = {
                        "run_id": run_ctx.run_id,
                        "run_dir": str(run_dir),
                        "model_name": run_ctx.model_name,
                    }
                    runtime.selected_run_dir = run_dir
                    runtime.live_state_by_run[self._run_key(run_dir)] = new_live_state(run_id=run_dir.name)
                if runtime.reporter:
                    runtime.reporter.set_run_dir(run_dir)
                run_thread_id = str(
                    (
                        resume_payload.get("thread_id")
                        if is_resume
                        else run_ctx.run_id
                    )
                    or run_ctx.run_id
                    or run_dir.name
                ).strip()
                self._write_active_runs(effective_lane, run_dir, workspace=ws)
                self._save_ui_prompt(run_dir, session_user_prompt, is_resume=is_resume)
                self._write_initial_run_state(
                    run_dir,
                    entrypoint=effective_lane,
                    user_prompt=session_user_prompt,
                    chat_session_id=message_session_id,
                    thread_id=run_thread_id,
                    is_resume=is_resume,
                )
                self._mark_sidebar_cache_dirty(workspace=ws)
                if is_resume:
                    if hasattr(runner, "aresume"):
                        result = await runner.aresume(human_feedback=resume_guidance)
                    else:
                        result = await asyncio.to_thread(runner.resume, human_feedback=resume_guidance)
                elif hasattr(runner, "arun"):
                    result = await runner.arun(
                        session_user_prompt,
                        entrypoint=effective_lane,
                        proposal_review=proposal_review,
                        chat_session_id=message_session_id,
                        thread_id=run_thread_id,
                        conversation_messages=conversation_messages,
                    )
                else:
                    result = await asyncio.to_thread(
                        runner.run,
                        session_user_prompt,
                        entrypoint=effective_lane,
                        proposal_review=proposal_review,
                        chat_session_id=message_session_id,
                        thread_id=run_thread_id,
                        conversation_messages=conversation_messages,
                    )
                return result

            try:
                task = loop.create_task(_execute())
                with self._lock:
                    runtime.run_task = task
                try:
                    result = loop.run_until_complete(task)
                except asyncio.CancelledError:
                    result = {"status": "interrupted_paused"}
                    if run_dir is not None:
                        self._write_interrupted_run_state(
                            run_dir,
                            entrypoint=effective_lane,
                            user_prompt=session_user_prompt,
                            chat_session_id=message_session_id,
                        )
                    if runtime.reporter:
                        runtime.reporter.emit(
                            make_event(
                                "RUN_PAUSED",
                                category="run",
                                payload={"status": "interrupted_paused", "phase": "task_cancelled"},
                                run_id=run_dir.name if run_dir is not None else None,
                            )
                        )
                with self._lock:
                    run_status = str((result or {}).get("status") or "done")
                    if run_status == "done":
                        runtime.run_status = "done"
                    else:
                        runtime.run_status = run_status
                        if run_status == "interrupted_paused" and not run_error:
                            runtime.run_error = ""
            except Exception as exc:
                run_error = _format_exception_for_ui(exc)
                with self._lock:
                    runtime.run_status = "error"
                    runtime.run_error = run_error
                if runtime.reporter:
                    runtime.reporter.emit(make_event(
                        "RUN_END",
                        level="error",
                        category="run",
                        payload={"status": "error", "error": run_error},
                        ))
            finally:
                with self._lock:
                    runtime.run_task = None
                    runtime.run_loop = None
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                asyncio.set_event_loop(None)
                loop.close()
                if (not skip_summarize) and run_dir and run_dir.exists():
                    summarize_run(run_dir, run_error=run_error or None)
                    if run_error:
                        self._append_chat_message(
                            role="assistant",
                            content=f"Run `{run_dir.name}` ended with error:\n\n{run_error}",
                            kind="run_error",
                            source_run_id=run_dir.name,
                            session_id=message_session_id,
                            workspace=ws,
                        )
                    else:
                        response_text = self._read_chat_result_text(run_dir)
                        if response_text:
                            self._append_chat_message(
                                role="assistant",
                                content=response_text,
                                kind="run_result",
                                source_run_id=run_dir.name if run_dir else "",
                                session_id=message_session_id,
                                workspace=ws,
                            )
                    if runtime.reporter:
                        runtime.reporter.emit(make_event(
                            "RUN_SNAPSHOT_READY",
                            category="run",
                            payload={
                                "status": str(runtime.run_status or ""),
                                "run_id": run_dir.name,
                            },
                            run_id=run_dir.name,
                        ))
                if runtime.reporter:
                    runtime.reporter.close()
                self._mark_sidebar_cache_dirty(workspace=ws)

        self._append_chat_message(
            role="user",
            content=resume_guidance if is_resume else session_user_prompt,
            kind="chat",
            source_run_id=resume_target.name if is_resume and resume_target is not None else "",
            session_id=message_session_id,
            workspace=ws,
        )
        thread = threading.Thread(target=_run, daemon=True)
        with self._lock:
            runtime.run_thread = thread
        thread.start()
        return "Run started."

    def _resolve_resume_target(
        self,
        *,
        resume_run_name: str,
        workspace: Path,
    ) -> Tuple[Optional[Path], Optional[str], Optional[str]]:
        run_name = str(resume_run_name or "").strip()
        candidate: Optional[Path] = None
        if run_name:
            candidate = self._resolve_run_dir_by_name(run_name, workspace=workspace)
            if candidate is None:
                return None, None, "Invalid run selection"
        else:
            with self._lock:
                selected = self._runtime_for_workspace_unlocked(workspace, create=True).selected_run_dir
            if isinstance(selected, Path):
                sys_root = system_root(workspace=workspace).resolve()
                try:
                    selected_resolved = selected.expanduser().resolve()
                    selected_resolved.relative_to(sys_root)
                    candidate = selected_resolved
                except Exception:
                    candidate = None
        if candidate is None:
            return None, None, "Select a run to resume."
        lane, err = self._load_run_lane(candidate)
        if err:
            return None, None, err
        return candidate, lane, None

    @staticmethod
    def _load_run_lane(run_dir: Path) -> Tuple[Optional[str], Optional[str]]:
        state_path = run_dir / RUN_STATE_FILE
        if not state_path.exists():
            return None, f"Selected run is not resumable (missing {RUN_STATE_FILE}): {run_dir.name}"
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, f"Invalid {RUN_STATE_FILE} in selected run: {exc}"
        if not isinstance(data, dict):
            return None, f"Invalid {RUN_STATE_FILE} in selected run: expected JSON object"
        raw_lane = str(data.get("entrypoint") or "research").strip() or "research"
        lane = LANE_ALIASES.get(raw_lane, raw_lane)
        if lane not in SUPPORTED_LANES:
            return None, f"Invalid entrypoint in selected run {RUN_STATE_FILE}: {lane}"
        return lane, None

    @staticmethod
    def _resolve_run_dir_by_name(run_name: str, *, workspace: Path) -> Optional[Path]:
        name = str(run_name or "").strip()
        if not name:
            return None
        runs_root = system_root(workspace=workspace) / "runs"
        candidate = (runs_root / name).resolve()
        sys_root = system_root(workspace=workspace).resolve()
        try:
            candidate.relative_to(sys_root)
        except ValueError:
            return None
        if not candidate.exists() or not candidate.is_dir():
            return None
        return candidate

    def request_interrupt_current_run(self, *, note: str = "", workspace: Optional[Path] = None) -> str:
        with self._lock:
            runtime = self._runtime_for_workspace_unlocked(self._workspace_path(workspace), create=True)
            run_thread = runtime.run_thread
            run_control = runtime.run_control
            reporter = runtime.reporter
            info = dict(runtime.run_info)
            run_loop = runtime.run_loop
            run_task = runtime.run_task
        if not run_thread or not run_thread.is_alive():
            return "No running run to interrupt."
        if run_control is None:
            return "Run control is unavailable."
        snapshot = run_control.request_interrupt(source="ui", note=note or "")
        if reporter is not None:
            reporter.emit(make_event(
                "INTERRUPT_REQUESTED",
                category="run",
                payload={
                    "source": snapshot.get("source", "ui"),
                    "note": snapshot.get("note", ""),
                    "run_id": info.get("run_id", ""),
                },
                run_id=info.get("run_id") or None,
            ))
        if run_loop is not None and run_task is not None:
            try:
                def _cancel_task() -> None:
                    if run_task.done():
                        return
                    run_task.cancel()

                run_loop.call_soon_threadsafe(_cancel_task)
                ack = run_control.ack_interrupt(
                    phase="task_cancel_requested",
                    details={"method": "task.cancel"},
                )
                if reporter is not None:
                    reporter.emit(make_event(
                        "INTERRUPT_ACKED",
                        category="run",
                        payload={
                            "phase": ack.get("phase", "task_cancel_requested"),
                            "details": ack.get("details", {}),
                            "run_id": info.get("run_id", ""),
                        },
                        run_id=info.get("run_id") or None,
                    ))
            except RuntimeError:
                pass
        return "Interrupt requested."

    def interrupt_status(self, *, workspace: Optional[Path] = None) -> Dict[str, Any]:
        with self._lock:
            runtime = self._runtime_for_workspace_unlocked(self._workspace_path(workspace), create=True)
            run_control = runtime.run_control
            running = bool(runtime.run_thread and runtime.run_thread.is_alive())
            status = runtime.run_status
        if run_control is None:
            return {"running": running, "run_status": status, "interrupt": {}}
        snap = run_control.snapshot()
        return {"running": running, "run_status": status, "interrupt": snap}

    def get_prompt(self) -> Optional[Dict[str, Any]]:
        return None

    def submit_prompt(self, prompt_id: str, text: str) -> str:
        _ = (prompt_id, text)
        return "Prompt-response HITL interface has been removed."

    def get_events(self) -> Tuple[List[Dict[str, Any]], int]:
        reporter = self.reporter
        if not reporter:
            return [], self.last_event_seq
        events, seq = reporter.get_events_since(self.last_event_seq)
        self.last_event_seq = seq
        return events, seq

    def run_status_text(self, *, workspace: Optional[Path] = None) -> str:
        with self._lock:
            runtime = self._runtime_for_workspace_unlocked(self._workspace_path(workspace), create=True)
            status = runtime.run_status
            error = runtime.run_error
            info = runtime.run_info
            selected_run = runtime.selected_run_dir
        status = self._display_status(status, selected_run)
        if info:
            parts = [f"run_id={info.get('run_id','')}", f"model={info.get('model_name','')}"]
            return f"{status} | {' '.join(parts)}{(' | ' + error) if error else ''}"
        return f"{status}{(' | ' + error) if error else ''}"

    def get_selected_run_dir(self, *, workspace: Optional[Path] = None) -> Optional[Path]:
        with self._lock:
            return self._runtime_for_workspace_unlocked(self._workspace_path(workspace), create=True).selected_run_dir

    def current_workspace_path(self) -> str:
        with self._lock:
            return str(self.workspace) if self.workspace else ""

    def _chat_store(self, *, workspace: Optional[Path] = None) -> Optional[ChatSessionStore]:
        ws = self._workspace_path(workspace)
        if ws is None:
            return None
        return ChatSessionStore(workspace=ws)

    def ensure_active_chat_session(self, *, workspace: Optional[Path] = None) -> str:
        ws = self._workspace_path(workspace)
        with self._lock:
            current = str(self._runtime_for_workspace_unlocked(ws, create=True).active_chat_session_id or "").strip()
        store = self._chat_store(workspace=ws)
        if store is None:
            return ""
        if current:
            store.ensure_session(current)
            return current
        session_id = store.get_active_session_id()
        with self._lock:
            self._runtime_for_workspace_unlocked(ws, create=True).active_chat_session_id = session_id
        return session_id

    def list_chat_sessions(self, *, workspace: Optional[Path] = None) -> List[Tuple[str, str]]:
        store = self._chat_store(workspace=workspace)
        if store is None:
            return []
        out: List[Tuple[str, str]] = []
        for item in store.list_sessions():
            session_id = str(item.get("session_id") or "").strip()
            if not session_id:
                continue
            title = str(item.get("title") or session_id).strip() or session_id
            suffix = " (active)" if bool(item.get("is_active")) else ""
            out.append((f"{title}{suffix}", session_id))
        return out

    def create_chat_session(self, *, workspace: Optional[Path] = None) -> str:
        ws = self._workspace_path(workspace)
        store = self._chat_store(workspace=ws)
        if store is None:
            return ""
        session_id = store.create_active_session()
        with self._lock:
            self._runtime_for_workspace_unlocked(ws, create=True).active_chat_session_id = session_id
        return session_id

    def select_chat_session(self, session_id: str, *, workspace: Optional[Path] = None) -> str:
        sid = str(session_id or "").strip()
        if not sid:
            return ""
        ws = self._workspace_path(workspace)
        store = self._chat_store(workspace=ws)
        if store is None:
            return ""
        store.set_active_session_id(sid)
        with self._lock:
            self._runtime_for_workspace_unlocked(ws, create=True).active_chat_session_id = sid
        return sid

    def get_chat_messages(self, *, limit: Optional[int] = None, workspace: Optional[Path] = None) -> List[Dict[str, Any]]:
        ws = self._workspace_path(workspace)
        session_id = self.ensure_active_chat_session(workspace=ws)
        store = self._chat_store(workspace=ws)
        if store is None or not session_id:
            return []
        persisted = store.chat_messages(session_id, limit=limit)
        return self._merge_missing_run_results(session_id=session_id, persisted_messages=persisted, workspace=ws)

    def current_chat_session_id(self, *, workspace: Optional[Path] = None) -> str:
        return self.ensure_active_chat_session(workspace=workspace)

    def _append_chat_message(
        self,
        *,
        role: str,
        content: str,
        kind: str = "chat",
        source_run_id: str = "",
        source_prompt_id: str = "",
        meta: Optional[Dict[str, Any]] = None,
        session_id: str = "",
        workspace: Optional[Path] = None,
    ) -> None:
        target_session_id = str(session_id or "").strip() or self.ensure_active_chat_session(workspace=workspace)
        store = self._chat_store(workspace=workspace)
        if store is None or not target_session_id:
            return
        try:
            store.append_message(
                target_session_id,
                role=role,
                content=content,
                kind=kind,
                source_run_id=source_run_id,
                source_prompt_id=source_prompt_id,
                meta=meta,
            )
        except Exception:
            return

    def build_thread_binding(self, *, lane: str, workspace: Optional[Path] = None) -> Dict[str, Any]:
        session_id = self.ensure_active_chat_session(workspace=workspace)
        history = self._conversation_messages_for_session(
            session_id=session_id,
            max_messages=_CHAT_HISTORY_MAX_MESSAGES,
            workspace=workspace,
        )
        return {
            "session_id": session_id,
            "thread_id": "",
            "history_messages": len(history),
            "history_turns": len(history) // 2,
        }

    def entry_context_status_text(self, *, lane: str, current_prompt: str = "", workspace: Optional[Path] = None) -> str:
        _ = current_prompt
        pack = self.build_thread_binding(lane=lane, workspace=workspace)
        session_id = str(pack.get("session_id") or self.ensure_active_chat_session(workspace=workspace)).strip()
        lane_text = str(lane or "research").strip() or "research"
        history_turns = int(pack.get("history_turns") or 0)
        return f"Session `{session_id}` | new deepagent thread per run | replay last `{history_turns}` turns | lane `{lane_text}`"

    def read_memory_index(self, *, source: str = "all", workspace: Optional[Path] = None) -> str:
        workspace = self._workspace_path(workspace)
        if workspace is None:
            return ""
        db_path = system_root(workspace=workspace) / "deepagent_memory.sqlite"
        if not db_path.exists():
            return "No persistent memory recorded yet."
        selected_kinds = self._memory_source_kinds(source)
        try:
            conn = sqlite3.connect(str(db_path), timeout=5, check_same_thread=False)
            conn.row_factory = sqlite3.Row
        except Exception:
            return "Persistent memory database exists but could not be opened."
        try:
            available_prefixes = [
                str(row["prefix"] or "").strip()
                for row in conn.execute("SELECT DISTINCT prefix FROM store ORDER BY prefix ASC").fetchall()
                if str(row["prefix"] or "").strip()
            ]
            selected_prefixes = self._memory_prefixes_for_workspace(
                workspace,
                available_prefixes=available_prefixes,
                kinds=selected_kinds,
            )
            if not selected_prefixes:
                rows = []
            elif len(selected_prefixes) == 1:
                rows = conn.execute(
                    "SELECT prefix, key, value FROM store WHERE prefix = ? ORDER BY key ASC",
                    (selected_prefixes[0],),
                ).fetchall()
            else:
                placeholders = ", ".join("?" for _ in selected_prefixes)
                rows = conn.execute(
                    f"SELECT prefix, key, value FROM store WHERE prefix IN ({placeholders}) ORDER BY prefix ASC, key ASC",
                    tuple(selected_prefixes),
                ).fetchall()
        except Exception:
            return "Persistent memory database exists but could not be read."
        finally:
            try:
                conn.close()
            except Exception:
                pass
        if not rows:
            return "No persistent memory recorded yet."
        sections: List[str] = []
        sections.append(f"# Persistent Memory\n")
        sections.append(f"Workspace: `{workspace.name}`")
        sections.append("Source: `deepagents`")
        sections.append("Namespace(s):")
        for prefix in selected_prefixes:
            sections.append(f"- `{prefix}`")
        last_prefix = ""
        for row in rows:
            prefix = str(row["prefix"] or "").strip()
            if len(selected_prefixes) > 1 and prefix and prefix != last_prefix:
                sections.append("")
                sections.append(f"### Namespace `{prefix}`")
            key = str(row["key"] or "").strip() or "/unknown"
            content = self._decode_memory_store_value(row["value"])
            sections.append("")
            sections.append(f"## {key}")
            sections.append(content or "(empty)")
            last_prefix = prefix
        return "\n".join(sections).strip()

    def read_artifacts(self, *, workspace: Optional[Path] = None):
        run_dir = self.get_selected_run_dir(workspace=workspace)
        if run_dir is None:
            return []
        state = self._read_run_state_payload(run_dir)
        return list(state.get("artifacts") or [])

    def read_task_state(self, run_dir: Optional[Path], *, workspace: Optional[Path] = None) -> str:
        ws = self._workspace_path(workspace)
        if ws is None or not run_dir:
            return ""
        return io.read_json_pretty(
            run_dir / RUN_STATE_FILE,
            scope="metadata",
            project_space=ws,
            max_chars=MAX_TEXT_PREVIEW_CHARS,
        )

    def read_proposal(self, run_dir: Optional[Path], *, workspace: Optional[Path] = None) -> str:
        ws = self._workspace_path(workspace)
        if ws is None or not run_dir:
            return ""
        return io.read_text(
            run_dir / "proposal.md",
            scope="metadata",
            project_space=ws,
            max_chars=MAX_TEXT_PREVIEW_CHARS,
        )

    def read_result_text(self, run_dir: Optional[Path]) -> str:
        payload = self._read_run_state_payload(run_dir)
        if not payload:
            return ""
        final_answer = str(payload.get("final_answer") or "").strip()
        if final_answer:
            return final_answer
        parts: List[str] = []
        summary = str(payload.get("summary") or "").strip()
        if summary:
            parts.extend(["## Summary", summary])
        facts = payload.get("facts") if isinstance(payload.get("facts"), list) else []
        if facts:
            parts.extend(["", "## Facts"])
            parts.extend(f"- {str(item).strip()}" for item in facts if str(item).strip())
        artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), list) else []
        if artifacts:
            parts.extend(["", "## Files"])
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                path = str(item.get("path") or "").strip()
                if path:
                    parts.append(f"- `{path}`")
        return "\n".join(parts).strip()

    def read_todo_items(self, run_dir: Optional[Path]) -> List[str]:
        payload = self._read_run_state_payload(run_dir)
        if not payload:
            return []
        items = payload.get("todo_items")
        if not isinstance(items, list):
            return []
        out: List[str] = []
        for item in items:
            text = str(item or "").strip()
            if text:
                out.append(text)
        return out

    def _read_chat_result_text(self, run_dir: Optional[Path]) -> str:
        if run_dir is None:
            return ""
        payload = self._read_run_state_payload(run_dir)
        if payload:
            final_answer = str(payload.get("final_answer") or "").strip()
            if final_answer:
                return final_answer
            summary = str(payload.get("summary") or "").strip()
            if summary:
                return summary
        summary = snapshot_summary(run_dir)
        return str(summary.get("summary") or "").strip()

    @staticmethod
    def _read_run_state_payload(run_dir: Optional[Path]) -> Dict[str, Any]:
        if run_dir is None:
            return {}
        path = run_dir / RUN_STATE_FILE
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def read_trace(self, run_dir: Optional[Path], trace_name: str, *, workspace: Optional[Path] = None) -> str:
        ws = self._workspace_path(workspace)
        if ws is None or not run_dir:
            return ""
        return io.tail_jsonl(run_dir / trace_name, project_space=ws, max_lines=MAX_TRACE_LINES)

    def read_usage_summary(self, run_dir: Optional[Path]) -> Dict[str, Any]:
        if not run_dir:
            return {}
        try:
            return load_usage_summary(run_dir)
        except Exception:
            return {}

    def read_ui_events(
        self,
        run_dir: Optional[Path],
        *,
        limit: int = 200,
        before_seq: int = 0,
        after_seq: int = 0,
    ) -> Dict[str, Any]:
        if not run_dir:
            return {"events": [], "has_more": False, "min_seq": 0, "max_seq": 0}
        path = Path(run_dir) / "ui_events.jsonl"
        if not path.exists():
            return {"events": [], "has_more": False, "min_seq": 0, "max_seq": 0}
        capped_limit = min(1000, max(1, int(limit or 200)))
        rows: List[Dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle, start=1):
                    text = line.strip()
                    if not text:
                        continue
                    try:
                        item = json.loads(text)
                    except Exception:
                        continue
                    if not isinstance(item, dict):
                        continue
                    try:
                        seq = int(item.get("seq") or index)
                    except Exception:
                        seq = index
                    item["seq"] = seq
                    rows.append(item)
        except Exception:
            return {"events": [], "has_more": False, "min_seq": 0, "max_seq": 0}

        rows.sort(key=lambda item: int(item.get("seq") or 0))
        if after_seq > 0:
            matching = [item for item in rows if int(item.get("seq") or 0) > int(after_seq)]
            page = matching[:capped_limit]
            has_more = len(matching) > len(page)
        elif before_seq > 0:
            matching = [item for item in rows if int(item.get("seq") or 0) < int(before_seq)]
            page = matching[-capped_limit:]
            has_more = len(matching) > len(page)
        else:
            matching = rows
            page = matching[-capped_limit:]
            has_more = len(matching) > len(page)
        return {
            "events": page,
            "has_more": has_more,
            "min_seq": int(page[0].get("seq") or 0) if page else 0,
            "max_seq": int(page[-1].get("seq") or 0) if page else 0,
        }

    def update_live_state(
        self,
        run_dir: Optional[Path],
        events: List[Dict[str, Any]],
        *,
        live_llm_enabled: bool,
        workspace: Optional[Path] = None,
    ) -> Dict[str, Any]:
        if run_dir is None:
            return {}
        ws = self._workspace_path(workspace)
        with self._lock:
            runtime = self._runtime_for_workspace_unlocked(ws, create=True)
            reporter = runtime.reporter
        active_run = reporter.get_run_dir() if reporter else None
        if reporter and active_run and active_run == run_dir:
            snapshot = reporter.get_snapshot()
            live_state = snapshot.get("live_state")
            return dict(live_state) if isinstance(live_state, dict) else {}
        key = self._run_key(run_dir)
        with self._lock:
            state = runtime.live_state_by_run.get(key)
        if state is None:
            state = new_live_state(run_id=run_dir.name)
        state, changed = apply_events(
            state,
            events,
            max_recent_toolcalls=20,
            max_recent_events=80,
            max_journal_items=LIVE_SUMMARY_MAX_JOURNAL_ITEMS,
        )
        self._hydrate_live_state_task_from_task_state(run_dir, state)
        terminal_status = self._terminal_status_from_task_state(run_dir)
        if terminal_status:
            self._apply_terminal_status(state, terminal_status)
        effective_live_llm = bool(live_llm_enabled) and not bool(terminal_status)
        if events and changed and should_refresh_live_summary(
            state,
            events,
            min_interval_s=LIVE_SUMMARY_MIN_INTERVAL_SEC,
            tool_event_batch=LIVE_SUMMARY_TOOL_EVENT_BATCH,
        ):
            state["live_summary"] = summarize_live_state(
                state,
                enabled=effective_live_llm,
                max_events=LIVE_SUMMARY_MAX_EVENTS,
                max_params_chars=LIVE_SUMMARY_MAX_PARAMS_CHARS,
                max_journal_items=LIVE_SUMMARY_MAX_JOURNAL_ITEMS,
                timeout_s=LIVE_SUMMARY_TIMEOUT_SEC,
            )
        elif not state.get("live_summary"):
            state["live_summary"] = summarize_live_state(
                state,
                enabled=False,
                max_events=LIVE_SUMMARY_MAX_EVENTS,
                max_params_chars=LIVE_SUMMARY_MAX_PARAMS_CHARS,
                max_journal_items=LIVE_SUMMARY_MAX_JOURNAL_ITEMS,
                timeout_s=LIVE_SUMMARY_TIMEOUT_SEC,
            )
        if (not effective_live_llm) and isinstance(state.get("live_summary"), dict):
            if str(state["live_summary"].get("source") or "") == "llm":
                state["live_summary"] = summarize_live_state(
                    state,
                    enabled=False,
                    max_events=LIVE_SUMMARY_MAX_EVENTS,
                    max_params_chars=LIVE_SUMMARY_MAX_PARAMS_CHARS,
                    max_journal_items=LIVE_SUMMARY_MAX_JOURNAL_ITEMS,
                    timeout_s=LIVE_SUMMARY_TIMEOUT_SEC,
                )
        if events:
            self._update_active_tool_elapsed(state)
        with self._lock:
            self._runtime_for_workspace_unlocked(ws, create=True).live_state_by_run[key] = state
        return self._public_live_state(state)

    def snapshot_live_state(self, run_dir: Optional[Path], *, workspace: Optional[Path] = None) -> Dict[str, Any]:
        if run_dir is None:
            return {}
        ws = self._workspace_path(workspace)
        with self._lock:
            runtime = self._runtime_for_workspace_unlocked(ws, create=True)
            reporter = runtime.reporter
        active_run = reporter.get_run_dir() if reporter else None
        if reporter and active_run and active_run == run_dir:
            snapshot = reporter.get_snapshot()
            live_state = snapshot.get("live_state")
            return dict(live_state) if isinstance(live_state, dict) else {}
        state = new_live_state(run_id=run_dir.name)
        self._hydrate_live_state_task_from_task_state(run_dir, state)
        terminal_status = self._terminal_status_from_task_state(run_dir)
        if terminal_status:
            self._apply_terminal_status(state, terminal_status)
        key = self._run_key(run_dir)
        with self._lock:
            cached = runtime.live_state_by_run.get(key)
        if isinstance(cached, dict) and isinstance(cached.get("live_summary"), dict):
            state["live_summary"] = cached.get("live_summary")
        else:
            state["live_summary"] = summarize_live_state(
                state,
                enabled=False,
                max_events=LIVE_SUMMARY_MAX_EVENTS,
                max_params_chars=LIVE_SUMMARY_MAX_PARAMS_CHARS,
                max_journal_items=LIVE_SUMMARY_MAX_JOURNAL_ITEMS,
                timeout_s=LIVE_SUMMARY_TIMEOUT_SEC,
            )
        return self._public_live_state(state)

    def _hydrate_live_state_task_from_task_state(self, run_dir: Optional[Path], state: Dict[str, Any]) -> None:
        if run_dir is None:
            return
        if str(state.get("current_task_goal") or "").strip() and str(state.get("current_phase") or "").strip():
            return
        path = run_dir / RUN_STATE_FILE
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        status = str(payload.get("status") or "").strip().lower()
        if status:
            state["status"] = status
        work_label = str(payload.get("text_preview") or "").strip()
        if work_label and not str(state.get("current_task_goal") or "").strip():
            state["current_task_goal"] = work_label
        phase = str(payload.get("phase") or payload.get("current_phase") or "").strip()
        if phase and not str(state.get("current_phase") or "").strip():
            state["current_phase"] = phase
        if str(state.get("current_task_goal") or "").strip():
            return
        todo_items = payload.get("todo_items")
        if isinstance(todo_items, list):
            rows: List[Dict[str, str]] = []
            for item in todo_items:
                goal = str(item or "").strip()
                if goal:
                    rows.append({"content": goal, "status": "pending"})
                    if not str(state.get("current_task_goal") or "").strip():
                        state["current_task_goal"] = goal
            if rows:
                state["todo_rows"] = rows
                state["todo_items"] = [str(row.get("content") or "").strip() for row in rows if str(row.get("content") or "").strip()]

    @staticmethod
    def _terminal_status_from_task_state(run_dir: Optional[Path]) -> str:
        if run_dir is None:
            return ""
        path = run_dir / RUN_STATE_FILE
        if not path.exists():
            return ""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        if not isinstance(payload, dict):
            return ""
        status = str(payload.get("status") or "").strip().lower()
        if status in {"done", "failure", "needs_intervention", "interrupted_paused", "awaiting_human_feedback"}:
            return status
        return ""

    @staticmethod
    def _apply_terminal_status(state: Dict[str, Any], status: str) -> None:
        state["status"] = status
        state["active_toolcall"] = None
        if status == "interrupted_paused":
            phase = str(state.get("current_phase") or "")
            state["current_phase"] = phase if phase.startswith("paused") else "paused"
            return
        if status == "awaiting_human_feedback":
            state["current_phase"] = "waiting_human"
            return
        state["current_phase"] = "finalizing"

    def list_run_cards(self, *, workspace: Optional[Path] = None) -> List[Dict[str, Any]]:
        snapshot = self.get_sidebar_snapshot(workspace=workspace)
        cards = snapshot.get("cards")
        return list(cards) if isinstance(cards, list) else []

    def _resolve_resume_dir(self, lane: str, *, workspace: Path) -> Optional[str]:
        sys_root = system_root(workspace=workspace).resolve()

        # Priority 1: explicit selected run in UI.
        with self._lock:
            selected = self._runtime_for_workspace_unlocked(workspace, create=True).selected_run_dir
        if isinstance(selected, Path):
            try:
                selected_resolved = selected.expanduser().resolve()
                selected_resolved.relative_to(sys_root)
                if (selected_resolved / RUN_STATE_FILE).exists():
                    return str(selected_resolved)
            except Exception:
                pass

        # Priority 2: active run by lane pointer.
        active_runs_path = sys_root / "active_runs.json"
        if active_runs_path.exists():
            try:
                active_runs = json.loads(active_runs_path.read_text(encoding="utf-8"))
            except Exception:
                active_runs = {}
            if isinstance(active_runs, dict):
                lane_run = active_runs.get(lane)
                if lane_run:
                    candidate = Path(lane_run)
                    if not candidate.is_absolute():
                        candidate = (sys_root / lane_run).resolve()
                    else:
                        candidate = candidate.resolve()
                    try:
                        candidate.relative_to(sys_root)
                        if (candidate / RUN_STATE_FILE).exists():
                            return str(candidate)
                    except Exception:
                        pass

        # Priority 3: latest resumable run.
        runs_root = sys_root / "runs"
        if not runs_root.exists():
            return None
        candidates = [d for d in runs_root.iterdir() if d.is_dir() and (d / RUN_STATE_FILE).exists()]
        if not candidates:
            return None
        candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        return str(candidates[0].resolve())

    def _write_active_runs(self, lane: str, run_dir: Path, *, workspace: Path) -> None:
        sys_root = system_root(workspace=workspace)
        active_runs_path = sys_root / "active_runs.json"
        try:
            active_runs = json.loads(active_runs_path.read_text(encoding="utf-8"))
        except Exception:
            active_runs = {}
        if not isinstance(active_runs, dict):
            active_runs = {}
        try:
            rel_run = run_dir.relative_to(sys_root)
            active_runs[lane] = str(rel_run)
        except Exception:
            active_runs[lane] = str(run_dir)
        try:
            active_runs_path.write_text(json.dumps(active_runs, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return

    def _write_initial_run_state(
        self,
        run_dir: Path,
        *,
        entrypoint: str,
        user_prompt: str,
        chat_session_id: str,
        thread_id: str,
        is_resume: bool,
    ) -> None:
        if not run_dir:
            return
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        payload = self._read_run_state_payload(run_dir)
        if str(payload.get("status") or "").strip().lower() in {
            "done",
            "failure",
            "error",
            "needs_intervention",
        }:
            return
        payload.update(
            {
                "schema_version": int(payload.get("schema_version") or 1),
                "entrypoint": str(payload.get("entrypoint") or entrypoint or "research").strip() or "research",
                "status": "running",
                "phase": "executing" if is_resume else str(payload.get("phase") or "executing"),
                "user_prompt": str(payload.get("user_prompt") or user_prompt or ""),
                "chat_session_id": str(payload.get("chat_session_id") or chat_session_id or ""),
                "thread_id": str(payload.get("thread_id") or thread_id or ""),
                "text_preview": str(payload.get("text_preview") or user_prompt or "")[:280],
                "proposal_review": bool(payload.get("proposal_review") or False),
                "resume": bool(is_resume),
            }
        )
        try:
            (run_dir / RUN_STATE_FILE).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return

    def _write_interrupted_run_state(
        self,
        run_dir: Path,
        *,
        entrypoint: str,
        user_prompt: str,
        chat_session_id: str,
    ) -> None:
        if not run_dir:
            return
        payload = self._read_run_state_payload(run_dir)
        if str(payload.get("status") or "").strip().lower() in {
            "done",
            "failure",
            "error",
            "needs_intervention",
        }:
            return
        payload.update(
            {
                "schema_version": int(payload.get("schema_version") or 1),
                "entrypoint": str(payload.get("entrypoint") or entrypoint or "research").strip() or "research",
                "status": "interrupted_paused",
                "phase": "interrupted",
                "pending_human_input": None,
                "proposal_review": False,
                "proposal_revision_count": 0,
                "text_preview": "Run interrupted by user.",
                "user_prompt": str(payload.get("user_prompt") or user_prompt or ""),
                "chat_session_id": str(payload.get("chat_session_id") or chat_session_id or ""),
                "final_answer": "",
                "summary": "Run interrupted by user.",
                "facts": [],
            }
        )
        try:
            (run_dir / RUN_STATE_FILE).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            return

    def _list_workspace_choices(self, root: Path) -> List[Tuple[str, str]]:
        choices: List[Tuple[str, str]] = []
        if not root.exists():
            return choices
        root_is_project = self._looks_like_project_space(root)
        if root_is_project:
            choices.append((root.name, root.name))
        for entry in sorted(root.iterdir(), key=lambda p: p.name):
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith("."):
                continue
            if root_is_project and name in {"files", "metadata"}:
                continue
            choices.append((name, name))
        return choices

    @staticmethod
    def _looks_like_project_space(path: Path) -> bool:
        try:
            resolved = path.expanduser().resolve()
        except Exception:
            return False
        return (resolved / "files").is_dir() and (resolved / "metadata").is_dir()

    @staticmethod
    def _run_key(run_dir: Path) -> str:
        try:
            return str(run_dir.expanduser().resolve())
        except Exception:
            return str(run_dir)

    @staticmethod
    def _save_ui_prompt(run_dir: Path, prompt: str, *, is_resume: bool) -> None:
        if not run_dir or not run_dir.exists():
            return
        try:
            (run_dir / "ui_prompt.txt").write_text(prompt or "", encoding="utf-8")
        except Exception:
            pass

    @staticmethod
    def _project_id_for_workspace(workspace: Path) -> str:
        resolved = Path(workspace).expanduser().resolve()
        digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:12]
        return f"project_ws_{digest}"

    @staticmethod
    def _memory_source_kinds(source: str) -> tuple[str, ...]:
        token = str(source or "all").strip().lower()
        _ = token
        return ("filesystem",)

    def _memory_prefixes_for_workspace(
        self,
        workspace: Path,
        *,
        available_prefixes: Optional[List[str]] = None,
        kinds: Optional[tuple[str, ...]] = None,
    ) -> List[str]:
        candidates: List[str] = []
        seen: set[str] = set()
        selected_kinds = tuple(kinds or ("filesystem",))

        def _append_project_id(project_id: str) -> None:
            pid = str(project_id or "").strip()
            if not pid:
                return
            for kind in selected_kinds:
                prefix = ".".join(("catmaster", pid, kind))
                if prefix not in seen:
                    candidates.append(prefix)
                    seen.add(prefix)

        with self._lock:
            selected_run = self._runtime_for_workspace_unlocked(workspace, create=True).selected_run_dir
        selected_meta = self._read_run_meta_payload(selected_run)
        _append_project_id(str(selected_meta.get("project_id") or ""))

        runs_root = system_root(workspace=workspace) / "runs"
        if runs_root.exists():
            run_dirs = [p for p in runs_root.iterdir() if p.is_dir()]
            for run_dir in sorted(run_dirs, key=lambda p: p.name, reverse=True):
                meta = self._read_run_meta_payload(run_dir)
                _append_project_id(str(meta.get("project_id") or ""))

        _append_project_id(self._project_id_for_workspace(workspace))

        available = [str(item or "").strip() for item in list(available_prefixes or []) if str(item or "").strip()]
        if not available:
            return candidates
        available_set = set(available)
        matched = [prefix for prefix in candidates if prefix in available_set]
        if matched:
            return matched
        if len(available) == 1:
            return available
        return []

    @staticmethod
    def _read_run_meta_payload(run_dir: Optional[Path]) -> Dict[str, Any]:
        if run_dir is None:
            return {}
        path = run_dir / "meta.json"
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _update_active_tool_elapsed(state: Dict[str, Any]) -> None:
        active = state.get("active_toolcall")
        if not isinstance(active, dict):
            return
        started = active.get("started_ts")
        if not isinstance(started, (int, float)):
            return
        active["elapsed_sec"] = max(0, int(time.time() - float(started)))

    @staticmethod
    def _public_live_state(state: Dict[str, Any]) -> Dict[str, Any]:
        public: Dict[str, Any] = {}
        for key, value in state.items():
            if str(key).startswith("_"):
                continue
            public[key] = value
        return public

    def _workspace_path(self, workspace: Optional[Path] = None) -> Optional[Path]:
        if isinstance(workspace, Path):
            return workspace.expanduser().resolve()
        with self._lock:
            ws = self.workspace
        return ws.resolve() if isinstance(ws, Path) else None

    def _merge_missing_run_results(
        self,
        *,
        session_id: str,
        persisted_messages: List[Dict[str, Any]],
        workspace: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        rows = [dict(item) for item in (persisted_messages or []) if isinstance(item, dict)]
        ws = self._workspace_path(workspace)
        if ws is None:
            return rows
        runs_root = system_root(workspace=ws) / "runs"
        if not runs_root.exists():
            return rows
        existing_run_ids = {
            str(item.get("source_run_id") or "").strip()
            for item in rows
            if str(item.get("kind") or "").strip() == "run_result"
        }
        synthetic: List[Dict[str, Any]] = []
        for run_dir in sorted((path for path in runs_root.iterdir() if path.is_dir()), key=lambda item: item.name):
            payload = self._read_run_state_payload(run_dir)
            if not payload:
                continue
            run_session_id = str(payload.get("chat_session_id") or "").strip()
            if run_session_id != session_id:
                continue
            run_id = run_dir.name
            if run_id in existing_run_ids:
                continue
            result_text = self._read_chat_result_text(run_dir)
            if not result_text:
                continue
            synthetic.append(
                {
                    "role": "assistant",
                    "content": result_text,
                    "kind": "run_result",
                    "created_at": None,
                    "source_run_id": run_id,
                    "__user_prompt": str(payload.get("user_prompt") or ""),
                }
            )
        if not synthetic:
            return rows
        return self._insert_synthetic_run_results(rows, synthetic)

    @staticmethod
    def _normalize_chat_pair_text(value: Any) -> str:
        return " ".join(str(value or "").split()).strip()

    @staticmethod
    def _user_message_has_run_result(rows: List[Dict[str, Any]], user_index: int) -> bool:
        for item in rows[user_index + 1:]:
            if str(item.get("role") or "") == "user":
                return False
            if str(item.get("kind") or "").strip() == "run_result":
                return True
        return False

    def _insert_synthetic_run_results(
        self,
        rows: List[Dict[str, Any]],
        synthetic: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        out = [dict(item) for item in rows]
        for result in synthetic:
            prompt_text = self._normalize_chat_pair_text(result.get("__user_prompt"))
            insert_after = -1
            if prompt_text:
                for idx, item in enumerate(out):
                    if item.get("__paired_synthetic_run_result"):
                        continue
                    if str(item.get("role") or "") != "user":
                        continue
                    if self._user_message_has_run_result(out, idx):
                        continue
                    if self._normalize_chat_pair_text(item.get("content")) == prompt_text:
                        insert_after = idx
                        break
            if insert_after < 0:
                for idx, item in enumerate(out):
                    if item.get("__paired_synthetic_run_result"):
                        continue
                    if str(item.get("role") or "") == "user":
                        if self._user_message_has_run_result(out, idx):
                            continue
                        insert_after = idx
                        break
            clean_result = {
                key: value
                for key, value in result.items()
                if not str(key).startswith("__")
            }
            if insert_after < 0:
                out.append(clean_result)
                continue
            out[insert_after]["__paired_synthetic_run_result"] = True
            out.insert(insert_after + 1, clean_result)
        return [
            {
                key: value
                for key, value in item.items()
                if not str(key).startswith("__")
            }
            for item in out
        ]

    @staticmethod
    def _decode_memory_store_value(raw: Any) -> str:
        payload = raw
        if isinstance(raw, memoryview):
            payload = raw.tobytes()
        if isinstance(payload, (bytes, bytearray)):
            try:
                payload = payload.decode("utf-8")
            except Exception:
                payload = payload.decode("utf-8", errors="replace")
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return ""
            try:
                payload = json.loads(text)
            except Exception:
                return text
        if isinstance(payload, dict):
            content = payload.get("content")
            if isinstance(content, list):
                return "\n".join(str(item) for item in content).strip()
            if isinstance(content, str):
                return content.strip()
            return json.dumps(payload, ensure_ascii=False, indent=2)
        return str(payload or "").strip()

    def _display_status(self, base_status: str, selected_run: Optional[Path]) -> str:
        status = str(base_status or "").strip() or "unknown"
        if status in {"starting", "running", "interrupting"}:
            return status
        run_status = self._load_task_state_status(selected_run)
        if run_status:
            return run_status
        return status

    @staticmethod
    def _load_task_state_status(run_dir: Optional[Path]) -> str:
        if run_dir is None:
            return ""
        path = run_dir / RUN_STATE_FILE
        if not path.exists():
            return ""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return ""
        if not isinstance(payload, dict):
            return ""
        status = str(payload.get("status") or "").strip().lower()
        if status in {
            "awaiting_human_feedback",
            "running",
            "starting",
            "done",
            "blocked",
            "error",
            "failure",
            "needs_intervention",
            "interrupted_paused",
        }:
            return status
        return ""

    def _load_prompt_from_run_dir(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        _ = run_dir
        return None

    @staticmethod
    def _read_task_state_payload(run_dir: Optional[Path]) -> Dict[str, Any]:
        if run_dir is None:
            return {}
        state_path = run_dir / RUN_STATE_FILE
        if not state_path.exists():
            return {}
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _annotate_prompt_payload(
        self,
        run_dir: Optional[Path],
        pending: Dict[str, Any],
    ) -> Dict[str, Any]:
        _ = run_dir
        return dict(pending)

    def _conversation_messages_for_session(
        self,
        *,
        session_id: str,
        max_messages: int,
        workspace: Optional[Path] = None,
    ) -> List[Dict[str, str]]:
        store = self._chat_store(workspace=workspace)
        if store is None or not session_id:
            return []
        pack = store.build_history_pack(session_id, recent_messages=max_messages)
        out: List[Dict[str, str]] = []
        for item in pack.recent_messages:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if role not in {"user", "assistant"} or not content:
                continue
            out.append({"role": role, "content": content})
        return out

    @staticmethod
    def _submit_prompt_via_file(run_dir: Path, *, prompt_id: str, text: str) -> bool:
        _ = (run_dir, prompt_id, text)
        return False

    @staticmethod
    def _has_submitted_feedback(
        run_dir: Path,
        *,
        prompt_id: str,
        newer_than: float,
    ) -> bool:
        _ = (run_dir, prompt_id, newer_than)
        return False
