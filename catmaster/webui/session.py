from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import sqlite3
import sys
import threading
import time
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
from .web_reporter import PromptBroker, WebReporter

RUN_MODE_NEW = "new_run"
RUN_MODE_RESUME_SELECTED = "resume_selected_run"
SUPPORTED_LANES = {"research", "experiment", "writing"}

_RECENT_SESSION_TURNS = 3


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
    target = str(lane or "research").strip() or "research"
    if target == "research":
        return "ResearchSpecialist entry context."
    if target == "writing":
        return "WritingSpecialist entry context."
    return "ExperimentSpecialist entry context."


class WebSession:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.workspace_root: Optional[Path] = None
        self.workspace: Optional[Path] = None
        self.reporter: Optional[WebReporter] = None
        self.broker: Optional[PromptBroker] = None
        self.run_thread: Optional[threading.Thread] = None
        self.run_control: Optional[RunControl] = None
        self.run_status: str = "idle"
        self.run_error: str = ""
        self.run_info: Dict[str, Any] = {}
        self.selected_run_dir: Optional[Path] = None
        self.last_event_seq: int = 0
        self.current_prompt_id: str = ""
        self._last_submitted_prompt_ts: float = 0.0
        self.event_lines: List[str] = []
        self.live_state_by_run: Dict[str, Dict[str, Any]] = {}
        self.active_chat_session_id: str = ""
        self._sidebar_cache: Dict[str, Any] = {
            "workspace": "",
            "runs": [],
            "cards": [],
            "generated_at": 0.0,
        }
        self._sidebar_cache_dirty: bool = True
        self._bg_refresh_thread: Optional[threading.Thread] = None
        self._bg_refresh_stop = threading.Event()

    def set_workspace_root(self, path: str) -> Tuple[bool, str, List[Tuple[str, str]]]:
        try:
            root = Path(path).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return False, f"Failed to open project-space root: {exc}", []
        with self._lock:
            self.workspace_root = root
        return True, f"Project-space root: {root}", self._list_workspace_choices(root)

    def _mark_sidebar_cache_dirty(self) -> None:
        with self._lock:
            self._sidebar_cache_dirty = True

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

    def _refresh_sidebar_cache_if_needed(self, *, force: bool = False) -> None:
        ws = self._workspace_path()
        if ws is None:
            with self._lock:
                self._sidebar_cache = {"workspace": "", "runs": [], "cards": [], "generated_at": time.time()}
                self._sidebar_cache_dirty = False
            return
        with self._lock:
            stale = (time.time() - float(self._sidebar_cache.get("generated_at") or 0.0)) >= max(2, SIDEBAR_POLL_INTERVAL)
            dirty = self._sidebar_cache_dirty
            cached_ws = str(self._sidebar_cache.get("workspace") or "")
        if not force and not dirty and not stale and cached_ws == str(ws):
            return
        snapshot = self._compute_sidebar_snapshot(ws)
        with self._lock:
            self._sidebar_cache = snapshot
            self._sidebar_cache_dirty = False

    def _background_refresh_loop(self) -> None:
        while not self._bg_refresh_stop.wait(timeout=1.0):
            try:
                self._refresh_sidebar_cache_if_needed(force=False)
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

    def get_sidebar_snapshot(self) -> Dict[str, Any]:
        self._refresh_sidebar_cache_if_needed(force=False)
        with self._lock:
            snapshot = dict(self._sidebar_cache)
            snapshot["runs"] = list(snapshot.get("runs") or [])
            snapshot["cards"] = list(snapshot.get("cards") or [])
        return snapshot

    def open_workspace(self, path: str, *, create: bool = True) -> Tuple[bool, str]:
        if self.run_thread and self.run_thread.is_alive():
            return False, "Run in progress; stop it before switching project space."
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
            self.workspace = ws
            self.selected_run_dir = None
            self.last_event_seq = 0
            self.event_lines = []
            self.live_state_by_run = {}
            self.run_info = {}
            self.run_control = None
            self.active_chat_session_id = ""
            if self.run_status != "running":
                self.run_status = "idle"
            self._sidebar_cache_dirty = True
        self.ensure_active_chat_session()
        self._ensure_background_refresh_thread()
        return True, f"Project space: {ws}"

    def open_workspace_by_name(self, name: str) -> Tuple[bool, str]:
        root = self.workspace_root
        if root is None:
            return False, "Project-space root not set."
        if not name:
            return False, "Select a project space first."
        candidate = str(name or "").strip()
        if candidate in {".", root.name} and self._looks_like_project_space(root):
            target = root.resolve()
        else:
            raw = Path(candidate).expanduser()
            if raw.is_absolute():
                target = raw.resolve()
            else:
                target = (root / candidate).resolve()
        return self.open_workspace(str(target), create=False)

    def create_workspace(self, name: str) -> Tuple[bool, str]:
        root = self.workspace_root
        if root is None:
            return False, "Project-space root not set."
        if not name:
            return False, "Project-space name is required."
        target = (root / name).resolve()
        return self.open_workspace(str(target), create=True)

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

    def list_runs(self) -> List[Tuple[str, str]]:
        snapshot = self.get_sidebar_snapshot()
        runs = snapshot.get("runs")
        return list(runs) if isinstance(runs, list) else []

    def select_run(self, run_name: str) -> str:
        if not run_name:
            return ""
        ws = self._workspace_path()
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
            if self.selected_run_dir is not None:
                try:
                    current = self.selected_run_dir.resolve()
                except Exception:
                    current = self.selected_run_dir
                if current == candidate:
                    return f"Selected run: {candidate.name}"
            self.selected_run_dir = candidate
            self.last_event_seq = 0
            self.event_lines = []
            self._sidebar_cache_dirty = True
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
    ) -> str:
        mode = str(run_mode or RUN_MODE_NEW).strip()
        if mode not in {RUN_MODE_NEW, RUN_MODE_RESUME_SELECTED}:
            return f"Invalid run mode: {mode}"
        requested_lane = str(lane or "research").strip() or "research"
        if requested_lane not in SUPPORTED_LANES:
            requested_lane = "research"
        is_resume = mode == RUN_MODE_RESUME_SELECTED
        resume_feedback = (prompt or "").strip() if is_resume else ""
        session_user_prompt = str(prompt or "")
        effective_prompt_text = session_user_prompt
        session_context = {"session_id": "", "context_text": "", "estimated_tokens": 0}

        with self._lock:
            ws = self.workspace
            if ws is None:
                return "Open a project space first."
            if self.run_thread and self.run_thread.is_alive():
                return "Run already in progress."

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
        if not is_resume:
            session_context = self.build_session_context(
                current_prompt=session_user_prompt,
                lane=effective_lane,
            )
            effective_prompt_text = session_user_prompt
        else:
            effective_prompt_text = resume_feedback

        with self._lock:
            self.run_status = "starting"
            self.run_error = ""
            self.last_event_seq = 0
            self.event_lines = []
            self.live_state_by_run = {}
            self.broker = PromptBroker()
            self.reporter = WebReporter(broker=self.broker, max_events=2000)
            self.run_control = RunControl()
            if resume_target is not None:
                self.selected_run_dir = resume_target
        def _run() -> None:
            run_dir: Optional[Path] = None
            run_error = ""
            skip_summarize = False
            try:
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
                    reporter=self.reporter,
                    run_control=self.run_control,
                    project_id=project_id,
                    run_dir=Path(resume_dir) if resume_dir else None,
                    preferred_entrypoint=effective_lane,
                )
                runner = built.runner
                run_ctx = built.run_context
                run_dir = run_ctx.run_dir
                with self._lock:
                    self.run_status = "running"
                    self.run_info = {
                        "run_id": run_ctx.run_id,
                        "run_dir": str(run_dir),
                        "model_name": run_ctx.model_name,
                    }
                    self.selected_run_dir = run_dir
                    self.live_state_by_run[self._run_key(run_dir)] = new_live_state(run_id=run_dir.name)
                if self.reporter:
                    self.reporter.set_run_dir(run_dir)
                self._write_active_runs(effective_lane, run_dir, workspace=ws)
                self._save_ui_prompt(run_dir, session_user_prompt, is_resume=is_resume)
                self._mark_sidebar_cache_dirty()
                if is_resume:
                    result = runner.resume(human_feedback=resume_feedback)
                else:
                    result = runner.run(
                        effective_prompt_text,
                        entrypoint=effective_lane,
                        proposal_review=proposal_review,
                        session_context_text=str(session_context.get("context_text") or ""),
                        chat_session_id=str(session_context.get("session_id") or ""),
                        entry_context_tokens_estimate=int(session_context.get("estimated_tokens") or 0),
                    )
                with self._lock:
                    run_status = str((result or {}).get("status") or "done")
                    if run_status == "done":
                        self.run_status = "done"
                    else:
                        self.run_status = run_status
            except Exception as exc:
                run_error = _format_exception_for_ui(exc)
                with self._lock:
                    self.run_status = "error"
                    self.run_error = run_error
                if self.reporter:
                    self.reporter.emit(make_event(
                        "RUN_END",
                        level="error",
                        category="run",
                        payload={"status": "error", "error": run_error},
                    ))
            finally:
                if (not skip_summarize) and run_dir and run_dir.exists():
                    summarize_run(run_dir, run_error=run_error or None)
                    if run_error:
                        self._append_chat_message(
                            role="assistant",
                            content=f"Run `{run_dir.name}` ended with error:\n\n{run_error}",
                            kind="run_error",
                            source_run_id=run_dir.name,
                        )
                    else:
                        response_text = self._read_chat_result_text(run_dir)
                        if response_text:
                            self._append_chat_message(
                                role="assistant",
                                content=response_text,
                                kind="run_result",
                                source_run_id=run_dir.name if run_dir else "",
                            )
                    if self.reporter:
                        self.reporter.emit(make_event(
                            "RUN_SNAPSHOT_READY",
                            category="run",
                            payload={
                                "status": str(self.run_status or ""),
                                "run_id": run_dir.name,
                            },
                            run_id=run_dir.name,
                        ))
                if self.reporter:
                    self.reporter.close()
                self._mark_sidebar_cache_dirty()

        self._append_chat_message(
            role="user",
            content=session_user_prompt,
            kind="hitl" if is_resume else "chat",
            source_run_id=resume_target.name if is_resume and resume_target is not None else "",
        )
        self.run_thread = threading.Thread(target=_run, daemon=True)
        self.run_thread.start()
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
                selected = self.selected_run_dir
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
        lane = str(data.get("entrypoint") or "research").strip() or "research"
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

    def submit_prompt(self, prompt_id: str, text: str) -> str:
        if not prompt_id:
            return "No active prompt."
        ok = False
        reporter = self.reporter
        if reporter is not None:
            ok = reporter.submit_prompt(prompt_id, text)
        elif self.broker:
            ok = self.broker.submit(prompt_id, text)
        if ok:
            with self._lock:
                self._last_submitted_prompt_ts = time.time()
        return "Submitted." if ok else "Prompt not found."

    def request_interrupt_current_run(self, *, note: str = "") -> str:
        with self._lock:
            run_thread = self.run_thread
            run_control = self.run_control
            reporter = self.reporter
            info = dict(self.run_info)
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
        return "Interrupt requested."

    def interrupt_status(self) -> Dict[str, Any]:
        with self._lock:
            run_control = self.run_control
            running = bool(self.run_thread and self.run_thread.is_alive())
            status = self.run_status
        if run_control is None:
            return {"running": running, "run_status": status, "interrupt": {}}
        snap = run_control.snapshot()
        return {"running": running, "run_status": status, "interrupt": snap}

    def get_prompt(self) -> Optional[Dict[str, Any]]:
        if self._in_submit_grace_period():
            return None
        run_dir = self.get_selected_run_dir()
        broker = self.broker
        if broker:
            pending = broker.get_pending()
            if isinstance(pending, dict):
                return self._annotate_prompt_payload(run_dir, pending)
        if run_dir is not None:
            pending = self._load_prompt_from_run_dir(run_dir)
            return pending
        return None

    _SUBMIT_GRACE_SEC = 15

    def _in_submit_grace_period(self) -> bool:
        with self._lock:
            ts = self._last_submitted_prompt_ts
        if ts <= 0:
            return False
        return time.time() - ts < self._SUBMIT_GRACE_SEC

    def get_events(self) -> Tuple[List[Dict[str, Any]], int]:
        reporter = self.reporter
        if not reporter:
            return [], self.last_event_seq
        events, seq = reporter.get_events_since(self.last_event_seq)
        self.last_event_seq = seq
        return events, seq

    def run_status_text(self) -> str:
        with self._lock:
            status = self.run_status
            error = self.run_error
            info = self.run_info
            selected_run = self.selected_run_dir
        status = self._display_status(status, selected_run)
        if info:
            parts = [f"run_id={info.get('run_id','')}", f"model={info.get('model_name','')}"]
            return f"{status} | {' '.join(parts)}{(' | ' + error) if error else ''}"
        return f"{status}{(' | ' + error) if error else ''}"

    def get_selected_run_dir(self) -> Optional[Path]:
        with self._lock:
            return self.selected_run_dir

    def current_workspace_path(self) -> str:
        with self._lock:
            return str(self.workspace) if self.workspace else ""

    def _chat_store(self) -> Optional[ChatSessionStore]:
        ws = self._workspace_path()
        if ws is None:
            return None
        return ChatSessionStore(workspace=ws)

    def ensure_active_chat_session(self) -> str:
        with self._lock:
            current = str(self.active_chat_session_id or "").strip()
        store = self._chat_store()
        if store is None:
            return ""
        if current:
            store.ensure_session(current)
            return current
        session_id = store.get_active_session_id()
        with self._lock:
            self.active_chat_session_id = session_id
        return session_id

    def list_chat_sessions(self) -> List[Tuple[str, str]]:
        store = self._chat_store()
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

    def create_chat_session(self) -> str:
        store = self._chat_store()
        if store is None:
            return ""
        session_id = store.create_active_session()
        with self._lock:
            self.active_chat_session_id = session_id
        return session_id

    def select_chat_session(self, session_id: str) -> str:
        sid = str(session_id or "").strip()
        if not sid:
            return ""
        store = self._chat_store()
        if store is None:
            return ""
        store.set_active_session_id(sid)
        with self._lock:
            self.active_chat_session_id = sid
        return sid

    def get_chat_messages(self, *, limit: int = 40) -> List[Dict[str, str]]:
        session_id = self.ensure_active_chat_session()
        store = self._chat_store()
        if store is None or not session_id:
            return []
        return store.chat_messages(session_id, limit=limit)

    def current_chat_session_id(self) -> str:
        return self.ensure_active_chat_session()

    def _append_chat_message(
        self,
        *,
        role: str,
        content: str,
        kind: str = "chat",
        source_run_id: str = "",
        source_prompt_id: str = "",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        session_id = self.ensure_active_chat_session()
        store = self._chat_store()
        if store is None or not session_id:
            return
        try:
            store.append_message(
                session_id,
                role=role,
                content=content,
                kind=kind,
                source_run_id=source_run_id,
                source_prompt_id=source_prompt_id,
                meta=meta,
            )
        except Exception:
            return

    def _ensure_prompt_logged_to_chat(self, pending: Optional[Dict[str, Any]]) -> None:
        if not isinstance(pending, dict):
            return
        prompt_id = str(pending.get("prompt_id") or "")
        if not prompt_id:
            return
        payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
        kind = str(pending.get("kind") or "").strip() or "hitl"
        body_parts = [str(payload.get("title") or "").strip(), str(payload.get("body") or "").strip()]
        # Prompt display text is rendered in components; store the raw payload body here.
        title = ""
        if kind == "proposal_review":
            title = "Revised Proposal Review" if bool(payload.get("is_revised")) else "Proposal Review"
            body_parts = [str(payload.get("proposal_description") or "").strip()]
            meta_lines: List[str] = []
            if payload.get("run_id"):
                label = "same run" if bool(payload.get("is_revised")) else "run"
                meta_lines.append(f"{label}: {payload.get('run_id')}")
            if payload.get("reason"):
                meta_lines.append(f"reason: {payload.get('reason')}")
            todo = payload.get("todo") if isinstance(payload.get("todo"), list) else []
            if todo:
                meta_lines.append("work packages:")
                meta_lines.extend(f"{idx + 1}. {item}" for idx, item in enumerate(todo))
            if meta_lines:
                body_parts.append("\n".join(meta_lines))
        elif kind == "hitl":
            title = "HITL Feedback Required"
            report_text = str(payload.get("report_text") or "").strip()
            report_path = str(payload.get("report_path") or "").strip()
            body_parts = [report_text]
            if report_path:
                body_parts.append(f"report: {report_path}")
        elif kind == "interrupt_feedback":
            title = "Interrupt Guidance Required"
            body_parts = [str(payload.get("guidance") or "").strip()]
        content = title.strip()
        body_text = "\n\n".join(part for part in body_parts if part)
        if body_text:
            content = f"{content}\n\n{body_text}" if content else body_text
        session_id = self.ensure_active_chat_session()
        store = self._chat_store()
        if store is None or not session_id:
            return
        try:
            store.ensure_prompt_message(
                session_id,
                prompt_id=prompt_id,
                title=title,
                body=body_text,
                meta_text="",
                run_id=str(payload.get("run_id") or ""),
            )
        except Exception:
            return

    def build_session_context(self, *, current_prompt: str, lane: str) -> Dict[str, Any]:
        session_id = self.ensure_active_chat_session()
        store = self._chat_store()
        if store is None or not session_id:
            return {
                "session_id": "",
                "context_text": "",
                "estimated_tokens": _estimate_tokens(_entry_system_prompt(lane) + "\n" + str(current_prompt or "")),
            }
        messages = store.list_messages(session_id)
        turns: list[tuple[str, str]] = []
        current_query = ""
        for item in messages:
            role = str(item.get("role") or "").strip().lower()
            kind = str(item.get("kind") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            if role == "user" and kind == "chat":
                current_query = content
                continue
            if role == "assistant" and kind == "run_result":
                if current_query:
                    turns.append((current_query, content))
                    current_query = ""
        recent_turns = turns[-_RECENT_SESSION_TURNS:]
        if not recent_turns:
            return {
                "session_id": session_id,
                "context_text": "",
                "estimated_tokens": _estimate_tokens(_entry_system_prompt(lane) + "\n" + str(current_prompt or "")),
            }
        lines: List[str] = []
        lines.append("Relevant conversation history:")
        for query, answer in recent_turns:
            lines.append(f"User: {query}")
            lines.append(f"Assistant: {answer}")
        context_text = "\n".join(lines).strip()
        estimated_tokens = _estimate_tokens(
            "\n".join(
                [
                    _entry_system_prompt(lane),
                    context_text,
                    str(current_prompt or ""),
                ]
            )
        )
        return {
            "session_id": session_id,
            "context_text": context_text,
            "estimated_tokens": estimated_tokens,
        }

    def entry_context_status_text(self, *, lane: str, current_prompt: str = "") -> str:
        pack = self.build_session_context(current_prompt=current_prompt, lane=lane)
        session_id = str(pack.get("session_id") or self.ensure_active_chat_session()).strip()
        estimated_tokens = int(pack.get("estimated_tokens") or 0)
        lane_text = str(lane or "research").strip() or "research"
        return f"Session `{session_id}` | entry context est. `{estimated_tokens}` tokens | lane `{lane_text}`"

    def read_memory_index(self) -> str:
        workspace = self._workspace_path()
        if workspace is None:
            return ""
        db_path = system_root(workspace=workspace) / "deepagent_memory.sqlite"
        if not db_path.exists():
            return "No persistent memory recorded yet."
        project_id = self._project_id_for_workspace(workspace)
        prefix = ".".join(("catmaster", project_id, "filesystem"))
        try:
            conn = sqlite3.connect(str(db_path), timeout=5, check_same_thread=False)
            conn.row_factory = sqlite3.Row
        except Exception:
            return "Persistent memory database exists but could not be opened."
        try:
            rows = conn.execute(
                "SELECT key, value FROM store WHERE prefix = ? ORDER BY key ASC",
                (prefix,),
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
        sections.append(f"Namespace: `{prefix}`")
        for row in rows:
            key = str(row["key"] or "").strip() or "/unknown"
            content = self._decode_memory_store_value(row["value"])
            sections.append("")
            sections.append(f"## {key}")
            sections.append(content or "(empty)")
        return "\n".join(sections).strip()

    def read_artifacts(self):
        run_dir = self.get_selected_run_dir()
        if run_dir is None:
            return []
        state = self._read_run_state_payload(run_dir)
        return list(state.get("artifacts") or [])

    def read_task_state(self, run_dir: Optional[Path]) -> str:
        ws = self._workspace_path()
        if ws is None or not run_dir:
            return ""
        return io.read_json_pretty(
            run_dir / RUN_STATE_FILE,
            scope="metadata",
            project_space=ws,
            max_chars=MAX_TEXT_PREVIEW_CHARS,
        )

    def read_proposal(self, run_dir: Optional[Path]) -> str:
        ws = self._workspace_path()
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

    def read_trace(self, run_dir: Optional[Path], trace_name: str) -> str:
        ws = self._workspace_path()
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

    def update_live_state(
        self,
        run_dir: Optional[Path],
        events: List[Dict[str, Any]],
        *,
        live_llm_enabled: bool,
    ) -> Dict[str, Any]:
        if run_dir is None:
            return {}
        reporter = self.reporter
        active_run = reporter.get_run_dir() if reporter else None
        if reporter and active_run and active_run == run_dir:
            snapshot = reporter.get_snapshot()
            live_state = snapshot.get("live_state")
            return dict(live_state) if isinstance(live_state, dict) else {}
        key = self._run_key(run_dir)
        with self._lock:
            state = self.live_state_by_run.get(key)
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
            self.live_state_by_run[key] = state
        return self._public_live_state(state)

    def snapshot_live_state(self, run_dir: Optional[Path]) -> Dict[str, Any]:
        if run_dir is None:
            return {}
        reporter = self.reporter
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
            cached = self.live_state_by_run.get(key)
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

    def list_run_cards(self) -> List[Dict[str, Any]]:
        snapshot = self.get_sidebar_snapshot()
        cards = snapshot.get("cards")
        return list(cards) if isinstance(cards, list) else []

    def _resolve_resume_dir(self, lane: str, *, workspace: Path) -> Optional[str]:
        sys_root = system_root(workspace=workspace).resolve()

        # Priority 1: explicit selected run in UI.
        with self._lock:
            selected = self.selected_run_dir
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

    def _workspace_path(self) -> Optional[Path]:
        with self._lock:
            ws = self.workspace
        return ws.resolve() if isinstance(ws, Path) else None

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
        if status not in {"running", "starting", "paused"}:
            return status
        pending = self.get_prompt()
        if isinstance(pending, dict):
            return "awaiting_human_feedback"
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
        }:
            return status
        return ""

    def _load_prompt_from_run_dir(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        state_path = run_dir / RUN_STATE_FILE
        if not state_path.exists():
            return None
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(state, dict):
            return None
        status = str(state.get("status") or "").strip().lower()
        if status != "awaiting_human_feedback":
            return None
        pending_input = state.get("pending_human_input")
        if not isinstance(pending_input, dict):
            return None

        interrupt_type = str(pending_input.get("kind") or "")
        prompt_kind = "interrupt_feedback"
        payload: Dict[str, Any]
        if interrupt_type == "proposal_review":
            prompt_kind = "proposal_review"
            payload = {
                "todo": list(state.get("todo_items") or []),
                "proposal_description": self.read_proposal(run_dir),
            }
        else:
            payload = {
                "guidance": "Provide feedback.",
                "run_id": str(self.run_info.get("run_id") or run_dir.name),
                "phase": interrupt_type,
            }

        prompt_id = f"snapshot::{run_dir.name}::{interrupt_type or 'interrupt'}"
        return self._annotate_prompt_payload(run_dir, {
            "prompt_id": prompt_id,
            "kind": prompt_kind,
            "payload": payload,
            "created_at": time.time(),
            "source": "run_state_snapshot",
        })

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
        annotated = dict(pending)
        payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
        payload = dict(payload)
        run_id = run_dir.name if run_dir is not None else ""
        prompt_id = str(pending.get("prompt_id") or "")
        if run_id and not payload.get("run_id"):
            payload["run_id"] = run_id
        if prompt_id and not payload.get("prompt_id"):
            payload["prompt_id"] = prompt_id

        if str(pending.get("kind") or "") == "proposal_review":
            state = self._read_task_state_payload(run_dir)
            history = list(state.get("hitl_history") or [])
            had_task_intervention = any(
                isinstance(item, dict)
                and (str(item.get("interrupt_type") or "") == "task_intervention" or bool(item.get("task_id")))
                for item in history
            )
            if had_task_intervention:
                payload["is_revised"] = True
                payload.setdefault("reason", "replanning after HITL")

        annotated["payload"] = payload
        return annotated

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
