from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from catmaster.tools.base import ensure_project_space_layout, system_root, workspace_root
from catmaster.runtime import RunControl
from catmaster.ui import make_event

from . import io
from .constants import (
    LIVE_SUMMARY_MAX_EVENTS,
    LIVE_SUMMARY_MAX_JOURNAL_ITEMS,
    LIVE_SUMMARY_MAX_PARAMS_CHARS,
    LIVE_SUMMARY_MIN_INTERVAL_SEC,
    LIVE_SUMMARY_TIMEOUT_SEC,
    LIVE_SUMMARY_TOOL_EVENT_BATCH,
    MAX_EVENT_FEED,
    MAX_TEXT_PREVIEW_CHARS,
    MAX_TRACE_LINES,
)
from .live_state import apply_events, new_live_state, should_refresh_live_summary
from .live_summary_service import summarize_live_state
from .summary_service import snapshot_summary, summarize_run
from .web_reporter import PromptBroker, WebReporter

RUN_MODE_NEW = "new_run"
RUN_MODE_RESUME_SELECTED = "resume_selected_run"
SUPPORTED_LANES = {"fast", "standard"}


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
        self.event_lines: List[str] = []
        self.live_state_by_run: Dict[str, Dict[str, Any]] = {}

    def set_workspace_root(self, path: str) -> Tuple[bool, str, List[Tuple[str, str]]]:
        try:
            root = Path(path).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return False, f"Failed to open project-space root: {exc}", []
        with self._lock:
            self.workspace_root = root
        return True, f"Project-space root: {root}", self._list_workspace_choices(root)

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
            if self.run_status != "running":
                self.run_status = "idle"
        return True, f"Project space: {ws}"

    def open_workspace_by_name(self, name: str) -> Tuple[bool, str]:
        root = self.workspace_root
        if root is None:
            return False, "Project-space root not set."
        if not name:
            return False, "Select a project space first."
        target = (root / name).resolve()
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
            if self.run_status != "running":
                self.run_status = "idle"
        return True, f"Project space cleared: {ws}"

    def list_workspaces(self) -> List[Tuple[str, str]]:
        root = self.workspace_root
        if root is None:
            return []
        return self._list_workspace_choices(root)

    def list_runs(self) -> List[Tuple[str, str]]:
        ws = self._workspace_path()
        if ws is None:
            return []
        runs_root = system_root(workspace=ws) / "runs"
        if not runs_root.exists():
            return []
        runs: List[Tuple[str, str]] = []
        for run_dir in sorted(runs_root.iterdir(), key=lambda p: p.name, reverse=True):
            if not run_dir.is_dir():
                continue
            display = run_dir.name
            meta_path = run_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    model = meta.get("model_name") or ""
                    start = meta.get("start_time") or ""
                    if model or start:
                        display = f"{run_dir.name} | {model} | {start}"
                except Exception:
                    pass
            runs.append((display, run_dir.name))
        return runs

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
    ) -> str:
        mode = str(run_mode or RUN_MODE_NEW).strip()
        if mode not in {RUN_MODE_NEW, RUN_MODE_RESUME_SELECTED}:
            return f"Invalid run mode: {mode}"
        requested_lane = str(lane or "standard").strip() or "standard"
        if requested_lane not in SUPPORTED_LANES:
            requested_lane = "standard"
        is_resume = mode == RUN_MODE_RESUME_SELECTED
        resume_feedback = (prompt or "").strip() if is_resume else ""

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
            effective_lane = resume_lane or "standard"

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
            try:
                from catmaster.llm.config import LLMProfile
                from catmaster.llm.factory import build_chat_model
                from catmaster.agents.graph import GraphRunner
                from catmaster.runtime.run_context import RunContext
                from catmaster.runtime.memory_store import MemoryStore
                from catmaster.runtime.local_tool_backend import LocalToolBackend
                from catmaster.runtime.tool_executor import ToolExecutor
                from catmaster.runtime.artifact_store import ArtifactStore
                from catmaster.runtime.trace_store import TraceStore
                from catmaster.tools.registry import get_tool_registry

                llm_profile = LLMProfile.from_env_or_file(llm_config)
                run_ctx = RunContext.create(
                    workspace=ws,
                    run_dir=Path(resume_dir) if resume_dir else None,
                    model_name=llm_profile.main.model,
                )
                if self.run_control is not None:
                    self.run_control.run_id = run_ctx.run_id
                run_dir = run_ctx.run_dir
                if self.reporter:
                    self.reporter.set_run_dir(run_dir)
                self._write_active_runs(effective_lane, run_dir, workspace=ws)

                memory_store = MemoryStore.create_default(workspace=ws)
                memory_store.ensure_exists()
                registry = get_tool_registry()
                tool_backend = LocalToolBackend(
                    registry=registry,
                    tool_executor=ToolExecutor(registry),
                    artifact_store=ArtifactStore(run_ctx.run_dir),
                    trace_store=TraceStore(run_ctx.run_dir),
                    role="langgraph",
                    workspace=ws,
                )

                runner = GraphRunner(
                    task_runner_model=build_chat_model(llm_profile.config_for_role("task_runner")),
                    proposal_model=build_chat_model(llm_profile.config_for_role("proposal")),
                    director_model=build_chat_model(llm_profile.config_for_role("director")),
                    memory_patch_model=build_chat_model(llm_profile.config_for_role("memory_patch")),
                    summary_model=build_chat_model(llm_profile.config_for_role("summary")),
                    registry=registry,
                    memory_store=memory_store,
                    run_context=run_ctx,
                    reporter=self.reporter,
                    tool_backend=tool_backend,
                    run_control=self.run_control,
                    max_task_steps=llm_profile.agent_runtime.max_tool_calls,
                    max_plan_steps=llm_profile.agent_runtime.max_tool_calls,
                    recursion_limit=llm_profile.agent_runtime.recursion_limit,
                    stream_debug_console=os.environ.get("CATMASTER_STREAM_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"},
                    print_state_messages=llm_profile.agent_runtime.print_state_messages,
                )
                with self._lock:
                    self.run_status = "running"
                    self.run_info = {
                        "run_id": run_ctx.run_id,
                        "run_dir": str(run_dir),
                        "model_name": run_ctx.model_name,
                    }
                    self.selected_run_dir = run_dir
                    self.live_state_by_run[self._run_key(run_dir)] = new_live_state(run_id=run_dir.name)
                result = runner.run(
                    prompt,
                    lane=effective_lane,
                )
                with self._lock:
                    run_status = str((result or {}).get("status") or "done")
                    if run_status == "interrupted_paused":
                        self.run_status = "paused"
                    elif run_status in {"done", "failure", "needs_intervention"}:
                        self.run_status = "done"
                    else:
                        self.run_status = run_status
            except Exception as exc:
                run_error = str(exc)
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
                if self.reporter:
                    self.reporter.close()
                if run_dir and run_dir.exists():
                    summarize_run(run_dir, run_error=run_error or None)

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
        state_path = run_dir / "task_state.json"
        if not state_path.exists():
            return None, f"Selected run is not resumable (missing task_state.json): {run_dir.name}"
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, f"Invalid task_state.json in selected run: {exc}"
        if not isinstance(data, dict):
            return None, "Invalid task_state.json in selected run: expected JSON object"
        lane = str(data.get("lane") or "standard").strip() or "standard"
        if lane not in SUPPORTED_LANES:
            return None, f"Invalid lane in selected run task_state.json: {lane}"
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
        broker = self.broker
        if not prompt_id:
            return "No active prompt."
        ok = False
        if broker:
            ok = broker.submit(prompt_id, text)
            if not ok:
                ok = broker.submit_persisted(prompt_id, text)
        else:
            run_dir = self.get_selected_run_dir()
            if run_dir is not None:
                ok = self._submit_prompt_via_file(run_dir, prompt_id=prompt_id, text=text)
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
        broker = self.broker
        if broker:
            pending = broker.get_pending()
            if isinstance(pending, dict):
                return pending
        run_dir = self.get_selected_run_dir()
        if run_dir is not None:
            return self._load_prompt_from_run_dir(run_dir)
        return None

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

    def read_memory_index(self) -> str:
        ws = self._workspace_path()
        if ws is None:
            return ""
        return io.read_text(
            workspace_root(ws) / "MEMORY" / "MEMORY.md",
            scope="files",
            project_space=ws,
            max_chars=MAX_TEXT_PREVIEW_CHARS,
        )

    def read_artifacts(self):
        ws = self._workspace_path()
        if ws is None:
            return io.read_key_files_table(Path("/__catmaster_missing__/MEMORY/topics/FILES.md"))
        return io.read_key_files_table(workspace_root(ws) / "MEMORY" / "topics" / "FILES.md", project_space=ws)

    def read_task_state(self, run_dir: Optional[Path]) -> str:
        ws = self._workspace_path()
        if ws is None or not run_dir:
            return ""
        return io.read_json_pretty(
            run_dir / "task_state.json",
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

    def read_final_report(self, run_dir: Optional[Path]) -> str:
        text, _ = self.read_final_report_with_source(run_dir)
        return text

    def read_final_report_with_source(self, run_dir: Optional[Path]) -> Tuple[str, str]:
        ws = self._workspace_path()
        if ws is None:
            return "(unavailable) Open a project space first.", "unavailable"
        if run_dir:
            text = io.read_text(
                run_dir / "reports" / "FINAL_REPORT.md",
                scope="metadata",
                project_space=ws,
                max_chars=MAX_TEXT_PREVIEW_CHARS,
            )
            if not text.startswith("(unavailable)"):
                return text, f"selected_run:{run_dir.name}"

        latest_run = self._resolve_latest_run_dir(workspace=ws)
        if latest_run:
            text = io.read_text(
                latest_run / "reports" / "FINAL_REPORT.md",
                scope="metadata",
                project_space=ws,
                max_chars=MAX_TEXT_PREVIEW_CHARS,
            )
            if not text.startswith("(unavailable)"):
                return text, f"latest_run:{latest_run.name}"

        # Legacy fallback for older runs/workspaces.
        text = io.read_text(
            workspace_root(ws) / "reports" / "FINAL_REPORT.md",
            scope="files",
            project_space=ws,
            max_chars=MAX_TEXT_PREVIEW_CHARS,
        )
        if not text.startswith("(unavailable)"):
            return text, "legacy_workspace_report"
        return text, "unavailable"

    def _resolve_latest_run_dir(self, *, workspace: Path) -> Optional[Path]:
        latest_link = workspace_root(workspace) / "reports" / "latest_run"
        if not latest_link.exists():
            return None

        # Current behavior: latest_run is a copied directory snapshot.
        if latest_link.is_dir() and not latest_link.is_symlink():
            try:
                return latest_link.resolve()
            except Exception:
                return latest_link

        try:
            if latest_link.is_symlink():
                target = latest_link.resolve()
                if target.exists() and target.is_dir():
                    return target
        except Exception:
            pass

        # Fallback when latest_run is a plain-text pointer.
        if latest_link.is_file():
            try:
                raw = latest_link.read_text(encoding="utf-8").strip()
            except Exception:
                raw = ""
            if not raw:
                return None
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = (system_root(workspace=workspace) / candidate).resolve()
            else:
                candidate = candidate.resolve()
            sys_root = system_root(workspace=workspace).resolve()
            try:
                candidate.relative_to(sys_root)
            except ValueError:
                return None
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    def read_trace(self, run_dir: Optional[Path], trace_name: str) -> str:
        ws = self._workspace_path()
        if ws is None or not run_dir:
            return ""
        return io.tail_jsonl(run_dir / trace_name, project_space=ws, max_lines=MAX_TRACE_LINES)

    def read_ui_events_from_file(self, run_dir: Optional[Path]) -> str:
        ws = self._workspace_path()
        if ws is None or not run_dir:
            return ""
        return io.tail_jsonl(run_dir / "ui_events.jsonl", project_space=ws, max_lines=MAX_EVENT_FEED)

    def read_ui_events_objects(self, run_dir: Optional[Path], *, max_lines: int = MAX_EVENT_FEED) -> List[Dict[str, Any]]:
        if not run_dir:
            return []
        path = run_dir / "ui_events.jsonl"
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        for raw in lines[-max_lines:]:
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                out.append(payload)
        return out

    def update_live_state(
        self,
        run_dir: Optional[Path],
        events: List[Dict[str, Any]],
        *,
        live_llm_enabled: bool,
    ) -> Dict[str, Any]:
        if run_dir is None:
            return {}
        key = self._run_key(run_dir)
        with self._lock:
            state = self.live_state_by_run.get(key)
        if state is None:
            state = new_live_state(run_id=run_dir.name)
            if not events:
                history = self.read_ui_events_objects(run_dir, max_lines=MAX_EVENT_FEED)
                state, _ = apply_events(
                    state,
                    history,
                    max_recent_toolcalls=20,
                    max_recent_events=80,
                    max_journal_items=LIVE_SUMMARY_MAX_JOURNAL_ITEMS,
                )
        state, changed = apply_events(
            state,
            events,
            max_recent_toolcalls=20,
            max_recent_events=80,
            max_journal_items=LIVE_SUMMARY_MAX_JOURNAL_ITEMS,
        )
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
        events = self.read_ui_events_objects(run_dir, max_lines=MAX_EVENT_FEED)
        state = new_live_state(run_id=run_dir.name)
        state, _ = apply_events(
            state,
            events,
            max_recent_toolcalls=20,
            max_recent_events=80,
            max_journal_items=LIVE_SUMMARY_MAX_JOURNAL_ITEMS,
        )
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

    @staticmethod
    def _terminal_status_from_task_state(run_dir: Optional[Path]) -> str:
        if run_dir is None:
            return ""
        path = run_dir / "task_state.json"
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
        ws = self._workspace_path()
        if ws is None:
            return []
        cards: List[Dict[str, Any]] = []
        runs_root = system_root(workspace=ws) / "runs"
        if not runs_root.exists():
            return []
        for run_dir in sorted(runs_root.iterdir(), key=lambda p: p.name, reverse=True):
            if not run_dir.is_dir():
                continue
            summary = snapshot_summary(run_dir)
            meta = {}
            meta_path = run_dir / "meta.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            cards.append(
                {
                    "run_name": run_dir.name,
                    "headline": str(summary.get("headline") or run_dir.name),
                    "summary": str(summary.get("summary") or ""),
                    "next_actions": summary.get("next_actions") if isinstance(summary.get("next_actions"), list) else [],
                    "status": str(summary.get("status") or ""),
                    "source": str(summary.get("source") or "rule"),
                    "model_name": str(meta.get("model_name") or ""),
                    "start_time": str(meta.get("start_time") or ""),
                    "project_space": str(meta.get("workspace") or ""),
                }
            )
        return cards

    def _resolve_resume_dir(self, lane: str, *, workspace: Path) -> Optional[str]:
        sys_root = system_root(workspace=workspace).resolve()

        # Priority 1: explicit selected run in UI.
        with self._lock:
            selected = self.selected_run_dir
        if isinstance(selected, Path):
            try:
                selected_resolved = selected.expanduser().resolve()
                selected_resolved.relative_to(sys_root)
                if (selected_resolved / "task_state.json").exists():
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
                        if (candidate / "task_state.json").exists():
                            return str(candidate)
                    except Exception:
                        pass

        # Priority 3: latest resumable run.
        runs_root = sys_root / "runs"
        if not runs_root.exists():
            return None
        candidates = [d for d in runs_root.iterdir() if d.is_dir() and (d / "task_state.json").exists()]
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
        for entry in sorted(root.iterdir(), key=lambda p: p.name):
            if not entry.is_dir():
                continue
            name = entry.name
            if name.startswith("."):
                continue
            choices.append((name, name))
        return choices

    @staticmethod
    def _run_key(run_dir: Path) -> str:
        try:
            return str(run_dir.expanduser().resolve())
        except Exception:
            return str(run_dir)

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
        path = run_dir / "task_state.json"
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
            "failure",
            "needs_intervention",
            "interrupted_paused",
        }:
            return status
        return ""

    def _load_prompt_from_run_dir(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        hitl_prompt = run_dir / "hitl" / "pending_prompt.json"
        if hitl_prompt.exists():
            try:
                payload = json.loads(hitl_prompt.read_text(encoding="utf-8"))
            except Exception:
                payload = None
            if isinstance(payload, dict):
                prompt_id = str(payload.get("prompt_id") or "")
                prompt_mtime = 0.0
                try:
                    prompt_mtime = float(hitl_prompt.stat().st_mtime)
                except Exception:
                    prompt_mtime = 0.0
                # If feedback was already submitted for this prompt and persisted,
                # do not keep showing the stale pending prompt.
                if self._has_submitted_feedback(
                    run_dir,
                    prompt_id=prompt_id,
                    newer_than=prompt_mtime,
                ):
                    return None
                return payload

        state_path = run_dir / "task_state.json"
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
        state_mtime = 0.0
        try:
            state_mtime = float(state_path.stat().st_mtime)
        except Exception:
            state_mtime = 0.0
        # Snapshot fallback should not re-open a prompt after feedback was submitted.
        if self._has_submitted_feedback(run_dir, prompt_id="", newer_than=state_mtime):
            return None
        interrupt_payload = state.get("last_interrupt")
        if not isinstance(interrupt_payload, dict):
            return None

        interrupt_type = str(interrupt_payload.get("type") or "")
        prompt_kind = "interrupt_feedback"
        payload: Dict[str, Any]
        if interrupt_type == "proposal_review":
            prompt_kind = "proposal_review"
            payload = {
                "todo": list(interrupt_payload.get("work_packages") or []),
                "proposal_description": str(interrupt_payload.get("proposal_md") or ""),
            }
        elif interrupt_type == "task_intervention":
            prompt_kind = "hitl"
            payload = {
                "report_text": str(interrupt_payload.get("task_summary") or ""),
                "report_path": str(interrupt_payload.get("task_id") or ""),
            }
        else:
            payload = {
                "guidance": str(interrupt_payload.get("message") or "Provide feedback."),
                "run_id": str(self.run_info.get("run_id") or run_dir.name),
                "phase": interrupt_type,
            }

        prompt_id = str(interrupt_payload.get("prompt_id") or f"snapshot::{run_dir.name}::{interrupt_type or 'interrupt'}")
        return {
            "prompt_id": prompt_id,
            "kind": prompt_kind,
            "payload": payload,
            "created_at": time.time(),
            "source": "task_state_snapshot",
        }

    @staticmethod
    def _submit_prompt_via_file(run_dir: Path, *, prompt_id: str, text: str) -> bool:
        pending_path = run_dir / "hitl" / "pending_prompt.json"
        response_path = run_dir / "hitl" / "pending_response.json"
        if not pending_path.exists():
            return False
        try:
            pending = json.loads(pending_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(pending, dict):
            return False
        if str(pending.get("prompt_id") or "") != str(prompt_id or ""):
            return False
        payload = {
            "prompt_id": str(prompt_id),
            "text": str(text or ""),
            "submitted_at": time.time(),
        }
        try:
            response_path.parent.mkdir(parents=True, exist_ok=True)
            response_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                # Clear stale UI prompt immediately; graph-side broker still consumes
                # pending_response.json for resume.
                pending_path.unlink(missing_ok=True)
            except Exception:
                pass
            return True
        except Exception:
            return False

    @staticmethod
    def _has_submitted_feedback(
        run_dir: Path,
        *,
        prompt_id: str,
        newer_than: float,
    ) -> bool:
        response_path = run_dir / "hitl" / "pending_response.json"
        if not response_path.exists():
            return False
        try:
            raw = json.loads(response_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(raw, dict):
            return False
        resp_prompt_id = str(raw.get("prompt_id") or "")
        if prompt_id and resp_prompt_id != prompt_id:
            return False
        submitted_at = raw.get("submitted_at")
        if not isinstance(submitted_at, (int, float)):
            return bool((not prompt_id) or (resp_prompt_id == prompt_id))
        if newer_than > 0 and float(submitted_at) + 1e-6 < float(newer_than):
            return False
        return True
