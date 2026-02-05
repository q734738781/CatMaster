from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from catmaster.agents.orchestrator import Orchestrator
from catmaster.llm.config import LLMProfile
from catmaster.tools.base import system_root
from catmaster.ui import make_event

from . import io
from .constants import MAX_EVENT_FEED, MAX_TEXT_PREVIEW_CHARS, MAX_TRACE_LINES
from .web_reporter import PromptBroker, WebReporter


class WebSession:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.workspace_root: Optional[Path] = None
        self.workspace: Optional[Path] = None
        self.reporter: Optional[WebReporter] = None
        self.broker: Optional[PromptBroker] = None
        self.run_thread: Optional[threading.Thread] = None
        self.run_status: str = "idle"
        self.run_error: str = ""
        self.run_info: Dict[str, Any] = {}
        self.selected_run_dir: Optional[Path] = None
        self.last_event_seq: int = 0
        self.current_prompt_id: str = ""
        self.event_lines: List[str] = []

    def set_workspace_root(self, path: str) -> Tuple[bool, str, List[Tuple[str, str]]]:
        try:
            root = Path(path).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return False, f"Failed to open workspace root: {exc}", []
        with self._lock:
            self.workspace_root = root
        return True, f"Workspace root: {root}", self._list_workspace_choices(root)

    def open_workspace(self, path: str, *, create: bool = True) -> Tuple[bool, str]:
        if self.run_thread and self.run_thread.is_alive():
            return False, "Run in progress; stop it before switching workspace."
        try:
            ws = Path(path).expanduser().resolve()
            if ws.exists():
                if not ws.is_dir():
                    return False, f"Workspace is not a directory: {ws}"
            else:
                if not create:
                    return False, f"Workspace does not exist: {ws}"
                ws.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            return False, f"Failed to open workspace: {exc}"
        os.environ["CATMASTER_WORKSPACE"] = str(ws)
        system_root().mkdir(parents=True, exist_ok=True)
        with self._lock:
            self.workspace = ws
            self.selected_run_dir = None
            self.last_event_seq = 0
            self.event_lines = []
            self.run_info = {}
            if self.run_status != "running":
                self.run_status = "idle"
        return True, f"Workspace: {ws}"

    def open_workspace_by_name(self, name: str) -> Tuple[bool, str]:
        root = self.workspace_root
        if root is None:
            return False, "Workspace root not set."
        if not name:
            return False, "Select a workspace first."
        target = (root / name).resolve()
        return self.open_workspace(str(target), create=False)

    def create_workspace(self, name: str) -> Tuple[bool, str]:
        root = self.workspace_root
        if root is None:
            return False, "Workspace root not set."
        if not name:
            return False, "Workspace name is required."
        target = (root / name).resolve()
        return self.open_workspace(str(target), create=True)

    def list_workspaces(self) -> List[Tuple[str, str]]:
        root = self.workspace_root
        if root is None:
            return []
        return self._list_workspace_choices(root)

    def list_runs(self) -> List[Tuple[str, str]]:
        if self.workspace is None:
            return []
        runs_root = system_root() / "runs"
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
        runs_root = system_root() / "runs"
        candidate = (runs_root / run_name).resolve()
        sys_root = system_root().resolve()
        if not str(candidate).startswith(str(sys_root)) or not candidate.exists():
            return "Invalid run selection"
        with self._lock:
            self.selected_run_dir = candidate
            self.last_event_seq = 0
            self.event_lines = []
        return f"Selected run: {candidate.name}"

    def start_run(
        self,
        *,
        prompt: str,
        lane: str,
        resume: bool,
        plan_review: bool,
        log_llm: bool,
        full_auto_major: bool,
        llm_config: Optional[str] = None,
    ) -> str:
        with self._lock:
            if self.workspace is None:
                return "Open a workspace first."
            if self.run_thread and self.run_thread.is_alive():
                return "Run already in progress."
            self.run_status = "starting"
            self.run_error = ""
            self.last_event_seq = 0
            self.event_lines = []
            self.broker = PromptBroker()
            self.reporter = WebReporter(broker=self.broker, max_events=2000)
        resume_dir = self._resolve_resume_dir(lane) if resume else None

        def _run() -> None:
            try:
                llm_profile = LLMProfile.from_env_or_file(llm_config)
                orch = Orchestrator(
                    llm_profile=llm_profile,
                    reporter=self.reporter,
                    log_llm_console=False,
                    resume=resume,
                    resume_dir=resume_dir,
                )
                if self.reporter:
                    self.reporter.set_run_dir(orch.run_context.run_dir)
                self._write_active_runs(lane, orch.run_context.run_dir)
                with self._lock:
                    self.run_status = "running"
                    self.run_info = {
                        "run_id": orch.run_context.run_id,
                        "run_dir": str(orch.run_context.run_dir),
                        "model_name": orch.run_context.model_name,
                    }
                    self.selected_run_dir = orch.run_context.run_dir
                orch.run(
                    prompt,
                    log_llm=log_llm,
                    plan_review=plan_review,
                    lane=lane,
                    full_auto_major=full_auto_major,
                )
                with self._lock:
                    self.run_status = "done"
            except Exception as exc:
                with self._lock:
                    self.run_status = "error"
                    self.run_error = str(exc)
                if self.reporter:
                    self.reporter.emit(make_event(
                        "RUN_END",
                        level="error",
                        category="run",
                        payload={"status": "error", "error": str(exc)},
                    ))
            finally:
                if self.reporter:
                    self.reporter.close()

        self.run_thread = threading.Thread(target=_run, daemon=True)
        self.run_thread.start()
        return "Run started."

    def submit_prompt(self, prompt_id: str, text: str) -> str:
        broker = self.broker
        if not broker:
            return "No active prompt broker."
        if not prompt_id:
            return "No active prompt."
        ok = broker.submit(prompt_id, text)
        return "Submitted." if ok else "Prompt not found."

    def get_prompt(self) -> Optional[Dict[str, Any]]:
        broker = self.broker
        if not broker:
            return None
        return broker.get_pending()

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

    def read_whiteboard(self) -> str:
        return io.read_text(system_root() / "whiteboard.md", view="system", max_chars=MAX_TEXT_PREVIEW_CHARS)

    def read_artifacts(self):
        return io.read_artifacts_csv(system_root() / "artifacts.csv")

    def read_task_state(self, run_dir: Optional[Path]) -> str:
        if not run_dir:
            return ""
        return io.read_json_pretty(run_dir / "task_state.json", view="system", max_chars=MAX_TEXT_PREVIEW_CHARS)

    def read_proposal(self, run_dir: Optional[Path]) -> str:
        if not run_dir:
            return ""
        return io.read_text(run_dir / "proposal.md", view="system", max_chars=MAX_TEXT_PREVIEW_CHARS)

    def read_final_report(self, run_dir: Optional[Path]) -> str:
        if not run_dir:
            return ""
        return io.read_text(run_dir / "reports" / "FINAL_REPORT.md", view="system", max_chars=MAX_TEXT_PREVIEW_CHARS)

    def read_trace(self, run_dir: Optional[Path], trace_name: str) -> str:
        if not run_dir:
            return ""
        return io.tail_jsonl(run_dir / trace_name, max_lines=MAX_TRACE_LINES)

    def read_ui_events_from_file(self, run_dir: Optional[Path]) -> str:
        if not run_dir:
            return ""
        return io.tail_jsonl(run_dir / "ui_events.jsonl", max_lines=MAX_EVENT_FEED)

    def _resolve_resume_dir(self, lane: str) -> Optional[str]:
        active_runs_path = system_root() / "active_runs.json"
        if not active_runs_path.exists():
            return None
        try:
            active_runs = json.loads(active_runs_path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if not isinstance(active_runs, dict):
            return None
        lane_run = active_runs.get(lane)
        if not lane_run:
            return None
        candidate = Path(lane_run)
        if not candidate.is_absolute():
            candidate = (system_root() / lane_run).resolve()
        return str(candidate)

    def _write_active_runs(self, lane: str, run_dir: Path) -> None:
        sys_root = system_root()
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
