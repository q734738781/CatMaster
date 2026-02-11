from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from catmaster.tools.base import system_root, workspace_root
from catmaster.ui import make_event

from . import io
from .constants import MAX_EVENT_FEED, MAX_TEXT_PREVIEW_CHARS, MAX_TRACE_LINES
from .summary_service import snapshot_summary, summarize_run
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
        system_root(workspace=ws).mkdir(parents=True, exist_ok=True)
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

    def clear_workspace(self) -> Tuple[bool, str]:
        with self._lock:
            ws = self.workspace
            root = self.workspace_root
            running = self.run_thread and self.run_thread.is_alive()
        if running:
            return False, "Run in progress; stop it before clearing the workspace."
        if ws is None:
            return False, "Open a workspace first."
        try:
            ws = ws.resolve()
            if root is not None:
                root = root.resolve()
                try:
                    ws.relative_to(root)
                except ValueError:
                    return False, f"Workspace path is outside workspace root: {ws}"
            if not ws.exists() or not ws.is_dir():
                return False, f"Workspace does not exist: {ws}"
            for entry in ws.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
        except Exception as exc:
            return False, f"Failed to clear workspace: {exc}"
        with self._lock:
            self.selected_run_dir = None
            self.last_event_seq = 0
            self.event_lines = []
            if self.run_status != "running":
                self.run_status = "idle"
        return True, f"Workspace cleared: {ws}"

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
            return "Open a workspace first."
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
            ws = self.workspace
            if ws is None:
                return "Open a workspace first."
            if self.run_thread and self.run_thread.is_alive():
                return "Run already in progress."
            self.run_status = "starting"
            self.run_error = ""
            self.last_event_seq = 0
            self.event_lines = []
            self.broker = PromptBroker()
            self.reporter = WebReporter(broker=self.broker, max_events=2000)
        resume_dir = self._resolve_resume_dir(lane, workspace=ws) if resume else None

        def _run() -> None:
            run_dir: Optional[Path] = None
            run_error = ""
            try:
                from catmaster.agents.orchestrator import Orchestrator
                from catmaster.llm.config import LLMProfile

                llm_profile = LLMProfile.from_env_or_file(llm_config)
                orch = Orchestrator(
                    llm_profile=llm_profile,
                    reporter=self.reporter,
                    log_llm_console=False,
                    workspace=str(ws),
                    resume=resume,
                    resume_dir=resume_dir,
                )
                run_dir = orch.run_context.run_dir
                if self.reporter:
                    self.reporter.set_run_dir(run_dir)
                self._write_active_runs(lane, run_dir, workspace=ws)
                with self._lock:
                    self.run_status = "running"
                    self.run_info = {
                        "run_id": orch.run_context.run_id,
                        "run_dir": str(run_dir),
                        "model_name": orch.run_context.model_name,
                    }
                    self.selected_run_dir = run_dir
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
                # Generate UI summary only when a run is finalized.
                if run_dir and run_dir.exists():
                    summarize_run(run_dir, run_error=run_error or None)

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
        ws = self._workspace_path()
        if ws is None:
            return ""
        return io.read_text(
            system_root(workspace=ws) / "whiteboard.md",
            view="system",
            workspace=ws,
            max_chars=MAX_TEXT_PREVIEW_CHARS,
        )

    def read_artifacts(self):
        ws = self._workspace_path()
        if ws is None:
            return io.read_artifacts_csv(Path("/__catmaster_missing__/artifacts.csv"))
        return io.read_artifacts_csv(system_root(workspace=ws) / "artifacts.csv", workspace=ws)

    def read_task_state(self, run_dir: Optional[Path]) -> str:
        ws = self._workspace_path()
        if ws is None or not run_dir:
            return ""
        return io.read_json_pretty(
            run_dir / "task_state.json",
            view="system",
            workspace=ws,
            max_chars=MAX_TEXT_PREVIEW_CHARS,
        )

    def read_proposal(self, run_dir: Optional[Path]) -> str:
        ws = self._workspace_path()
        if ws is None or not run_dir:
            return ""
        return io.read_text(
            run_dir / "proposal.md",
            view="system",
            workspace=ws,
            max_chars=MAX_TEXT_PREVIEW_CHARS,
        )

    def read_final_report(self, run_dir: Optional[Path]) -> str:
        text, _ = self.read_final_report_with_source(run_dir)
        return text

    def read_final_report_with_source(self, run_dir: Optional[Path]) -> Tuple[str, str]:
        ws = self._workspace_path()
        if ws is None:
            return "(unavailable) Open a workspace first.", "unavailable"
        if run_dir:
            text = io.read_text(
                run_dir / "reports" / "FINAL_REPORT.md",
                view="system",
                workspace=ws,
                max_chars=MAX_TEXT_PREVIEW_CHARS,
            )
            if not text.startswith("(unavailable)"):
                return text, f"selected_run:{run_dir.name}"

        latest_run = self._resolve_latest_run_dir(workspace=ws)
        if latest_run:
            text = io.read_text(
                latest_run / "reports" / "FINAL_REPORT.md",
                view="system",
                workspace=ws,
                max_chars=MAX_TEXT_PREVIEW_CHARS,
            )
            if not text.startswith("(unavailable)"):
                return text, f"latest_run:{latest_run.name}"

        # Legacy fallback for older runs/workspaces.
        text = io.read_text(
            workspace_root(ws) / "reports" / "FINAL_REPORT.md",
            view="user",
            workspace=ws,
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
        return io.tail_jsonl(run_dir / trace_name, workspace=ws, max_lines=MAX_TRACE_LINES)

    def read_ui_events_from_file(self, run_dir: Optional[Path]) -> str:
        ws = self._workspace_path()
        if ws is None or not run_dir:
            return ""
        return io.tail_jsonl(run_dir / "ui_events.jsonl", workspace=ws, max_lines=MAX_EVENT_FEED)

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
                    "workspace": str(meta.get("workspace") or ""),
                }
            )
        return cards

    def _resolve_resume_dir(self, lane: str, *, workspace: Path) -> Optional[str]:
        active_runs_path = system_root(workspace=workspace) / "active_runs.json"
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
        sys_root = system_root(workspace=workspace).resolve()
        if not candidate.is_absolute():
            candidate = (sys_root / lane_run).resolve()
        else:
            candidate = candidate.resolve()
        try:
            candidate.relative_to(sys_root)
        except ValueError:
            return None
        return str(candidate)

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

    def _workspace_path(self) -> Optional[Path]:
        with self._lock:
            ws = self.workspace
        return ws.resolve() if isinstance(ws, Path) else None
