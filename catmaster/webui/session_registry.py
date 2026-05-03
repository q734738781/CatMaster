from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import threading
from urllib.parse import quote_plus
import uuid
from typing import Dict, Optional, Tuple

from .session import WebSession

_CTX_RE = re.compile(r"^[A-Za-z0-9_-]{6,80}$")


@dataclass
class BootstrapState:
    ctx: str
    project_space_root: str
    project_space_name: str
    project_space_path: str
    run_name: str
    status: str


class SessionRegistry:
    def __init__(self, default_project_space_root: str | Path) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, WebSession] = {}
        self.default_project_space_root = Path(default_project_space_root).expanduser().resolve()
        self.default_project_space_root.mkdir(parents=True, exist_ok=True)

    def get_session(self, ctx: str) -> WebSession:
        key = self.normalize_ctx(ctx)
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = WebSession()
                self._sessions[key] = session
        return session

    def normalize_ctx(self, ctx: Optional[str]) -> str:
        value = (ctx or "").strip()
        if _CTX_RE.match(value):
            return value
        return f"ctx_{uuid.uuid4().hex[:12]}"

    def bootstrap(self, *, ctx: Optional[str], project_space: Optional[str], run: Optional[str]) -> BootstrapState:
        key = self.normalize_ctx(ctx)
        session = self.get_session(key)
        _, root_msg, _ = session.set_workspace_root(str(self.default_project_space_root))

        status_parts = [root_msg]
        project_space_name = ""
        project_space_path = session.current_workspace_path()
        missing_requested_project_space = False

        project_space_value = (project_space or "").strip()
        if project_space_value:
            target_path, resolved_project_space_name = self._resolve_project_space_target(project_space_value)
            if target_path is None:
                missing_requested_project_space = True
                status_parts.append(f"Project space does not exist: {project_space_value}")
            else:
                project_space_name = resolved_project_space_name
                ok, msg = session.open_workspace(str(target_path), create=False, set_current=True)
                status_parts.append(msg)
                if ok:
                    project_space_path = str(target_path.resolve())
                    project_space_name = self._project_space_name_from_path(project_space_path) or project_space_name

        run_name = (run or "").strip()
        if run_name:
            run_msg = session.select_run(run_name)
            if run_msg:
                status_parts.append(run_msg)

        if not project_space_name and not missing_requested_project_space:
            project_space_name = self._project_space_name_from_path(session.current_workspace_path()) or ""

        return BootstrapState(
            ctx=key,
            project_space_root=str(self.default_project_space_root),
            project_space_name=project_space_name,
            project_space_path=project_space_path,
            run_name=run_name,
            status="\n".join([part for part in status_parts if part]).strip(),
        )

    def project_space_name_for_session(self, session: WebSession) -> str:
        return self._project_space_name_from_path(session.current_workspace_path()) or ""

    def monitor_url(self, *, ctx: str, project_space: str = "", run: str = "") -> str:
        params = [f"ctx={quote_plus(ctx)}"]
        if project_space:
            params.append(f"project_space={quote_plus(project_space)}")
        if run:
            params.append(f"run={quote_plus(run)}")
        return "/monitor/?" + "&".join(params)

    def _resolve_project_space_target(self, value: str) -> Tuple[Optional[Path], str]:
        raw = Path(value).expanduser()
        if raw.is_absolute():
            resolved = raw.resolve()
            if resolved.exists() and resolved.is_dir():
                return resolved, resolved.name
            return None, value

        if value in {".", self.default_project_space_root.name} and self._looks_like_project_space(self.default_project_space_root):
            return self.default_project_space_root, self.default_project_space_root.name

        candidate = (self.default_project_space_root / value).resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate, value

        return None, value

    def _project_space_name_from_path(self, path: str) -> Optional[str]:
        if not path:
            return None
        try:
            resolved = Path(path).expanduser().resolve()
            if resolved == self.default_project_space_root:
                if self._looks_like_project_space(resolved):
                    return self.default_project_space_root.name
                return ""
            return str(resolved.relative_to(self.default_project_space_root))
        except Exception:
            return Path(path).name if path else None

    @staticmethod
    def _looks_like_project_space(path: Path) -> bool:
        try:
            resolved = Path(path).expanduser().resolve()
        except Exception:
            return False
        return (resolved / "files").is_dir() and (resolved / "metadata").is_dir()


__all__ = ["BootstrapState", "SessionRegistry"]
