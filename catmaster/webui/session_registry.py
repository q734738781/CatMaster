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

    def get_session(self, ctx: str, *, username: str = "admin") -> WebSession:
        key = self.normalize_ctx(ctx)
        owner = self.normalize_owner(username)
        session_key = f"{owner}:{key}"
        with self._lock:
            session = self._sessions.get(session_key)
            if session is None:
                session = WebSession()
                self._sessions[session_key] = session
        return session

    @staticmethod
    def normalize_owner(username: str) -> str:
        value = str(username or "").strip().lower()
        return value or "admin"

    def normalize_ctx(self, ctx: Optional[str]) -> str:
        value = (ctx or "").strip()
        if _CTX_RE.match(value):
            return value
        return f"ctx_{uuid.uuid4().hex[:12]}"

    def bootstrap(
        self,
        *,
        ctx: Optional[str],
        project_space: Optional[str],
        run: Optional[str],
        username: str = "admin",
        project_space_root: str | Path | None = None,
        default_project_space: str = "",
        auto_open_default: bool = False,
    ) -> BootstrapState:
        key = self.normalize_ctx(ctx)
        root = Path(project_space_root).expanduser().resolve() if project_space_root is not None else self.default_project_space_root
        root.mkdir(parents=True, exist_ok=True)
        session = self.get_session(key, username=username)
        _, root_msg, _ = session.set_workspace_root(str(root))

        status_parts = [root_msg]
        project_space_name = ""
        project_space_path = session.current_workspace_path()
        missing_requested_project_space = False

        project_space_value = (project_space or "").strip()
        if not project_space_value and auto_open_default:
            project_space_value = str(default_project_space or "default").strip() or "default"
        if project_space_value:
            target_path, resolved_project_space_name = self._resolve_project_space_target(project_space_value, root=root)
            if target_path is None:
                if auto_open_default and project_space_value == (str(default_project_space or "default").strip() or "default"):
                    target_path = (root / project_space_value).resolve()
                    project_space_name = resolved_project_space_name
                    ok, msg = session.open_workspace(str(target_path), create=True, set_current=True)
                    status_parts.append(msg)
                    if ok:
                        project_space_path = str(target_path.resolve())
                        project_space_name = self._project_space_name_from_path(project_space_path, root=root) or project_space_name
                else:
                    missing_requested_project_space = True
                    status_parts.append(f"Project space does not exist: {project_space_value}")
            else:
                project_space_name = resolved_project_space_name
                ok, msg = session.open_workspace(str(target_path), create=False, set_current=True)
                status_parts.append(msg)
                if ok:
                    project_space_path = str(target_path.resolve())
                    project_space_name = self._project_space_name_from_path(project_space_path, root=root) or project_space_name

        run_name = (run or "").strip()
        if run_name:
            run_msg = session.select_run(run_name)
            if run_msg:
                status_parts.append(run_msg)

        if not project_space_name and not missing_requested_project_space:
            project_space_name = self._project_space_name_from_path(session.current_workspace_path(), root=root) or ""

        return BootstrapState(
            ctx=key,
            project_space_root=str(root),
            project_space_name=project_space_name,
            project_space_path=project_space_path,
            run_name=run_name,
            status="\n".join([part for part in status_parts if part]).strip(),
        )

    def project_space_name_for_session(self, session: WebSession) -> str:
        return self._project_space_name_from_path(session.current_workspace_path(), root=session.workspace_root) or ""

    def monitor_url(self, *, ctx: str, project_space: str = "", run: str = "") -> str:
        params = [f"ctx={quote_plus(ctx)}"]
        if project_space:
            params.append(f"project_space={quote_plus(project_space)}")
        if run:
            params.append(f"run={quote_plus(run)}")
        return "/monitor/?" + "&".join(params)

    def _resolve_project_space_target(self, value: str, *, root: Path | None = None) -> Tuple[Optional[Path], str]:
        workspace_root = Path(root).expanduser().resolve() if root is not None else self.default_project_space_root
        raw = Path(value).expanduser()
        if raw.is_absolute():
            resolved = raw.resolve()
            try:
                resolved.relative_to(workspace_root)
            except ValueError:
                return None, value
            if resolved.exists() and resolved.is_dir():
                return resolved, resolved.name
            return None, value

        if value in {".", workspace_root.name} and self._looks_like_project_space(workspace_root):
            return workspace_root, workspace_root.name

        candidate = (workspace_root / value).resolve()
        try:
            candidate.relative_to(workspace_root)
        except ValueError:
            return None, value
        if candidate.exists() and candidate.is_dir():
            return candidate, value

        return None, value

    def _project_space_name_from_path(self, path: str, *, root: Path | str | None = None) -> Optional[str]:
        if not path:
            return None
        try:
            workspace_root = Path(root).expanduser().resolve() if root is not None else self.default_project_space_root
            resolved = Path(path).expanduser().resolve()
            if resolved == workspace_root:
                if self._looks_like_project_space(resolved):
                    return workspace_root.name
                return ""
            return str(resolved.relative_to(workspace_root))
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
