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
    workspace_root: str
    workspace_name: str
    workspace_path: str
    run_name: str
    status: str


class SessionRegistry:
    def __init__(self, default_workspace_root: str | Path) -> None:
        self._lock = threading.Lock()
        self._sessions: Dict[str, WebSession] = {}
        self.default_workspace_root = Path(default_workspace_root).expanduser().resolve()
        self.default_workspace_root.mkdir(parents=True, exist_ok=True)

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

    def bootstrap(self, *, ctx: Optional[str], ws: Optional[str], run: Optional[str]) -> BootstrapState:
        key = self.normalize_ctx(ctx)
        session = self.get_session(key)
        _, root_msg, _ = session.set_workspace_root(str(self.default_workspace_root))

        status_parts = [root_msg]
        workspace_name = ""
        workspace_path = session.current_workspace_path()

        ws_value = (ws or "").strip()
        if ws_value:
            target_path, workspace_name = self._resolve_workspace_target(ws_value)
            if target_path is None:
                status_parts.append(f"Workspace does not exist: {ws_value}")
            else:
                ok, msg = session.open_workspace(str(target_path), create=False)
                status_parts.append(msg)
                if ok:
                    workspace_path = session.current_workspace_path()
                    workspace_name = self._workspace_name_from_path(workspace_path) or workspace_name

        run_name = (run or "").strip()
        if run_name:
            run_msg = session.select_run(run_name)
            if run_msg:
                status_parts.append(run_msg)

        if not workspace_name:
            workspace_name = self._workspace_name_from_path(session.current_workspace_path()) or ""

        return BootstrapState(
            ctx=key,
            workspace_root=str(self.default_workspace_root),
            workspace_name=workspace_name,
            workspace_path=session.current_workspace_path(),
            run_name=run_name,
            status="\n".join([part for part in status_parts if part]).strip(),
        )

    def workspace_name_for_session(self, session: WebSession) -> str:
        return self._workspace_name_from_path(session.current_workspace_path()) or ""

    def monitor_url(self, *, ctx: str, ws: str = "", run: str = "") -> str:
        params = [f"ctx={quote_plus(ctx)}"]
        if ws:
            params.append(f"ws={quote_plus(ws)}")
        if run:
            params.append(f"run={quote_plus(run)}")
        return "/monitor/?" + "&".join(params)

    def _resolve_workspace_target(self, value: str) -> Tuple[Optional[Path], str]:
        raw = Path(value).expanduser()
        if raw.is_absolute():
            resolved = raw.resolve()
            if resolved.exists() and resolved.is_dir():
                return resolved, resolved.name
            return None, value

        candidate = (self.default_workspace_root / value).resolve()
        if candidate.exists() and candidate.is_dir():
            return candidate, value

        return None, value

    def _workspace_name_from_path(self, path: str) -> Optional[str]:
        if not path:
            return None
        try:
            resolved = Path(path).expanduser().resolve()
            return str(resolved.relative_to(self.default_workspace_root))
        except Exception:
            return Path(path).name if path else None


__all__ = ["BootstrapState", "SessionRegistry"]
