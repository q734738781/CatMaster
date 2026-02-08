from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
import gradio as gr
import uvicorn

from .pages_home import build_home_page
from .pages_monitor import build_monitor_page
from .session_registry import SessionRegistry


def _make_theme() -> Optional[Any]:
    themes = getattr(gr, "themes", None)
    if themes is None:
        return None
    origin = getattr(themes, "Origin", None)
    if origin is None:
        return None
    try:
        return origin(font=["IBM Plex Sans", "Space Grotesk", "sans-serif"])
    except TypeError:
        try:
            return origin()
        except Exception:
            return None
    except Exception:
        return None


def create_app(*, workspace: str) -> FastAPI:
    default_workspace = str(Path(workspace).expanduser().resolve())
    registry = SessionRegistry(default_workspace_root=default_workspace)
    theme = _make_theme()

    home_page = build_home_page(registry=registry, default_workspace=default_workspace, theme=theme)
    monitor_page = build_monitor_page(registry=registry, default_workspace=default_workspace, theme=theme)

    app = FastAPI(title="CatMaster WebUI")

    @app.get("/monitor", include_in_schema=False)
    def _monitor_redirect(request: Request):
        query = request.url.query
        target = "/monitor/"
        if query:
            target = f"{target}?{query}"
        return RedirectResponse(url=target, status_code=307)

    app = gr.mount_gradio_app(app, monitor_page, path="/monitor")
    app = gr.mount_gradio_app(app, home_page, path="/")
    return app


def launch(*, host: str = "127.0.0.1", port: int = 7860, workspace: Optional[str] = None) -> None:
    if workspace is None:
        workspace = str(Path.cwd() / "workspace")
    app = create_app(workspace=workspace)
    uvicorn.run(app, host=host, port=port, log_level="info")


__all__ = ["create_app", "launch"]
