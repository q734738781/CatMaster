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
    soft = getattr(themes, "Soft", None)
    if soft is None:
        return None
    try:
        return soft(
            primary_hue="indigo",
            secondary_hue="violet",
            neutral_hue="gray",
            radius_size="lg",
            text_size="lg",
            font=["IBM Plex Sans", "Space Grotesk", "sans-serif"],
            font_mono=["IBM Plex Mono", "monospace"],
        )
    except TypeError:
        try:
            return soft()
        except Exception:
            return None
    except Exception:
        return None


def create_app(*, project_space_root: str) -> FastAPI:
    default_project_space_root = str(Path(project_space_root).expanduser().resolve())
    registry = SessionRegistry(default_project_space_root=default_project_space_root)
    theme = _make_theme()

    home_page = build_home_page(registry=registry, default_workspace=default_project_space_root, theme=theme)
    monitor_page = build_monitor_page(registry=registry, default_workspace=default_project_space_root, theme=theme)

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


def launch(
    *,
    host: str = "127.0.0.1",
    port: int = 7860,
    project_space_root: Optional[str] = None,
    timeout_keep_alive: int = 0,
    timeout_graceful_shutdown: int = 0,
) -> None:
    if project_space_root is None:
        project_space_root = str(Path.cwd() / "project_space")
    app = create_app(project_space_root=project_space_root)
    run_kwargs = {
        "host": host,
        "port": port,
        "log_level": "info",
        # Default to immediate shutdown to avoid lingering "Waiting for connections to close".
        "timeout_keep_alive": max(0, int(timeout_keep_alive)),
        "timeout_graceful_shutdown": max(0, int(timeout_graceful_shutdown)),
    }
    try:
        uvicorn.run(app, **run_kwargs)
    except TypeError:
        # Backward compatibility for older uvicorn versions that may not
        # support timeout_graceful_shutdown.
        run_kwargs.pop("timeout_graceful_shutdown", None)
        uvicorn.run(app, **run_kwargs)


__all__ = ["create_app", "launch"]
