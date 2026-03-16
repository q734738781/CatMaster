from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .session_registry import SessionRegistry


def _serialize_choices(choices: list[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"label": str(label), "value": str(value)} for label, value in choices]


def _serialize_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for card in cards:
        out.append(
            {
                "run_name": str(card.get("run_name") or ""),
                "headline": str(card.get("headline") or ""),
                "summary": str(card.get("summary") or ""),
                "next_actions": [str(item) for item in list(card.get("next_actions") or []) if str(item).strip()],
                "status": str(card.get("status") or ""),
                "source": str(card.get("source") or ""),
                "model_name": str(card.get("model_name") or ""),
                "start_time": str(card.get("start_time") or ""),
                "project_space": str(card.get("project_space") or ""),
            }
        )
    return out


def _coerce_int(value: Any, default: int = 0) -> int:
    if value is None:
        return int(default)
    text = str(value).strip()
    if not text:
        return int(default)
    try:
        return int(text)
    except Exception:
        return int(default)


def _runtime_snapshot(session) -> dict[str, Any]:
    reporter = session.reporter
    if reporter is None:
        return {
            "active": False,
            "run_name": "",
            "seq": 0,
            "live_state": {},
            "llm": {},
            "graph": {},
            "prompt": None,
            "usage_totals": {},
            "recent_events": [],
        }
    snapshot = reporter.get_snapshot()
    return {
        "active": True,
        "run_name": str(snapshot.get("run_name") or ""),
        "seq": int(snapshot.get("seq") or 0),
        "live_state": snapshot.get("live_state") if isinstance(snapshot.get("live_state"), dict) else {},
        "llm": snapshot.get("llm") if isinstance(snapshot.get("llm"), dict) else {},
        "graph": snapshot.get("graph") if isinstance(snapshot.get("graph"), dict) else {},
        "prompt": snapshot.get("prompt") if isinstance(snapshot.get("prompt"), dict) else None,
        "usage_totals": snapshot.get("usage_totals") if isinstance(snapshot.get("usage_totals"), dict) else {},
        "recent_events": snapshot.get("recent_events") if isinstance(snapshot.get("recent_events"), list) else [],
    }


def _active_run_name(session, runtime: dict[str, Any] | None = None) -> str:
    runtime_dict = runtime if isinstance(runtime, dict) else {}
    runtime_run = str(runtime_dict.get("run_name") or "").strip()
    if runtime_run:
        return runtime_run
    info = getattr(session, "run_info", None)
    if isinstance(info, dict):
        run_id = str(info.get("run_id") or "").strip()
        if run_id:
            return run_id
    return ""


def _stream_patch(session, runtime: dict[str, Any]) -> dict[str, Any]:
    run_name = str(runtime.get("run_name") or "").strip()
    active_run = _active_run_name(session, runtime)
    selected_run = run_name
    run_dir = None
    workspace_path = str(session.current_workspace_path() or "").strip()
    if run_name and workspace_path:
        resolved = session._resolve_run_dir_by_name(run_name, workspace=Path(workspace_path))
        if resolved is not None:
            run_dir = resolved
    if run_dir is None:
        run_dir = session.get_selected_run_dir()
        selected_run = run_dir.name if run_dir is not None else selected_run
    usage_summary = dict(runtime.get("usage_totals") or {})
    if not usage_summary and run_dir is not None:
        usage_summary = session.read_usage_summary(run_dir)
    return {
        "active_run": active_run,
        "selected_run": selected_run,
        "chat_messages": session.get_chat_messages(limit=40),
        "cards": _serialize_cards(session.list_run_cards()),
        "usage_summary": usage_summary,
        "proposal": session.read_proposal(run_dir),
        "todo_items": session.read_todo_items(run_dir),
        "result_text": session.read_result_text(run_dir),
    }


def _pick_selected_run(session, requested_run: str = "") -> str:
    selected = str(requested_run or "").strip()
    runs = session.list_runs()
    run_names = {value for _, value in runs}
    if selected and selected in run_names:
        session.select_run(selected)
        return selected
    current = session.get_selected_run_dir()
    current_name = current.name if current is not None else ""
    if current_name and current_name in run_names:
        return current_name
    if runs:
        fallback = runs[0][1]
        session.select_run(fallback)
        return fallback
    return ""


def _run_dir_for_name(session, run_name: str):
    selected = _pick_selected_run(session, run_name)
    if not selected:
        return None, ""
    return session.get_selected_run_dir(), selected


def _build_snapshot(*, registry: SessionRegistry, ctx: str, lane: str = "research", run_name: str = "") -> dict[str, Any]:
    session = registry.get_session(ctx)
    selected_run = _pick_selected_run(session, run_name)
    run_dir = session.get_selected_run_dir()
    runtime = _runtime_snapshot(session)
    active_run = _active_run_name(session, runtime)
    runtime_matches_selection = bool(selected_run) and selected_run == str(runtime.get("run_name") or "")

    if runtime_matches_selection:
        live_state = dict(runtime.get("live_state") or {})
        prompt = runtime.get("prompt")
        events = list(runtime.get("recent_events") or [])
        usage_summary = dict(runtime.get("usage_totals") or {})
        llm = dict(runtime.get("llm") or {})
        graph = dict(runtime.get("graph") or {})
    else:
        live_state = session.snapshot_live_state(run_dir)
        prompt = session.get_prompt() if str(session._load_task_state_status(run_dir) or "") == "awaiting_human_feedback" else None
        events = []
        usage_summary = session.read_usage_summary(run_dir)
        llm = live_state.get("llm") if isinstance(live_state.get("llm"), dict) else {}
        graph = {"node": str(live_state.get("current_node") or ""), "message_count": 0, "tool_calls": [], "text_preview": ""}

    cards = _serialize_cards(session.list_run_cards())
    prompt_payload = prompt if isinstance(prompt, dict) else None
    return {
        "ctx": ctx,
        "workspace_root": str(registry.default_project_space_root),
        "workspace_path": session.current_workspace_path(),
        "workspace_name": registry.project_space_name_for_session(session),
        "workspaces": _serialize_choices(session.list_workspaces()),
        "chat_sessions": _serialize_choices(session.list_chat_sessions()),
        "current_chat_session": session.current_chat_session_id(),
        "runs": _serialize_choices(session.list_runs()),
        "active_run": active_run,
        "selected_run": selected_run,
        "cards": cards,
        "run_status": str(session.run_status or "idle"),
        "run_status_text": session.run_status_text(),
        "run_info": dict(session.run_info or {}),
        "live_state": live_state,
        "llm": llm,
        "graph": graph,
        "prompt": prompt_payload,
        "events": events[-120:],
        "usage_summary": usage_summary,
        "proposal": session.read_proposal(run_dir),
        "todo_items": session.read_todo_items(run_dir),
        "result_text": session.read_result_text(run_dir),
        "chat_messages": session.get_chat_messages(limit=40),
        "entry_context_status": session.entry_context_status_text(lane=lane),
        "runtime": runtime,
        "can_submit_prompt": bool(runtime_matches_selection and prompt_payload),
    }


def _apply_chat_session_view(snapshot: dict[str, Any], *, active_run: str = "") -> dict[str, Any]:
    active = str(active_run or snapshot.get("active_run") or "").strip()
    if active:
        snapshot["selected_run"] = active
        return snapshot
    snapshot["selected_run"] = ""
    snapshot["live_state"] = {}
    snapshot["llm"] = {}
    snapshot["graph"] = {"node": "", "message_count": 0, "tool_calls": [], "text_preview": ""}
    snapshot["prompt"] = None
    snapshot["events"] = []
    snapshot["usage_summary"] = {}
    snapshot["proposal"] = ""
    snapshot["todo_items"] = []
    snapshot["result_text"] = ""
    snapshot["can_submit_prompt"] = False
    return snapshot


def _build_details(*, registry: SessionRegistry, ctx: str, run_name: str) -> dict[str, Any]:
    session = registry.get_session(ctx)
    run_dir, selected_run = _run_dir_for_name(session, run_name)
    artifacts = session.read_artifacts()
    if hasattr(artifacts, "to_dict"):
        artifact_rows = artifacts.to_dict(orient="records")
    else:
        artifact_rows = list(artifacts or [])
    return {
        "selected_run": selected_run,
        "memory": session.read_memory_index(),
        "artifacts": artifact_rows,
        "proposal": session.read_proposal(run_dir),
        "task_state": session.read_task_state(run_dir),
        "trace_event": session.read_trace(run_dir, "event_trace.jsonl"),
        "trace_tool": session.read_trace(run_dir, "tool_trace.jsonl"),
        "trace_patch": session.read_trace(run_dir, "patch_trace.jsonl"),
    }


def _build_memory(*, registry: SessionRegistry, ctx: str, run_name: str = "") -> dict[str, Any]:
    session = registry.get_session(ctx)
    _run_dir, selected_run = _run_dir_for_name(session, run_name)
    return {
        "selected_run": selected_run,
        "memory": session.read_memory_index(),
    }


def _nav_html(view: str) -> str:
    home_class = "nav-link active" if view == "home" else "nav-link"
    monitor_class = "nav-link active" if view == "monitor" else "nav-link"
    return (
        '<nav class="nav-bar">'
        f'<a class="{home_class}" href="/">Home</a>'
        f'<a class="{monitor_class}" href="/monitor/">Monitor</a>'
        "</nav>"
    )


def _page_html(*, view: str) -> str:
    boot = json.dumps({"view": view}, ensure_ascii=False)
    title = "CatMaster" if view == "home" else "CatMaster Monitor"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <link rel="stylesheet" href="/static/app.css" />
</head>
<body>
  <div id="app"></div>
  <script>window.CATMASTER_BOOT = {boot};</script>
  <script type="module" src="/static/app.js"></script>
</body>
</html>"""


async def _json_body(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def create_app(*, project_space_root: str) -> FastAPI:
    default_project_space_root = str(Path(project_space_root).expanduser().resolve())
    registry = SessionRegistry(default_project_space_root=default_project_space_root)
    app = FastAPI(title="CatMaster WebUI")
    static_dir = Path(__file__).resolve().parent / "static"
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/monitor", include_in_schema=False)
    def _monitor_redirect(request: Request):
        query = request.url.query
        target = "/monitor/"
        if query:
            target = f"{target}?{query}"
        return RedirectResponse(url=target, status_code=307)

    @app.get("/", response_class=HTMLResponse)
    def _home_page() -> str:
        return _page_html(view="home")

    @app.get("/monitor/", response_class=HTMLResponse)
    def _monitor_page() -> str:
        return _page_html(view="monitor")

    @app.get("/api/bootstrap")
    def _bootstrap(
        ctx: Optional[str] = None,
        project_space: Optional[str] = None,
        run: Optional[str] = None,
        lane: str = "research",
    ):
        state = registry.bootstrap(ctx=ctx, project_space=project_space, run=run)
        snapshot = _build_snapshot(registry=registry, ctx=state.ctx, lane=lane, run_name=state.run_name)
        snapshot["status_message"] = state.status
        snapshot["workspace_root"] = state.project_space_root
        return JSONResponse(snapshot)

    @app.get("/api/session/{ctx}/snapshot")
    def _session_snapshot(ctx: str, lane: str = "research", run: str = ""):
        return JSONResponse(_build_snapshot(registry=registry, ctx=ctx, lane=lane, run_name=run))

    @app.get("/api/session/{ctx}/details")
    def _session_details(ctx: str, run: str = ""):
        return JSONResponse(_build_details(registry=registry, ctx=ctx, run_name=run))

    @app.get("/api/session/{ctx}/memory")
    def _session_memory(ctx: str, run: str = ""):
        return JSONResponse(_build_memory(registry=registry, ctx=ctx, run_name=run))

    @app.post("/api/session/{ctx}/workspace/open")
    async def _workspace_open(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        root_path = str(payload.get("root_path") or registry.default_project_space_root)
        session.set_workspace_root(root_path)
        ok, message = session.open_workspace_by_name(str(payload.get("workspace") or ""))
        snapshot = _build_snapshot(registry=registry, ctx=ctx, lane=str(payload.get("lane") or "research"))
        snapshot["ok"] = ok
        snapshot["status_message"] = message
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/workspace/refresh")
    async def _workspace_refresh(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        ok, message, _choices = session.set_workspace_root(str(payload.get("root_path") or registry.default_project_space_root))
        snapshot = _build_snapshot(registry=registry, ctx=ctx, lane=str(payload.get("lane") or "research"))
        snapshot["ok"] = ok
        snapshot["status_message"] = message
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/workspace/create")
    async def _workspace_create(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        root_path = str(payload.get("root_path") or registry.default_project_space_root)
        session.set_workspace_root(root_path)
        ok, message = session.create_workspace(str(payload.get("workspace") or ""))
        snapshot = _build_snapshot(registry=registry, ctx=ctx, lane=str(payload.get("lane") or "research"))
        snapshot["ok"] = ok
        snapshot["status_message"] = message
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/run/select")
    async def _run_select(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        message = session.select_run(str(payload.get("run_name") or ""))
        snapshot = _build_snapshot(
            registry=registry,
            ctx=ctx,
            lane=str(payload.get("lane") or "research"),
            run_name=str(payload.get("run_name") or ""),
        )
        snapshot["status_message"] = message
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/chat/create")
    async def _chat_create(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        session_id = session.create_chat_session()
        snapshot = _build_snapshot(registry=registry, ctx=ctx, lane=str(payload.get("lane") or "research"))
        snapshot = _apply_chat_session_view(snapshot)
        snapshot["status_message"] = f"Started new chat session: {session_id}"
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/chat/select")
    async def _chat_select(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        session_id = session.select_chat_session(str(payload.get("session_id") or ""))
        snapshot = _build_snapshot(registry=registry, ctx=ctx, lane=str(payload.get("lane") or "research"))
        snapshot = _apply_chat_session_view(snapshot)
        snapshot["status_message"] = f"Switched to chat session: {session_id}" if session_id else "Chat session not found."
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/run/start")
    async def _run_start(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        message = session.start_run(
            prompt=str(payload.get("prompt") or ""),
            lane=str(payload.get("lane") or "research"),
            run_mode=str(payload.get("run_mode") or "new_run"),
            resume_run_name=str(payload.get("resume_run_name") or ""),
            proposal_review=bool(payload.get("proposal_review", False)),
            log_llm=bool(payload.get("log_llm", False)),
            full_auto_major=bool(payload.get("full_auto_major", False)),
            seed_hypotheses=str(payload.get("seed_hypotheses") or ""),
            exploration_policy=str(payload.get("exploration_policy") or "anchored"),
            writing_mode=str(payload.get("writing_mode") or "none"),
            output_format=str(payload.get("output_format") or "md"),
            target_section=str(payload.get("target_section") or ""),
            max_cycles=int(payload.get("max_cycles") or 6),
            max_literature_queries=int(payload.get("max_literature_queries") or 4),
            max_fast_runs=int(payload.get("max_fast_runs") or 3),
            max_standard_runs=int(payload.get("max_standard_runs") or 2),
            allow_deep_report=bool(payload.get("allow_deep_report", False)),
        )
        snapshot = _build_snapshot(registry=registry, ctx=ctx, lane=str(payload.get("lane") or "research"))
        snapshot["status_message"] = message
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/run/interrupt")
    async def _run_interrupt(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        message = session.request_interrupt_current_run(note=str(payload.get("note") or ""))
        snapshot = _build_snapshot(registry=registry, ctx=ctx, lane=str(payload.get("lane") or "research"))
        snapshot["status_message"] = message
        return JSONResponse(snapshot)

    @app.post("/api/session/{ctx}/prompt/respond")
    async def _prompt_respond(ctx: str, request: Request):
        payload = await _json_body(request)
        session = registry.get_session(ctx)
        message = session.submit_prompt(str(payload.get("prompt_id") or ""), str(payload.get("text") or ""))
        snapshot = _build_snapshot(
            registry=registry,
            ctx=ctx,
            lane=str(payload.get("lane") or "research"),
            run_name=str(payload.get("run_name") or ""),
        )
        snapshot["status_message"] = message
        return JSONResponse(snapshot)

    @app.get("/api/session/{ctx}/stream")
    async def _session_stream(ctx: str, request: Request, last_seq: str = "0"):
        async def _event_stream():
            seq = _coerce_int(last_seq, 0)
            while True:
                if await request.is_disconnected():
                    break
                session = registry.get_session(ctx)
                reporter = session.reporter
                if reporter is None:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(1.0)
                    continue
                events, seq = await asyncio.to_thread(reporter.wait_for_events_since, seq, timeout_s=10.0)
                if not events:
                    yield ": keepalive\n\n"
                    continue
                runtime = _runtime_snapshot(session)
                for event in events:
                    envelope = {
                        "event": event,
                        "runtime": runtime,
                        "run_status": str(session.run_status or "idle"),
                        "run_status_text": session.run_status_text(),
                    }
                    envelope.update(_stream_patch(session, runtime))
                    yield f"id: {int(event.get('seq') or seq)}\ndata: {json.dumps(envelope, ensure_ascii=False)}\n\n"

        return StreamingResponse(_event_stream(), media_type="text/event-stream")

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
        "timeout_keep_alive": max(0, int(timeout_keep_alive)),
        "timeout_graceful_shutdown": max(0, int(timeout_graceful_shutdown)),
    }
    try:
        uvicorn.run(app, **run_kwargs)
    except TypeError:
        run_kwargs.pop("timeout_graceful_shutdown", None)
        uvicorn.run(app, **run_kwargs)


__all__ = ["create_app", "launch"]
