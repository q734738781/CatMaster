from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from .session_registry import SessionRegistry


# ---------------------------------------------------------------------------
# Status badge (animated pulsing dot for running, pill background)
# ---------------------------------------------------------------------------

_STATUS_COLORS: Dict[str, str] = {
    "running": "#10b981",
    "starting": "#10b981",
    "done": "#6b7280",
    "error": "#ef4444",
    "failure": "#ef4444",
    "paused": "#f59e0b",
    "interrupted_paused": "#f59e0b",
    "interrupting": "#f59e0b",
    "awaiting_human_feedback": "#6366f1",
    "needs_intervention": "#f97316",
    "idle": "#9ca3af",
    "unknown": "#9ca3af",
}

_PULSING_STATUSES = {"running", "starting", "interrupting"}


def status_color(status: str) -> str:
    return _STATUS_COLORS.get(status, "#94a3b8")


def status_badge_html(status: str, *, large: bool = False) -> str:
    color = status_color(status)
    label = escape(status.replace("_", " "))
    dot_size = "12px" if large else "10px"
    font_size = "0.92rem" if large else "0.82rem"
    pad = "3px 12px 3px 10px" if large else "2px 10px 2px 8px"
    pulse = (
        f"animation:cm-pulse 1.6s ease-in-out infinite;"
        if status in _PULSING_STATUSES
        else ""
    )
    return (
        f'<span style="display:inline-flex;align-items:center;gap:7px;'
        f"background:{color}14;border:1px solid {color}30;border-radius:999px;"
        f'padding:{pad};white-space:nowrap;">'
        f'<span style="width:{dot_size};height:{dot_size};border-radius:50%;'
        f"background:{color};display:inline-block;flex-shrink:0;"
        f'box-shadow:0 0 6px {color}60;{pulse}"></span>'
        f'<span style="font-size:{font_size};font-weight:600;color:{color};'
        f'letter-spacing:.01em;">{label}</span>'
        f"</span>"
    )


# ---------------------------------------------------------------------------
# Inline SVG logo mark
# ---------------------------------------------------------------------------

_LOGO_SVG = (
    '<svg width="28" height="28" viewBox="0 0 28 28" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M14 2L25.66 8.5V21.5L14 28L2.34 21.5V8.5L14 2Z" fill="url(#cm_g)" stroke="rgba(255,255,255,.25)" stroke-width=".5"/>'
    '<text x="14" y="18" text-anchor="middle" font-size="11" font-weight="700" fill="#fff" font-family="IBM Plex Sans,sans-serif">CM</text>'
    '<defs><linearGradient id="cm_g" x1="2" y1="2" x2="26" y2="28">'
    '<stop offset="0%" stop-color="#818cf8"/><stop offset="100%" stop-color="#7c3aed"/>'
    '</linearGradient></defs></svg>'
)


# ---------------------------------------------------------------------------
# Navigation header
# ---------------------------------------------------------------------------

def nav_header_html(active: str, project_space: str = "") -> str:
    def _cls(page: str) -> str:
        return ' class="cm-nav-link active"' if page == active else ' class="cm-nav-link"'

    ps_escaped = escape(project_space) if project_space else ""
    ctx_chip = (
        f'<span class="cm-nav-chip">{ps_escaped}</span>'
        if ps_escaped
        else ""
    )
    return (
        f'<nav class="cm-nav">'
        f'<span class="cm-nav-brand">{_LOGO_SVG}'
        f'<span class="cm-nav-title">CatMaster</span></span>'
        f'<span class="cm-nav-links">'
        f'<a href="/"{_cls("home")}>Workbench</a>'
        f'<a href="/monitor/"{_cls("monitor")}>Monitor</a>'
        f"</span>"
        f"{ctx_chip}"
        f"</nav>"
    )


# ---------------------------------------------------------------------------
# Compact status bar (home page)
# ---------------------------------------------------------------------------

def compact_status_bar_html(
    status: str,
    run_info: str,
    model_name: str,
    monitor_url: str,
    project_space: str,
) -> str:
    badge = status_badge_html(status)
    info_escaped = escape(run_info) if run_info else ""
    model_escaped = escape(model_name) if model_name else ""
    ps_escaped = escape(project_space) if project_space else ""
    right_parts: list[str] = []
    if model_escaped:
        right_parts.append(f'<span class="cm-status-model">{model_escaped}</span>')
    if ps_escaped:
        right_parts.append(f'<span class="cm-status-ps">{ps_escaped}</span>')
    right_parts.append(
        f'<a class="cm-status-link" href="{monitor_url}" target="_blank">'
        f"Monitor &rarr;</a>"
    )
    return (
        f'<div class="cm-status-bar">'
        f'<div class="cm-status-left">'
        f"{badge}"
        f'<span class="cm-status-info">{info_escaped}</span>'
        f"</div>"
        f'<div class="cm-status-right">'
        + "".join(right_parts)
        + "</div></div>"
    )


# ---------------------------------------------------------------------------
# Shared CSS -- rich visual design
# ---------------------------------------------------------------------------

SHARED_CSS = """\
@keyframes cm-pulse {
  0%,100% { opacity:1; box-shadow:0 0 6px currentColor; }
  50% { opacity:.5; box-shadow:0 0 14px currentColor; }
}

.gradio-container {
  background: #f7f8fa !important;
  min-height: 100vh;
}

/* ---- navigation bar ---- */
.cm-nav {
  display:flex; align-items:center; justify-content:space-between;
  padding:8px 20px;
  background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
  border-radius: 0 0 12px 12px;
  box-shadow: 0 2px 12px rgba(0,0,0,.12);
  margin-bottom: 8px;
}
.cm-nav-brand { display:inline-flex; align-items:center; gap:8px; }
.cm-nav-title {
  font-size:1.1rem; font-weight:700; color:#fff;
  letter-spacing:-0.01em;
}
.cm-nav-links { display:flex; gap:2px; }
.cm-nav-link {
  text-decoration:none; font-size:0.82rem; font-weight:600;
  color:rgba(255,255,255,.6); padding:5px 12px; border-radius:6px;
  transition: all .15s ease;
}
.cm-nav-link:hover { color:#fff; background:rgba(255,255,255,.08); }
.cm-nav-link.active { color:#fff; background:rgba(255,255,255,.12); }
.cm-nav-chip {
  font-size:0.72rem; color:rgba(255,255,255,.75);
  background:rgba(255,255,255,.08); padding:3px 10px;
  border-radius:999px; border:1px solid rgba(255,255,255,.1);
}

/* ---- compact status bar ---- */
.cm-status-bar {
  display:flex; align-items:center; justify-content:space-between;
  padding:10px 16px; border-radius:10px;
  background:#fff; border:1px solid #e8eaed;
  margin-bottom:8px;
  box-shadow: 0 1px 3px rgba(0,0,0,.04);
}
.cm-status-left { display:flex; align-items:center; gap:12px; }
.cm-status-info { font-size:0.8rem; color:#5f6368; }
.cm-status-right { display:flex; align-items:center; gap:10px; }
.cm-status-model { font-size:0.75rem; color:#80868b; font-family:monospace; }
.cm-status-ps { font-size:0.75rem; color:#80868b; }
.cm-status-link {
  font-size:0.78rem; font-weight:600; color:#4f46e5;
  text-decoration:none; padding:3px 10px; border-radius:6px;
  background:rgba(79,70,229,.06); transition:all .12s;
}
.cm-status-link:hover { background:rgba(79,70,229,.12); }

/* ---- shell container ---- */
.cm-shell { max-width:1440px; margin:0 auto; }
.cm-top-align { align-items:flex-start !important; }

/* ---- right panel ---- */
.cm-right-card {
  background:#fff; border:1px solid #e8eaed;
  border-radius:12px; padding:14px; margin-bottom:8px;
  box-shadow: 0 1px 3px rgba(0,0,0,.03);
}
.cm-right-card-header {
  display:flex; align-items:center; gap:8px;
  font-size:0.85rem; font-weight:700; color:#202124;
  margin-bottom:10px; padding-bottom:8px;
  border-bottom:1px solid #f1f3f4;
}
.cm-right-card-icon {
  width:22px; height:22px; display:flex; align-items:center;
  justify-content:center; border-radius:6px; font-size:0.75rem;
  font-weight:700;
}
.cm-plan-icon { background:#e8f5e9; color:#2e7d32; }
.cm-results-icon { background:#fff3e0; color:#ef6c00; }

/* ---- content / report area ---- */
.cm-content-area {
  background:#fff; border:1px solid #e8eaed;
  border-radius:12px; padding:20px 24px;
  box-shadow: 0 1px 3px rgba(0,0,0,.03);
  min-height:300px;
}

/* ---- scrollable containers ---- */
.cm-scroll-md,.cm-scroll-html,.cm-scroll-code {
  max-height:600px; overflow-y:auto; overflow-x:hidden; padding-right:6px;
}
.cm-right-scroll {
  max-height:260px; overflow-y:auto; overflow-x:hidden; padding-right:4px;
}

/* ---- HITL panel ---- */
.cm-hitl-panel {
  border:2px solid #f59e0b; border-radius:12px; padding:14px;
  background: linear-gradient(135deg, rgba(245,158,11,.04) 0%, rgba(252,211,77,.02) 100%);
  box-shadow: 0 0 12px rgba(245,158,11,.08);
  animation: cm-hitl-glow 2.2s ease-in-out infinite;
}
@keyframes cm-hitl-glow {
  0%,100% { box-shadow: 0 0 12px rgba(245,158,11,.08); }
  50% { box-shadow: 0 0 24px rgba(245,158,11,.14); }
}

/* ---- panels (monitor) ---- */
.cm-panel {
  background: rgba(255,255,255,.92);
  border:1px solid #e8eaed; border-radius:12px;
  padding:14px;
  box-shadow: 0 1px 6px rgba(0,0,0,.03);
  transition: box-shadow .2s ease;
}
.cm-panel:hover { box-shadow: 0 4px 16px rgba(0,0,0,.06); }

/* ---- event feed (monitor) ---- */
.cm-event-feed {
  max-height:560px; overflow-y:auto; overflow-x:hidden;
  padding-right:6px; font-family:var(--font-mono);
  scroll-behavior:smooth;
}

/* ---- sidebar run list ---- */
.cm-sidebar-runs {
  display:flex; flex-direction:column; gap:4px;
  max-height:400px; overflow-y:auto; padding-right:4px;
}
.cm-run-item {
  display:flex; align-items:center; gap:8px;
  padding:8px 10px; border-radius:8px;
  background:transparent; border:1px solid transparent;
  transition: all .12s ease; cursor:default;
}
.cm-run-item:hover {
  background:rgba(79,70,229,.04); border-color:#e8eaed;
}
.cm-run-item-active {
  background:rgba(79,70,229,.06) !important;
  border-color:rgba(79,70,229,.25) !important;
}
.cm-run-dot {
  width:8px; height:8px; border-radius:50%; flex-shrink:0;
}
.cm-run-info { flex:1; min-width:0; }
.cm-run-title {
  font-size:0.78rem; font-weight:600; color:#202124;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}
.cm-run-meta {
  font-size:0.68rem; color:#80868b; margin-top:1px;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
}

/* ---- divider ---- */
.cm-divider {
  height:1px; border:none; margin:10px 0;
  background: linear-gradient(90deg, transparent 0%, #e8eaed 30%, #e8eaed 70%, transparent 100%);
}
"""


# ---------------------------------------------------------------------------
# Workspace controls (Sidebar)
# ---------------------------------------------------------------------------

@dataclass
class WorkspaceComponents:
    root_box: gr.Textbox
    workspace_list: gr.Dropdown
    current_box: gr.Textbox
    status_md: gr.Markdown
    sidebar: Any


def build_workspace_controls(
    *,
    registry: SessionRegistry,
    default_workspace: str,
    ctx_state: gr.State,
) -> WorkspaceComponents:
    with gr.Sidebar(label="Project Space", open=True) as sidebar:
        gr.Markdown("### Project Space")
        root_box = gr.Textbox(label="Root", value=default_workspace)
        refresh_btn = gr.Button("Refresh")
        workspace_list = gr.Dropdown(label="Project Spaces", choices=[])
        with gr.Row():
            open_btn = gr.Button("Open", scale=1)
            create_btn = gr.Button("Create", variant="primary", scale=1)
        new_name = gr.Textbox(label="New Name")
        current_box = gr.Textbox(label="Current", interactive=False)
        status_md = gr.Markdown("")

    def _refresh(root_path: str, ctx: str) -> Tuple[Any, str, str]:
        session = registry.get_session(ctx)
        ok, msg, choices = session.set_workspace_root(root_path)
        ps_name = registry.project_space_name_for_session(session)
        return (
            gr.update(choices=choices, value=ps_name if ok and ps_name else None),
            msg,
            session.current_workspace_path(),
        )

    def _open(root_path: str, name: str, ctx: str) -> Tuple[str, str]:
        session = registry.get_session(ctx)
        session.set_workspace_root(root_path)
        _, msg = session.open_workspace_by_name(name)
        return msg, session.current_workspace_path()

    def _create(root_path: str, name: str, ctx: str) -> Tuple[Any, str, str, str]:
        session = registry.get_session(ctx)
        ok, msg, choices = session.set_workspace_root(root_path)
        if not ok:
            return gr.update(choices=[], value=None), msg, session.current_workspace_path(), name
        created, create_msg = session.create_workspace(name)
        if created:
            choices = session.list_workspaces()
        ps_name = registry.project_space_name_for_session(session)
        return (
            gr.update(choices=choices, value=ps_name if created else None),
            create_msg or msg,
            session.current_workspace_path(),
            "",
        )

    refresh_btn.click(
        _refresh,
        inputs=[root_box, ctx_state],
        outputs=[workspace_list, status_md, current_box],
    )
    open_btn.click(
        _open,
        inputs=[root_box, workspace_list, ctx_state],
        outputs=[status_md, current_box],
    )
    create_btn.click(
        _create,
        inputs=[root_box, new_name, ctx_state],
        outputs=[workspace_list, status_md, current_box, new_name],
    )

    return WorkspaceComponents(
        root_box=root_box,
        workspace_list=workspace_list,
        current_box=current_box,
        status_md=status_md,
        sidebar=sidebar,
    )


# ---------------------------------------------------------------------------
# HITL prompt group
# ---------------------------------------------------------------------------

@dataclass
class HITLComponents:
    group: gr.Group
    title_md: gr.Markdown
    body_md: gr.Markdown
    meta_md: gr.Markdown
    prompt_input: gr.Textbox
    prompt_status: gr.Markdown
    prompt_id_box: gr.Textbox


def build_hitl_group(
    *,
    registry: SessionRegistry,
    ctx_state: gr.State,
) -> HITLComponents:
    with gr.Group(visible=False) as group:
        with gr.Column(elem_classes=["cm-hitl-panel"]):
            gr.Markdown("## Input Required")
            title_md = gr.Markdown()
            body_md = gr.Markdown()
            meta_md = gr.Markdown()
            prompt_input = gr.Textbox(label="Your feedback", lines=3)
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                approve_btn = gr.Button("Approve (yes)")
            prompt_status = gr.Markdown("")
            prompt_id_box = gr.Textbox(visible=False)

    def _submit(pid: str, text: str, ctx: str) -> Tuple[str, str, Any]:
        session = registry.get_session(ctx)
        status = session.submit_prompt(pid, text)
        return status, "", gr.update(visible=False)

    def _approve(pid: str, ctx: str) -> Tuple[str, str, Any]:
        session = registry.get_session(ctx)
        status = session.submit_prompt(pid, "yes")
        return status, "", gr.update(visible=False)

    submit_btn.click(
        _submit,
        inputs=[prompt_id_box, prompt_input, ctx_state],
        outputs=[prompt_status, prompt_input, group],
    )
    approve_btn.click(
        _approve,
        inputs=[prompt_id_box, ctx_state],
        outputs=[prompt_status, prompt_input, group],
    )

    return HITLComponents(
        group=group,
        title_md=title_md,
        body_md=body_md,
        meta_md=meta_md,
        prompt_input=prompt_input,
        prompt_status=prompt_status,
        prompt_id_box=prompt_id_box,
    )


# ---------------------------------------------------------------------------
# Prompt payload unpacking (shared between pages)
# ---------------------------------------------------------------------------

@dataclass
class PromptDisplay:
    visible: bool
    title: str
    body: str
    meta: str
    prompt_id: str


def unpack_prompt(pending: Optional[Dict[str, Any]]) -> PromptDisplay:
    if not pending:
        return PromptDisplay(visible=False, title="", body="", meta="", prompt_id="")
    prompt_id = pending.get("prompt_id", "")
    kind = pending.get("kind", "")
    payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
    title = "Input Required"
    body = ""
    meta_lines: list[str] = []
    run_id = str(payload.get("run_id") or "")
    prompt_id_text = str(payload.get("prompt_id") or prompt_id or "")
    if kind == "proposal_review":
        is_revised = bool(payload.get("is_revised"))
        title = "Revised Proposal Review" if is_revised else "Proposal Review"
        body = payload.get("proposal_description", "") or ""
        todo = payload.get("todo", []) or []
        if is_revised:
            if run_id:
                meta_lines.append(f"same run: `{run_id}`")
            reason = str(payload.get("reason") or "replanning after HITL")
            meta_lines.append(f"reason: {reason}")
        elif run_id:
            meta_lines.append(f"run: `{run_id}`")
        if prompt_id_text:
            meta_lines.append(f"prompt id: `{prompt_id_text}`")
        if isinstance(todo, list) and todo:
            meta_lines.append("Work packages:")
            meta_lines.extend(
                f"{i + 1}. {item}" for i, item in enumerate(todo)
            )
    elif kind == "hitl":
        title = "HITL Feedback Required"
        body = payload.get("report_text", "") or ""
        rp = payload.get("report_path", "") or ""
        if run_id:
            meta_lines.append(f"run: `{run_id}`")
        if prompt_id_text:
            meta_lines.append(f"prompt id: `{prompt_id_text}`")
        if rp:
            meta_lines.append(f"report: `{rp}`")
    elif kind == "interrupt_feedback":
        title = "Interrupt Guidance Required"
        body = payload.get("guidance", "") or "Run was interrupted."
        run_id = payload.get("run_id", "") or ""
        phase = payload.get("phase", "") or ""
        if run_id:
            meta_lines.append(f"run: `{run_id}`")
        if prompt_id_text:
            meta_lines.append(f"prompt id: `{prompt_id_text}`")
        if phase:
            meta_lines.append(f"phase: `{phase}`")
    meta = "\n".join(meta_lines)
    return PromptDisplay(visible=True, title=title, body=body, meta=meta, prompt_id=prompt_id)


__all__ = [
    "SHARED_CSS",
    "HITLComponents",
    "PromptDisplay",
    "WorkspaceComponents",
    "build_hitl_group",
    "build_workspace_controls",
    "compact_status_bar_html",
    "nav_header_html",
    "status_badge_html",
    "status_color",
    "unpack_prompt",
]
