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
# Hero status banner (home page)
# ---------------------------------------------------------------------------

def hero_banner_html(
    status: str,
    run_info: str,
    monitor_url: str,
    project_space: str,
) -> str:
    badge = status_badge_html(status, large=True)
    ps = escape(project_space) if project_space else "(none)"
    info = escape(run_info) if run_info else ""
    return (
        f'<div class="cm-hero">'
        f'<div class="cm-hero-left">'
        f'<div class="cm-hero-status">{badge}</div>'
        f'<div class="cm-hero-info">{info}</div>'
        f"</div>"
        f'<div class="cm-hero-right">'
        f'<span class="cm-hero-ps">Project: <b>{ps}</b></span>'
        f'<a class="cm-hero-monitor-link" href="{monitor_url}" target="_blank">'
        f"Open Monitor &rarr;</a>"
        f"</div>"
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Shared CSS -- rich visual design
# ---------------------------------------------------------------------------

SHARED_CSS = """\
/* ---- pulsing dot animation ---- */
@keyframes cm-pulse {
  0%,100% { opacity:1; box-shadow:0 0 6px currentColor; }
  50% { opacity:.5; box-shadow:0 0 14px currentColor; }
}

/* ---- page background: warm neutral, very subtle violet tint ---- */
.gradio-container {
  background: linear-gradient(160deg, #faf9fb 0%, #f5f3ff 30%, #f9fafb 70%, #f3f4f6 100%) !important;
  min-height: 100vh;
}

/* ---- navigation bar: dark slate, clean and professional ---- */
.cm-nav {
  display:flex; align-items:center; justify-content:space-between;
  padding:10px 24px;
  background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #1e293b 100%);
  border-radius: 0 0 14px 14px;
  box-shadow: 0 4px 20px rgba(30,27,75,.22), 0 1px 3px rgba(0,0,0,.1);
  margin-bottom: 12px;
}
.cm-nav-brand {
  display:inline-flex; align-items:center; gap:10px;
}
.cm-nav-title {
  font-size:1.3rem; font-weight:800; color:#e0e7ff;
  letter-spacing:-0.02em; text-shadow:0 1px 4px rgba(0,0,0,.2);
}
.cm-nav-links { display:flex; gap:4px; }
.cm-nav-link {
  text-decoration:none; font-size:0.88rem; font-weight:600;
  color:rgba(224,231,255,.6); padding:5px 14px; border-radius:8px;
  transition: all .18s ease;
}
.cm-nav-link:hover { color:#e0e7ff; background:rgba(255,255,255,.08); }
.cm-nav-link.active {
  color:#fff; background:rgba(129,140,248,.2);
  box-shadow: inset 0 -2px 0 0 #818cf8;
}
.cm-nav-chip {
  font-size:0.76rem; color:rgba(224,231,255,.8);
  background:rgba(255,255,255,.1); padding:3px 12px;
  border-radius:999px; border:1px solid rgba(255,255,255,.08);
}

/* ---- hero banner: subtle indigo tint ---- */
.cm-hero {
  display:flex; justify-content:space-between; align-items:center;
  padding:14px 20px; border-radius:14px; margin-bottom:10px;
  background: linear-gradient(135deg, rgba(99,102,241,.06) 0%, rgba(139,92,246,.04) 100%);
  border:1px solid rgba(99,102,241,.12);
}
.cm-hero-left { display:flex; align-items:center; gap:16px; }
.cm-hero-status { flex-shrink:0; }
.cm-hero-info { font-size:0.84rem; color:var(--body-text-color-subdued,#6b7280); }
.cm-hero-right { display:flex; align-items:center; gap:16px; text-align:right; }
.cm-hero-ps { font-size:0.82rem; color:var(--body-text-color-subdued,#6b7280); }
.cm-hero-monitor-link {
  font-size:0.82rem; font-weight:600; color:#4f46e5;
  text-decoration:none; padding:5px 14px; border-radius:8px;
  background:rgba(99,102,241,.08); transition:all .15s;
}
.cm-hero-monitor-link:hover {
  background:rgba(99,102,241,.15); color:#4338ca;
}

/* ---- shell container ---- */
.cm-shell { max-width:1360px; margin:0 auto; }

/* ---- glassmorphism panels ---- */
.cm-panel {
  background: rgba(255,255,255,.88);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(229,231,235,.6);
  border-radius: 16px;
  padding: 16px;
  box-shadow: 0 4px 24px rgba(0,0,0,.04), 0 1px 2px rgba(0,0,0,.03);
  transition: box-shadow .2s ease, transform .15s ease;
}
.cm-panel:hover {
  box-shadow: 0 8px 32px rgba(0,0,0,.07), 0 2px 4px rgba(0,0,0,.04);
  transform: translateY(-1px);
}

.cm-top-align { align-items:flex-start !important; }

/* ---- scrollable containers ---- */
.cm-scroll-md,.cm-scroll-html,.cm-scroll-code {
  max-height:560px; overflow-y:auto; overflow-x:hidden; padding-right:6px;
}

/* ---- HITL panel: warm amber attention ---- */
.cm-hitl-panel {
  border:2px solid #f59e0b; border-radius:16px; padding:16px;
  background: linear-gradient(135deg, rgba(245,158,11,.05) 0%, rgba(252,211,77,.03) 100%);
  box-shadow: 0 0 16px rgba(245,158,11,.08);
  animation: cm-hitl-glow 2.2s ease-in-out infinite;
}
@keyframes cm-hitl-glow {
  0%,100% { box-shadow: 0 0 16px rgba(245,158,11,.08); }
  50% { box-shadow: 0 0 28px rgba(245,158,11,.16); }
}

/* ---- typography refinements ---- */
.cm-panel h3, .cm-panel h4 {
  letter-spacing: -0.02em;
  color: var(--body-text-color, #111827);
}

/* ---- faded divider ---- */
.cm-divider {
  height:1px; border:none; margin:12px 0;
  background: linear-gradient(90deg, transparent 0%, #e5e7eb 30%, #e5e7eb 70%, transparent 100%);
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
    with gr.Sidebar(label="Project Space", open=False) as sidebar:
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

    def _submit(pid: str, text: str, ctx: str) -> Tuple[str, str]:
        session = registry.get_session(ctx)
        status = session.submit_prompt(pid, text)
        return status, ""

    def _approve(pid: str, ctx: str) -> Tuple[str, str]:
        session = registry.get_session(ctx)
        status = session.submit_prompt(pid, "yes")
        return status, ""

    submit_btn.click(
        _submit,
        inputs=[prompt_id_box, prompt_input, ctx_state],
        outputs=[prompt_status, prompt_input],
    )
    approve_btn.click(
        _approve,
        inputs=[prompt_id_box, ctx_state],
        outputs=[prompt_status, prompt_input],
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
    meta = ""
    if kind == "proposal_review":
        title = "Proposal Review"
        body = payload.get("proposal_description", "") or ""
        todo = payload.get("todo", []) or []
        if isinstance(todo, list) and todo:
            meta = "Work packages:\n" + "\n".join(
                f"{i + 1}. {item}" for i, item in enumerate(todo)
            )
    elif kind == "hitl":
        title = "HITL Feedback Required"
        body = payload.get("report_text", "") or ""
        rp = payload.get("report_path", "") or ""
        meta = f"Report: {rp}" if rp else ""
    elif kind == "interrupt_feedback":
        title = "Interrupt Guidance Required"
        body = payload.get("guidance", "") or "Run was interrupted."
        run_id = payload.get("run_id", "") or ""
        phase = payload.get("phase", "") or ""
        bits = [f"run_id={run_id}" if run_id else "", f"phase={phase}" if phase else ""]
        meta = " ".join(b for b in bits if b)
    return PromptDisplay(visible=True, title=title, body=body, meta=meta, prompt_id=prompt_id)


__all__ = [
    "SHARED_CSS",
    "HITLComponents",
    "PromptDisplay",
    "WorkspaceComponents",
    "build_hitl_group",
    "build_workspace_controls",
    "hero_banner_html",
    "nav_header_html",
    "status_badge_html",
    "status_color",
    "unpack_prompt",
]
