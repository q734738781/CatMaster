from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from .session_registry import SessionRegistry


# ---------------------------------------------------------------------------
# Status badge
# ---------------------------------------------------------------------------

_STATUS_COLORS: Dict[str, str] = {
    "running": "#16a34a",
    "starting": "#16a34a",
    "done": "#64748b",
    "error": "#dc2626",
    "failure": "#dc2626",
    "paused": "#d97706",
    "interrupted_paused": "#d97706",
    "interrupting": "#d97706",
    "awaiting_human_feedback": "#2563eb",
    "needs_intervention": "#ea580c",
    "idle": "#94a3b8",
    "unknown": "#94a3b8",
}


def status_badge_html(status: str) -> str:
    color = _STATUS_COLORS.get(status, "#94a3b8")
    label = status.replace("_", " ")
    return (
        f'<span style="display:inline-flex;align-items:center;gap:6px;">'
        f'<span style="width:8px;height:8px;border-radius:50%;background:{color};display:inline-block;"></span>'
        f'<code style="font-size:0.85rem;">{label}</code>'
        f"</span>"
    )


# ---------------------------------------------------------------------------
# Navigation header
# ---------------------------------------------------------------------------

_NAV_CSS = """\
.cm-nav{display:flex;align-items:center;justify-content:space-between;padding:6px 16px;
border-bottom:1px solid var(--border-color-primary);margin-bottom:8px;}
.cm-nav-title{font-size:1.15rem;font-weight:700;color:var(--body-text-color);}
.cm-nav-links{display:flex;gap:18px;}
.cm-nav-links a{text-decoration:none;font-size:0.92rem;font-weight:500;
color:var(--body-text-color-subdued);transition:color .15s;}
.cm-nav-links a:hover,.cm-nav-links a.active{color:var(--color-accent);}
.cm-nav-ctx{font-size:0.82rem;color:var(--body-text-color-subdued);}
"""


def nav_header_html(active: str, project_space: str = "") -> str:
    def _cls(page: str) -> str:
        return ' class="active"' if page == active else ""

    ctx_chip = f'<span class="cm-nav-ctx">{project_space}</span>' if project_space else ""
    return (
        f'<nav class="cm-nav">'
        f'<span class="cm-nav-title">CatMaster</span>'
        f'<span class="cm-nav-links">'
        f'<a href="/"{_cls("home")}>Workbench</a>'
        f'<a href="/monitor/"{_cls("monitor")}>Monitor</a>'
        f"</span>"
        f"{ctx_chip}"
        f"</nav>"
    )


# ---------------------------------------------------------------------------
# Shared CSS (layout helpers only -- no theme overrides)
# ---------------------------------------------------------------------------

SHARED_CSS = (
    _NAV_CSS
    + """\
.cm-shell{max-width:1320px;margin:0 auto;}
.cm-panel{background:var(--background-fill-primary);border:1px solid var(--border-color-primary);
border-radius:12px;padding:12px;box-shadow:0 4px 16px rgba(15,23,42,.04);}
.cm-top-align{align-items:flex-start !important;}
.cm-scroll-md{max-height:560px;overflow-y:auto;overflow-x:hidden;padding-right:6px;}
.cm-scroll-html{max-height:560px;overflow-y:auto;overflow-x:hidden;padding-right:6px;}
.cm-scroll-code{max-height:560px;overflow-y:auto;overflow-x:hidden;padding-right:6px;}
.cm-hitl-panel{border:2px solid #d97706;border-radius:12px;padding:12px;
background:color-mix(in srgb, #d97706 6%, var(--background-fill-primary));}
.cm-autoscroll{scroll-behavior:smooth;}
"""
)


# ---------------------------------------------------------------------------
# Workspace controls
# ---------------------------------------------------------------------------

@dataclass
class WorkspaceComponents:
    root_box: gr.Textbox
    workspace_list: gr.Dropdown
    current_box: gr.Textbox
    status_md: gr.Markdown
    accordion: gr.Accordion


def build_workspace_controls(
    *,
    registry: SessionRegistry,
    default_workspace: str,
    ctx_state: gr.State,
) -> WorkspaceComponents:
    with gr.Accordion("Project Space", open=False) as accordion:
        with gr.Row():
            root_box = gr.Textbox(label="Project Space Root", value=default_workspace, scale=4)
            refresh_btn = gr.Button("Refresh", scale=1)
        with gr.Row():
            workspace_list = gr.Dropdown(label="Project Spaces", choices=[], scale=3)
            open_btn = gr.Button("Open", scale=1)
            new_name = gr.Textbox(label="New Project Space", scale=2)
            create_btn = gr.Button("Create", variant="primary", scale=1)
        current_box = gr.Textbox(label="Current Project Space", interactive=False)
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
        accordion=accordion,
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
    "nav_header_html",
    "status_badge_html",
    "unpack_prompt",
]
