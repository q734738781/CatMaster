from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr

from .constants import EVENT_POLL_INTERVAL
from .session_registry import SessionRegistry

_HOME_CSS = """
:root {
  --cm-bg: radial-gradient(1200px 700px at 10% 0%, #f5f8ff 0%, #eef3f9 45%, #e8edf4 100%);
  --cm-card: #ffffff;
  --cm-border: #dbe2ec;
  --cm-text: #1d2939;
  --cm-subtle: #526173;
  --cm-accent: #0d6e6e;
}
.gradio-container {
  background: var(--cm-bg);
  color: var(--cm-text);
  font-family: "IBM Plex Sans", "Noto Sans", sans-serif;
}
.cm-shell {
  max-width: 1280px;
  margin: 0 auto;
}
.cm-top-align {
  align-items: flex-start !important;
}
.cm-panel {
  background: var(--cm-card);
  border: 1px solid var(--cm-border);
  border-radius: 14px;
  padding: 12px;
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
}
.cm-mini-note {
  color: var(--cm-subtle);
  font-size: 0.92rem;
}
.cm-scroll-markdown {
  max-height: 560px;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 6px;
}
"""


def build_home_page(*, registry: SessionRegistry, default_workspace: str, theme: Optional[Any] = None) -> gr.Blocks:
    def _prompt_box_for_mode(run_mode: str) -> Dict[str, Any]:
        if run_mode == "resume_selected_run":
            return gr.update(
                label="Interrupt Guidance (optional)",
                placeholder=(
                    "Resuming interrupted run: provide guidance for next action. "
                    "Leave empty to continue with no feedback."
                ),
            )
        return gr.update(
            label="User Request",
            placeholder="Describe what CatMaster should do.",
        )

    def _monitor_link(ctx: str, project_space_name: str, run_name: str) -> str:
        url = registry.monitor_url(ctx=ctx, project_space=project_space_name, run=run_name)
        return (
            f'<div class="cm-mini-note">'
            f'<a href="{url}" target="_blank">Open Monitor in new tab</a>'
            f"</div>"
        )

    def _on_load(request: gr.Request) -> Tuple[str, str, gr.Dropdown, str, str, str, str, str, str, gr.Dropdown]:
        params = dict(getattr(request, "query_params", {}) or {})
        state = registry.bootstrap(
            ctx=params.get("ctx"),
            project_space=params.get("project_space"),
            run=params.get("run"),
        )
        session = registry.get_session(state.ctx)
        runs = session.list_runs()
        selected = state.run_name or (runs[0][1] if runs else "")
        if selected:
            session.select_run(selected)
        run_dir = session.get_selected_run_dir()
        final_report, _ = session.read_final_report_with_source(run_dir)
        workspaces = session.list_workspaces()
        project_space_name = registry.project_space_name_for_session(session)
        run_info = session.run_status_text()
        return (
            state.ctx,
            state.project_space_root,
            gr.update(choices=workspaces, value=project_space_name or None),
            state.project_space_path,
            state.status,
            _monitor_link(state.ctx, project_space_name, selected),
            run_info,
            final_report,
            selected,
            gr.update(choices=runs, value=selected or None),
        )

    def _refresh_workspaces(root_path: str, ctx: str) -> Tuple[gr.Dropdown, str, str, str]:
        session = registry.get_session(ctx)
        ok, msg, choices = session.set_workspace_root(root_path)
        project_space_name = registry.project_space_name_for_session(session)
        return (
            gr.update(choices=choices, value=project_space_name if ok and project_space_name else None),
            msg,
            session.current_workspace_path(),
            _monitor_link(ctx, project_space_name, ""),
        )

    def _open_workspace(root_path: str, name: str, ctx: str) -> Tuple[str, str, str]:
        session = registry.get_session(ctx)
        session.set_workspace_root(root_path)
        _, msg = session.open_workspace_by_name(name)
        project_space_name = registry.project_space_name_for_session(session)
        return msg, session.current_workspace_path(), _monitor_link(ctx, project_space_name, "")

    def _create_workspace(root_path: str, name: str, ctx: str) -> Tuple[gr.Dropdown, str, str, str, str]:
        session = registry.get_session(ctx)
        ok, msg, choices = session.set_workspace_root(root_path)
        if not ok:
            return gr.update(choices=[], value=None), msg, session.current_workspace_path(), _monitor_link(ctx, "", ""), name
        created, create_msg = session.create_workspace(name)
        if created:
            choices = session.list_workspaces()
        project_space_name = registry.project_space_name_for_session(session)
        return (
            gr.update(choices=choices, value=project_space_name if created else None),
            create_msg if create_msg else msg,
            session.current_workspace_path(),
            _monitor_link(ctx, project_space_name, ""),
            "",
        )

    def _start_run(
        prompt: str,
        lane: str,
        run_mode: str,
        resume_run_name: str,
        proposal_review: bool,
        log_llm: bool,
        full_auto_major: bool,
        ctx: str,
    ) -> str:
        session = registry.get_session(ctx)
        return session.start_run(
            prompt=prompt,
            lane=lane,
            run_mode=run_mode,
            resume_run_name=resume_run_name,
            proposal_review=proposal_review,
            log_llm=log_llm,
            full_auto_major=full_auto_major,
        )

    def _select_run(run_name: str, ctx: str) -> Tuple[str, str]:
        selected = (run_name or "").strip()
        if not selected:
            return "", ""
        session = registry.get_session(ctx)
        msg = session.select_run(selected)
        if msg.startswith("Invalid") or msg.startswith("Open"):
            return msg, ""
        return msg, selected

    def _on_run_mode_change(run_mode: str, selected_run: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        resume_mode = run_mode == "resume_selected_run"
        return (
            _prompt_box_for_mode(run_mode),
            gr.update(visible=not resume_mode),
            gr.update(visible=resume_mode, value=(selected_run or None)),
        )

    def _submit_prompt(prompt_id: str, text: str, ctx: str) -> Tuple[str, str]:
        session = registry.get_session(ctx)
        status = session.submit_prompt(prompt_id, text)
        return status, ""

    def _submit_approve(prompt_id: str, ctx: str) -> Tuple[str, str]:
        session = registry.get_session(ctx)
        status = session.submit_prompt(prompt_id, "yes")
        return status, ""

    def _interrupt_run(note: str, ctx: str) -> str:
        session = registry.get_session(ctx)
        return session.request_interrupt_current_run(note=note or "")

    def _poll_home(ctx: str, selected_run: str) -> Tuple[str, str, str, str, Any, str, str, str, str, gr.Dropdown]:
        session = registry.get_session(ctx)
        runs = session.list_runs()
        selected = (selected_run or "").strip()
        run_names = {name for _, name in runs}

        if not selected and runs:
            selected = runs[0][1]
        if selected and selected not in run_names:
            selected = ""
        if selected:
            session.select_run(selected)

        run_dir = session.get_selected_run_dir()
        final_report, _ = session.read_final_report_with_source(run_dir)
        run_info = session.run_status_text()
        project_space_name = registry.project_space_name_for_session(session)

        pending = session.get_prompt()
        prompt_visible = False
        prompt_title = ""
        prompt_body = ""
        prompt_meta = ""
        prompt_id = ""
        if pending:
            prompt_visible = True
            prompt_id = pending.get("prompt_id", "")
            kind = pending.get("kind", "")
            payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
            if kind == "proposal_review":
                prompt_title = "Proposal Review"
                prompt_body = payload.get("proposal_description", "") or ""
                todo = payload.get("todo", []) or []
                if isinstance(todo, list) and todo:
                    prompt_meta = "Next tasks:\n" + "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(todo)])
            elif kind == "hitl":
                prompt_title = "HITL Feedback Required"
                prompt_body = payload.get("report_text", "") or ""
                report_path = payload.get("report_path", "") or ""
                prompt_meta = f"Report: {report_path}" if report_path else ""
            elif kind == "interrupt_feedback":
                prompt_title = "Interrupt Guidance Required"
                prompt_body = payload.get("guidance", "") or "Run was interrupted."
                run_id = payload.get("run_id", "") or ""
                phase = payload.get("phase", "") or ""
                bits = [f"run_id={run_id}" if run_id else "", f"phase={phase}" if phase else ""]
                prompt_meta = " ".join([b for b in bits if b])
            else:
                prompt_title = "Input Required"

        return (
            run_info,
            final_report,
            _monitor_link(ctx, project_space_name, selected),
            prompt_title,
            prompt_body,
            prompt_meta,
            gr.update(visible=prompt_visible),
            prompt_id,
            selected,
            gr.update(choices=runs, value=selected or None),
        )

    _ = theme
    with gr.Blocks() as page:
        gr.HTML(f"<style>{_HOME_CSS}</style>")
        ctx_state = gr.State("")
        selected_run_state = gr.State("")

        with gr.Column(elem_classes=["cm-shell"]):
            gr.Markdown("# CatMaster Workbench")
            with gr.Row():
                workspace_root_box = gr.Textbox(label="Project Space Root", value=default_workspace)
                refresh_workspaces_btn = gr.Button("Refresh")

            with gr.Row():
                workspace_list = gr.Dropdown(label="Project Spaces", choices=[])
                open_workspace_btn = gr.Button("Open")
                new_workspace_name = gr.Textbox(label="New Project Space")
                create_workspace_btn = gr.Button("Create", variant="primary")

            with gr.Row():
                current_workspace_box = gr.Textbox(label="Current Project Space", interactive=False)
                monitor_link_html = gr.HTML("")

            status_box = gr.Markdown(elem_classes=["cm-mini-note"])

            with gr.Row(equal_height=True, elem_classes=["cm-top-align"]):
                with gr.Column(scale=5, elem_classes=["cm-panel"]):
                    gr.Markdown("## Prompt")
                    prompt_box = gr.Textbox(label="User Request", lines=6, placeholder="Describe what CatMaster should do.")
                    with gr.Row():
                        run_mode_box = gr.Dropdown(
                            label="Run Mode",
                            choices=[("New Run", "new_run"), ("Resume Selected Run", "resume_selected_run")],
                            value="new_run",
                        )
                        lane_box = gr.Dropdown(label="Lane", choices=["fast", "standard"], value="standard")
                    with gr.Row():
                        resume_run_box = gr.Dropdown(label="Resume Run", choices=[], visible=False)
                    with gr.Row():
                        start_btn = gr.Button("Start Run", variant="primary")
                        interrupt_btn = gr.Button("Interrupt")
                    with gr.Accordion("Advanced", open=False):
                        with gr.Row():
                            proposal_review_box = gr.Checkbox(label="Proposal Review", value=True)
                            log_llm_box = gr.Checkbox(label="Log LLM", value=False)
                            full_auto_major_box = gr.Checkbox(label="Full Auto Major", value=False)
                    run_info = gr.Markdown("")

                with gr.Column(scale=5, elem_classes=["cm-panel"]):
                    gr.Markdown("## Final Report")
                    final_report_md = gr.Markdown(
                        "Waiting for run output...",
                        elem_classes=["cm-scroll-markdown"],
                    )

            with gr.Group(visible=False) as prompt_group:
                with gr.Column(elem_classes=["cm-panel"]):
                    gr.Markdown("## Input Required")
                    prompt_title_md = gr.Markdown()
                    prompt_body_md = gr.Markdown()
                    prompt_meta_md = gr.Markdown()
                    prompt_input = gr.Textbox(label="Your feedback", lines=3)
                    with gr.Row():
                        submit_btn = gr.Button("Submit", variant="primary")
                        approve_btn = gr.Button("Approve (yes)")
                    prompt_status = gr.Markdown("")
                    prompt_id_box = gr.Textbox(visible=False)

        refresh_workspaces_btn.click(
            _refresh_workspaces,
            inputs=[workspace_root_box, ctx_state],
            outputs=[workspace_list, status_box, current_workspace_box, monitor_link_html],
        )

        open_workspace_btn.click(
            _open_workspace,
            inputs=[workspace_root_box, workspace_list, ctx_state],
            outputs=[status_box, current_workspace_box, monitor_link_html],
        )

        create_workspace_btn.click(
            _create_workspace,
            inputs=[workspace_root_box, new_workspace_name, ctx_state],
            outputs=[workspace_list, status_box, current_workspace_box, monitor_link_html, new_workspace_name],
        )

        start_btn.click(
            _start_run,
            inputs=[prompt_box, lane_box, run_mode_box, resume_run_box, proposal_review_box, log_llm_box, full_auto_major_box, ctx_state],
            outputs=[status_box],
        )

        interrupt_btn.click(
            _interrupt_run,
            inputs=[prompt_box, ctx_state],
            outputs=[status_box],
        )

        run_mode_box.change(
            _on_run_mode_change,
            inputs=[run_mode_box, selected_run_state],
            outputs=[prompt_box, lane_box, resume_run_box],
            queue=False,
        )

        resume_run_box.change(
            _select_run,
            inputs=[resume_run_box, ctx_state],
            outputs=[status_box, selected_run_state],
            queue=False,
        )

        submit_btn.click(
            _submit_prompt,
            inputs=[prompt_id_box, prompt_input, ctx_state],
            outputs=[prompt_status, prompt_input],
        )

        approve_btn.click(
            _submit_approve,
            inputs=[prompt_id_box, ctx_state],
            outputs=[prompt_status, prompt_input],
        )

        timer = gr.Timer(EVENT_POLL_INTERVAL)
        timer.tick(
            _poll_home,
            inputs=[ctx_state, selected_run_state],
            outputs=[
                run_info,
                final_report_md,
                monitor_link_html,
                prompt_title_md,
                prompt_body_md,
                prompt_meta_md,
                prompt_group,
                prompt_id_box,
                selected_run_state,
                resume_run_box,
            ],
            queue=False,
            trigger_mode="always_last",
        )

        page.load(
            _on_load,
            inputs=None,
            outputs=[
                ctx_state,
                workspace_root_box,
                workspace_list,
                current_workspace_box,
                status_box,
                monitor_link_html,
                run_info,
                final_report_md,
                selected_run_state,
                resume_run_box,
            ],
            queue=False,
        )

    return page


__all__ = ["build_home_page"]
