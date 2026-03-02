from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gradio as gr

from .components import (
    SHARED_CSS,
    build_hitl_group,
    build_workspace_controls,
    hero_banner_html,
    nav_header_html,
    unpack_prompt,
)
from .constants import HOME_POLL_INTERVAL
from .session_registry import SessionRegistry


def build_home_page(
    *,
    registry: SessionRegistry,
    default_workspace: str,
    theme: Optional[Any] = None,
) -> gr.Blocks:

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

    def _monitor_url(ctx: str, project_space_name: str, run_name: str) -> str:
        return registry.monitor_url(
            ctx=ctx, project_space=project_space_name, run=run_name,
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_load(request: gr.Request) -> Tuple[
        str, str, Any, str, str, str, str, str, Any, str,
    ]:
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
        ps_name = registry.project_space_name_for_session(session)
        run_info = session.run_status_text()
        run_status = session.run_status or "idle"
        return (
            state.ctx,
            state.project_space_root,
            gr.update(choices=workspaces, value=ps_name or None),
            state.project_space_path,
            state.status,
            hero_banner_html(
                run_status, run_info,
                _monitor_url(state.ctx, ps_name, selected), ps_name,
            ),
            final_report,
            selected,
            gr.update(choices=runs, value=selected or None),
            nav_header_html("home", ps_name),
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

    def _on_run_mode_change(
        run_mode: str, selected_run: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        resume_mode = run_mode == "resume_selected_run"
        return (
            _prompt_box_for_mode(run_mode),
            gr.update(visible=not resume_mode),
            gr.update(visible=resume_mode, value=(selected_run or None)),
        )

    def _interrupt_run(note: str, ctx: str) -> str:
        session = registry.get_session(ctx)
        return session.request_interrupt_current_run(note=note or "")

    def _poll_home(
        ctx: str, selected_run: str,
    ) -> Tuple[str, str, str, str, str, Any, str, str, Any, str]:
        try:
            return _poll_home_inner(ctx, selected_run)
        except Exception as exc:
            gr.Warning(f"Poll error: {exc}")
            return (
                "", "", "", "", "",
                gr.update(), "", selected_run,
                gr.update(), "",
            )

    def _poll_home_inner(
        ctx: str, selected_run: str,
    ) -> Tuple[str, str, str, str, str, Any, str, str, Any, str]:
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
        run_status = session.run_status or "idle"
        ps_name = registry.project_space_name_for_session(session)

        prompt = unpack_prompt(session.get_prompt())

        return (
            hero_banner_html(
                run_status, run_info,
                _monitor_url(ctx, ps_name, selected), ps_name,
            ),
            prompt.title,
            prompt.body,
            prompt.meta,
            prompt.prompt_id,
            gr.update(visible=prompt.visible),
            final_report,
            selected,
            gr.update(choices=runs, value=selected or None),
            nav_header_html("home", ps_name),
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    with gr.Blocks(theme=theme, title="CatMaster Workbench") as page:
        gr.HTML(f"<style>{SHARED_CSS}</style>")
        ctx_state = gr.State("")
        selected_run_state = gr.State("")

        # -- Sidebar workspace controls --
        ws = build_workspace_controls(
            registry=registry,
            default_workspace=default_workspace,
            ctx_state=ctx_state,
        )

        nav_html = gr.HTML(nav_header_html("home"))

        with gr.Column(elem_classes=["cm-shell"]):

            # -- Hero status banner --
            hero_html = gr.HTML(hero_banner_html("idle", "", "/monitor/", ""))

            # -- HITL prompt (prominent, above content) --
            hitl = build_hitl_group(registry=registry, ctx_state=ctx_state)

            # -- Main content: prompt + final report --
            with gr.Row(equal_height=True, elem_classes=["cm-top-align"]):
                with gr.Column(scale=5, elem_classes=["cm-panel"]):
                    gr.Markdown("### Prompt")
                    prompt_box = gr.Textbox(
                        label="User Request",
                        lines=6,
                        placeholder="Describe what CatMaster should do.",
                    )
                    with gr.Row():
                        run_mode_box = gr.Dropdown(
                            label="Run Mode",
                            choices=[
                                ("New Run", "new_run"),
                                ("Resume Selected Run", "resume_selected_run"),
                            ],
                            value="new_run",
                        )
                        lane_box = gr.Dropdown(
                            label="Lane",
                            choices=["fast", "standard"],
                            value="standard",
                        )
                    resume_run_box = gr.Dropdown(
                        label="Resume Run", choices=[], visible=False,
                    )
                    with gr.Row():
                        start_btn = gr.Button("Start Run", variant="primary")
                        interrupt_btn = gr.Button("Interrupt")
                    with gr.Accordion("Advanced", open=False):
                        with gr.Row():
                            proposal_review_box = gr.Checkbox(label="Proposal Review", value=True)
                            log_llm_box = gr.Checkbox(label="Log LLM", value=False)
                            full_auto_major_box = gr.Checkbox(label="Full Auto Major", value=False)

                with gr.Column(scale=5, elem_classes=["cm-panel"]):
                    gr.Markdown("### Final Report")
                    final_report_md = gr.Markdown(
                        "Waiting for run output...",
                        elem_classes=["cm-scroll-md"],
                    )

        # ------------------------------------------------------------------
        # Event wiring
        # ------------------------------------------------------------------

        start_btn.click(
            _start_run,
            inputs=[
                prompt_box, lane_box, run_mode_box, resume_run_box,
                proposal_review_box, log_llm_box, full_auto_major_box,
                ctx_state,
            ],
            outputs=[ws.status_md],
        )

        interrupt_btn.click(
            _interrupt_run,
            inputs=[prompt_box, ctx_state],
            outputs=[ws.status_md],
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
            outputs=[ws.status_md, selected_run_state],
            queue=False,
        )

        _poll_outputs = [
            hero_html,
            hitl.title_md,
            hitl.body_md,
            hitl.meta_md,
            hitl.prompt_id_box,
            hitl.group,
            final_report_md,
            selected_run_state,
            resume_run_box,
            nav_html,
        ]

        timer = gr.Timer(HOME_POLL_INTERVAL)
        timer.tick(
            _poll_home,
            inputs=[ctx_state, selected_run_state],
            outputs=_poll_outputs,
            queue=False,
            trigger_mode="always_last",
        )

        page.load(
            _on_load,
            inputs=None,
            outputs=[
                ctx_state,
                ws.root_box,
                ws.workspace_list,
                ws.current_box,
                ws.status_md,
                hero_html,
                final_report_md,
                selected_run_state,
                resume_run_box,
                nav_html,
            ],
            queue=False,
        )

    return page


__all__ = ["build_home_page"]
