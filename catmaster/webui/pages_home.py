from __future__ import annotations

from typing import Any, Dict, List, Tuple

import gradio as gr

from .components import (
    SHARED_CSS,
    build_hitl_group,
    compact_status_bar_html,
    nav_header_html,
    unpack_prompt,
)
from .constants import ACTIVE_POLL_INTERVAL, SIDEBAR_POLL_INTERVAL
from .session_registry import SessionRegistry
from .view_utils import (
    render_live_tracker_markdown,
    render_sidebar_run_cards_html,
)

_PLAN_HEADER = (
    '<div class="cm-right-card-header">'
    '<span class="cm-right-card-icon cm-plan-icon">P</span>Plan</div>'
)
_RESULTS_HEADER = (
    '<div class="cm-right-card-header">'
    '<span class="cm-right-card-icon cm-results-icon">R</span>Results</div>'
)
_PLAN_ACTIVE_STATUSES = {"running", "starting", "interrupting", "paused"}


def _truncate_inline(text: str, max_chars: int = 160) -> str:
    compact = " ".join(str(text or "").split()).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max(0, max_chars - 3)].rstrip() + "..."


def _running_chat_placeholder(live_state: Dict[str, Any]) -> str:
    task_goal = _truncate_inline(str(live_state.get("current_task_goal") or ""), max_chars=180)
    phase = _truncate_inline(str(live_state.get("current_phase") or ""), max_chars=80)
    if task_goal:
        return (
            "Run in progress.\n\n"
            f"Current task: {task_goal}\n\n"
            "Follow the right-side Results card for live execution details."
        )
    if phase:
        return (
            "Run in progress.\n\n"
            f"Current phase: {phase}\n\n"
            "Follow the right-side Results card for live execution details."
        )
    return (
        "Run in progress.\n\n"
        "Follow the right-side Results card for live execution details."
    )


def build_home_page(
    *,
    registry: SessionRegistry,
    default_workspace: str,
) -> gr.Blocks:

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _monitor_url(ctx: str, ps_name: str, run_name: str) -> str:
        return registry.monitor_url(ctx=ctx, project_space=ps_name, run=run_name)

    def _get_live_state(session, run_dir):
        reporter = session.reporter
        active_run_dir = reporter.get_run_dir() if reporter else None
        if reporter and active_run_dir and run_dir and active_run_dir == run_dir:
            events, _ = session.get_events()
            return session.update_live_state(
                run_dir, events, live_llm_enabled=False,
            )
        return session.snapshot_live_state(run_dir)

    def _chat_messages_for_home(session, live_state: Dict[str, Any], run_status: str) -> List[Dict[str, str]]:
        messages = list(session.get_chat_messages(limit=40))
        if run_status in ("running", "starting", "interrupting"):
            messages.append({"role": "assistant", "content": _running_chat_placeholder(live_state)})
        return messages

    def _chat_session_updates(session, *, lane: str) -> tuple[Dict[str, Any], str]:
        choices = session.list_chat_sessions()
        current = session.current_chat_session_id()
        return gr.update(choices=choices, value=(current or None)), session.entry_context_status_text(lane=lane)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_load(request: gr.Request) -> tuple:
        params = dict(getattr(request, "query_params", {}) or {})
        state = registry.bootstrap(
            ctx=params.get("ctx"),
            project_space=params.get("project_space"),
            run=params.get("run"),
        )
        session = registry.get_session(state.ctx)
        snapshot = session.get_sidebar_snapshot()
        runs = list(snapshot.get("runs") or [])
        cards = list(snapshot.get("cards") or [])
        selected = state.run_name or (runs[0][1] if runs else "")
        if selected:
            session.select_run(selected)
        run_dir = session.get_selected_run_dir()
        workspaces = session.list_workspaces()
        ps_name = registry.project_space_name_for_session(session)
        run_info = session.run_status_text()
        run_status = session.run_status or "idle"
        model_name = str((session.run_info or {}).get("model_name", ""))
        is_running = run_status in ("running", "starting", "interrupting")

        reporter = session.reporter
        active_reporter_dir = reporter.get_run_dir() if reporter else None
        live_dir = active_reporter_dir if (is_running and active_reporter_dir) else run_dir
        live_state = _get_live_state(session, live_dir)
        chat_messages = _chat_messages_for_home(session, live_state, run_status)
        chat_session_update, chat_context_status = _chat_session_updates(session, lane="standard")

        sidebar_html = render_sidebar_run_cards_html(
            cards, selected_run=selected, search_text="",
        )
        detail_dir = live_dir if (is_running and active_reporter_dir) else run_dir
        proposal = session.read_proposal(detail_dir) if run_status in _PLAN_ACTIVE_STATUSES else ""
        tracker_md = render_live_tracker_markdown(live_state)

        return (
            state.ctx,                                   # ctx_state
            state.project_space_root,                    # ws_root_box
            gr.update(choices=workspaces, value=ps_name or None),  # ws_list
            state.project_space_path,                    # ws_current_box
            state.status,                                # ws_status_md
            compact_status_bar_html(
                run_status, run_info, model_name,
                _monitor_url(state.ctx, ps_name, selected), ps_name,
            ),                                           # status_bar_html
            chat_session_update,                         # chat_session_dropdown
            chat_context_status,                         # chat_context_status_md
            chat_messages,                               # chatbot
            selected,                                    # selected_run_state
            gr.update(choices=runs, value=selected or None),  # runs_dropdown
            gr.update(choices=runs, value=selected or None),  # resume_run_box
            nav_header_html("home", ps_name),            # nav_html
            sidebar_html,                                # sidebar_cards_html
            proposal or "No plan yet.\n\nPlans appear for multi-step tasks.",
            tracker_md,                                  # results_md
        )

    def _start_run(
        prompt: str,
        lane: str,
        run_mode: str,
        resume_run_name: str,
        proposal_review: bool,
        log_llm: bool,
        full_auto_major: bool,
        seed_hypotheses: str,
        exploration_policy: str,
        writing_mode: str,
        target_section: str,
        max_cycles: int,
        max_literature_queries: int,
        max_fast_runs: int,
        max_standard_runs: int,
        allow_deep_report: bool,
        ctx: str,
    ) -> Tuple[str, str, str]:
        if not (prompt or "").strip():
            return "Please enter a request.", prompt, ""
        session = registry.get_session(ctx)
        msg = session.start_run(
            prompt=prompt,
            lane=lane,
            run_mode=run_mode,
            resume_run_name=resume_run_name,
            proposal_review=proposal_review,
            log_llm=log_llm,
            full_auto_major=full_auto_major,
            seed_hypotheses=seed_hypotheses,
            exploration_policy=exploration_policy,
            writing_mode=writing_mode,
            target_section=target_section,
            max_cycles=max_cycles,
            max_literature_queries=max_literature_queries,
            max_fast_runs=max_fast_runs,
            max_standard_runs=max_standard_runs,
            allow_deep_report=allow_deep_report,
        )
        return msg, "", session.entry_context_status_text(lane=lane)

    def _create_chat_session(ctx: str, lane: str) -> tuple:
        session = registry.get_session(ctx)
        session.create_chat_session()
        chat_session_update, chat_context_status = _chat_session_updates(session, lane=lane)
        return (
            chat_session_update,
            _chat_messages_for_home(session, {}, session.run_status or "idle"),
            chat_context_status,
            "Started a new chat session.",
        )

    def _select_chat_session(session_id: str, lane: str, ctx: str) -> tuple:
        session = registry.get_session(ctx)
        if session_id:
            session.select_chat_session(session_id)
        chat_session_update, chat_context_status = _chat_session_updates(session, lane=lane)
        return (
            chat_session_update,
            _chat_messages_for_home(session, {}, session.run_status or "idle"),
            chat_context_status,
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
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        resume_mode = run_mode == "resume_selected_run"
        if resume_mode:
            prompt_update = gr.update(
                placeholder=(
                    "Resuming interrupted run: provide guidance. "
                    "Leave empty to continue."
                ),
            )
        else:
            prompt_update = gr.update(
                placeholder="Describe what CatMaster should do...",
            )
        return prompt_update, gr.update(visible=resume_mode, value=(selected_run or None))

    def _interrupt_run(ctx: str) -> str:
        session = registry.get_session(ctx)
        return session.request_interrupt_current_run()

    def _on_lane_change(lane: str, ctx: str) -> Tuple[Dict[str, Any], str]:
        session = registry.get_session(ctx)
        return (gr.update(visible=(lane == "research")), session.entry_context_status_text(lane=lane))

    def _poll_home_active(
        ctx: str, selected_run: str, lane: str,
    ) -> tuple:
        try:
            return _poll_home_active_inner(ctx, selected_run, lane)
        except Exception as exc:
            gr.Warning(f"Poll error: {exc}")
            return ("",) * 5 + (gr.update(),) + ([],) + ("",) * 4

    def _poll_home_active_inner(
        ctx: str, selected_run: str, lane: str,
    ) -> tuple:
        session = registry.get_session(ctx)
        selected = (selected_run or "").strip()
        if selected:
            session.select_run(selected)

        run_dir = session.get_selected_run_dir()
        run_info = session.run_status_text()
        run_status = session.run_status or "idle"
        ps_name = registry.project_space_name_for_session(session)
        model_name = str((session.run_info or {}).get("model_name", ""))
        is_running = run_status in ("running", "starting", "interrupting")

        prompt_display = unpack_prompt(session.get_prompt())

        reporter = session.reporter
        active_reporter_dir = reporter.get_run_dir() if reporter else None
        live_dir = active_reporter_dir if (is_running and active_reporter_dir) else run_dir
        live_state = _get_live_state(session, live_dir)
        chat_messages = _chat_messages_for_home(session, live_state, run_status)
        detail_dir = live_dir if (is_running and active_reporter_dir) else run_dir
        proposal = session.read_proposal(detail_dir) if run_status in _PLAN_ACTIVE_STATUSES else ""
        tracker_md = render_live_tracker_markdown(live_state)
        chat_context_status = session.entry_context_status_text(lane=lane)

        return (
            compact_status_bar_html(
                run_status, run_info, model_name,
                _monitor_url(ctx, ps_name, selected), ps_name,
            ),
            prompt_display.title,
            prompt_display.body,
            prompt_display.meta,
            prompt_display.prompt_id,
            gr.update(visible=prompt_display.visible),
            chat_messages,
            chat_context_status,
            proposal or "No plan yet.\n\nPlans appear for multi-step tasks.",
            tracker_md,
        )

    def _poll_home_sidebar(
        ctx: str, selected_run: str, search_text: str,
    ) -> tuple:
        try:
            return _poll_home_sidebar_inner(ctx, selected_run, search_text)
        except Exception as exc:
            gr.Warning(f"Sidebar poll error: {exc}")
            return gr.update(), "", "", selected_run

    def _poll_home_sidebar_inner(
        ctx: str, selected_run: str, search_text: str,
    ) -> tuple:
        session = registry.get_session(ctx)
        snapshot = session.get_sidebar_snapshot()
        runs = list(snapshot.get("runs") or [])
        cards = list(snapshot.get("cards") or [])
        selected = (selected_run or "").strip()
        run_names = {name for _, name in runs}
        if not selected and runs:
            selected = runs[0][1]
        if selected and selected not in run_names:
            selected = ""
        if selected:
            session.select_run(selected)
        ps_name = registry.project_space_name_for_session(session)
        sidebar_html = render_sidebar_run_cards_html(
            cards, selected_run=selected, search_text=search_text,
        )
        return (
            gr.update(choices=runs, value=(selected if selected else None), allow_custom_value=True),
            gr.update(choices=runs, value=(selected if selected else None), allow_custom_value=True),
            sidebar_html,
            nav_header_html("home", ps_name),
            selected,
        )

    # ------------------------------------------------------------------
    # Workspace callbacks
    # ------------------------------------------------------------------

    def _ws_refresh(root_path: str, ctx: str) -> Tuple[Any, str, str]:
        session = registry.get_session(ctx)
        ok, msg, choices = session.set_workspace_root(root_path)
        ps_name = registry.project_space_name_for_session(session)
        return (
            gr.update(choices=choices, value=ps_name if ok and ps_name else None),
            msg,
            session.current_workspace_path(),
        )

    def _ws_open(root_path: str, name: str, ctx: str) -> Tuple[str, str]:
        session = registry.get_session(ctx)
        session.set_workspace_root(root_path)
        _, msg = session.open_workspace_by_name(name)
        return msg, session.current_workspace_path()

    def _ws_create(root_path: str, name: str, ctx: str) -> Tuple[Any, str, str, str]:
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

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    with gr.Blocks(title="CatMaster") as page:
        gr.HTML(f"<style>{SHARED_CSS}</style>")
        ctx_state = gr.State("")
        selected_run_state = gr.State("")

        # -- Left Sidebar --
        with gr.Sidebar(label="CatMaster", open=True):
            with gr.Accordion("Project Space", open=False):
                ws_root_box = gr.Textbox(label="Root", value=default_workspace)
                ws_list = gr.Dropdown(label="Projects", choices=[])
                with gr.Row():
                    ws_open_btn = gr.Button("Open", size="sm")
                    ws_create_btn = gr.Button("Create", variant="primary", size="sm")
                ws_new_name = gr.Textbox(label="New Name")
                ws_refresh_btn = gr.Button("Refresh", size="sm")
            ws_current_box = gr.Textbox(visible=False)
            ws_status_md = gr.Markdown("")

            gr.HTML('<hr class="cm-divider">')
            gr.Markdown("### Chat Sessions")
            chat_session_dropdown = gr.Dropdown(label="Active Session", choices=[])
            new_chat_session_btn = gr.Button("New Session", size="sm")

            gr.HTML('<hr class="cm-divider">')
            gr.Markdown("### Tasks")
            search_box = gr.Textbox(
                placeholder="Search tasks...", label="", container=False,
            )
            runs_dropdown = gr.Dropdown(label="Select Task", choices=[])
            sidebar_cards_html = gr.HTML("")

            with gr.Accordion("Run Settings", open=False):
                run_mode_box = gr.Dropdown(
                    label="Run Mode",
                    choices=[
                        ("New Run", "new_run"),
                        ("Resume Selected Run", "resume_selected_run"),
                    ],
                    value="new_run",
                )
                resume_run_box = gr.Dropdown(
                    label="Resume Run", choices=[], visible=False,
                )
                lane_box = gr.Dropdown(
                    choices=["fast", "standard", "research", "writing"],
                    value="standard",
                    label="Lane",
                )
                with gr.Row():
                    proposal_review_box = gr.Checkbox(label="Proposal Review", value=True)
                    log_llm_box = gr.Checkbox(label="Log LLM", value=False)
                    full_auto_major_box = gr.Checkbox(label="Full Auto Major", value=False)
                with gr.Group(visible=False) as research_controls_group:
                    seed_hypotheses_box = gr.Textbox(
                        label="Seed Hypotheses", lines=3,
                        placeholder="One hypothesis per line.",
                    )
                    with gr.Row():
                        exploration_policy_box = gr.Dropdown(
                            label="Exploration Policy",
                            choices=["anchored", "local_expand", "open"],
                            value="anchored",
                        )
                        writing_mode_box = gr.Dropdown(
                            label="Writing Mode",
                            choices=["none", "internal_report", "paper_outline",
                                     "section_draft", "full_draft"],
                            value="none",
                        )
                    target_section_box = gr.Textbox(
                        label="Target Section",
                        placeholder="Used when Writing Mode = section_draft",
                    )
                    with gr.Row():
                        max_cycles_box = gr.Number(label="Max Cycles", value=6, precision=0)
                        max_literature_box = gr.Number(label="Max Lit Queries", value=4, precision=0)
                    with gr.Row():
                        max_fast_runs_box = gr.Number(label="Max Fast Runs", value=3, precision=0)
                        max_standard_runs_box = gr.Number(label="Max Std Runs", value=2, precision=0)
                    allow_deep_report_box = gr.Checkbox(label="Allow Deep Report", value=False)

        # -- Navigation --
        nav_html = gr.HTML(nav_header_html("home"))

        # -- Main Content --
        with gr.Column(elem_classes=["cm-shell"]):

            status_bar_html = gr.HTML(
                compact_status_bar_html("idle", "", "", "/monitor/", ""),
            )

            hitl = build_hitl_group(registry=registry, ctx_state=ctx_state)

            with gr.Row(equal_height=False, elem_classes=["cm-top-align"]):

                # -- Center: Conversation --
                with gr.Column(scale=7):
                    chatbot = gr.Chatbot(
                        value=[],
                        height=520,
                        buttons=["copy", "copy_all"],
                        placeholder="Start a new task below to begin the conversation.",
                    )
                    chat_context_status_md = gr.Markdown("")
                    with gr.Row():
                        prompt_box = gr.Textbox(
                            placeholder="Describe what CatMaster should do...",
                            label="",
                            lines=2,
                            scale=5,
                        )
                    with gr.Row():
                        start_btn = gr.Button(
                            "Send", variant="primary", scale=3,
                        )
                        interrupt_btn = gr.Button("Stop", scale=1)

                # -- Right: Plan + Results --
                with gr.Column(scale=3):
                    with gr.Group(elem_classes=["cm-right-card"]):
                        gr.HTML(_PLAN_HEADER)
                        plan_md = gr.Markdown(
                            "No plan yet.\n\nPlans appear for multi-step tasks.",
                            elem_classes=["cm-right-scroll"],
                        )
                    with gr.Group(elem_classes=["cm-right-card"]):
                        gr.HTML(_RESULTS_HEADER)
                        results_md = gr.Markdown(
                            "No results yet.",
                            elem_classes=["cm-right-scroll"],
                        )

        # ------------------------------------------------------------------
        # Event wiring
        # ------------------------------------------------------------------

        ws_refresh_btn.click(
            _ws_refresh,
            inputs=[ws_root_box, ctx_state],
            outputs=[ws_list, ws_status_md, ws_current_box],
        )
        ws_open_btn.click(
            _ws_open,
            inputs=[ws_root_box, ws_list, ctx_state],
            outputs=[ws_status_md, ws_current_box],
        )
        ws_create_btn.click(
            _ws_create,
            inputs=[ws_root_box, ws_new_name, ctx_state],
            outputs=[ws_list, ws_status_md, ws_current_box, ws_new_name],
        )

        start_btn.click(
            _start_run,
            inputs=[
                prompt_box, lane_box, run_mode_box, resume_run_box,
                proposal_review_box, log_llm_box, full_auto_major_box,
                seed_hypotheses_box, exploration_policy_box, writing_mode_box,
                target_section_box,
                max_cycles_box, max_literature_box, max_fast_runs_box,
                max_standard_runs_box, allow_deep_report_box,
                ctx_state,
            ],
            outputs=[ws_status_md, prompt_box, chat_context_status_md],
        )

        interrupt_btn.click(
            _interrupt_run,
            inputs=[ctx_state],
            outputs=[ws_status_md],
        )

        run_mode_box.change(
            _on_run_mode_change,
            inputs=[run_mode_box, selected_run_state],
            outputs=[prompt_box, resume_run_box],
            queue=False,
        )

        lane_box.change(
            _on_lane_change,
            inputs=[lane_box, ctx_state],
            outputs=[research_controls_group, chat_context_status_md],
            queue=False,
        )

        new_chat_session_btn.click(
            _create_chat_session,
            inputs=[ctx_state, lane_box],
            outputs=[chat_session_dropdown, chatbot, chat_context_status_md, ws_status_md],
            queue=False,
        )

        chat_session_dropdown.change(
            _select_chat_session,
            inputs=[chat_session_dropdown, lane_box, ctx_state],
            outputs=[chat_session_dropdown, chatbot, chat_context_status_md],
            queue=False,
        )

        runs_dropdown.change(
            _select_run,
            inputs=[runs_dropdown, ctx_state],
            outputs=[ws_status_md, selected_run_state],
            queue=False,
        )

        resume_run_box.change(
            _select_run,
            inputs=[resume_run_box, ctx_state],
            outputs=[ws_status_md, selected_run_state],
            queue=False,
        )

        # -- Poll --
        _active_poll_outputs = [
            status_bar_html,
            hitl.title_md,
            hitl.body_md,
            hitl.meta_md,
            hitl.prompt_id_box,
            hitl.group,
            chatbot,
            chat_context_status_md,
            plan_md,
            results_md,
        ]

        active_timer = gr.Timer(ACTIVE_POLL_INTERVAL)
        active_timer.tick(
            _poll_home_active,
            inputs=[ctx_state, selected_run_state, lane_box],
            outputs=_active_poll_outputs,
            queue=False,
            trigger_mode="always_last",
        )

        sidebar_timer = gr.Timer(SIDEBAR_POLL_INTERVAL)
        sidebar_timer.tick(
            _poll_home_sidebar,
            inputs=[ctx_state, selected_run_state, search_box],
            outputs=[runs_dropdown, resume_run_box, sidebar_cards_html, nav_html, selected_run_state],
            queue=False,
            trigger_mode="always_last",
        )

        page.load(
            _on_load,
            inputs=None,
            outputs=[
                ctx_state,
                ws_root_box,
                ws_list,
                ws_current_box,
                ws_status_md,
                status_bar_html,
                chat_session_dropdown,
                chat_context_status_md,
                chatbot,
                selected_run_state,
                runs_dropdown,
                resume_run_box,
                nav_html,
                sidebar_cards_html,
                plan_md,
                results_md,
            ],
            queue=False,
        )

    return page


__all__ = ["build_home_page"]
