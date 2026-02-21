from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from catmaster.tools.base import workspace_root
from .constants import EVENT_POLL_INTERVAL, LIVE_SUMMARY_ENABLED_DEFAULT, MAX_EVENT_FEED
from .session_registry import SessionRegistry
from .view_utils import format_event_line, render_live_tracker_markdown, render_run_cards_html

_MONITOR_CSS = """
:root {
  --cm-bg: linear-gradient(165deg, #eef4f8 0%, #e7edf3 42%, #e2e8ef 100%);
  --cm-card: #ffffff;
  --cm-border: #d6deea;
  --cm-text: #10212f;
  --cm-subtle: #5b6b7d;
  --cm-accent: #035f87;
}
.gradio-container {
  background: var(--cm-bg);
  color: var(--cm-text);
  font-family: "Space Grotesk", "IBM Plex Sans", sans-serif;
}
.cm-monitor-shell {
  max-width: 1480px;
  margin: 0 auto;
}
.cm-top-align {
  align-items: flex-start !important;
}
.cm-panel {
  background: var(--cm-card);
  border: 1px solid var(--cm-border);
  border-radius: 14px;
  padding: 10px;
  box-shadow: 0 10px 24px rgba(15, 23, 42, 0.05);
}
.cm-card-list {
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.cm-card {
  border: 1px solid var(--cm-border);
  border-radius: 12px;
  padding: 10px;
  background: #fefefe;
  transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
}
.cm-card-link {
  text-decoration: none;
  color: inherit;
  display: block;
}
.cm-card-link:hover .cm-card {
  transform: translateY(-1px);
  border-color: #0d7a7a;
  box-shadow: 0 8px 20px rgba(13, 122, 122, 0.12);
}
.cm-card.cm-active {
  border-color: #0d7a7a;
  box-shadow: 0 0 0 1px #0d7a7a inset;
}
.cm-card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 0.84rem;
  color: var(--cm-subtle);
}
.cm-badge {
  border: 1px solid var(--cm-border);
  border-radius: 999px;
  padding: 1px 8px;
  font-size: 0.75rem;
}
.cm-card h4 {
  margin: 0.35rem 0;
  font-size: 0.95rem;
}
.cm-card p {
  margin: 0;
  font-size: 0.85rem;
  color: var(--cm-subtle);
}
.cm-actions {
  margin: 0.55rem 0;
  padding-left: 1rem;
}
.cm-actions li {
  margin: 0.15rem 0;
  font-size: 0.83rem;
}
.cm-meta {
  font-size: 0.76rem;
  color: var(--cm-subtle);
}
.cm-empty {
  color: var(--cm-subtle);
  font-size: 0.9rem;
  border: 1px dashed var(--cm-border);
  border-radius: 12px;
  padding: 12px;
}
.cm-scroll-html {
  max-height: 560px;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 6px;
}
.cm-scroll-code {
  max-height: 560px;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 6px;
}
.cm-scroll-markdown {
  max-height: 560px;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 6px;
}
.cm-live-summary-box {
  max-height: 560px;
  overflow-y: auto;
  overflow-x: hidden;
  padding-right: 6px;
}
.cm-live-summary-box pre,
.cm-live-summary-box code {
  white-space: pre-wrap;
  word-break: break-word;
}
"""


def build_monitor_page(*, registry: SessionRegistry, default_workspace: str, theme: Optional[Any] = None) -> gr.Blocks:
    def _workspace_dir(session) -> Optional[Path]:
        raw = session.current_workspace_path()
        if not raw:
            return None
        try:
            p = workspace_root(Path(raw).expanduser().resolve())
        except Exception:
            return None
        if not p.exists() or not p.is_dir():
            return None
        return p

    def _fileexplorer_filter_to_glob(filter_text: str) -> str:
        raw = (filter_text or "").strip()
        if not raw:
            return "**/*"
        glob_meta = set("*?[]{}")
        is_glob_like = any(ch in raw for ch in glob_meta) or ("/" in raw) or ("\\" in raw)
        if is_glob_like:
            return raw
        return f"**/*{raw}*"

    def _fileexplorer_ignore_glob(*, include_hidden: bool) -> Optional[str]:
        if include_hidden:
            return None
        return "{**/.*,**/.*/**}"

    def _read_workspace_file(session, selected_path: Any) -> str:
        base = _workspace_dir(session)
        if base is None:
            return "(no project space opened) Open a project space first."
        if selected_path is None:
            return ""
        if isinstance(selected_path, (list, tuple)):
            if not selected_path:
                return ""
            selected_path = selected_path[0]
        if not isinstance(selected_path, str) or not selected_path.strip():
            return ""
        try:
            raw = Path(selected_path).expanduser()
            target = raw.resolve() if raw.is_absolute() else (base / raw).resolve()
        except Exception:
            return f"(unavailable) Invalid path: {selected_path}"
        if base not in target.parents and target != base:
            return f"(blocked) Path is outside project files root: {selected_path}"
        try:
            rel_path = str(target.relative_to(base))
        except Exception:
            rel_path = str(target)
        if not target.exists():
            return f"(unavailable) Path does not exist:\n{target}"
        if target.is_dir():
            return f"(directory)\n{rel_path}"
        max_bytes = 200_000
        try:
            with target.open("rb") as f:
                data = f.read(max_bytes + 1)
        except Exception as exc:
            return f"(unavailable) Failed to read file:\n{rel_path}\n\n{exc}"
        truncated = len(data) > max_bytes
        if truncated:
            data = data[:max_bytes]
        text = data.decode("utf-8", errors="replace")
        if truncated:
            text += "\n\n...(truncated)..."
        return f"# {rel_path}\n\n{text}"

    def _cards_markdown(cards: List[Dict[str, Any]], selected_run: str) -> str:
        selected = next((c for c in cards if c.get("run_name") == selected_run), None)
        if not selected:
            return ""
        actions = selected.get("next_actions") if isinstance(selected.get("next_actions"), list) else []
        action_lines = "\n".join([f"- {item}" for item in actions[:3]]) if actions else "- (none)"
        return (
            f"### {selected.get('headline', selected_run)}\n"
            f"**Status**: `{selected.get('status', 'unknown')}`\n\n"
            f"{selected.get('summary', '')}\n\n"
            f"**Next Actions**\n{action_lines}"
        )

    def _monitor_link(ctx: str, project_space_name: str, run_name: str) -> str:
        url = registry.monitor_url(ctx=ctx, project_space=project_space_name, run=run_name)
        return f'<a href="{url}" target="_blank">Permalink</a>'

    def _on_load(request: gr.Request) -> Tuple[str, str, gr.Dropdown, str, str, str, str]:
        params = dict(getattr(request, "query_params", {}) or {})
        state = registry.bootstrap(ctx=params.get("ctx"), project_space=params.get("project_space"), run=params.get("run"))
        session = registry.get_session(state.ctx)
        workspaces = session.list_workspaces()
        project_space_name = registry.project_space_name_for_session(session)
        return (
            state.ctx,
            state.project_space_root,
            gr.update(choices=workspaces, value=project_space_name or None),
            state.project_space_path,
            state.status,
            state.run_name,
            _monitor_link(state.ctx, project_space_name, state.run_name),
        )

    def _refresh_workspaces(root_path: str, ctx: str) -> Tuple[gr.Dropdown, str, str]:
        session = registry.get_session(ctx)
        ok, msg, choices = session.set_workspace_root(root_path)
        project_space_name = registry.project_space_name_for_session(session)
        return gr.update(choices=choices, value=project_space_name if ok and project_space_name else None), msg, session.current_workspace_path()

    def _open_workspace(root_path: str, name: str, ctx: str) -> Tuple[str, str]:
        session = registry.get_session(ctx)
        session.set_workspace_root(root_path)
        _, msg = session.open_workspace_by_name(name)
        return msg, session.current_workspace_path()

    def _create_workspace(root_path: str, name: str, ctx: str) -> Tuple[gr.Dropdown, str, str, str]:
        session = registry.get_session(ctx)
        ok, msg, choices = session.set_workspace_root(root_path)
        if not ok:
            return gr.update(choices=[], value=None), msg, session.current_workspace_path(), name
        created, create_msg = session.create_workspace(name)
        if created:
            choices = session.list_workspaces()
        project_space_name = registry.project_space_name_for_session(session)
        return gr.update(choices=choices, value=project_space_name if created else None), (create_msg if create_msg else msg), session.current_workspace_path(), ""

    def _sync_and_render(ctx: str, selected_run: str, search_text: str, live_llm_enabled: bool) -> Tuple[str, str, gr.Dropdown, str, str, str, str, str, str, str, Any, str, str, str]:
        session = registry.get_session(ctx)
        runs = session.list_runs()
        run_names = {name for _, name in runs}
        selected = (selected_run or "").strip()
        if selected and selected not in run_names:
            selected = ""
        if not selected and runs:
            selected = runs[0][1]
        if selected:
            session.select_run(selected)

        run_dir = session.get_selected_run_dir()
        live_state: Dict[str, Any] = {}
        reporter = session.reporter
        active_run_dir = reporter.get_run_dir() if reporter else None
        if reporter and active_run_dir and run_dir and active_run_dir == run_dir:
            events, _ = session.get_events()
            if events:
                for event in events:
                    session.event_lines.append(format_event_line(event))
                if len(session.event_lines) > MAX_EVENT_FEED:
                    session.event_lines = session.event_lines[-MAX_EVENT_FEED:]
            event_feed = "\n".join(session.event_lines)
            live_state = session.update_live_state(run_dir, events, live_llm_enabled=bool(live_llm_enabled))
        else:
            event_feed = session.read_ui_events_from_file(run_dir)
            live_state = session.snapshot_live_state(run_dir)

        run_info = session.run_status_text()
        run_select_status = f"Selected run: {selected}" if selected else "No run selected."
        project_space_name = registry.project_space_name_for_session(session)

        cards = session.list_run_cards()
        cards_html = render_run_cards_html(
            cards,
            selected_run=selected,
            search_text=search_text,
            run_link_builder=lambda run_name: registry.monitor_url(ctx=ctx, project_space=project_space_name, run=run_name),
        )
        summary_parts = [part for part in [_cards_markdown(cards, selected), render_live_tracker_markdown(live_state)] if part]
        summary_md = "\n\n---\n\n".join(summary_parts)

        final_report, report_source = session.read_final_report_with_source(run_dir)
        report_source_md = f"<small>Report Source: `{report_source}`</small>"

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
            if kind == "plan_review":
                prompt_title = "Plan Review"
                prompt_body = payload.get("plan_description", "") or ""
                todo = payload.get("todo", []) or []
                if isinstance(todo, list) and todo:
                    prompt_meta = "Work packages:\n" + "\n".join([f"{idx + 1}. {item}" for idx, item in enumerate(todo)])
            elif kind == "hitl":
                prompt_title = "HITL Feedback Required"
                prompt_body = payload.get("report_text", "") or ""
                report_path = payload.get("report_path", "") or ""
                prompt_meta = f"Report path: {report_path}" if report_path else ""
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
            run_select_status,
            gr.update(choices=runs, value=selected or None),
            cards_html,
            event_feed,
            summary_md,
            final_report,
            report_source_md,
            prompt_title,
            prompt_body,
            gr.update(visible=prompt_visible),
            prompt_id,
            _monitor_link(ctx, project_space_name, selected),
            selected,
        )

    def _refresh_details(ctx: str, selected_run: str) -> Tuple[str, Any, str, str, str, str, str]:
        session = registry.get_session(ctx)
        selected = (selected_run or "").strip()
        if selected:
            session.select_run(selected)
        run_dir = session.get_selected_run_dir()
        return (
            session.read_memory_index(),
            session.read_artifacts(),
            session.read_proposal(run_dir),
            session.read_task_state(run_dir),
            session.read_trace(run_dir, "event_trace.jsonl"),
            session.read_trace(run_dir, "tool_trace.jsonl"),
            session.read_trace(run_dir, "patch_trace.jsonl"),
        )

    def _submit_prompt(prompt_id: str, text: str, ctx: str) -> Tuple[str, str]:
        session = registry.get_session(ctx)
        status = session.submit_prompt(prompt_id, text)
        return status, ""

    def _submit_approve(prompt_id: str, ctx: str) -> Tuple[str, str]:
        session = registry.get_session(ctx)
        status = session.submit_prompt(prompt_id, "yes")
        return status, ""

    def _interrupt_run(ctx: str) -> str:
        session = registry.get_session(ctx)
        return session.request_interrupt_current_run()

    def _refresh_files_ui(ctx: str, filter_text: str, include_hidden: bool) -> Tuple[gr.FileExplorer, str, str]:
        session = registry.get_session(ctx)
        root = _workspace_dir(session)
        if root is None:
            return (
                gr.FileExplorer(label="Project files", root_dir=str(Path.cwd()), glob="**/*", file_count="single", interactive=True, height=420),
                "Project files root: *(not opened yet)*",
                "",
            )
        glob = _fileexplorer_filter_to_glob(filter_text)
        ignore_glob = _fileexplorer_ignore_glob(include_hidden=bool(include_hidden))
        status = (
            f"Project files root: `{root}`\n\n"
            f"glob: `{glob}`\n\n"
            f"ignore_glob: `{ignore_glob}`" if ignore_glob else f"Project files root: `{root}`\n\nglob: `{glob}`\n\nignore_glob: *(none)*"
        )
        explorer = gr.FileExplorer(
            label="Project files",
            root_dir=str(root),
            glob=glob,
            ignore_glob=ignore_glob,
            file_count="single",
            interactive=True,
            height=420,
        )
        return explorer, status, ""

    def _fileexplorer_select(ctx: str, selected_path: Any, filter_text: str, include_hidden: bool) -> Tuple[str, str]:
        session = registry.get_session(ctx)
        content = _read_workspace_file(session, selected_path)
        root = _workspace_dir(session)
        glob = _fileexplorer_filter_to_glob(filter_text)
        ignore_glob = _fileexplorer_ignore_glob(include_hidden=bool(include_hidden))
        status = (
            f"Project files root: `{root}`\n\nglob: `{glob}`\n\nignore_glob: `{ignore_glob}`" if ignore_glob else f"Project files root: `{root}`\n\nglob: `{glob}`\n\nignore_glob: *(none)*"
        )
        return content, status

    _ = theme
    with gr.Blocks() as page:
        gr.HTML(f"<style>{_MONITOR_CSS}</style>")
        ctx_state = gr.State("")
        selected_run_state = gr.State("")

        with gr.Column(elem_classes=["cm-monitor-shell"]):
            gr.Markdown("# CatMaster Monitor")
            with gr.Row():
                workspace_root_box = gr.Textbox(label="Project Space Root", value=default_workspace)
                refresh_workspaces_btn = gr.Button("Refresh")

            with gr.Row():
                workspace_list = gr.Dropdown(label="Project Spaces", choices=[])
                open_workspace_btn = gr.Button("Open")
                new_workspace_name = gr.Textbox(label="New Project Space")
                create_workspace_btn = gr.Button("Create", variant="primary")
                permalink_html = gr.HTML("")

            with gr.Row():
                current_workspace_box = gr.Textbox(label="Current Project Space", interactive=False)
                status_box = gr.Markdown("")

            with gr.Row():
                runs_dropdown = gr.Dropdown(label="Runs", choices=[])
                search_box = gr.Textbox(label="Search", placeholder="Filter by run/status/model/project_space")
                live_llm_toggle = gr.Checkbox(label="Live LLM summary", value=LIVE_SUMMARY_ENABLED_DEFAULT)
                refresh_monitor_btn = gr.Button("Refresh", variant="primary")
                interrupt_btn = gr.Button("Interrupt")

            run_info = gr.Markdown("")
            run_select_status = gr.Markdown("")

            with gr.Row(equal_height=True, elem_classes=["cm-top-align"]):
                with gr.Column(scale=4, elem_classes=["cm-panel"]):
                    gr.Markdown("## Run Inbox")
                    cards_html = gr.HTML("", elem_classes=["cm-scroll-html"])
                with gr.Column(scale=4, elem_classes=["cm-panel"]):
                    gr.Markdown("## Event Timeline")
                    event_feed = gr.Code(label="ui_events", lines=30, elem_classes=["cm-scroll-code"])
                with gr.Column(scale=4, elem_classes=["cm-panel"]):
                    gr.Markdown("## Run Detail")
                    run_summary_md = gr.Markdown(elem_classes=["cm-live-summary-box"])
                    report_source_md = gr.Markdown("<small>Report Source: `unavailable`</small>")
                    final_report_md = gr.Markdown(elem_classes=["cm-scroll-markdown"])

            with gr.Group(visible=False) as prompt_group:
                with gr.Column(elem_classes=["cm-panel"]):
                    gr.Markdown("## Input Required")
                    prompt_title_md = gr.Markdown()
                    prompt_body_md = gr.Markdown()
                    prompt_status = gr.Markdown("")
                    prompt_input = gr.Textbox(label="Your feedback", lines=3)
                    with gr.Row():
                        submit_btn = gr.Button("Submit", variant="primary")
                        approve_btn = gr.Button("Approve (yes)")
                    prompt_id_box = gr.Textbox(visible=False)

            with gr.Accordion("Advanced", open=False):
                refresh_detail_btn = gr.Button("Refresh Advanced")
                with gr.Row(equal_height=True, elem_classes=["cm-top-align"]):
                    with gr.Column(scale=1):
                        memory_md = gr.Markdown()
                    with gr.Column(scale=1):
                        artifacts_df = gr.Dataframe(interactive=False)
                proposal_md = gr.Markdown()
                task_state_text = gr.Code(language="json", lines=16)
                trace_event = gr.Code(label="event_trace.jsonl", language="json", lines=8)
                trace_tool = gr.Code(label="tool_trace.jsonl", language="json", lines=8)
                trace_patch = gr.Code(label="patch_trace.jsonl", language="json", lines=8)

                with gr.Row():
                    with gr.Column(scale=1):
                        files_filter = gr.Textbox(label="Filter (glob or substring)", value="**/*")
                        include_hidden = gr.Checkbox(label="Include hidden", value=False)
                        refresh_files_btn = gr.Button("Refresh File Explorer")
                        files_status = gr.Markdown("")
                        files_explorer = gr.FileExplorer(label="Project files", root_dir=default_workspace, glob="**/*", file_count="single", interactive=True, height=420)
                    with gr.Column(scale=2):
                        file_content = gr.Code(label="File content")

        page.load(
            _on_load,
            inputs=None,
            outputs=[
                ctx_state,
                workspace_root_box,
                workspace_list,
                current_workspace_box,
                status_box,
                selected_run_state,
                permalink_html,
            ],
            queue=False,
        ).then(
            _sync_and_render,
            inputs=[ctx_state, selected_run_state, search_box, live_llm_toggle],
            outputs=[
                run_info,
                run_select_status,
                runs_dropdown,
                cards_html,
                event_feed,
                run_summary_md,
                final_report_md,
                report_source_md,
                prompt_title_md,
                prompt_body_md,
                prompt_group,
                prompt_id_box,
                permalink_html,
                selected_run_state,
            ],
            queue=False,
        ).then(
            _refresh_details,
            inputs=[ctx_state, selected_run_state],
            outputs=[memory_md, artifacts_df, proposal_md, task_state_text, trace_event, trace_tool, trace_patch],
            queue=False,
        ).then(
            _refresh_files_ui,
            inputs=[ctx_state, files_filter, include_hidden],
            outputs=[files_explorer, files_status, file_content],
            queue=False,
        )

        refresh_workspaces_btn.click(
            _refresh_workspaces,
            inputs=[workspace_root_box, ctx_state],
            outputs=[workspace_list, status_box, current_workspace_box],
            queue=False,
        )

        open_workspace_btn.click(
            _open_workspace,
            inputs=[workspace_root_box, workspace_list, ctx_state],
            outputs=[status_box, current_workspace_box],
            queue=False,
        )

        create_workspace_btn.click(
            _create_workspace,
            inputs=[workspace_root_box, new_workspace_name, ctx_state],
            outputs=[workspace_list, status_box, current_workspace_box, new_workspace_name],
            queue=False,
        )

        refresh_monitor_btn.click(
            _sync_and_render,
            inputs=[ctx_state, selected_run_state, search_box, live_llm_toggle],
            outputs=[
                run_info,
                run_select_status,
                runs_dropdown,
                cards_html,
                event_feed,
                run_summary_md,
                final_report_md,
                report_source_md,
                prompt_title_md,
                prompt_body_md,
                prompt_group,
                prompt_id_box,
                permalink_html,
                selected_run_state,
            ],
            queue=False,
        )

        interrupt_btn.click(
            _interrupt_run,
            inputs=[ctx_state],
            outputs=[status_box],
            queue=False,
        )

        runs_dropdown.change(
            _sync_and_render,
            inputs=[ctx_state, runs_dropdown, search_box, live_llm_toggle],
            outputs=[
                run_info,
                run_select_status,
                runs_dropdown,
                cards_html,
                event_feed,
                run_summary_md,
                final_report_md,
                report_source_md,
                prompt_title_md,
                prompt_body_md,
                prompt_group,
                prompt_id_box,
                permalink_html,
                selected_run_state,
            ],
            queue=False,
        ).then(
            _refresh_details,
            inputs=[ctx_state, selected_run_state],
            outputs=[memory_md, artifacts_df, proposal_md, task_state_text, trace_event, trace_tool, trace_patch],
            queue=False,
        )

        search_box.change(
            _sync_and_render,
            inputs=[ctx_state, selected_run_state, search_box, live_llm_toggle],
            outputs=[
                run_info,
                run_select_status,
                runs_dropdown,
                cards_html,
                event_feed,
                run_summary_md,
                final_report_md,
                report_source_md,
                prompt_title_md,
                prompt_body_md,
                prompt_group,
                prompt_id_box,
                permalink_html,
                selected_run_state,
            ],
            queue=False,
        )

        refresh_detail_btn.click(
            _refresh_details,
            inputs=[ctx_state, selected_run_state],
            outputs=[memory_md, artifacts_df, proposal_md, task_state_text, trace_event, trace_tool, trace_patch],
            queue=False,
        )

        refresh_files_btn.click(
            _refresh_files_ui,
            inputs=[ctx_state, files_filter, include_hidden],
            outputs=[files_explorer, files_status, file_content],
            queue=False,
        )

        files_explorer.change(
            _fileexplorer_select,
            inputs=[ctx_state, files_explorer, files_filter, include_hidden],
            outputs=[file_content, files_status],
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
            _sync_and_render,
            inputs=[ctx_state, selected_run_state, search_box, live_llm_toggle],
            outputs=[
                run_info,
                run_select_status,
                runs_dropdown,
                cards_html,
                event_feed,
                run_summary_md,
                final_report_md,
                report_source_md,
                prompt_title_md,
                prompt_body_md,
                prompt_group,
                prompt_id_box,
                permalink_html,
                selected_run_state,
            ],
            queue=False,
            trigger_mode="always_last",
        )

    return page


__all__ = ["build_monitor_page"]
