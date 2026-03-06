from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from catmaster.tools.base import workspace_root

from .components import (
    SHARED_CSS,
    build_hitl_group,
    build_workspace_controls,
    nav_header_html,
    unpack_prompt,
)
from .constants import EVENT_POLL_INTERVAL, LIVE_SUMMARY_ENABLED_DEFAULT, MAX_EVENT_FEED
from .session_registry import SessionRegistry
from .view_utils import (
    format_event_html,
    render_cost_card_markdown,
    render_live_tracker_markdown,
    render_run_cards_html,
)


# ---------------------------------------------------------------------------
# CSS additions specific to the monitor (auto-scroll for event feed)
# ---------------------------------------------------------------------------

_MONITOR_EXTRA_CSS = """\
.cm-event-feed{max-height:560px;overflow-y:auto;overflow-x:hidden;padding-right:6px;
font-family:var(--font-mono);scroll-behavior:smooth;}
"""

_AUTOSCROLL_JS = """\
<script>
(function(){
  const obs = new MutationObserver(()=>{
    document.querySelectorAll('.cm-event-feed').forEach(el=>{
      el.scrollTop = el.scrollHeight;
    });
  });
  const target = document.querySelector('.cm-event-feed');
  if(target) obs.observe(target, {childList:true, subtree:true, characterData:true});
})();
</script>
"""


def build_monitor_page(
    *,
    registry: SessionRegistry,
    default_workspace: str,
    theme: Optional[Any] = None,
) -> gr.Blocks:

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

    def _monitor_link(ctx: str, ps_name: str, run_name: str) -> str:
        url = registry.monitor_url(ctx=ctx, project_space=ps_name, run=run_name)
        return f'<a href="{url}" target="_blank">Permalink</a>'

    # ------------------------------------------------------------------
    # Sync output contract
    # ------------------------------------------------------------------

    _SYNC_KEYS = [
        "run_info",
        "run_select_status",
        "runs_dropdown",
        "cards_html",
        "event_feed",
        "cost_md",
        "summary_md",
        "final_report",
        "report_source_md",
        "prompt_title",
        "prompt_body",
        "prompt_group_visible",
        "prompt_id",
        "permalink",
        "selected_run",
        "nav_html",
    ]

    def _sync_and_render(
        ctx: str, selected_run: str, search_text: str, live_llm_enabled: bool,
    ) -> tuple:
        try:
            return _sync_inner(ctx, selected_run, search_text, live_llm_enabled)
        except Exception as exc:
            gr.Warning(f"Sync error: {exc}")
            empty: Dict[str, Any] = {
                "run_info": "",
                "run_select_status": "",
                "runs_dropdown": gr.update(),
                "cards_html": "",
                "event_feed": "",
                "cost_md": "",
                "summary_md": "",
                "final_report": "",
                "report_source_md": "",
                "prompt_title": "",
                "prompt_body": "",
                "prompt_group_visible": gr.update(),
                "prompt_id": "",
                "permalink": "",
                "selected_run": selected_run,
                "nav_html": "",
            }
            return tuple(empty[k] for k in _SYNC_KEYS)

    def _sync_inner(
        ctx: str, selected_run: str, search_text: str, live_llm_enabled: bool,
    ) -> tuple:
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
                    session.event_lines.append(format_event_html(event))
                if len(session.event_lines) > MAX_EVENT_FEED:
                    session.event_lines = session.event_lines[-MAX_EVENT_FEED:]
            event_feed = "\n".join(session.event_lines)
            live_state = session.update_live_state(
                run_dir, events, live_llm_enabled=bool(live_llm_enabled),
            )
        else:
            raw_events = session.read_ui_events_objects(run_dir)
            event_feed = "\n".join(format_event_html(e) for e in raw_events[-MAX_EVENT_FEED:])
            live_state = session.snapshot_live_state(run_dir)

        run_info = session.run_status_text()
        run_select_status = f"Selected run: {selected}" if selected else "No run selected."
        ps_name = registry.project_space_name_for_session(session)
        usage_summary = session.read_usage_summary(run_dir)

        cards = session.list_run_cards()
        cards_html = render_run_cards_html(
            cards,
            selected_run=selected,
            search_text=search_text,
            run_link_builder=lambda rn: registry.monitor_url(
                ctx=ctx, project_space=ps_name, run=rn,
            ),
        )
        summary_parts = [
            part
            for part in [
                _cards_markdown(cards, selected),
                render_live_tracker_markdown(live_state),
            ]
            if part
        ]
        summary_md = "\n\n---\n\n".join(summary_parts)

        final_report, report_source = session.read_final_report_with_source(run_dir)
        report_source_md = f"<small>Report Source: <code>{report_source}</code></small>"

        prompt = unpack_prompt(session.get_prompt())

        result: Dict[str, Any] = {
            "run_info": run_info,
            "run_select_status": run_select_status,
            "runs_dropdown": gr.update(choices=runs, value=selected or None),
            "cards_html": cards_html,
            "event_feed": f'<div class="cm-event-feed">{event_feed}</div>',
            "cost_md": render_cost_card_markdown(usage_summary),
            "summary_md": summary_md,
            "final_report": final_report,
            "report_source_md": report_source_md,
            "prompt_title": prompt.title,
            "prompt_body": prompt.body,
            "prompt_group_visible": gr.update(visible=prompt.visible),
            "prompt_id": prompt.prompt_id,
            "permalink": _monitor_link(ctx, ps_name, selected),
            "selected_run": selected,
            "nav_html": nav_header_html("monitor", ps_name),
        }
        return tuple(result[k] for k in _SYNC_KEYS)

    # ------------------------------------------------------------------
    # Secondary callbacks
    # ------------------------------------------------------------------

    def _on_load(request: gr.Request) -> Tuple[str, str, Any, str, str, str, str]:
        params = dict(getattr(request, "query_params", {}) or {})
        state = registry.bootstrap(
            ctx=params.get("ctx"),
            project_space=params.get("project_space"),
            run=params.get("run"),
        )
        session = registry.get_session(state.ctx)
        workspaces = session.list_workspaces()
        ps_name = registry.project_space_name_for_session(session)
        return (
            state.ctx,
            state.project_space_root,
            gr.update(choices=workspaces, value=ps_name or None),
            state.project_space_path,
            state.status,
            state.run_name,
            _monitor_link(state.ctx, ps_name, state.run_name),
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

    def _interrupt_run(ctx: str) -> str:
        session = registry.get_session(ctx)
        return session.request_interrupt_current_run()

    def _refresh_files_ui(
        ctx: str, filter_text: str, include_hidden: bool,
    ) -> Tuple[Any, str, str]:
        session = registry.get_session(ctx)
        root = _workspace_dir(session)
        if root is None:
            return (
                gr.FileExplorer(
                    label="Project files",
                    root_dir=str(Path.cwd()),
                    glob="**/*",
                    file_count="single",
                    interactive=True,
                    height=420,
                ),
                "Project files root: *(not opened yet)*",
                "",
            )
        glob = _fileexplorer_filter_to_glob(filter_text)
        ignore_glob = _fileexplorer_ignore_glob(include_hidden=bool(include_hidden))
        status = f"Project files root: `{root}`\n\nglob: `{glob}`"
        if ignore_glob:
            status += f"\n\nignore_glob: `{ignore_glob}`"
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

    def _fileexplorer_select(
        ctx: str, selected_path: Any, filter_text: str, include_hidden: bool,
    ) -> Tuple[str, str]:
        session = registry.get_session(ctx)
        content = _read_workspace_file(session, selected_path)
        root = _workspace_dir(session)
        glob = _fileexplorer_filter_to_glob(filter_text)
        ignore_glob = _fileexplorer_ignore_glob(include_hidden=bool(include_hidden))
        status = f"Project files root: `{root}`\n\nglob: `{glob}`"
        if ignore_glob:
            status += f"\n\nignore_glob: `{ignore_glob}`"
        return content, status

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    with gr.Blocks(theme=theme, title="CatMaster Monitor") as page:
        gr.HTML(f"<style>{SHARED_CSS}\n{_MONITOR_EXTRA_CSS}</style>")
        gr.HTML(_AUTOSCROLL_JS)
        ctx_state = gr.State("")
        selected_run_state = gr.State("")

        # -- Sidebar workspace controls --
        ws = build_workspace_controls(
            registry=registry,
            default_workspace=default_workspace,
            ctx_state=ctx_state,
        )

        nav_html = gr.HTML(nav_header_html("monitor"))

        with gr.Column(elem_classes=["cm-shell"]):

            # -- Run selector bar --
            with gr.Row():
                runs_dropdown = gr.Dropdown(label="Runs", choices=[], scale=3)
                search_box = gr.Textbox(
                    label="Search",
                    placeholder="Filter by run/status/model",
                    scale=2,
                )
                live_llm_toggle = gr.Checkbox(
                    label="Live LLM summary",
                    value=LIVE_SUMMARY_ENABLED_DEFAULT,
                    scale=1,
                )
                refresh_monitor_btn = gr.Button("Refresh", variant="primary", scale=1)
                interrupt_btn = gr.Button("Interrupt", scale=1)

            with gr.Row():
                run_info = gr.Markdown("")
                run_select_status = gr.Markdown("")
                permalink_html = gr.HTML("")

            # -- HITL prompt (always visible above tabs) --
            hitl = build_hitl_group(registry=registry, ctx_state=ctx_state)

            # -- Tabbed main content --
            with gr.Tabs():

                with gr.Tab("Runs"):
                    cards_html = gr.HTML("", elem_classes=["cm-scroll-html"])

                with gr.Tab("Live View"):
                    with gr.Row(equal_height=True, elem_classes=["cm-top-align"]):
                        with gr.Column(scale=1):
                            gr.Markdown("#### Event Timeline")
                            event_feed = gr.HTML("", elem_classes=["cm-event-feed"])
                        with gr.Column(scale=1):
                            cost_md = gr.Markdown(elem_classes=["cm-scroll-md"])
                            gr.Markdown("#### Live Tracker")
                            run_summary_md = gr.Markdown(
                                elem_classes=["cm-scroll-md"],
                            )

                with gr.Tab("Detail"):
                    report_source_md = gr.Markdown(
                        "<small>Report Source: <code>unavailable</code></small>",
                    )
                    final_report_md = gr.Markdown(elem_classes=["cm-scroll-md"])

                with gr.Tab("Advanced"):
                    refresh_detail_btn = gr.Button("Refresh Advanced")
                    with gr.Row(equal_height=True, elem_classes=["cm-top-align"]):
                        with gr.Column(scale=1):
                            memory_md = gr.Markdown()
                        with gr.Column(scale=1):
                            artifacts_df = gr.Dataframe(interactive=False)
                    proposal_md = gr.Markdown()
                    task_state_text = gr.Code(language="json", lines=16)
                    trace_event = gr.Code(
                        label="event_trace.jsonl", language="json", lines=8,
                    )
                    trace_tool = gr.Code(
                        label="tool_trace.jsonl", language="json", lines=8,
                    )
                    trace_patch = gr.Code(
                        label="patch_trace.jsonl", language="json", lines=8,
                    )

                with gr.Tab("Files"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            files_filter = gr.Textbox(
                                label="Filter (glob or substring)", value="**/*",
                            )
                            include_hidden = gr.Checkbox(
                                label="Include hidden", value=False,
                            )
                            refresh_files_btn = gr.Button("Refresh File Explorer")
                            files_status = gr.Markdown("")
                            files_explorer = gr.FileExplorer(
                                label="Project files",
                                root_dir=default_workspace,
                                glob="**/*",
                                file_count="single",
                                interactive=True,
                                height=420,
                            )
                        with gr.Column(scale=2):
                            file_content = gr.Code(label="File content")

        # ------------------------------------------------------------------
        # Sync outputs
        # ------------------------------------------------------------------

        _sync_outputs = [
            run_info,
            run_select_status,
            runs_dropdown,
            cards_html,
            event_feed,
            cost_md,
            run_summary_md,
            final_report_md,
            report_source_md,
            hitl.title_md,
            hitl.body_md,
            hitl.group,
            hitl.prompt_id_box,
            permalink_html,
            selected_run_state,
            nav_html,
        ]

        _sync_inputs = [ctx_state, selected_run_state, search_box, live_llm_toggle]

        # ------------------------------------------------------------------
        # Event wiring
        # ------------------------------------------------------------------

        page.load(
            _on_load,
            inputs=None,
            outputs=[
                ctx_state,
                ws.root_box,
                ws.workspace_list,
                ws.current_box,
                ws.status_md,
                selected_run_state,
                permalink_html,
            ],
            queue=False,
        )

        refresh_monitor_btn.click(
            _sync_and_render,
            inputs=_sync_inputs,
            outputs=_sync_outputs,
            queue=False,
        )

        interrupt_btn.click(
            _interrupt_run,
            inputs=[ctx_state],
            outputs=[ws.status_md],
            queue=False,
        )

        runs_dropdown.change(
            _sync_and_render,
            inputs=[ctx_state, runs_dropdown, search_box, live_llm_toggle],
            outputs=_sync_outputs,
            queue=False,
        ).then(
            _refresh_details,
            inputs=[ctx_state, selected_run_state],
            outputs=[
                memory_md, artifacts_df, proposal_md, task_state_text,
                trace_event, trace_tool, trace_patch,
            ],
            queue=False,
        )

        search_box.change(
            _sync_and_render,
            inputs=_sync_inputs,
            outputs=_sync_outputs,
            queue=False,
        )

        refresh_detail_btn.click(
            _refresh_details,
            inputs=[ctx_state, selected_run_state],
            outputs=[
                memory_md, artifacts_df, proposal_md, task_state_text,
                trace_event, trace_tool, trace_patch,
            ],
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

        timer = gr.Timer(EVENT_POLL_INTERVAL)
        timer.tick(
            _sync_and_render,
            inputs=_sync_inputs,
            outputs=_sync_outputs,
            queue=False,
            trigger_mode="always_last",
        )

    return page


__all__ = ["build_monitor_page"]
