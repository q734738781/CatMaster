from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

from .constants import EVENT_POLL_INTERVAL, MAX_EVENT_FEED
from .session import WebSession


_SESSION = WebSession()


def _truncate(value: Any, max_len: int = 140) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


def _summarize_event(event: Dict[str, Any]) -> str:
    name = event.get("name", "")
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    if name == "RUN_INIT_DONE":
        run_id = event.get("run_id") or payload.get("run_id", "")
        model = payload.get("model_name", "")
        return f"run_id={run_id} model={model}".strip()
    if name == "PLAN_CREATED":
        n_items = payload.get("n_items")
        return f"{n_items} tasks" if n_items is not None else "Plan created"
    if name == "PLAN_REVIEW_REVISED":
        n_items = payload.get("n_items")
        return f"Plan revised ({n_items} tasks)" if n_items is not None else "Plan revised"
    if name == "TASK_START":
        goal = payload.get("goal", "")
        return f"{event.get('task_id','')}: {_truncate(goal, 120)}".strip()
    if name == "TASK_END":
        outcome = payload.get("outcome", "")
        return f"{event.get('task_id','')}: outcome={outcome}".strip()
    if name == "TASK_DECISION":
        action = payload.get("action", "")
        method = payload.get("method", "")
        params = payload.get("params_compact", "")
        parts = [action]
        if method:
            parts.append(method)
        if params:
            parts.append(params)
        return " | ".join(parts)
    if name == "TOOL_CALL_START":
        tool = payload.get("tool", "")
        params = payload.get("params_compact", "")
        return f"{tool} {params}".strip()
    if name == "TOOL_CALL_END":
        tool = payload.get("tool", "")
        status = payload.get("status", "")
        return f"{tool} status={status}".strip()
    if name == "TASK_SUMMARY":
        outcome = payload.get("outcome", "")
        summary = payload.get("summary_snippet", "")
        return f"{outcome} - {_truncate(summary, 120)}".strip()
    if name == "FINAL_SUMMARY_DONE":
        return "Final report generated"
    if name == "RUN_END":
        status = payload.get("status", "")
        return f"status={status}".strip()
    return name


def _format_event_line(event: Dict[str, Any]) -> str:
    ts = event.get("ts")
    if ts:
        try:
            stamp = datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
        except Exception:
            stamp = "--:--:--"
    else:
        stamp = "--:--:--"
    summary = _summarize_event(event)
    return f"{stamp} {event.get('name','')} {summary}".rstrip()


def _refresh_runs() -> Tuple[List[Tuple[str, str]], str]:
    runs = _SESSION.list_runs()
    return runs, "Runs refreshed."

def _refresh_workspaces(root_path: str) -> Tuple[gr.Dropdown, str]:
    ok, msg, choices = _SESSION.set_workspace_root(root_path)
    if not ok:
        return gr.update(choices=[], value=None), msg
    return gr.update(choices=choices, value=None), msg


def _open_workspace(root_path: str, name: str) -> Tuple[gr.Dropdown, str, str]:
    _SESSION.set_workspace_root(root_path)
    ok, msg = _SESSION.open_workspace_by_name(name)
    runs = _SESSION.list_runs() if ok else []
    current = _SESSION.current_workspace_path()
    return gr.update(choices=runs, value=None), msg, current


def _create_workspace(root_path: str, name: str) -> Tuple[gr.Dropdown, gr.Dropdown, str, str, str]:
    ok, msg, choices = _SESSION.set_workspace_root(root_path)
    if not ok:
        return (
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            msg,
            "",
            name,
        )
    created, create_msg = _SESSION.create_workspace(name)
    runs = _SESSION.list_runs() if created else []
    current = _SESSION.current_workspace_path()
    status = create_msg if create_msg else msg
    value = name if created else None
    return (
        gr.update(choices=choices, value=value),
        gr.update(choices=runs, value=None),
        status,
        current,
        "",
    )


def _select_run(run_name: str) -> str:
    return _SESSION.select_run(run_name)


def _start_run(
    prompt: str,
    lane: str,
    resume: bool,
    plan_review: bool,
    log_llm: bool,
    full_auto_major: bool,
) -> str:
    return _SESSION.start_run(
        prompt=prompt,
        lane=lane,
        resume=resume,
        plan_review=plan_review,
        log_llm=log_llm,
        full_auto_major=full_auto_major,
    )


def _submit_prompt(prompt_id: str, text: str) -> Tuple[str, str]:
    status = _SESSION.submit_prompt(prompt_id, text)
    return status, ""


def _submit_approve(prompt_id: str) -> Tuple[str, str]:
    status = _SESSION.submit_prompt(prompt_id, "yes")
    return status, ""


def _poll_updates() -> Tuple[
    str,
    str,
    str,
    str,
    Any,
    str,
    str,
    str,
    str,
    str,
    str,
    str,
    bool,
    str,
]:
    run_dir = _SESSION.get_selected_run_dir()
    reporter = _SESSION.reporter
    active_run_dir = reporter.get_run_dir() if reporter else None

    if reporter and active_run_dir and run_dir and active_run_dir == run_dir:
        events, _ = _SESSION.get_events()
        if events:
            for event in events:
                line = _format_event_line(event)
                _SESSION.event_lines.append(line)
            if len(_SESSION.event_lines) > MAX_EVENT_FEED:
                _SESSION.event_lines = _SESSION.event_lines[-MAX_EVENT_FEED:]
        feed_text = "\n".join(_SESSION.event_lines)
    else:
        feed_text = _SESSION.read_ui_events_from_file(run_dir)

    run_status = _SESSION.run_status_text()
    whiteboard = _SESSION.read_whiteboard()
    proposal = _SESSION.read_proposal(run_dir)
    artifacts = _SESSION.read_artifacts()
    task_state = _SESSION.read_task_state(run_dir)
    trace_event = _SESSION.read_trace(run_dir, "event_trace.jsonl")
    trace_tool = _SESSION.read_trace(run_dir, "tool_trace.jsonl")
    trace_patch = _SESSION.read_trace(run_dir, "patch_trace.jsonl")
    final_report = _SESSION.read_final_report(run_dir)

    pending = _SESSION.get_prompt()
    prompt_visible = False
    prompt_title = ""
    prompt_body = ""
    prompt_meta = ""
    prompt_id_value = ""
    if pending:
        prompt_visible = True
        prompt_id_value = pending.get("prompt_id", "")
        kind = pending.get("kind", "")
        payload = pending.get("payload") if isinstance(pending.get("payload"), dict) else {}
        if kind == "plan_review":
            prompt_title = "Plan Review"
            prompt_body = payload.get("plan_description", "") or ""
            todo = payload.get("todo", []) or []
            if isinstance(todo, list) and todo:
                prompt_meta = "Work packages:\n" + "\n".join(
                    [f"{idx+1}. {item}" for idx, item in enumerate(todo)]
                )
        elif kind == "hitl":
            prompt_title = "HITL Feedback Required"
            prompt_body = payload.get("report_text", "") or ""
            report_path = payload.get("report_path", "") or ""
            prompt_meta = f"Report path: {report_path}" if report_path else ""
        else:
            prompt_title = "Input Required"
    if prompt_visible:
        run_status = f"[INPUT REQUIRED] {run_status}"
    return (
        run_status,
        feed_text,
        whiteboard,
        proposal,
        artifacts,
        task_state,
        trace_event,
        trace_tool,
        trace_patch,
        final_report,
        prompt_title,
        prompt_body,
        prompt_meta,
        gr.update(visible=prompt_visible),
        prompt_id_value,
    )


def launch(*, host: str = "127.0.0.1", port: int = 7860, workspace: Optional[str] = None) -> None:
    if workspace is None:
        workspace = str(Path.cwd() / "workspace")
    with gr.Blocks() as demo:
        gr.Markdown("# CatMaster Web Workbench")

        with gr.Row():
            workspace_root_box = gr.Textbox(
                label="Workspace Root",
                placeholder="Path to workspace root",
                value=workspace,
            )
            refresh_workspaces_btn = gr.Button("Refresh Workspaces")
            status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Row():
            workspace_list = gr.Dropdown(label="Workspaces", choices=[])
            open_workspace_btn = gr.Button("Open Workspace")
            new_workspace_name = gr.Textbox(label="New Workspace Name")
            create_workspace_btn = gr.Button("Create Workspace")
            current_workspace_box = gr.Textbox(label="Current Workspace", interactive=False)

        with gr.Row():
            prompt_box = gr.Textbox(label="User Request", lines=4)
            lane_box = gr.Dropdown(label="Lane", choices=["fast", "standard"], value="standard")
            resume_box = gr.Checkbox(label="Resume", value=False)
            plan_review_box = gr.Checkbox(label="Plan Review", value=True)
            log_llm_box = gr.Checkbox(label="Log LLM", value=False)
            full_auto_major_box = gr.Checkbox(label="Full Auto Major", value=False)
            start_btn = gr.Button("Start Run")

        with gr.Row():
            run_info = gr.Textbox(label="Run Info", interactive=False)

        with gr.Group(visible=False) as prompt_group:
            gr.Markdown("## INPUT REQUIRED (Plan Review / HITL)")
            prompt_title_md = gr.Markdown()
            prompt_body_md = gr.Markdown()
            prompt_meta_md = gr.Markdown()
            prompt_input = gr.Textbox(label="Your Feedback", lines=3)
            with gr.Row():
                submit_btn = gr.Button("Submit")
                approve_btn = gr.Button("Approve (yes)")
            prompt_status = gr.Textbox(label="Prompt Status", interactive=False)
            prompt_id_box = gr.Textbox(visible=False)

        with gr.Row():
            with gr.Column(scale=1):
                runs_dropdown = gr.Dropdown(label="Runs", choices=[])
                run_select_status = gr.Textbox(label="Run Selection", interactive=False)

            with gr.Column(scale=2):
                event_feed = gr.Textbox(label="Event Feed", lines=24, interactive=False)

            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Whiteboard"):
                        whiteboard_md = gr.Markdown()
                    with gr.TabItem("Plan/Proposal"):
                        proposal_md = gr.Markdown()
                    with gr.TabItem("Artifacts"):
                        artifacts_df = gr.Dataframe(interactive=False)
                    with gr.TabItem("Task State"):
                        task_state_text = gr.Textbox(lines=20, interactive=False)
                    with gr.TabItem("Traces"):
                        trace_event = gr.Textbox(label="event_trace.jsonl", lines=8, interactive=False)
                        trace_tool = gr.Textbox(label="tool_trace.jsonl", lines=8, interactive=False)
                        trace_patch = gr.Textbox(label="patch_trace.jsonl", lines=8, interactive=False)
                    with gr.TabItem("Final Report"):
                        final_report_md = gr.Markdown()

        refresh_workspaces_btn.click(
            _refresh_workspaces,
            inputs=[workspace_root_box],
            outputs=[workspace_list, status_box],
        )
        open_workspace_btn.click(
            _open_workspace,
            inputs=[workspace_root_box, workspace_list],
            outputs=[runs_dropdown, status_box, current_workspace_box],
        )
        create_workspace_btn.click(
            _create_workspace,
            inputs=[workspace_root_box, new_workspace_name],
            outputs=[workspace_list, runs_dropdown, status_box, current_workspace_box, new_workspace_name],
        )
        # Runs are refreshed after opening/creating a workspace.
        runs_dropdown.change(
            _select_run,
            inputs=[runs_dropdown],
            outputs=[run_select_status],
        )
        start_btn.click(
            _start_run,
            inputs=[prompt_box, lane_box, resume_box, plan_review_box, log_llm_box, full_auto_major_box],
            outputs=[status_box],
        )

        submit_btn.click(
            _submit_prompt,
            inputs=[prompt_id_box, prompt_input],
            outputs=[prompt_status, prompt_input],
        )
        approve_btn.click(
            _submit_approve,
            inputs=[prompt_id_box],
            outputs=[prompt_status, prompt_input],
        )

        timer = gr.Timer(EVENT_POLL_INTERVAL)
        timer.tick(
            _poll_updates,
            inputs=[],
            outputs=[
                run_info,
                event_feed,
                whiteboard_md,
                proposal_md,
                artifacts_df,
                task_state_text,
                trace_event,
                trace_tool,
                trace_patch,
                final_report_md,
                prompt_title_md,
                prompt_body_md,
                prompt_meta_md,
                prompt_group,
                prompt_id_box,
            ],
        )

        if workspace:
            ok, msg, choices = _SESSION.set_workspace_root(workspace)
            status_box.value = msg
            workspace_root_box.value = workspace
            workspace_list.choices = choices

    demo.queue().launch(server_name=host, server_port=port)
