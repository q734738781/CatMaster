from __future__ import annotations

from pathlib import Path
from typing import Any

from catmaster.webui.thread_models import ThreadRole, ThreadStatus, research_automation_paused

from .common import humanize_agent_name, redact_internal_text
from .messages import project_current_active_parts, project_messages
from .models import PublicResearchActivity

_ACTIVE_LAUNCH_STATUSES = {"claimed", "submitting", "running", "unknown"}
_ACTION_REQUIRED_STATES = {
    "waiting_review",
    "operationally_incomplete",
    "blocked",
}


def _thread_status(value: Any) -> ThreadStatus:
    try:
        return ThreadStatus(value)
    except ValueError:
        return ThreadStatus.IDLE


def _activity_for_thread_status(value: Any) -> str:
    status = _thread_status(value)
    if status is ThreadStatus.IDLE:
        return "waiting_continue"
    if status is ThreadStatus.INTERRUPTED:
        return "waiting_review"
    if status in {ThreadStatus.ERROR, ThreadStatus.STOPPED}:
        return "operationally_incomplete"
    return "running"


def _latest_progress_and_active_parts(
    *,
    thread_store: Any,
    thread_id: str,
    workspace: Path,
) -> tuple[str, list[Any]]:
    if not thread_id:
        return "", []
    try:
        rows = [
            message.model_dump(mode="json")
            for message in thread_store.list_current_turn_messages(thread_id)
        ]
    except (KeyError, ValueError):
        return "", []
    active_parts = project_current_active_parts(rows, workspace=workspace)
    latest_progress = ""
    for message in reversed(project_messages(rows, workspace=workspace)):
        for part in reversed(message.parts):
            if part.type == "progress" and not part.items:
                latest_progress = str(part.summary or part.text or "").strip()
                if latest_progress:
                    return latest_progress, active_parts
    return latest_progress, active_parts


def project_research_session_activity(
    root_thread: Any,
    *,
    threads: list[Any],
    graph_service: Any,
    workspace: Path,
) -> PublicResearchActivity:
    """Project one Research Session from durable graph and child-thread state."""

    root_id = str(root_thread.thread_id)
    children = [
        thread
        for thread in threads
        if str(getattr(thread, "parent_thread_id", "") or "") == root_id
    ]
    descendant_ids = {root_id}
    while True:
        found = {str(thread.thread_id) for thread in threads
                 if str(getattr(thread, "parent_thread_id", "") or "") in descendant_ids}
        if found.issubset(descendant_ids):
            break
        descendant_ids.update(found)
    descendants = [thread for thread in threads
                   if str(thread.thread_id) in descendant_ids and str(thread.thread_id) != root_id]
    executions = [
        thread
        for thread in descendants
        if ThreadRole(thread.thread_role) is ThreadRole.RESEARCH_EXECUTION
        or (getattr(thread, "meta", None) or {}).get("background_task")
    ]
    planning_threads = [
        thread
        for thread in children
        if ThreadRole(thread.thread_role) is ThreadRole.RESEARCH_PLANNING
    ]
    graph_id = str(getattr(root_thread, "active_research_graph_id", "") or "")
    updated_at = max(
        [float(getattr(root_thread, "updated_at", 0.0) or 0.0)]
        + [float(getattr(thread, "updated_at", 0.0) or 0.0) for thread in descendants]
    )
    if not graph_id:
        return PublicResearchActivity(
            state="idle",
            execution_count=len(executions),
            decision_round_count=len(planning_threads),
            updated_at=updated_at,
        )

    try:
        graph = graph_service.store.get_graph(graph_id)
        snapshot = graph_service.store.get_snapshot(graph_id)
    except KeyError:
        return PublicResearchActivity(
            state="operationally_incomplete",
            current_title="Attached Research Graph is unavailable",
            action_required=True,
            execution_count=len(executions),
            decision_round_count=len(planning_threads),
            updated_at=updated_at,
        )

    updated_at = max(updated_at, float(graph.get("updated_at") or 0.0))
    orchestration_root_id = str(graph.get("orchestration_thread_id") or "")
    owns_orchestration = not orchestration_root_id or orchestration_root_id == root_id
    nodes_by_id = {
        str(node["node_id"]): node for node in list(snapshot.get("nodes") or [])
    }
    latest_result = graph_service.latest_session_result(
        graph_id,
        session_root_thread_id=root_id,
    )
    updated_at = max(
        updated_at,
        float(latest_result.get("updated_at") or 0.0),
    )
    launches = sorted(
        list(snapshot.get("launches") or []),
        key=lambda row: float(row.get("updated_at") or row.get("created_at") or 0.0),
        reverse=True,
    )
    execution_thread_ids = {thread.thread_id for thread in executions}
    session_launches = [
        launch
        for launch in launches
        if str(launch.get("session_root_thread_id") or "") == root_id
        or (
            not str(launch.get("session_root_thread_id") or "")
            and str(launch.get("thread_id") or "") in execution_thread_ids
        )
    ]
    active_launch = next(
        (
            launch
            for launch in session_launches
            if str(launch.get("status") or "") in _ACTIVE_LAUNCH_STATUSES
        ),
        None,
    )
    state = "idle"
    current_title = ""
    active_child_thread_id = ""
    active_child_role = ""
    decision_summary = ""

    planning = graph_service.store.latest_planning_preview(
        graph_id,
        current_revision_only=False,
    )
    if planning is not None:
        planning_root_id = str(planning.get("session_root_thread_id") or "")
        planning_thread_id = str(planning.get("thread_id") or "")
        owned_planning_ids = {thread.thread_id for thread in planning_threads}
        if (
            (planning_root_id and planning_root_id != root_id)
            or (
                not planning_root_id
                and planning_thread_id
                and planning_thread_id not in owned_planning_ids
            )
        ):
            planning = None
    planning_preview = dict(planning.get("preview") or {}) if planning else {}
    selection = dict(planning_preview.get("selection") or {})
    if selection:
        decision_summary = str(
            selection.get("reason")
            or selection.get("unresolved_tradeoff")
            or ""
        ).strip()
    elif planning is not None and str(planning.get("status") or "") == "no_change":
        decision_summary = str(planning_preview.get("no_change_reason") or "").strip()

    if active_launch is not None:
        active_child_thread_id = str(active_launch.get("thread_id") or "")
        active_child_role = ThreadRole.RESEARCH_EXECUTION.value
        node = nodes_by_id.get(str(active_launch.get("experiment_node_id") or ""))
        current_title = str((node or {}).get("title") or "Running Experiment")
        if not active_child_thread_id:
            state = "running"
        else:
            try:
                active_child = graph_service.thread_store.get_thread(
                    active_child_thread_id
                )
            except (KeyError, ValueError):
                state = "operationally_incomplete"
            else:
                state = _activity_for_thread_status(active_child.status)
        updated_at = max(
            updated_at,
            float(active_launch.get("updated_at") or 0.0),
        )
    else:
        active_comparison = dict(selection.get("active_comparison") or {})
        comparison_thread_id = str(active_comparison.get("thread_id") or "")
        if comparison_thread_id:
            active_child_thread_id = comparison_thread_id
            active_child_role = ThreadRole.RESEARCH_COMPARISON.value
            current_title = "Comparing ready Experiments"
            try:
                comparison_thread = graph_service.thread_store.get_thread(
                    comparison_thread_id
                )
            except KeyError:
                state = "operationally_incomplete"
            else:
                child_activity = _activity_for_thread_status(
                    comparison_thread.status
                )
                state = (
                    child_activity
                    if child_activity in _ACTION_REQUIRED_STATES
                    else "comparing"
                )
        else:
            current_planning_thread_id = str(
                (planning or {}).get("thread_id") or ""
            )
            live_planning = next(
                (
                    thread
                    for thread in sorted(
                        planning_threads,
                        key=lambda row: float(row.updated_at or 0.0),
                        reverse=True,
                    )
                    if (
                        not current_planning_thread_id
                        or thread.thread_id == current_planning_thread_id
                    )
                    if _thread_status(thread.status)
                    in {
                        ThreadStatus.RUNNING,
                        ThreadStatus.STOPPING,
                        ThreadStatus.INTERRUPTED,
                        ThreadStatus.ERROR,
                    }
                ),
                None,
            )
            if live_planning is not None:
                active_child_thread_id = live_planning.thread_id
                active_child_role = ThreadRole.RESEARCH_PLANNING.value
                current_title = f"Plan next step for {graph['title']}"
                child_activity = _activity_for_thread_status(live_planning.status)
                state = (
                    child_activity
                    if child_activity in _ACTION_REQUIRED_STATES
                    else "planning"
                )
            elif bool(graph.get("completed")):
                state = "completed"
                current_title = str(graph.get("title") or "Research completed")
            elif planning is not None and str(planning.get("status") or "") == "no_change":
                state = "waiting"
                current_title = "No further bounded branch was justified; research is not complete"
            elif not owns_orchestration:
                state = "idle"
                current_title = "No background activity in this Research Session"
            elif session_launches and str(session_launches[0].get("status") or "") == "blocked":
                state = "blocked"
                node = nodes_by_id.get(
                    str(session_launches[0].get("experiment_node_id") or "")
                )
                current_title = str((node or {}).get("title") or "Experiment blocked")
            elif research_automation_paused(root_thread, graph):
                state = "paused"
                current_title = str(graph.get("title") or "Research paused")
            elif str(selection.get("status") or "") in {"pending", "comparing"}:
                state = "comparing"
                current_title = "Comparing ready Experiments"
            elif str(selection.get("status") or "") == "wait":
                state = "waiting"
                current_title = "Waiting — research is not complete"
            else:
                # A native Research root owns this stage. An open auto graph
                # does not imply a separate planner exists or is running.
                state = _activity_for_thread_status(root_thread.status)
                current_title = str(graph.get("title") or "Research")
                decisions = [row for row in snapshot.get("decisions", [])
                             if row.get("thread_id") == root_id]
                if decisions:
                    decision = max(decisions, key=lambda row: row["updated_at"])
                    body, review = decision["body"], decision["review"]
                    decision_summary = str(review.get("assessment") or body["reason"])
                    current_run = str(getattr(root_thread, "active_run_id", "") or "")
                    if state not in _ACTION_REQUIRED_STATES and (
                        not current_run or decision.get("run_id") == current_run
                    ):
                        disposition = body["disposition"]
                        if disposition == "stalled":
                            if not review:
                                state, current_title = "reviewing", "Independent scientific reconsideration"
                            elif review.get("remedy_experiment_id") and not (
                                body["validation_result_ids"] or body["exception"]
                            ):
                                state, current_title = "validating", "Bounded validation within the authorized stage"
                        elif disposition == "parked":
                            state, current_title = "parked", body["problem"]
                        elif disposition in {"waiting", "boundary"}:
                            state, current_title = "waiting", body["reason"]

    background = [thread for thread in descendants
                  if (getattr(thread, "meta", None) or {}).get("background_task")
                  and _thread_status(thread.status) in {ThreadStatus.RUNNING, ThreadStatus.STOPPING}]
    if owns_orchestration and background and state not in {
        "completed", "waiting_review", "operationally_incomplete", "blocked",
    }:
        active_child = max(background, key=lambda thread: float(thread.updated_at or 0))
        active_child_thread_id = active_child.thread_id
        active_child_role = ThreadRole(active_child.thread_role).value
        state = "running"
        current_title = ", ".join(dict.fromkeys(
            humanize_agent_name(thread.meta.get("agent_name") or thread.entrypoint)
            for thread in background))
    progress_thread_id = active_child_thread_id or (root_id if owns_orchestration else "")
    if not progress_thread_id and active_launch is None and executions:
        progress_thread_id = max(
            executions,
            key=lambda row: float(row.updated_at or 0.0),
        ).thread_id
    latest_progress, active_parts = _latest_progress_and_active_parts(
        thread_store=graph_service.thread_store,
        thread_id=progress_thread_id,
        workspace=workspace,
    )
    if owns_orchestration and state == "waiting_continue" and (active_parts or background):
        # The graph summary may reflect child thread lifecycle projections, but
        # background cards come only from the execution-backed message snapshot.
        # Historical message parts must never revive stopped child activity.
        state = "running"
        titles = [part.title for part in active_parts if part.title]
        titles.extend(humanize_agent_name(thread.meta.get("agent_name") or thread.entrypoint)
                      for thread in background)
        current_title = ", ".join(dict.fromkeys(titles)) or "Research in progress"
    automation_paused = research_automation_paused(root_thread, graph)
    if (owns_orchestration and automation_paused and not active_child_thread_id
            and state not in {"running", "completed"}):
        state = "paused"
        current_title = str(graph.get("title") or "Research paused")
    return PublicResearchActivity(
        state=state,
        current_title=current_title,
        latest_progress=latest_progress,
        decision_summary=decision_summary,
        latest_result_title=redact_internal_text(
            latest_result.get("title"),
            workspace=workspace,
            limit=300,
        ),
        latest_result_summary=redact_internal_text(
            latest_result.get("summary"),
            workspace=workspace,
            limit=900,
        ),
        latest_report_artifact_id=str(
            latest_result.get("report_artifact_id") or ""
        ),
        latest_report_path=str(latest_result.get("report_path") or ""),
        latest_report_renderer=str(latest_result.get("report_renderer") or ""),
        action_required=state in _ACTION_REQUIRED_STATES,
        automation_paused=automation_paused,
        active_child_thread_id=active_child_thread_id,
        active_child_role=active_child_role,
        execution_count=len(executions),
        decision_round_count=len(planning_threads),
        updated_at=updated_at,
        active_parts=active_parts,
    )


__all__ = ["project_research_session_activity"]
