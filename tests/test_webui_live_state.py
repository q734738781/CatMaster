from __future__ import annotations

from catmaster.webui.live_state import apply_events, new_live_state, should_refresh_live_summary


def _event(name: str, *, ts: float, task_id: str = "", step_id: int | None = None, payload: dict | None = None) -> dict:
    return {
        "name": name,
        "ts": ts,
        "task_id": task_id,
        "step_id": step_id,
        "payload": payload or {},
    }


def test_apply_events_builds_live_tracker_state() -> None:
    state = new_live_state("run_001")
    events = [
        _event("TASKS_COMPILED", ts=1.0, payload={"n_tasks": 3}),
        _event("TASK_START", ts=2.0, task_id="task_01", payload={"goal": "run first step"}),
        _event(
            "TOOL_CALL_START",
            ts=3.0,
            task_id="task_01",
            step_id=0,
            payload={
                "tool": "bash",
                "params_compact": "cmd=echo hi",
                "params_full": {"cmd": "echo hi"},
                "toolcall_id": "task_01_s1_bash_0001",
            },
        ),
        _event(
            "TOOL_CALL_END",
            ts=6.0,
            task_id="task_01",
            step_id=0,
            payload={
                "tool": "bash",
                "status": "success",
                "highlights": "ok",
                "toolcall_id": "task_01_s1_bash_0001",
                "input_ref": "toolcalls/task_01_s1_bash_0001/input.json",
                "output_ref": "toolcalls/task_01_s1_bash_0001/output.json",
            },
        ),
        _event(
            "TASK_SUMMARY",
            ts=7.0,
            task_id="task_01",
            payload={"outcome": "success", "summary_snippet": "done", "artifacts": ["a.txt"]},
        ),
        _event(
            "TASK_JOURNAL_APPEND",
            ts=8.0,
            task_id="task_01",
            payload={"outcome": "success", "summary_snippet": "done", "artifacts": ["a.txt"]},
        ),
        _event("TASK_END", ts=9.0, task_id="task_01", payload={"outcome": "success"}),
    ]

    state, changed = apply_events(state, events)

    assert changed is True
    assert state["current_task_id"] == "task_01"
    assert state["progress"]["total"] == 3
    assert state["progress"]["completed"] == 1
    assert state["progress"]["pending"] == 2
    assert state["active_toolcall"] is None
    assert len(state["recent_toolcalls"]) == 1
    assert state["recent_toolcalls"][0]["tool"] == "bash"
    assert state["recent_toolcalls"][0]["duration_sec"] == 3
    assert len(state["journal_recent"]) == 1
    assert state["journal_recent"][0]["task_id"] == "task_01"


def test_should_refresh_live_summary_uses_tool_batch_and_interval() -> None:
    state = new_live_state("run_002")
    tool_event = [_event("TOOL_CALL_END", ts=10.0, task_id="task_01", payload={"tool": "bash", "status": "success"})]

    assert should_refresh_live_summary(state, tool_event, min_interval_s=8, tool_event_batch=5) is False
    assert should_refresh_live_summary(state, tool_event, min_interval_s=8, tool_event_batch=5) is False
    assert should_refresh_live_summary(state, tool_event, min_interval_s=8, tool_event_batch=5) is False
    assert should_refresh_live_summary(state, tool_event, min_interval_s=8, tool_event_batch=5) is False
    assert should_refresh_live_summary(state, tool_event, min_interval_s=8, tool_event_batch=5) is True

    task_trigger = [_event("TASK_END", ts=11.0, task_id="task_01", payload={"outcome": "success"})]
    assert should_refresh_live_summary(state, task_trigger, min_interval_s=100, tool_event_batch=5) is False
