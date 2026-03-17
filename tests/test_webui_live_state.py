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


def test_apply_events_tracks_llm_stream_and_graph_node() -> None:
    state = new_live_state("run_003")
    events = [
        _event("GRAPH_NODE_UPDATE", ts=1.0, payload={"node": "director", "text_preview": "thinking"}),
        _event("LLM_CALL_START", ts=2.0, payload={"model": "gpt-5", "phase": "react"}),
        _event("LLM_REASONING_DELTA", ts=2.5, payload={"model": "gpt-5", "phase": "react", "text": "Check spin state."}),
        _event("LLM_TOKEN_DELTA", ts=3.0, payload={"model": "gpt-5", "phase": "react", "text": "Hello"}),
        _event("LLM_TOKEN_DELTA", ts=4.0, payload={"model": "gpt-5", "phase": "react", "text": " world"}),
        _event("LLM_CALL_END", ts=5.0, payload={"model": "gpt-5", "phase": "react", "usage": {"output_tokens": 2}, "reasoning_text": "Check spin state."}),
    ]

    state, changed = apply_events(state, events)

    assert changed is True
    assert state["current_node"] == "director"
    assert state["llm"]["status"] == "completed"
    assert state["llm"]["text"] == "Hello world"
    assert state["llm"]["reasoning_text"] == "Check spin state."


def test_apply_events_uses_llm_end_preview_when_no_token_delta() -> None:
    state = new_live_state("run_004")
    events = [
        _event("LLM_CALL_START", ts=1.0, payload={"model": "gpt-5", "phase": "react"}),
        _event(
            "LLM_CALL_END",
            ts=2.0,
            payload={
                "model": "gpt-5",
                "phase": "react",
                "text_preview": "Progress: running a quick MACE relaxation now.",
                "usage": {"output_tokens": 12},
            },
        ),
    ]

    state, changed = apply_events(state, events)

    assert changed is True
    assert state["llm"]["status"] == "completed"
    assert state["llm"]["text"] == "Progress: running a quick MACE relaxation now."


def test_run_end_clears_live_llm_and_tool_state() -> None:
    state = new_live_state("run_005")
    events = [
        _event("LLM_CALL_START", ts=1.0, payload={"model": "gpt-5", "phase": "react"}),
        _event("LLM_REASONING_DELTA", ts=1.5, payload={"text": "Preparing final answer."}),
        _event("LLM_CALL_END", ts=2.0, payload={"text_preview": "## Summary\nDone."}),
        _event(
            "TOOL_CALL_START",
            ts=2.5,
            task_id="task_01",
            step_id=0,
            payload={"tool": "execute", "params_compact": "{\"command\":\"pwd\"}", "toolcall_id": "call_1"},
        ),
        _event("RUN_END", ts=3.0, payload={"status": "done"}),
    ]

    state, changed = apply_events(state, events)

    assert changed is True
    assert state["status"] == "done"
    assert state["active_toolcall"] is None
    assert state["recent_toolcalls"] == []
    assert state["llm"]["status"] == "idle"
    assert state["llm"]["text"] == ""
    assert state["llm"]["reasoning_text"] == ""


def test_write_todos_tool_start_updates_live_todo_panel() -> None:
    state = new_live_state("run_006")
    events = [
        _event(
            "TOOL_CALL_START",
            ts=1.0,
            payload={
                "tool": "write_todos",
                "params_full": {
                    "todos": [
                        {"content": "Inspect the current workspace", "status": "in_progress"},
                        {"content": "Run quick MACE relax", "status": "pending"},
                    ]
                },
            },
        )
    ]

    state, changed = apply_events(state, events)

    assert changed is True
    assert state["todo_items"] == ["Inspect the current workspace", "Run quick MACE relax"]
    assert state["todo_rows"][0]["status"] == "in_progress"


def test_apply_events_tracks_agent_scoped_live_state() -> None:
    state = new_live_state("run_007")
    events = [
        _event("LLM_CALL_START", ts=1.0, payload={"model": "gpt-5", "agent_name": "research_specialist"}),
        _event("LLM_REASONING_DELTA", ts=1.1, payload={"text": "Planning.", "agent_name": "research_specialist"}),
        _event(
            "TOOL_CALL_START",
            ts=2.0,
            payload={
                "tool": "write_todos",
                "toolcall_id": "call_todos",
                "params_full": {"todos": [{"content": "Check literature", "status": "in_progress"}]},
                "agent_name": "literature_agent",
            },
        ),
        _event(
            "TOOL_CALL_END",
            ts=3.0,
            payload={
                "tool": "write_todos",
                "status": "success",
                "toolcall_id": "call_todos",
                "agent_name": "literature_agent",
            },
        ),
        _event("RUN_END", ts=4.0, payload={"status": "done"}),
    ]

    state, changed = apply_events(state, events)

    assert changed is True
    assert state["agents"]["research_specialist"]["status"] == "completed"
    assert state["agents"]["research_specialist"]["started_ts"] == 1.0
    assert state["agents"]["research_specialist"]["completed_ts"] == 4.0
    assert state["agents"]["literature_agent"]["todo_rows"][0]["content"] == "Check literature"
    assert state["agents"]["literature_agent"]["recent_toolcalls"][0]["tool"] == "write_todos"


def test_agent_tool_end_marks_agent_completed_with_timestamps() -> None:
    state = new_live_state("run_008")
    events = [
        _event(
            "TOOL_CALL_START",
            ts=2.0,
            payload={
                "tool": "write_todos",
                "toolcall_id": "call_todos",
                "params_full": {"todos": [{"content": "Draft note", "status": "in_progress"}]},
                "agent_name": "writing_specialist",
            },
        ),
        _event(
            "TOOL_CALL_END",
            ts=5.0,
            payload={
                "tool": "write_todos",
                "status": "success",
                "toolcall_id": "call_todos",
                "agent_name": "writing_specialist",
            },
        ),
    ]

    state, changed = apply_events(state, events)

    assert changed is True
    agent = state["agents"]["writing_specialist"]
    assert agent["status"] == "completed"
    assert agent["started_ts"] == 2.0
    assert agent["completed_ts"] == 5.0
