from __future__ import annotations

from catmaster.webui.live_state import apply_events, new_live_state


def _event(name: str, *, ts: float, payload: dict | None = None, task_id: str = "") -> dict:
    return {
        "name": name,
        "ts": ts,
        "payload": payload or {},
        "task_id": task_id,
    }


def test_live_state_handles_interrupt_lifecycle() -> None:
    state = new_live_state("run_interrupt")
    events = [
        _event("INTERRUPT_REQUESTED", ts=1.0, payload={"source": "ui"}),
        _event("INTERRUPT_ACKED", ts=2.0, payload={"phase": "toolcall"}),
        _event("TOOL_CALL_INTERRUPTED", ts=3.0, payload={"tool": "bash_exec", "toolcall_id": "x1"}, task_id="task_01"),
        _event("RUN_PAUSED", ts=4.0, payload={"phase": "toolcall"}),
    ]

    state, changed = apply_events(state, events)

    assert changed is True
    assert state["status"] == "interrupted_paused"
    assert state["current_phase"].startswith("paused")
