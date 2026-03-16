from __future__ import annotations

from catmaster.runtime.run_control import RunControl


def test_run_control_interrupt_flow() -> None:
    control = RunControl(run_id="run_x")
    assert control.is_interrupt_requested() is False

    req = control.request_interrupt(source="ui", note="stop")
    assert req["requested"] is True
    assert control.is_interrupt_requested() is True

    ack = control.ack_interrupt(phase="toolcall", details={"tool": "bash"})
    assert ack["acked"] is True
    assert ack["phase"] == "toolcall"

    cleared = control.clear_interrupt()
    assert cleared["requested"] is False
    assert control.is_interrupt_requested() is False
