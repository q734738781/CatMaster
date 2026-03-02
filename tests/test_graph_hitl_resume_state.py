from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.graph import GraphRunner


class _DummyReporter:
    def is_live(self) -> bool:
        return True


def test_invoke_loop_marks_running_before_resume_after_hitl() -> None:
    runner = object.__new__(GraphRunner)
    runner.run_control = None
    runner.reporter = _DummyReporter()

    writes: list[dict] = []
    runner._write_task_state = lambda state, _lane: writes.append(dict(state))
    runner._emit = lambda *_args, **_kwargs: None
    runner._collect_human_feedback = lambda _payload: "yes"
    runner._publish_report = lambda *_args, **_kwargs: None

    outputs = [
        {"__interrupt__": [SimpleNamespace(value={"type": "proposal_review", "message": "review"})]},
        {"status": "done", "summary": "ok", "tasks": [], "observations": []},
    ]

    invoke_calls = {"idx": 0}

    async def _fake_invoke(_compiled, _graph_input, _config):
        idx = invoke_calls["idx"]
        invoke_calls["idx"] = idx + 1
        return outputs[idx]

    runner._ainvoke_graph_once = _fake_invoke

    result = asyncio.run(
        GraphRunner._ainvoke_loop(
            runner,
            compiled=object(),
            initial_state={"user_request": "u"},
            config={},
            workspace=None,
            lane="standard",
        )
    )

    assert result["status"] == "done"

    statuses = [str(state.get("status") or "") for state in writes]
    assert "awaiting_human_feedback" in statuses
    assert "running" in statuses
    assert "done" in statuses

    awaiting_idx = statuses.index("awaiting_human_feedback")
    running_idx = statuses.index("running")
    done_idx = statuses.index("done")
    assert awaiting_idx < running_idx < done_idx
