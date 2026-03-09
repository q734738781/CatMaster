from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents import graph
from catmaster.agents.graph import GraphRunner


class _DummyReporter:
    def is_live(self) -> bool:
        return True


def test_invoke_loop_marks_running_before_resume_after_hitl() -> None:
    runner = object.__new__(GraphRunner)
    runner.run_control = None
    runner.reporter = _DummyReporter()
    runner.run_policy = graph.GraphRunPolicy()
    runner.run_context = SimpleNamespace(run_id="run_test", run_dir=Path("/tmp/run_test"))

    writes: list[dict] = []
    runner._write_task_state = lambda state, _lane: writes.append(dict(state))
    runner._emit = lambda *_args, **_kwargs: None
    runner._collect_human_feedback = lambda _payload: "yes"
    runner._publish_report = lambda *_args, **_kwargs: {}
    runner._publish_run_export = lambda **_kwargs: {}
    runner._upsert_run_ledger = lambda **_kwargs: asyncio.sleep(0)

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


def test_proposal_review_node_appends_hitl_history_on_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph, "interrupt", lambda _payload: "Need explicit O2 reference convention.")

    cmd = graph._proposal_review_node(  # type: ignore[attr-defined]
        {
            "proposal_review_enabled": True,
            "proposal_approved": False,
            "proposal_md": "# proposal",
            "work_packages": ["wp1"],
            "hitl_history": [],
        }
    )

    assert cmd.goto == "run_proposal"
    history = list(cmd.update.get("hitl_history") or [])
    assert len(history) == 1
    assert history[0]["interrupt_type"] == "proposal_review"
    assert history[0]["feedback"] == "Need explicit O2 reference convention."
    assert history[0]["approved"] is False


def test_proposal_review_node_appends_hitl_history_on_approval(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(graph, "interrupt", lambda _payload: "approve")

    cmd = graph._proposal_review_node(  # type: ignore[attr-defined]
        {
            "proposal_review_enabled": True,
            "proposal_approved": False,
            "proposal_md": "# proposal",
            "work_packages": ["wp1", "wp2"],
            "hitl_history": [],
        }
    )

    assert cmd.goto == "run_director"
    history = list(cmd.update.get("hitl_history") or [])
    assert len(history) == 1
    assert history[0]["interrupt_type"] == "proposal_review"
    assert history[0]["feedback"] == "approve"
    assert history[0]["approved"] is True
