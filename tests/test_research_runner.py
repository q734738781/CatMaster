from __future__ import annotations

from types import SimpleNamespace

import pytest

from catmaster.agents.research_runner import ResearchRunner


def test_research_runner_lead_action_state_accepts_model_object() -> None:
    lead_action = SimpleNamespace(state="Conclude")
    assert ResearchRunner._lead_action_state(lead_action) == "Conclude"


def test_research_runner_lead_action_state_accepts_dict() -> None:
    assert ResearchRunner._lead_action_state({"state": "RunExperiment"}) == "RunExperiment"


@pytest.mark.anyio
async def test_research_runner_load_history_context_uses_all_project_runs() -> None:
    captured: dict[str, object] = {}

    class _HistoryReader:
        async def aload_context(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(context_text="history summary")

    runner = ResearchRunner.__new__(ResearchRunner)
    runner.history_reader = _HistoryReader()
    runner.run_context = SimpleNamespace(project_id="proj")

    summary = await ResearchRunner._load_history_context(runner, "coverage trends")

    assert summary == "history summary"
    assert captured["query"] == "coverage trends"
    assert captured["project_id"] == "proj"
    assert captured["lane"] is None
