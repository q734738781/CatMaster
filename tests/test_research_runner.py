from __future__ import annotations

from types import SimpleNamespace

import pytest

from catmaster.agents.research_runner import RESEARCH_TO_WRITER_HOUSE_PROMPT, ResearchRunner
from catmaster.agents.research_schemas import ResearchRequest


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


@pytest.mark.anyio
async def test_research_runner_writer_handoff_uses_house_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeWritingRunner:
        def __init__(self, **kwargs):
            captured["init"] = kwargs

        async def arun(self, request):
            captured["request"] = request
            return {"run_id": "write_001", "summary": "ok"}

    monkeypatch.setattr("catmaster.agents.research_runner.WritingRunner", _FakeWritingRunner)
    monkeypatch.setattr(
        "catmaster.agents.research_runner.RunContext.create",
        lambda **kwargs: SimpleNamespace(**kwargs, run_id="write_001"),
    )

    runner = ResearchRunner.__new__(ResearchRunner)
    runner.run_context = SimpleNamespace(
        workspace="/tmp/ws",
        project_id="proj",
        run_id="camp_001",
    )
    runner.llm_profile = SimpleNamespace(
        config_for_role=lambda role: SimpleNamespace(model=role, provider="openrouter", base_url="https://openrouter.ai/api/v1")
    )
    runner.reporter = None
    runner.run_ledger_store = None
    runner.history_reader = None
    runner.skills_runtime = None

    request = ResearchRequest(
        question="What controls CO adsorption on Fe(110)?",
        writing_mode="full_draft",
        target_section="Results and Discussion",
        campaign_title="Fe adsorption manuscript",
    )
    result = await ResearchRunner._launch_writing_handoff(runner, request_model=request)

    assert result["run_id"] == "write_001"
    writing_request = captured["request"]
    assert RESEARCH_TO_WRITER_HOUSE_PROMPT in writing_request.request
    assert "Research question: What controls CO adsorption on Fe(110)?" in writing_request.request
    assert "Preferred writing mode: full_draft." in writing_request.request
    assert "Prefer focusing on section: Results and Discussion." in writing_request.request
    assert "Preferred title direction: Fe adsorption manuscript." in writing_request.request
    assert writing_request.source_campaign_id == "camp_001"
