from __future__ import annotations

from types import SimpleNamespace

import pytest

from catmaster.agents.research_runner import RESEARCH_TO_WRITER_HOUSE_PROMPT, ResearchRunner
from catmaster.agents.research_schemas import ResearchRequest, RunWriterPayload


def test_research_runner_lead_action_state_accepts_model_object() -> None:
    lead_action = SimpleNamespace(state="Conclude")
    assert ResearchRunner._lead_action_state(lead_action) == "Conclude"


def test_research_runner_lead_action_state_accepts_dict() -> None:
    assert ResearchRunner._lead_action_state({"state": "RunExperiment"}) == "RunExperiment"


@pytest.mark.anyio
async def test_research_runner_execute_writer_request_uses_house_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
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
    runner.skills_runtime = None

    request = ResearchRequest(
        question="What controls CO adsorption on Fe(110)?",
        writing_mode="full_draft",
        output_format="tex",
        target_section="Results and Discussion",
        campaign_title="Fe adsorption manuscript",
        session_context_text="<chat_session_context>\nRecent conversation: User asked for a manuscript.\n</chat_session_context>",
        chat_session_id="chat_123",
        entry_context_tokens_estimate=321,
    )
    payload = RunWriterPayload(
        request="",
        writing_mode="full_draft",
        output_format="tex",
        target_section="Results and Discussion",
    )
    result = await ResearchRunner._execute_writer_request(
        runner,
        request_model=request,
        writer_payload=payload,
    )

    assert result["run_id"] == "write_001"
    assert result["final_output_path"] == ""
    writing_request = captured["request"]
    assert RESEARCH_TO_WRITER_HOUSE_PROMPT in writing_request.request
    assert "Research question: What controls CO adsorption on Fe(110)?" in writing_request.request
    assert "Preferred writing mode: full_draft." in writing_request.request
    assert "Preferred output format: tex." in writing_request.request
    assert "Prefer focusing on section: Results and Discussion." in writing_request.request
    assert "Preferred title direction: Fe adsorption manuscript." in writing_request.request
    assert writing_request.source_campaign_id == "camp_001"
    assert writing_request.writing_mode == "full_draft"
    assert writing_request.output_format == "tex"
    assert writing_request.target_section == "Results and Discussion"
    assert writing_request.session_context_text == request.session_context_text
    assert writing_request.chat_session_id == "chat_123"
    assert writing_request.entry_context_tokens_estimate == 321


def test_research_runner_build_writer_request_prefers_lead_payload() -> None:
    request = ResearchRequest(
        question="What controls CO adsorption on Fe(110)?",
        writing_mode="full_draft",
        output_format="tex",
        target_section="Results and Discussion",
        session_context_text="ctx",
        chat_session_id="chat_123",
        entry_context_tokens_estimate=321,
    )
    payload = RunWriterPayload(
        request="Write a concise markdown validation report from the current campaign evidence.",
        writing_mode="internal_report",
        output_format="md",
        target_section="Validation Summary",
    )

    writing_request = ResearchRunner._build_writer_request(
        request_model=request,
        source_campaign_id="camp_001",
        writer_payload=payload,
    )

    assert writing_request.request == payload.request
    assert writing_request.writing_mode == "internal_report"
    assert writing_request.output_format == "md"
    assert writing_request.target_section == "Validation Summary"
    assert writing_request.source_campaign_id == "camp_001"
    assert writing_request.session_context_text == "ctx"


@pytest.mark.anyio
async def test_research_runner_builds_planner_with_memory_tools_only(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    class _Tool:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Registry:
        def as_langchain_tools(self, **kwargs):
            _ = kwargs
            return [_Tool("memory_read_index")]

    class _FakeGraph:
        async def ainvoke(self, initial_state, config=None):
            _ = (initial_state, config)
            return {"status": "done", "summary": "ok"}

    def _fake_build_structured_agent(*, role, tools, skills_runtime, mounted_skill_tokens, **kwargs):
        _ = kwargs
        captured.setdefault("roles", []).append(role)
        captured["tools"] = [tool.name for tool in tools]
        captured["mounted"] = tuple(mounted_skill_tokens)
        captured["skills_runtime"] = skills_runtime
        return object()

    monkeypatch.setattr("catmaster.agents.research_runner.get_tool_registry", lambda: _Registry())
    monkeypatch.setattr("catmaster.agents.research_runner.build_chat_model", lambda cfg: object())
    monkeypatch.setattr("catmaster.agents.research_runner.build_research_graph", lambda **kwargs: _FakeGraph())
    monkeypatch.setattr(ResearchRunner, "_build_structured_agent", staticmethod(_fake_build_structured_agent))

    runner = ResearchRunner.__new__(ResearchRunner)
    runner.llm_profile = SimpleNamespace(
        config_for_role=lambda role: SimpleNamespace(model=role, provider="openrouter", base_url="https://openrouter.ai/api/v1"),
    )
    runner.run_context = SimpleNamespace(
        workspace=tmp_path,
        run_id="camp_001",
        run_dir=tmp_path / "metadata" / "runs" / "camp_001",
        project_id="proj",
    )
    runner.memory_store = SimpleNamespace()
    runner.store = SimpleNamespace()
    runner.literature_runner = object()
    runner.experiment_runner = object()
    runner.reporter = None
    runner.skills_runtime = None
    runner._write_task_state = lambda payload: None
    runner._read_board_cycle_index = lambda: 0
    runner._emit = lambda *args, **kwargs: None

    request = ResearchRequest(question="Q")
    result = await ResearchRunner._run_graph(
        runner,
        {"request": request.model_dump(), "status": "running"},
        request_model=request,
    )

    assert result["status"] == "done"
    assert captured["tools"] == ["memory_read_index"]
    assert captured["mounted"] == ()
    assert captured["roles"] == ["research_lead", "research_state_updater"]


@pytest.mark.anyio
async def test_research_runner_does_not_implicitly_start_writer_after_graph(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    class _FakeGraph:
        async def ainvoke(self, initial_state, config=None):
            _ = (initial_state, config)
            return {"status": "done", "summary": "research complete", "lead_action": {"state": "Conclude", "rationale": "done", "conclude": {"why_now": "done", "recommended_next_steps": [], "confidence": "medium", "memory_promotion_candidates": []}}}

    monkeypatch.setattr("catmaster.agents.research_runner.build_chat_model", lambda cfg: object())
    monkeypatch.setattr("catmaster.agents.research_runner.get_tool_registry", lambda: SimpleNamespace(as_langchain_tools=lambda **kwargs: []))
    monkeypatch.setattr("catmaster.agents.research_runner.build_research_graph", lambda **kwargs: _FakeGraph())
    monkeypatch.setattr(ResearchRunner, "_build_structured_agent", staticmethod(lambda **kwargs: object()))

    called: dict[str, bool] = {"writer": False}

    async def _unexpected_writer(*, request_model, writer_payload):
        _ = (request_model, writer_payload)
        called["writer"] = True
        return {"run_id": "write_001", "summary": "should not happen"}

    runner = ResearchRunner.__new__(ResearchRunner)
    runner.llm_profile = SimpleNamespace(
        config_for_role=lambda role: SimpleNamespace(model=role, provider="openrouter", base_url="https://openrouter.ai/api/v1"),
    )
    runner.run_context = SimpleNamespace(
        workspace=tmp_path,
        run_id="camp_001",
        run_dir=tmp_path / "metadata" / "runs" / "camp_001",
        project_id="proj",
    )
    runner.memory_store = SimpleNamespace()
    runner.store = SimpleNamespace()
    runner.literature_runner = object()
    runner.experiment_runner = object()
    runner.reporter = None
    runner.skills_runtime = None
    runner._write_task_state = lambda payload: None
    runner._read_board_cycle_index = lambda: 0
    runner._emit = lambda *args, **kwargs: None
    runner._execute_writer_request = _unexpected_writer

    request = ResearchRequest(question="Q", writing_mode="full_draft", output_format="tex")
    result = await ResearchRunner._run_graph(
        runner,
        {"request": request.model_dump(), "status": "running"},
        request_model=request,
    )

    assert result["status"] == "done"
    assert result["writing_run_id"] == ""
    assert called["writer"] is False
