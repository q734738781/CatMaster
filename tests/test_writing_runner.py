from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.agents.writing_runner import WritingRunner
from catmaster.agents.writing_schemas import WritingRequest


class _Tool:
    def __init__(self, name: str) -> None:
        self.name = name


class _Registry:
    def as_langchain_tools(self, **kwargs):
        _ = kwargs
        return [
            _Tool("apply_aider_edits"),
            _Tool("compile_text"),
            _Tool("polish_academic_prose"),
            _Tool("run_literature_research"),
        ]


class _FakeGraph:
    async def ainvoke(self, initial_state, config=None):
        _ = (initial_state, config)
        return {"status": "done", "summary": "ok"}


@pytest.mark.anyio
async def test_write_director_uses_local_tools_only_and_excludes_aider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_build_agent(*, tools, role, **kwargs):
        _ = kwargs
        captured[role] = [tool.name for tool in tools]
        return object()

    monkeypatch.setattr("catmaster.agents.writing_runner.get_tool_registry", lambda: _Registry())
    monkeypatch.setattr("catmaster.agents.writing_runner._build_agent", _fake_build_agent)
    monkeypatch.setattr("catmaster.agents.writing_runner.build_chat_model", lambda cfg: object())
    monkeypatch.setattr("catmaster.agents.writing_runner.build_writing_graph", lambda **kwargs: _FakeGraph())

    runner = WritingRunner.__new__(WritingRunner)
    runner.llm_profile = SimpleNamespace(
        config_for_role=lambda role: SimpleNamespace(model=role),
        writing=SimpleNamespace(author_name="CatMaster"),
    )
    runner.run_context = SimpleNamespace(
        workspace=tmp_path,
        run_dir=tmp_path / "metadata" / "runs" / "write_001",
        run_id="write_001",
        project_id="proj",
    )
    runner.reporter = None
    runner.skills_runtime = None
    runner.store = SimpleNamespace(ensure_exists=lambda: None)
    runner._write_task_state = lambda payload: None

    request = WritingRequest(
        request="Write from existing evidence.",
        source_campaign_id=None,
    )
    result = await WritingRunner._run_graph(
        runner,
        {"request": request.model_dump(), "status": "planning", "resume_mode": False},
    )

    assert result["status"] == "done"
    assert "compile_text" in captured["write_director"]
    assert "polish_academic_prose" in captured["write_director"]
    assert "apply_aider_edits" not in captured["write_director"]
    assert "polish_academic_prose" not in captured["section_writer"]


@pytest.mark.anyio
async def test_markdown_writing_request_does_not_expose_compile_tool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_build_agent(*, tools, role, **kwargs):
        _ = kwargs
        captured[role] = [tool.name for tool in tools]
        return object()

    monkeypatch.setattr("catmaster.agents.writing_runner.get_tool_registry", lambda: _Registry())
    monkeypatch.setattr("catmaster.agents.writing_runner._build_agent", _fake_build_agent)
    monkeypatch.setattr("catmaster.agents.writing_runner.build_chat_model", lambda cfg: object())
    monkeypatch.setattr("catmaster.agents.writing_runner.build_writing_graph", lambda **kwargs: _FakeGraph())

    runner = WritingRunner.__new__(WritingRunner)
    runner.llm_profile = SimpleNamespace(
        config_for_role=lambda role: SimpleNamespace(model=role),
        writing=SimpleNamespace(author_name="CatMaster"),
    )
    runner.run_context = SimpleNamespace(
        workspace=tmp_path,
        run_dir=tmp_path / "metadata" / "runs" / "write_001",
        run_id="write_001",
        project_id="proj",
    )
    runner.reporter = None
    runner.skills_runtime = None
    runner.store = SimpleNamespace(ensure_exists=lambda: None)
    runner._write_task_state = lambda payload: None

    request = WritingRequest(
        request="Write an internal report in markdown.",
        source_campaign_id=None,
        writing_mode="internal_report",
        output_format="md",
    )
    result = await WritingRunner._run_graph(
        runner,
        {"request": request.model_dump(), "status": "planning", "resume_mode": False},
    )

    assert result["status"] == "done"
    assert "compile_text" not in captured["write_director"]
