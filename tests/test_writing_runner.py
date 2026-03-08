from __future__ import annotations

from contextlib import asynccontextmanager
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
            _Tool("bash_exec"),
            _Tool("apply_aider_edits"),
            _Tool("polish_academic_prose"),
            _Tool("run_literature_research"),
        ]


class _FakeGraph:
    async def ainvoke(self, initial_state, config=None):
        _ = (initial_state, config)
        return {"status": "done", "summary": "ok"}


@pytest.mark.anyio
async def test_write_director_gets_bash_and_director_mcp_but_not_aider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_build_agent(*, tools, role, **kwargs):
        _ = kwargs
        captured[role] = [tool.name for tool in tools]
        return object()

    @asynccontextmanager
    async def _fake_open_mcp_runtime():
        yield SimpleNamespace(
            skill_mounts={},
            role_filtered_tools=lambda role: [_Tool(f"mcp_{role}")]
        )

    monkeypatch.setattr("catmaster.agents.writing_runner.get_tool_registry", lambda: _Registry())
    monkeypatch.setattr("catmaster.agents.writing_runner._build_agent", _fake_build_agent)
    monkeypatch.setattr("catmaster.agents.writing_runner.build_chat_model", lambda cfg: object())
    monkeypatch.setattr("catmaster.agents.writing_runner.build_writing_graph", lambda **kwargs: _FakeGraph())

    runner = WritingRunner.__new__(WritingRunner)
    runner.llm_profile = SimpleNamespace(
        config_for_role=lambda role: SimpleNamespace(model=role),
        mcp=SimpleNamespace(filesystem=SimpleNamespace(enabled=True)),
        writing=SimpleNamespace(author_name="CatMaster"),
    )
    runner.run_context = SimpleNamespace(
        workspace=tmp_path,
        run_dir=tmp_path / "metadata" / "runs" / "write_001",
        run_id="write_001",
        project_id="proj",
    )
    runner.reporter = None
    runner.run_ledger_store = None
    runner.history_reader = None
    runner.skills_runtime = None
    runner.store = SimpleNamespace(ensure_exists=lambda: None)
    runner._write_task_state = lambda payload: None
    runner._publish_final_report = lambda **kwargs: None
    runner._open_mcp_filesystem_runtime = _fake_open_mcp_runtime

    request = WritingRequest(
        request="Write from existing evidence.",
        source_campaign_id=None,
    )
    result = await WritingRunner._run_graph(
        runner,
        {"request": request.model_dump(), "status": "planning", "resume_mode": False},
    )

    assert result["status"] == "done"
    assert "bash_exec" in captured["write_director"]
    assert "mcp_director" in captured["write_director"]
    assert "polish_academic_prose" in captured["write_director"]
    assert "apply_aider_edits" not in captured["write_director"]
    assert "polish_academic_prose" not in captured["section_writer"]
