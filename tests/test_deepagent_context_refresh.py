from __future__ import annotations

import asyncio
from pathlib import Path

from deepagents.backends import LocalShellBackend
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from catmaster.runtime.deepagent_context_refresh import ReloadDeepAgentContextMiddleware


def _write_skill(root: Path, description: str) -> None:
    skill_dir = root / "skills" / "demo"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: demo\ndescription: {description}\n---\n\n# Demo\n",
        encoding="utf-8",
    )


def test_context_refresh_reloads_checkpointed_skills_and_memory(tmp_path: Path) -> None:
    _write_skill(tmp_path, "first description")
    (tmp_path / "AGENTS.md").write_text("first memory\n", encoding="utf-8")
    backend = LocalShellBackend(root_dir=tmp_path, virtual_mode=True)
    middleware = ReloadDeepAgentContextMiddleware(
        backend=backend,
        skills=["/skills/"],
        memory=["/AGENTS.md"],
    )

    stale_state = {
        "skills_metadata": [{"name": "demo", "description": "stale description"}],
        "skills_load_errors": ["stale error"],
        "memory_contents": {"/AGENTS.md": "stale memory"},
    }
    first = middleware.before_agent(stale_state, None, {})
    assert first["skills_metadata"][0]["description"] == "first description"
    assert first["memory_contents"]["/AGENTS.md"] == "first memory\n"
    assert first["skills_load_errors"] == []

    _write_skill(tmp_path, "second description")
    (tmp_path / "AGENTS.md").write_text("second memory\n", encoding="utf-8")
    checkpointed_state = {**stale_state, **first}
    second = middleware.before_agent(checkpointed_state, None, {})
    assert second["skills_metadata"][0]["description"] == "second description"
    assert second["memory_contents"]["/AGENTS.md"] == "second memory\n"


def test_context_refresh_async_path_reloads_files(tmp_path: Path) -> None:
    _write_skill(tmp_path, "async description")
    (tmp_path / "AGENTS.md").write_text("async memory\n", encoding="utf-8")
    middleware = ReloadDeepAgentContextMiddleware(
        backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=True),
        skills=["/skills/"],
        memory=["/AGENTS.md"],
    )

    update = asyncio.run(
        middleware.abefore_agent(
            {"skills_metadata": [], "memory_contents": {}},
            None,
            {},
        )
    )
    assert update["skills_metadata"][0]["description"] == "async description"
    assert update["memory_contents"]["/AGENTS.md"] == "async memory\n"


def test_context_refresh_async_hook_accepts_langchain_runtime_injection(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("runtime memory\n", encoding="utf-8")
    middleware = ReloadDeepAgentContextMiddleware(
        backend=LocalShellBackend(root_dir=tmp_path, virtual_mode=True),
        skills=[],
        memory=["/AGENTS.md"],
    )
    agent = create_agent(
        FakeMessagesListChatModel(responses=[AIMessage(content="ok")]),
        tools=[],
        middleware=[middleware],
    )

    result = asyncio.run(agent.ainvoke({"messages": [{"role": "user", "content": "ping"}]}))

    assert result["messages"][-1].content == "ok"
