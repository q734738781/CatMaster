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


def test_native_skill_discovery_keeps_bodies_and_references_on_demand(tmp_path: Path) -> None:
    """Staging/discovery must not eagerly inject optional procedure bodies."""
    from deepagents import create_deep_agent

    _write_skill(tmp_path, "A focused optional procedure")
    skill = tmp_path / "skills/demo/SKILL.md"
    skill.write_text(skill.read_text() + "\nBODY_ONLY_SENTINEL\nSee references/detail.md when needed.\n")
    reference = skill.parent / "references/detail.md"
    reference.parent.mkdir()
    reference.write_text("REFERENCE_ONLY_SENTINEL\n")
    backend = LocalShellBackend(root_dir=tmp_path, virtual_mode=True)
    requests = []

    class CapturingModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            requests.append(messages)
            return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    graph = create_deep_agent(
        model=CapturingModel(responses=[AIMessage(content="ok")]),
        backend=backend, skills=["/skills/"],
    )
    graph.invoke({"messages": [{"role": "user", "content": "A brief greeting."}]})
    system = "\n".join(str(m.content) for m in requests[0] if m.type == "system")
    assert "A focused optional procedure" in system and "/skills/demo/SKILL.md" in system
    assert "BODY_ONLY_SENTINEL" not in system
    assert "REFERENCE_ONLY_SENTINEL" not in system
    assert "BODY_ONLY_SENTINEL" in backend.read("/skills/demo/SKILL.md").file_data["content"]
    assert "REFERENCE_ONLY_SENTINEL" in backend.read("/skills/demo/references/detail.md").file_data["content"]
