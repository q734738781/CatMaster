from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents import graph
from langchain_core.messages import SystemMessage
from catmaster.runtime.skills.middleware import CatMasterSkillsMiddleware
from catmaster.runtime.skills.models import SkillMeta


class _DummySkillsRuntime:
    def __init__(self, skills: list[SkillMeta]) -> None:
        self._skills = skills
        self.refresh_calls = 0

    def refresh_catalog(self):
        self.refresh_calls += 1
        return list(self._skills)

    def visible_skills(self, role: str):
        if role == "task_runner":
            return list(self._skills)
        return []


class _DummyRequest:
    def __init__(self, system_prompt) -> None:
        self.system_prompt = system_prompt
        self.system_message = None

    def override(self, **overrides):
        system_message = overrides.get("system_message")
        next_request = _DummyRequest(system_message or self.system_prompt)
        next_request.system_message = system_message
        return next_request


class _DummyModel:
    pass


def test_skills_middleware_refreshes_and_appends_addendum(tmp_path) -> None:
    skill = SkillMeta(
        name="demo-skill",
        description="demo desc",
        file_path="skills/demo-skill/SKILL.md",
        abs_skill_dir=tmp_path / "skills" / "demo-skill",
        abs_skill_md=tmp_path / "skills" / "demo-skill" / "SKILL.md",
        compatibility="local",
        suggested_tools=["demo_tool"],
    )
    runtime = _DummySkillsRuntime([skill])
    middleware = CatMasterSkillsMiddleware(
        role="task_runner",
        skills_runtime=runtime,
        skills_mount_available=True,
    )

    middleware.before_agent({}, None)
    assert runtime.refresh_calls == 1
    assert middleware.tools == []

    req = _DummyRequest("Base system")
    out = middleware.wrap_model_call(req, lambda request: request)
    assert isinstance(out.system_message, SystemMessage)
    rendered = "\n".join(block["text"] for block in out.system_message.content_blocks if block.get("type") == "text")
    assert "Base system" in rendered
    assert "## Skills" in rendered
    assert "`demo-skill`" in rendered
    assert "@skills/demo-skill/SKILL.md" in rendered


def test_skills_middleware_marks_unavailable_mount_when_absent(tmp_path) -> None:
    skill = SkillMeta(
        name="demo-skill",
        description="demo desc",
        file_path="skills/demo-skill/SKILL.md",
        abs_skill_dir=tmp_path / "skills" / "demo-skill",
        abs_skill_md=tmp_path / "skills" / "demo-skill" / "SKILL.md",
        compatibility="local",
        suggested_tools=["demo_tool"],
    )
    runtime = _DummySkillsRuntime([skill])
    middleware = CatMasterSkillsMiddleware(
        role="task_runner",
        skills_runtime=runtime,
        skills_mount_available=False,
    )

    middleware.before_agent({}, None)
    out = middleware.wrap_model_call(_DummyRequest("Base system"), lambda request: request)
    rendered = "\n".join(block["text"] for block in out.system_message.content_blocks if block.get("type") == "text")
    assert "skills mount is unavailable" in rendered
    assert "@skills/demo-skill/SKILL.md" not in rendered


def test_skills_middleware_preserves_existing_system_message_blocks(tmp_path) -> None:
    skill = SkillMeta(
        name="demo-skill",
        description="demo desc",
        file_path="skills/demo-skill/SKILL.md",
        abs_skill_dir=tmp_path / "skills" / "demo-skill",
        abs_skill_md=tmp_path / "skills" / "demo-skill" / "SKILL.md",
        compatibility="local",
        suggested_tools=["demo_tool"],
    )
    runtime = _DummySkillsRuntime([skill])
    middleware = CatMasterSkillsMiddleware(
        role="task_runner",
        skills_runtime=runtime,
        skills_mount_available=True,
    )
    middleware.before_agent({}, None)

    base_message = SystemMessage(content_blocks=[{"type": "text", "text": "Base block"}], name="base")
    out = middleware.wrap_model_call(_DummyRequest(base_message), lambda request: request)
    assert isinstance(out.system_message, SystemMessage)
    assert out.system_message.name == "base"
    assert out.system_message.content_blocks[0]["text"] == "Base block"
    assert any("## Skills" in block.get("text", "") for block in out.system_message.content_blocks if block.get("type") == "text")


def test_build_role_middleware_task_runner_order(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeSelector:
        def __init__(self, *, model=None, max_tools=None, always_include=None):
            captured["model"] = model
            captured["max_tools"] = max_tools
            captured["always_include"] = list(always_include or [])

    monkeypatch.setattr(graph, "_load_llm_tool_selector_middleware", lambda: _FakeSelector)
    runtime = _DummySkillsRuntime([])

    chain = graph._build_role_middleware(
        role="task_runner",
        max_tool_calls=7,
        skills_runtime=runtime,
        skills_mount_available=True,
        selector_model=_DummyModel(),
        enable_selector=True,
    )
    assert isinstance(chain[0], CatMasterSkillsMiddleware)
    assert isinstance(chain[1], _FakeSelector)
    assert captured["max_tools"] == 20
    assert captured["always_include"] == graph._SKILL_FILESYSTEM_ALWAYS_INCLUDE
    assert len(chain) == 4

    memory_chain = graph._build_role_middleware(
        role="memory_patch",
        max_tool_calls=7,
        skills_runtime=runtime,
        skills_mount_available=False,
        selector_model=None,
        enable_selector=False,
    )
    assert all(not isinstance(item, CatMasterSkillsMiddleware) for item in memory_chain)
    assert len(memory_chain) == 2
