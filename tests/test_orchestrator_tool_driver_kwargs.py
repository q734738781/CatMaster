from __future__ import annotations

import sys
import types
from types import SimpleNamespace

sys.modules.setdefault("langchain_openai", types.SimpleNamespace(ChatOpenAI=object))
if "langchain_core.prompts" not in sys.modules:
    _lc_prompts = types.ModuleType("langchain_core.prompts")

    class _FakeChatPromptTemplate:
        @staticmethod
        def from_messages(_messages):
            return object()

    _lc_prompts.ChatPromptTemplate = _FakeChatPromptTemplate
    _lc_core = types.ModuleType("langchain_core")
    _lc_core.prompts = _lc_prompts
    sys.modules["langchain_core"] = _lc_core
    sys.modules["langchain_core.prompts"] = _lc_prompts

from catmaster.agents.orchestrator import Orchestrator
from catmaster.llm.config import LLMConfig, LLMProfile, ToolCallingConfig


def _orchestrator_for_kwargs(profile: LLMProfile) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    orch.llm_profile = profile
    orch.llm = SimpleNamespace(model_kwargs={})
    return orch


def test_orchestrator_tool_driver_kwargs_include_prompt_cache_retention() -> None:
    profile = LLMProfile(
        main=LLMConfig(
            tool_calling=ToolCallingConfig(prompt_cache_retention="24h"),
        )
    )
    orch = _orchestrator_for_kwargs(profile)

    kwargs = orch._tool_driver_kwargs()

    assert kwargs.get("prompt_cache_retention") == "24h"


def test_orchestrator_tool_driver_kwargs_skip_prompt_cache_retention_when_unset() -> None:
    profile = LLMProfile(
        main=LLMConfig(
            tool_calling=ToolCallingConfig(),
        )
    )
    orch = _orchestrator_for_kwargs(profile)

    kwargs = orch._tool_driver_kwargs()

    assert "prompt_cache_retention" not in kwargs


def test_orchestrator_proposal_function_tools_default_allowlist() -> None:
    profile = LLMProfile(
        main=LLMConfig(
            tool_calling=ToolCallingConfig(),
        )
    )
    orch = _orchestrator_for_kwargs(profile)
    orch.tool_backend = SimpleNamespace(list_function_tools=lambda: [
        {"name": "bash_exec"},
        {"name": "python_exec"},
        {"name": "write_note"},
    ])
    orch.tool_policy = SimpleNamespace(denied_tools=set())

    tools = orch._proposal_function_tools()

    assert [tool["name"] for tool in tools] == ["bash_exec", "python_exec"]


def test_orchestrator_proposal_function_tools_respects_denied() -> None:
    profile = LLMProfile(
        main=LLMConfig(
            tool_calling=ToolCallingConfig(),
        )
    )
    orch = _orchestrator_for_kwargs(profile)
    orch.tool_backend = SimpleNamespace(list_function_tools=lambda: [
        {"name": "bash_exec"},
        {"name": "python_exec"},
    ])
    orch.tool_policy = SimpleNamespace(denied_tools={"python_exec"})

    tools = orch._proposal_function_tools()

    assert [tool["name"] for tool in tools] == ["bash_exec"]


def test_orchestrator_proposal_function_tools_disabled_returns_empty() -> None:
    profile = LLMProfile(
        main=LLMConfig(
            tool_calling=ToolCallingConfig(proposal_browse_tools_enabled=False),
        )
    )
    orch = _orchestrator_for_kwargs(profile)
    orch.tool_backend = SimpleNamespace(list_function_tools=lambda: [
        {"name": "bash_exec"},
        {"name": "python_exec"},
    ])
    orch.tool_policy = SimpleNamespace(denied_tools=set())

    tools = orch._proposal_function_tools()

    assert tools == []
