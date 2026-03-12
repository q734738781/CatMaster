from __future__ import annotations

import importlib.util
import sys
import types
from types import SimpleNamespace

sys.modules.setdefault("langchain_openai", types.SimpleNamespace(ChatOpenAI=object))
try:
    _lc_prompts_spec = importlib.util.find_spec("langchain_core.prompts")
except (ModuleNotFoundError, ValueError):
    _lc_prompts_spec = None
if _lc_prompts_spec is None and "langchain_core.prompts" not in sys.modules:
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
from catmaster.runtime.tool_policy import ToolPolicy


def _profile_for_roles(
    *,
    task_runner_cfg: LLMConfig | None = None,
    proposal_cfg: LLMConfig | None = None,
) -> LLMProfile:
    task_runner_cfg = task_runner_cfg or LLMConfig(tool_calling=ToolCallingConfig())
    proposal_cfg = proposal_cfg or task_runner_cfg
    models = {
        "task_runner_model": task_runner_cfg,
        "proposal_model": proposal_cfg,
    }
    return LLMProfile(
        models=models,
        agents={
            "proposal": "proposal_model",
            "director": "task_runner_model",
            "task_runner": "task_runner_model",
            "memory_patch": "task_runner_model",
            "summary": "task_runner_model",
        },
    )


def _orchestrator_for_kwargs(profile: LLMProfile) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    orch.llm_profile = profile
    orch.summary_llm = SimpleNamespace(model_kwargs={})
    orch.llm = SimpleNamespace(model_kwargs={})
    orch._tool_drivers_by_role = {}
    orch.tool_driver = None
    orch._supports_builtin_tools = False
    return orch


def test_orchestrator_tool_driver_kwargs_include_request_options() -> None:
    profile = _profile_for_roles(
        task_runner_cfg=LLMConfig(
            tool_calling=ToolCallingConfig(
                driver="openai_responses",
                request_options={"prompt_cache_retention": "24h"},
            ),
        )
    )
    orch = _orchestrator_for_kwargs(profile)

    kwargs = orch._tool_driver_kwargs()

    assert kwargs.get("prompt_cache_retention") == "24h"


def test_orchestrator_tool_driver_kwargs_include_extra_body_for_chat_driver() -> None:
    profile = _profile_for_roles(
        task_runner_cfg=LLMConfig(
            tool_calling=ToolCallingConfig(
                driver="openai_chat_completions",
                extra_body={"provider": {"order": ["openai"]}},
            ),
        )
    )
    orch = _orchestrator_for_kwargs(profile)

    kwargs = orch._tool_driver_kwargs()

    assert kwargs.get("extra_body") == {"provider": {"order": ["openai"]}}


def test_orchestrator_proposal_function_tools_default_allowlist() -> None:
    profile = _profile_for_roles()
    orch = _orchestrator_for_kwargs(profile)
    orch.tool_backend = SimpleNamespace(list_function_tools=lambda: [
        {"name": "bash"},
        {"name": "write_note"},
    ])
    orch.tool_policy = SimpleNamespace(denied_tools=set())

    tools = orch._proposal_function_tools()

    assert [tool["name"] for tool in tools] == ["bash"]


def test_orchestrator_proposal_function_tools_respects_denied() -> None:
    profile = _profile_for_roles()
    orch = _orchestrator_for_kwargs(profile)
    orch.tool_backend = SimpleNamespace(list_function_tools=lambda: [
        {"name": "bash"},
        {"name": "write_note"},
    ])
    orch.tool_policy = SimpleNamespace(denied_tools={"bash"})

    tools = orch._proposal_function_tools()

    assert tools == []


def test_orchestrator_proposal_function_tools_disabled_returns_empty() -> None:
    profile = _profile_for_roles()
    profile.agent_policies.proposal.browse_tools_enabled = False
    orch = _orchestrator_for_kwargs(profile)
    orch.tool_backend = SimpleNamespace(list_function_tools=lambda: [
        {"name": "bash"},
        {"name": "write_note"},
    ])
    orch.tool_policy = SimpleNamespace(denied_tools=set())

    tools = orch._proposal_function_tools()

    assert tools == []


def test_orchestrator_tool_schema_respects_policy_filter() -> None:
    profile = _profile_for_roles()
    orch = _orchestrator_for_kwargs(profile)
    orch.tool_backend = SimpleNamespace(list_function_tools=lambda: [
        {"name": "bash"},
        {"name": "memory_apply_aider_edits"},
    ])
    orch.tool_policy = ToolPolicy(denied_tools={"memory_apply_aider_edits"})

    class _Registry:
        @staticmethod
        def get_tool_descriptions_for_llm(allowlist=None):
            return "\n".join(allowlist or [])

        @staticmethod
        def get_short_tool_descriptions_for_llm(allowlist=None):
            return "\n".join(allowlist or [])

    orch.registry = _Registry()

    text = orch._tool_schema()
    short = orch._tool_schema_short()
    assert "bash" in text
    assert "bash" in short
    assert "memory_apply_aider_edits" not in text
    assert "memory_apply_aider_edits" not in short
