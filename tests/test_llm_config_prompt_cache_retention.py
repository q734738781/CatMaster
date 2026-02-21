from __future__ import annotations

from pathlib import Path
import pytest

from catmaster.llm.config import LLMConfig, ToolCallingConfig
from catmaster.llm.config import LLMProfile


def test_tool_calling_config_parses_request_options_and_extra_body() -> None:
    cfg = ToolCallingConfig.from_dict({
        "profile": "openrouter_chat_completions",
        "driver": "openai_chat_completions",
        "request_options": {"metadata": {"k": "v"}},
        "extra_body": {"prompt_cache_retention": "24h"},
    })

    assert cfg.profile == "openrouter_chat_completions"
    assert cfg.driver == "openai_chat_completions"
    assert cfg.request_options == {"metadata": {"k": "v"}}
    assert cfg.extra_body == {"prompt_cache_retention": "24h"}


def test_llm_config_env_fallback_driver_openrouter(monkeypatch) -> None:
    monkeypatch.setenv("CATMASTER_TOOL_DRIVER", "openai_chat_completions")
    cfg = LLMConfig(provider="openrouter", tool_calling=ToolCallingConfig(driver="openai_responses"))

    cfg.apply_env_fallbacks()

    assert cfg.tool_calling.driver == "openai_chat_completions"


def test_llm_profile_reads_profiles_models_agents_and_policies(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join([
            "tool_calling_profiles:",
            "  openrouter_chat_completions:",
            "    driver: openai_chat_completions",
            "    parallel_tool_calls: false",
            "    supports_builtin_tools: false",
            "    request_options:",
            "      metadata:",
            "        source: profile",
            "    extra_body:",
            "      provider:",
            "        order: ['openai']",
            "models:",
            "  'openai/gpt-5.2:online':",
            "    provider: openrouter",
            "    model: openai/gpt-5.2:online",
            "    tool_calling:",
            "      profile: openrouter_chat_completions",
            "      request_options:",
            "        metadata:",
            "          run: model",
            "      extra_body:",
            "        prompt_cache_retention: 24h",
            "  'openai/gpt-5-nano':",
            "    provider: openrouter",
            "    model: openai/gpt-5-nano",
            "    tool_calling:",
            "      profile: openrouter_chat_completions",
            "agents:",
            "  proposal: 'openai/gpt-5.2:online'",
            "  director: 'openai/gpt-5.2:online'",
            "  task_runner: 'openai/gpt-5.2:online'",
            "  memory_patch: 'openai/gpt-5.2:online'",
            "  summary: 'openai/gpt-5-nano'",
            "agent_policies:",
            "  proposal:",
            "    browse_tools_enabled: false",
        ]),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))

    task_cfg = profile.config_for_role("task_runner")
    assert task_cfg.tool_calling.driver == "openai_chat_completions"
    assert task_cfg.tool_calling.request_options == {"metadata": {"run": "model"}}
    assert task_cfg.tool_calling.extra_body == {
        "provider": {"order": ["openai"]},
        "prompt_cache_retention": "24h",
    }
    assert profile.summary.model == "openai/gpt-5-nano"
    assert profile.agent_policies.proposal.browse_tools_enabled is False


def test_llm_profile_requires_tool_calling_profile_reference(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join([
            "tool_calling_profiles:",
            "  openrouter_chat_completions:",
            "    driver: openai_chat_completions",
            "models:",
            "  'openai/gpt-5.2:online':",
            "    provider: openrouter",
            "    model: openai/gpt-5.2:online",
            "    tool_calling: {}",
            "agents:",
            "  proposal: 'openai/gpt-5.2:online'",
            "  director: 'openai/gpt-5.2:online'",
            "  task_runner: 'openai/gpt-5.2:online'",
            "  memory_patch: 'openai/gpt-5.2:online'",
            "  summary: 'openai/gpt-5.2:online'",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        LLMProfile.from_env_or_file(str(cfg))


def test_llm_profile_rejects_legacy_main_summary_schema(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join([
            "main:",
            "  provider: openrouter",
            "  model: openai/gpt-5.2:online",
            "summary:",
            "  provider: openrouter",
            "  model: openai/gpt-5-nano",
        ]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        LLMProfile.from_env_or_file(str(cfg))
