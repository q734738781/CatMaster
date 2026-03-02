from __future__ import annotations

from pathlib import Path
import pytest

from catmaster.llm.config import LLMConfig, LLMProfile


def test_llm_config_parses_reasoning_and_provider_options() -> None:
    cfg = LLMConfig.from_dict(
        {
            "provider": "openrouter",
            "model": "openai/gpt-5.2:online",
            "reasoning": {"effort": "high"},
            "provider_options": {
                "openrouter": {
                    "extra_body": {"prompt_cache_retention": "24h"},
                },
                "openai": {
                    "request_options": {"timeout": 30},
                },
            },
        }
    )

    assert cfg.reasoning == {"effort": "high"}
    assert cfg.provider_options == {
        "openrouter": {"extra_body": {"prompt_cache_retention": "24h"}},
        "openai": {"request_options": {"timeout": 30}},
    }


def test_llm_config_env_fallback_sets_reasoning_effort(monkeypatch) -> None:
    monkeypatch.setenv("CATMASTER_REASONING_EFFORT", "medium")
    cfg = LLMConfig(provider="openrouter", model="openai/gpt-5.2:online")

    cfg.apply_env_fallbacks()

    assert cfg.reasoning == {"effort": "medium"}


def test_llm_profile_reads_models_agents_and_policies(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2:online':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2:online",
                "    reasoning:",
                "      effort: high",
                "    provider_options:",
                "      openrouter:",
                "        extra_body:",
                "          prompt_cache_retention: 24h",
                "  'openai/gpt-5-nano':",
                "    provider: openrouter",
                "    model: openai/gpt-5-nano",
                "agents:",
                "  proposal: 'openai/gpt-5.2:online'",
                "  director: 'openai/gpt-5.2:online'",
                "  task_runner: 'openai/gpt-5.2:online'",
                "  memory_patch: 'openai/gpt-5.2:online'",
                "  summary: 'openai/gpt-5-nano'",
                "agent_policies:",
                "  proposal:",
                "    browse_tools_enabled: false",
                "agent_runtime:",
                "  recursion_limit: 512",
                "  max_tool_calls: 72",
                "  print_state_messages: true",
                "  print_http_raw_post: true",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))

    task_cfg = profile.config_for_role("task_runner")
    assert task_cfg.reasoning == {"effort": "high"}
    assert task_cfg.provider_options == {
        "openrouter": {"extra_body": {"prompt_cache_retention": "24h"}},
    }
    assert task_cfg.print_http_raw_post is True
    assert profile.summary.model == "openai/gpt-5-nano"
    assert profile.summary.print_http_raw_post is True
    assert profile.agent_policies.proposal.browse_tools_enabled is False
    assert profile.agent_runtime.recursion_limit == 512
    assert profile.agent_runtime.max_tool_calls == 72
    assert profile.agent_runtime.print_state_messages is True
    assert profile.agent_runtime.print_http_raw_post is True


def test_llm_profile_agent_runtime_legacy_keys_are_rejected(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2:online':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2:online",
                "agents:",
                "  proposal: 'openai/gpt-5.2:online'",
                "  director: 'openai/gpt-5.2:online'",
                "  task_runner: 'openai/gpt-5.2:online'",
                "  memory_patch: 'openai/gpt-5.2:online'",
                "  summary: 'openai/gpt-5.2:online'",
                "agent_runtime:",
                "  termination_mode: control_tools",
                "  strict_control_contract: false",
                "  recursion_limit: 0",
                "  print_state_messages: false",
                "  print_http_raw_post: false",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="agent_runtime no longer supports"):
        LLMProfile.from_env_or_file(str(cfg))


def test_llm_profile_agent_runtime_recursion_limit_zero_expands(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2:online':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2:online",
                "agents:",
                "  proposal: 'openai/gpt-5.2:online'",
                "  director: 'openai/gpt-5.2:online'",
                "  task_runner: 'openai/gpt-5.2:online'",
                "  memory_patch: 'openai/gpt-5.2:online'",
                "  summary: 'openai/gpt-5.2:online'",
                "agent_runtime:",
                "  recursion_limit: 0",
                "  print_state_messages: false",
                "  print_http_raw_post: false",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))
    assert profile.agent_runtime.recursion_limit == 1_000_000
    assert profile.agent_runtime.max_tool_calls == 60
    assert profile.agent_runtime.print_state_messages is False
    assert profile.agent_runtime.print_http_raw_post is False


def test_llm_profile_from_env_ignores_legacy_runtime_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CATMASTER_LLM_PROVIDER", "openai")
    monkeypatch.setenv("CATMASTER_LLM_MODEL", "gpt-5.2")
    monkeypatch.setenv("CATMASTER_TERMINATION_MODE", "control_tools")
    monkeypatch.setenv("CATMASTER_STRICT_CONTROL_CONTRACT", "false")
    monkeypatch.setenv("CATMASTER_RECURSION_LIMIT", "256")
    monkeypatch.setenv("CATMASTER_MAX_TOOL_CALLS", "66")
    monkeypatch.setenv("CATMASTER_PRINT_HTTP_RAW_POST", "true")

    profile = LLMProfile.from_env()

    assert profile.agent_runtime.recursion_limit == 256
    assert profile.agent_runtime.max_tool_calls == 66
    assert profile.agent_runtime.print_http_raw_post is True


def test_llm_profile_rejects_legacy_tool_calling_profiles(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "tool_calling_profiles:",
                "  legacy:",
                "    driver: openai_chat_completions",
                "models:",
                "  'openai/gpt-5.2:online':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2:online",
                "agents:",
                "  proposal: 'openai/gpt-5.2:online'",
                "  director: 'openai/gpt-5.2:online'",
                "  task_runner: 'openai/gpt-5.2:online'",
                "  memory_patch: 'openai/gpt-5.2:online'",
                "  summary: 'openai/gpt-5.2:online'",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tool_calling_profiles"):
        LLMProfile.from_env_or_file(str(cfg))


def test_llm_profile_rejects_legacy_model_tool_calling(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2:online':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2:online",
                "    tool_calling:",
                "      profile: legacy",
                "agents:",
                "  proposal: 'openai/gpt-5.2:online'",
                "  director: 'openai/gpt-5.2:online'",
                "  task_runner: 'openai/gpt-5.2:online'",
                "  memory_patch: 'openai/gpt-5.2:online'",
                "  summary: 'openai/gpt-5.2:online'",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tool_calling"):
        LLMProfile.from_env_or_file(str(cfg))


def test_llm_profile_rejects_legacy_reasoning_effort(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2:online':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2:online",
                "    reasoning_effort: high",
                "agents:",
                "  proposal: 'openai/gpt-5.2:online'",
                "  director: 'openai/gpt-5.2:online'",
                "  task_runner: 'openai/gpt-5.2:online'",
                "  memory_patch: 'openai/gpt-5.2:online'",
                "  summary: 'openai/gpt-5.2:online'",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reasoning_effort"):
        LLMProfile.from_env_or_file(str(cfg))


def test_llm_profile_rejects_model_level_extra_body(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2:online':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2:online",
                "    extra_body:",
                "      prompt_cache_retention: 24h",
                "agents:",
                "  proposal: 'openai/gpt-5.2:online'",
                "  director: 'openai/gpt-5.2:online'",
                "  task_runner: 'openai/gpt-5.2:online'",
                "  memory_patch: 'openai/gpt-5.2:online'",
                "  summary: 'openai/gpt-5.2:online'",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="extra_body"):
        LLMProfile.from_env_or_file(str(cfg))


def test_llm_profile_rejects_legacy_main_summary_schema(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "main:",
                "  provider: openrouter",
                "  model: openai/gpt-5.2:online",
                "summary:",
                "  provider: openrouter",
                "  model: openai/gpt-5-nano",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        LLMProfile.from_env_or_file(str(cfg))
