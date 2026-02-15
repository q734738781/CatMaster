from __future__ import annotations

from pathlib import Path

from catmaster.llm.config import LLMConfig, ToolCallingConfig
from catmaster.llm.config import LLMProfile


def test_llm_config_parses_prompt_cache_retention() -> None:
    cfg = LLMConfig.from_dict({
        "provider": "openrouter",
        "model": "openai/gpt-5.2",
        "tool_calling": {
            "prompt_cache_retention": "24h",
        },
    })

    assert cfg.tool_calling.prompt_cache_retention == "24h"


def test_llm_config_invalid_prompt_cache_retention_warns(caplog) -> None:
    with caplog.at_level("WARNING"):
        cfg = LLMConfig.from_dict({
            "tool_calling": {
                "prompt_cache_retention": "7d",
            },
        })

    assert cfg.tool_calling.prompt_cache_retention is None
    assert "prompt_cache_retention" in caplog.text


def test_llm_config_env_fallback_prompt_cache_retention(monkeypatch) -> None:
    monkeypatch.setenv("CATMASTER_PROMPT_CACHE_RETENTION", "in_memory")
    cfg = LLMConfig(tool_calling=ToolCallingConfig())

    cfg.apply_env_fallbacks()

    assert cfg.tool_calling.prompt_cache_retention == "in_memory"


def test_llm_config_explicit_prompt_cache_retention_not_overridden(monkeypatch) -> None:
    monkeypatch.setenv("CATMASTER_PROMPT_CACHE_RETENTION", "in_memory")
    cfg = LLMConfig(tool_calling=ToolCallingConfig(prompt_cache_retention="24h"))

    cfg.apply_env_fallbacks()

    assert cfg.tool_calling.prompt_cache_retention == "24h"


def test_llm_config_proposal_browse_tools_default_true() -> None:
    cfg = LLMConfig.from_dict({})
    assert cfg.tool_calling.proposal_browse_tools_enabled is True


def test_llm_config_proposal_browse_tools_from_dict_false() -> None:
    cfg = LLMConfig.from_dict({
        "tool_calling": {
            "proposal_browse_tools_enabled": False,
        },
    })
    assert cfg.tool_calling.proposal_browse_tools_enabled is False


def test_llm_config_env_fallback_proposal_browse_tools(monkeypatch) -> None:
    monkeypatch.setenv("CATMASTER_PROPOSAL_BROWSE_TOOLS_ENABLED", "false")
    cfg = LLMConfig(tool_calling=ToolCallingConfig())

    cfg.apply_env_fallbacks()

    assert cfg.tool_calling.proposal_browse_tools_enabled is False


def test_llm_profile_reads_top_level_summary_override(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join([
            "main:",
            "  provider: openrouter",
            "  model: openai/gpt-5.2:online",
            "  tool_calling:",
            "    driver: openai_chat_completions",
            "summary:",
            "  provider: openrouter",
            "  model: openai/gpt-5-nano",
        ]),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))

    assert profile.summary is not None
    assert profile.summary.model == "openai/gpt-5-nano"
