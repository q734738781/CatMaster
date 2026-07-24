from __future__ import annotations

from pathlib import Path
import pytest

from catmaster.llm.config import LLMConfig, LLMProfile


def test_codex_oauth_template_routes_specialists_and_workers_by_reasoning_effort() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "llm_codex_oauth.template.yaml"
    profile = LLMProfile.from_env_or_file(str(config_path))

    def effort(role: str) -> str:
        cfg = profile.config_for_role(role)
        return str(cfg.provider_options["codex_oauth"]["chat_kwargs"]["reasoning"]["effort"])

    for role in (
        "proposal",
        "director",
        "research_lead",
        "hypothesis_proposer",
        "write_director",
        "write_reviewer",
        "literature_deep_research",
    ):
        assert effort(role) == "xhigh"
    for role in (
        "task_runner",
        "research_state_updater",
        "evidence_judge",
        "section_writer",
        "academic_polisher",
        "tex_compile_fixer",
        "memory_patch",
        "summary",
        "tool_selector",
        "image_analyzer",
        "self_evolution_proposer",
        "self_evolution_reviewer",
    ):
        assert effort(role) == "high"


def test_llm_config_parses_reasoning_and_provider_options() -> None:
    cfg = LLMConfig.from_dict(
        {
            "provider": "openrouter",
            "model": "openai/gpt-5.2",
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
    cfg = LLMConfig(provider="openrouter", model="openai/gpt-5.2")

    cfg.apply_env_fallbacks()

    assert cfg.reasoning == {"effort": "medium"}


def test_llm_config_env_fallback_sets_anthropic_key_env() -> None:
    cfg = LLMConfig(provider="anthropic", model="claude-sonnet-4-5-20250929")

    cfg.apply_env_fallbacks()

    assert cfg.api_key_env == "ANTHROPIC_API_KEY"


def test_llm_config_env_fallback_leaves_codex_oauth_keyless() -> None:
    cfg = LLMConfig(provider="codex_oauth", model="gpt-5.2-codex")

    cfg.apply_env_fallbacks()

    assert cfg.api_key_env == ""


def test_llm_profile_reads_models_agents_and_policies(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
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
                "  proposal: 'openai/gpt-5.2'",
                "  director: 'openai/gpt-5.2'",
                "  task_runner: 'openai/gpt-5.2'",
                "  memory_patch: 'openai/gpt-5.2'",
                "  summary: 'openai/gpt-5-nano'",
                "agent_policies:",
                "  proposal:",
                "    browse_tools_enabled: false",
                "agent_runtime:",
                "  recursion_limit: 512",
                "  max_tool_calls: 72",
                "  deepagent_context_trigger_token_cap: 180000",
                "  print_state_messages: true",
                "  print_http_raw_post: true",
                "writing:",
                "  author_name: 'CatMaster'",
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
    assert profile.agent_runtime.deepagent_context_trigger_token_cap == 180000
    assert profile.agent_runtime.print_state_messages is True
    assert profile.agent_runtime.print_http_raw_post is True
    assert profile.writing.author_name == "CatMaster"
    assert profile.peer_review_models == ["openai/gpt-5-nano"]


def test_llm_profile_accepts_current_specialist_alias_role_names(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'main-online':",
                "    provider: openrouter",
                "    model: openai/gpt-5.4",
                "  'summary-mini':",
                "    provider: openrouter",
                "    model: openai/gpt-5.4-mini",
                "agents:",
                "  proposal_agent: 'main-online'",
                "  planning_director: 'main-online'",
                "  experiment_specialist: 'main-online'",
                "  research_specialist: 'main-online'",
                "  writing_specialist: 'main-online'",
                "  writing_worker_agent: 'main-online'",
                "  peer_review_specialist: 'main-online'",
                "  litreview_agent: 'main-online'",
                "  memory_patcher: 'main-online'",
                "  run_summary: 'summary-mini'",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))

    assert profile.config_for_role("experiment_specialist").model == "openai/gpt-5.4"
    assert profile.config_for_role("task_runner").model == "openai/gpt-5.4"
    assert profile.config_for_role("litreview_agent").model == "openai/gpt-5.4"
    assert profile.config_for_role("literature_deep_research").model == "openai/gpt-5.4"
    assert profile.label_for_role("run_summary") == "summary-mini"
    assert profile.summary.model == "openai/gpt-5.4-mini"


def test_llm_profile_peer_review_models_must_reference_explicit_model_labels(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'peer-a':",
                "    provider: openrouter",
                "    model: google/gemini-3.1-pro",
                "  'peer-b':",
                "    provider: openrouter",
                "    model: openai/gpt-5.4",
                "agents:",
                "  proposal: 'peer-a'",
                "  director: 'peer-a'",
                "  task_runner: 'peer-a'",
                "  memory_patch: 'peer-a'",
                "  summary: 'peer-b'",
                "peer_review_models:",
                "  - 'peer-a'",
                "  - 'peer-b'",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))

    assert profile.peer_review_models == ["peer-a", "peer-b"]


def test_llm_profile_rejects_unknown_peer_review_model_labels(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'peer-a':",
                "    provider: openrouter",
                "    model: google/gemini-3.1-pro",
                "agents:",
                "  proposal: 'peer-a'",
                "  director: 'peer-a'",
                "  task_runner: 'peer-a'",
                "  memory_patch: 'peer-a'",
                "  summary: 'peer-a'",
                "peer_review_models:",
                "  - 'peer-a'",
                "  - 'missing-peer'",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="peer_review_models references unknown model label"):
        LLMProfile.from_env_or_file(str(cfg))


def test_llm_profile_tool_selector_fallbacks_to_task_runner(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "  'openai/gpt-5-nano':",
                "    provider: openrouter",
                "    model: openai/gpt-5-nano",
                "agents:",
                "  proposal: 'openai/gpt-5.2'",
                "  director: 'openai/gpt-5.2'",
                "  task_runner: 'openai/gpt-5-nano'",
                "  memory_patch: 'openai/gpt-5.2'",
                "  summary: 'openai/gpt-5.2'",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))
    assert profile.tool_selector.model == "openai/gpt-5-nano"


def test_llm_profile_scientific_campaign_roles_follow_research_fallbacks(
    tmp_path: Path,
) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  coordinator:",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "  judge:",
                "    provider: openrouter",
                "    model: openai/gpt-5-nano",
                "agents:",
                "  proposal: coordinator",
                "  director: coordinator",
                "  task_runner: coordinator",
                "  research_lead: coordinator",
                "  research_state_updater: judge",
                "  memory_patch: coordinator",
                "  summary: coordinator",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))

    assert profile.config_for_role("hypothesis_proposer").model == "openai/gpt-5.2"
    assert profile.config_for_role("evidence_judge").model == "openai/gpt-5-nano"


def test_llm_profile_tool_selector_can_use_dedicated_model(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "  'openai/gpt-5-nano':",
                "    provider: openrouter",
                "    model: openai/gpt-5-nano",
                "agents:",
                "  proposal: 'openai/gpt-5.2'",
                "  director: 'openai/gpt-5.2'",
                "  task_runner: 'openai/gpt-5.2'",
                "  tool_selector: 'openai/gpt-5-nano'",
                "  memory_patch: 'openai/gpt-5.2'",
                "  summary: 'openai/gpt-5.2'",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))
    assert profile.tool_selector.model == "openai/gpt-5-nano"


def test_llm_profile_image_analyzer_fallbacks_to_task_runner(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "  'openai/gpt-5-nano':",
                "    provider: openrouter",
                "    model: openai/gpt-5-nano",
                "agents:",
                "  proposal: 'openai/gpt-5.2'",
                "  director: 'openai/gpt-5.2'",
                "  task_runner: 'openai/gpt-5-nano'",
                "  memory_patch: 'openai/gpt-5.2'",
                "  summary: 'openai/gpt-5.2'",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))
    assert profile.image_analyzer.model == "openai/gpt-5-nano"


def test_llm_profile_image_analyzer_can_use_dedicated_model(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "  'openai/gpt-5-nano':",
                "    provider: openrouter",
                "    model: openai/gpt-5-nano",
                "agents:",
                "  proposal: 'openai/gpt-5.2'",
                "  director: 'openai/gpt-5.2'",
                "  task_runner: 'openai/gpt-5.2'",
                "  image_analyzer: 'openai/gpt-5-nano'",
                "  memory_patch: 'openai/gpt-5.2'",
                "  summary: 'openai/gpt-5.2'",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))
    assert profile.image_analyzer.model == "openai/gpt-5-nano"


def test_llm_profile_image_generation_can_use_dedicated_model_and_yaml_image_config(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "  'google/gemini-2.5-flash-image-preview':",
                "    provider: openrouter",
                "    model: google/gemini-2.5-flash-image-preview",
                "agents:",
                "  proposal: 'openai/gpt-5.2'",
                "  director: 'openai/gpt-5.2'",
                "  task_runner: 'openai/gpt-5.2'",
                "  memory_patch: 'openai/gpt-5.2'",
                "  summary: 'openai/gpt-5.2'",
                "image_generation:",
                "  model_label: 'google/gemini-2.5-flash-image-preview'",
                "  image_config:",
                "    aspect_ratio: '4:3'",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))
    assert profile.config_for_image_generation().model == "google/gemini-2.5-flash-image-preview"
    assert profile.image_generation.image_config == {"aspect_ratio": "4:3"}


def test_llm_profile_image_generation_falls_back_to_image_analyzer_when_omitted(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "  'openai/gpt-5-nano':",
                "    provider: openrouter",
                "    model: openai/gpt-5-nano",
                "agents:",
                "  proposal: 'openai/gpt-5.2'",
                "  director: 'openai/gpt-5.2'",
                "  task_runner: 'openai/gpt-5.2'",
                "  image_analyzer: 'openai/gpt-5-nano'",
                "  memory_patch: 'openai/gpt-5.2'",
                "  summary: 'openai/gpt-5.2'",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))
    assert profile.config_for_image_generation().model == "openai/gpt-5-nano"
    assert profile.image_generation.image_config == {}


def test_llm_profile_literature_deep_research_fallbacks_to_director(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "  'openai/gpt-5-nano':",
                "    provider: openrouter",
                "    model: openai/gpt-5-nano",
                "agents:",
                "  proposal: 'openai/gpt-5.2'",
                "  director: 'openai/gpt-5-nano'",
                "  task_runner: 'openai/gpt-5.2'",
                "  memory_patch: 'openai/gpt-5.2'",
                "  summary: 'openai/gpt-5.2'",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))
    assert profile.literature_deep_research.model == "openai/gpt-5-nano"


def test_llm_profile_agent_runtime_legacy_keys_are_rejected(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "agents:",
                "  proposal: 'openai/gpt-5.2'",
                "  director: 'openai/gpt-5.2'",
                "  task_runner: 'openai/gpt-5.2'",
                "  memory_patch: 'openai/gpt-5.2'",
                "  summary: 'openai/gpt-5.2'",
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
                "  'openai/gpt-5.2':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "agents:",
                "  proposal: 'openai/gpt-5.2'",
                "  director: 'openai/gpt-5.2'",
                "  task_runner: 'openai/gpt-5.2'",
                "  memory_patch: 'openai/gpt-5.2'",
                "  summary: 'openai/gpt-5.2'",
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
    assert profile.agent_runtime.max_tool_calls == 120
    assert profile.agent_runtime.deepagent_context_trigger_token_cap == 258_000
    assert profile.agent_runtime.print_state_messages is False
    assert profile.agent_runtime.print_http_raw_post is False


def test_llm_profile_agent_runtime_context_trigger_cap_can_be_disabled(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "agents:",
                "  proposal: 'openai/gpt-5.2'",
                "  director: 'openai/gpt-5.2'",
                "  task_runner: 'openai/gpt-5.2'",
                "  memory_patch: 'openai/gpt-5.2'",
                "  summary: 'openai/gpt-5.2'",
                "agent_runtime:",
                "  deepagent_context_trigger_token_cap: null",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))

    assert profile.agent_runtime.deepagent_context_trigger_token_cap is None


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
                "  'openai/gpt-5.2':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "agents:",
                "  proposal: 'openai/gpt-5.2'",
                "  director: 'openai/gpt-5.2'",
                "  task_runner: 'openai/gpt-5.2'",
                "  memory_patch: 'openai/gpt-5.2'",
                "  summary: 'openai/gpt-5.2'",
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
                "  'openai/gpt-5.2':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "    tool_calling:",
                "      profile: legacy",
                "agents:",
                "  proposal: 'openai/gpt-5.2'",
                "  director: 'openai/gpt-5.2'",
                "  task_runner: 'openai/gpt-5.2'",
                "  memory_patch: 'openai/gpt-5.2'",
                "  summary: 'openai/gpt-5.2'",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="tool_calling"):
        LLMProfile.from_env_or_file(str(cfg))


def test_llm_profile_accepts_official_reasoning_effort(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'deepseek-v4-pro':",
                "    provider: oai_compatible",
                "    model: deepseek-v4-pro",
                "    reasoning_effort: high",
                "agents:",
                "  proposal: 'deepseek-v4-pro'",
                "  director: 'deepseek-v4-pro'",
                "  task_runner: 'deepseek-v4-pro'",
                "  memory_patch: 'deepseek-v4-pro'",
                "  summary: 'deepseek-v4-pro'",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))

    assert profile.config_for_role("task_runner").reasoning_effort == "high"


def test_llm_profile_rejects_model_level_extra_body(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  'openai/gpt-5.2':",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "    extra_body:",
                "      prompt_cache_retention: 24h",
                "agents:",
                "  proposal: 'openai/gpt-5.2'",
                "  director: 'openai/gpt-5.2'",
                "  task_runner: 'openai/gpt-5.2'",
                "  memory_patch: 'openai/gpt-5.2'",
                "  summary: 'openai/gpt-5.2'",
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
                "  model: openai/gpt-5.2",
                "summary:",
                "  provider: openrouter",
                "  model: openai/gpt-5-nano",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        LLMProfile.from_env_or_file(str(cfg))
