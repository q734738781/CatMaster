from __future__ import annotations

from pathlib import Path
import pytest

from catmaster.llm.config import LLMConfig, LLMProfile


def test_all_shipped_llm_profiles_load_without_removed_budget_fields() -> None:
    config_root = Path(__file__).resolve().parents[1] / "configs"
    shipped = (
        "llm.template.yaml",
        "llm.full.template.yaml",
        "llm_codex_oauth.template.yaml",
        "llm_gemini.yaml",
        "llm_mimo.yaml",
        "llm_modele.yaml",
        "llm_sonnet.yaml",
    )

    for name in shipped:
        profile = LLMProfile.from_env_or_file(str(config_root / name))
        assert not hasattr(profile.agent_runtime, "recursion_limit")
        assert not hasattr(profile.agent_runtime, "max_tool_calls")
        assert not hasattr(profile.literature, "budgets")
        assert not hasattr(profile.literature, "auto_default_depth")


def test_codex_oauth_template_routes_coordinators_and_workers_by_role() -> None:
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
        "literature_deep_research",
    ):
        assert profile.config_for_role(role).model == "gpt-6-astra"
        assert effort(role) == "high"
    for role in (
        "write_director", "section_writer", "presentation_worker",
        "plot_worker", "write_reviewer", "tex_compile_fixer",
    ):
        assert profile.config_for_role(role).model == "gpt-6-astra"
        assert effort(role) == "medium"
    assert profile.config_for_role("literature_worker").model == "gpt-6-luna"
    assert effort("literature_worker") == "xhigh"
    assert profile.config_for_role("thread_title").model == "gpt-6-luna"
    assert effort("thread_title") == "low"
    assert profile.config_for_role("thread_title").max_output_tokens is None
    assert profile.label_for_role("thread_title") != profile.label_for_role("literature_worker")
    for role in (
        "task_runner",
        "research_state_updater",
        "evidence_judge",
        "memory_patch",
        "summary",
        "tool_selector",
        "image_analyzer",
        "self_evolution_proposer",
        "self_evolution_reviewer",
    ):
        assert effort(role) == "medium"


def test_thread_title_role_remains_unbound_when_omitted(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  main:",
                "    provider: openrouter",
                "    model: openai/gpt-5.2",
                "agents:",
                "  proposal: main",
                "  director: main",
                "  task_runner: main",
                "  memory_patch: main",
                "  summary: main",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))

    assert "thread_title" not in profile.agents
    with pytest.raises(ValueError, match="thread_title"):
        profile.config_for_role("thread_title")


@pytest.mark.parametrize("name", ["llm.template.yaml", "llm.full.template.yaml", "llm_codex_oauth.template.yaml"])
def test_templates_build_astra_with_role_reasoning_without_sampling(name, monkeypatch):
    from langchain_core.messages import HumanMessage
    from catmaster.llm.factory import build_chat_model

    # An explicit null must override even a configured global temperature.
    monkeypatch.setenv("CATMASTER_TEMPERATURE", "0.7")
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    profile = LLMProfile.from_env_or_file(str(Path(__file__).resolve().parents[1] / "configs" / name))
    roles = {role: "medium" for role in (
        "write_director", "section_writer", "presentation_worker",
        "plot_worker", "write_reviewer", "tex_compile_fixer",
    )}
    if name == "llm_codex_oauth.template.yaml":
        roles.update({role: "high" for role in (
            "proposal", "director", "research_lead", "hypothesis_proposer",
            "literature_deep_research",
        )})
        roles["task_runner"] = "medium"
    for role, expected_effort in roles.items():
        cfg = profile.config_for_role(role)
        assert cfg.temperature is None
        model = build_chat_model(cfg)
        if cfg.provider == "codex_oauth":
            payload = model._get_request_payload([HumanMessage("hello")], _codex_headers={})
            assert payload["model"] == "gpt-6-astra"
        else:
            _, payload = model._create_message_dicts([HumanMessage("hello")], None)
            assert payload["model"] == "openai/gpt-6-astra"
        assert payload["reasoning"]["effort"] == expected_effort
        assert "temperature" not in payload
        assert "top_p" not in payload


@pytest.mark.parametrize("temperature_line,expected", [("", 0.7), ("    temperature: null\n", None), ("    temperature: 0.2\n", 0.2)])
def test_yaml_temperature_distinguishes_omission_from_null(tmp_path, monkeypatch, temperature_line, expected):
    monkeypatch.setenv("CATMASTER_TEMPERATURE", "0.7")
    path = tmp_path / "llm.yaml"
    path.write_text(
        "models:\n  main:\n    provider: openrouter\n    model: openai/gpt-6-astra\n"
        + temperature_line
        + "agents:\n"
        + "".join(f"  {role}: main\n" for role in ("proposal", "director", "task_runner", "memory_patch", "summary"))
    )
    assert LLMProfile.from_env_or_file(str(path)).main.temperature == expected


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
                "  litreview_worker_agent: 'main-online'",
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
    assert profile.config_for_role("litreview_worker_agent").model == "openai/gpt-5.4"
    assert profile.config_for_role("literature_worker").model == "openai/gpt-5.4"
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
    assert profile.literature_worker.model == "openai/gpt-5-nano"


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
                "  print_state_messages: false",
                "  print_http_raw_post: false",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="agent_runtime no longer supports"):
        LLMProfile.from_env_or_file(str(cfg))


def test_llm_profile_rejects_removed_numeric_runtime_budget_fields(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  main:",
                "    provider: openai",
                "    model: gpt-5.2",
                "agents:",
                "  proposal: main",
                "  director: main",
                "  task_runner: main",
                "  memory_patch: main",
                "  summary: main",
                "agent_runtime:",
                "  max_tool_calls: 12",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="prescribed numeric budgets"):
        LLMProfile.from_env_or_file(str(cfg))


def test_llm_profile_rejects_removed_literature_budget_fields(tmp_path: Path) -> None:
    cfg = tmp_path / "llm.yaml"
    cfg.write_text(
        "\n".join(
            [
                "models:",
                "  main:",
                "    provider: openai",
                "    model: gpt-5.2",
                "agents:",
                "  proposal: main",
                "  director: main",
                "  task_runner: main",
                "  memory_patch: main",
                "  summary: main",
                "literature:",
                "  budgets: {}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="depth or numeric budget"):
        LLMProfile.from_env_or_file(str(cfg))


def test_llm_profile_agent_runtime_reads_diagnostic_flags(tmp_path: Path) -> None:
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
                "  print_state_messages: true",
                "  print_http_raw_post: true",
            ]
        ),
        encoding="utf-8",
    )

    profile = LLMProfile.from_env_or_file(str(cfg))
    assert profile.agent_runtime.print_state_messages is True
    assert profile.agent_runtime.print_http_raw_post is True


def test_llm_profile_from_env_reads_http_debug_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CATMASTER_LLM_PROVIDER", "openai")
    monkeypatch.setenv("CATMASTER_LLM_MODEL", "gpt-5.2")
    monkeypatch.setenv("CATMASTER_PRINT_HTTP_RAW_POST", "true")

    profile = LLMProfile.from_env()

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


def test_research_challenger_model_binding_compatibility(tmp_path):
    import yaml
    source = Path(__file__).resolve().parents[1] / "configs/llm_codex_oauth.template.yaml"
    data = yaml.safe_load(source.read_text())
    data['agents'].pop('research_challenger')
    data['agents']['hypothesis_proposer'] = data['agents']['task_runner']
    path = tmp_path / 'legacy.yaml'
    path.write_text(yaml.safe_dump(data))
    old = LLMProfile.from_env_or_file(str(path))
    assert old.label_for_role('research_challenger') == old.label_for_role('hypothesis_proposer')
    data['agents']['research_challenger'] = data['agents']['research_lead']
    path.write_text(yaml.safe_dump(data))
    new = LLMProfile.from_env_or_file(str(path))
    assert new.label_for_role('research_challenger') == new.label_for_role('research_lead')
    assert new.label_for_role('research_challenger') != new.label_for_role('hypothesis_proposer')
    new.agents.pop('research_challenger')
    new.agents.pop('hypothesis_proposer')
    assert new.label_for_role('research_challenger') == new.label_for_role('research_lead')
