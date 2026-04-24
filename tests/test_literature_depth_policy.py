from __future__ import annotations

from catmaster.llm.config import LiteratureRuntimeConfig
from catmaster.runtime.literature import resolve_depth


def test_resolve_depth_auto_defaults_to_configured_depth() -> None:
    config = LiteratureRuntimeConfig.from_dict({"auto_default_depth": "quick"})
    depth = resolve_depth("Give me representative papers on CO adsorption on Fe(110).", config=config)
    assert depth == "quick"


def test_resolve_depth_explicit_depth_has_priority() -> None:
    config = LiteratureRuntimeConfig.from_dict({"auto_default_depth": "quick"})
    depth = resolve_depth(
        "Prepare a deep research survey and benchmark landscape for Fe-based single-atom ORR catalysis.",
        requested_depth="deep_report",
        config=config,
    )
    assert depth == "deep_report"


def test_resolve_depth_none_for_task_runner_role() -> None:
    config = LiteratureRuntimeConfig.from_dict({"auto_default_depth": "focused"})
    depth = resolve_depth("Find supporting evidence for this setting.", role="task_runner", config=config)
    assert depth == "none"


def test_resolve_depth_auto_is_capped_to_standard_for_proposal() -> None:
    config = LiteratureRuntimeConfig.from_dict(
        {
            "auto_default_depth": "deep_report",
            "role_auto_max": {"proposal": "standard"},
        }
    )
    depth = resolve_depth(
        "Prepare a deep research survey and benchmark landscape for Fe-based single-atom ORR catalysis.",
        role="proposal",
        config=config,
    )
    assert depth == "standard"


def test_resolve_depth_auto_is_capped_to_focused_for_director() -> None:
    config = LiteratureRuntimeConfig.from_dict(
        {
            "auto_default_depth": "deep_report",
            "role_auto_max": {"director": "focused"},
        }
    )
    depth = resolve_depth(
        "Prepare a deep research survey and benchmark landscape for Fe-based single-atom ORR catalysis.",
        role="director",
        config=config,
    )
    assert depth == "focused"


def test_resolve_depth_accepts_current_specialist_aliases_in_role_caps() -> None:
    config = LiteratureRuntimeConfig.from_dict(
        {
            "auto_default_depth": "deep_report",
            "role_auto_max": {
                "research_specialist": "focused",
                "research_specialist_fast_lane": "standard",
            },
        }
    )
    lead_depth = resolve_depth(
        "Prepare a deep research survey and benchmark landscape for Fe-based single-atom ORR catalysis.",
        role="research_lead",
        config=config,
    )
    fast_depth = resolve_depth(
        "Prepare a deep research survey and benchmark landscape for Fe-based single-atom ORR catalysis.",
        role="fast_director",
        config=config,
    )
    assert lead_depth == "focused"
    assert fast_depth == "standard"


def test_resolve_depth_auto_can_use_internal_preferred_depth_flag() -> None:
    config = LiteratureRuntimeConfig.from_dict({"auto_default_depth": "quick"})
    depth = resolve_depth(
        "Need benchmark conventions.",
        flags={"preferred_depth": "standard"},
        config=config,
    )
    assert depth == "standard"


def test_literature_runtime_config_parses_agent_step_budgets() -> None:
    config = LiteratureRuntimeConfig.from_dict(
        {
            "budgets": {
                "quick": {"agent_step_budget": 4},
                "focused": {"agent_step_budget": 8},
                "deep_report": {"agent_step_budget": 16},
            }
        }
    )
    assert config.budget_for_depth("quick").agent_step_budget == 4
    assert config.budget_for_depth("focused").agent_step_budget == 8
    assert config.budget_for_depth("deep_report").agent_step_budget == 16
