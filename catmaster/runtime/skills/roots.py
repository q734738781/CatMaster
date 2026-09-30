from __future__ import annotations


# One source of truth for every skill root mounted by the active specialist
# runtime. Self-evolution and catalog tooling import this list instead of
# maintaining smaller copies that silently remove valid skill owners.
ACTIVE_SKILL_GROUPS: tuple[str, ...] = (
    "materials_worker",
    "dynamics_worker",
    "ml_worker",
    "orca_xtb_worker",
    "atomistic",
    "research_specialist",
    "research_reasoning",
    "litreview_agent",
    "research_execution",
    "execution",
    "writing_specialist",
    "writing_quality",
    "plot_worker",
    "presentation_worker",
)


__all__ = ["ACTIVE_SKILL_GROUPS"]
