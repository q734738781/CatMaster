from __future__ import annotations

ROLE_SKILL_NAMES: dict[str, list[str]] = {
    "proposal": [
        "computational-heterogeneous-catalysis",
        "literature-grounding",
        "catalysis-prior-art-and-benchmarking",
        "materials-discovery-and-bulk-selection",
        "slab-construction-and-surface-modeling",
        "adsorption-site-screening",
        "structure-visual-inspection",
        "thermo-free-energy-and-reporting",
        "transition-state-neb",
    ],
    "director": [
        "computational-heterogeneous-catalysis",
        "literature-grounding",
        "catalysis-prior-art-and-benchmarking",
        "materials-discovery-and-bulk-selection",
        "slab-construction-and-surface-modeling",
        "adsorbate-and-intermediate-generation",
        "adsorption-site-screening",
        "mace-screening-and-relaxation",
        "structure-visual-inspection",
        "vasp-input-preparation",
        "vasp-batch-execution",
        "transition-state-neb",
        "thermo-free-energy-and-reporting",
    ],
    "fast_director": [
        "computational-heterogeneous-catalysis",
        "literature-grounding",
        "catalysis-prior-art-and-benchmarking",
        "materials-discovery-and-bulk-selection",
        "slab-construction-and-surface-modeling",
        "adsorbate-and-intermediate-generation",
        "adsorption-site-screening",
        "mace-screening-and-relaxation",
        "structure-visual-inspection",
        "vasp-input-preparation",
        "vasp-batch-execution",
        "transition-state-neb",
        "thermo-free-energy-and-reporting",
    ],
    "task_runner": [
        "materials-discovery-and-bulk-selection",
        "slab-construction-and-surface-modeling",
        "adsorbate-and-intermediate-generation",
        "adsorption-site-screening",
        "mace-screening-and-relaxation",
        "structure-visual-inspection",
        "vasp-input-preparation",
        "vasp-batch-execution",
        "transition-state-neb",
        "thermo-free-energy-and-reporting",
    ],
    "research_lead": [
        "computational-heterogeneous-catalysis",
        "literature-grounding",
        "catalysis-prior-art-and-benchmarking",
        "materials-discovery-and-bulk-selection",
        "slab-construction-and-surface-modeling",
        "adsorption-site-screening",
        "structure-visual-inspection",
        "transition-state-neb",
        "thermo-free-energy-and-reporting",
    ],
    "research_state_updater": [
        "computational-heterogeneous-catalysis",
        "literature-grounding",
        "catalysis-prior-art-and-benchmarking",
        "materials-discovery-and-bulk-selection",
        "slab-construction-and-surface-modeling",
        "adsorption-site-screening",
        "structure-visual-inspection",
        "transition-state-neb",
        "thermo-free-energy-and-reporting",
    ],
    "write_director": [],
    "section_writer": [],
    "write_reviewer": [],
}


def role_visible_skill_names(role: str) -> list[str]:
    return list(ROLE_SKILL_NAMES.get(str(role or "").strip(), []))


__all__ = ["ROLE_SKILL_NAMES", "role_visible_skill_names"]
