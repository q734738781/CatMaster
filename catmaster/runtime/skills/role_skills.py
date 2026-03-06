from __future__ import annotations

ROLE_SKILL_NAMES: dict[str, list[str]] = {
    "proposal": [
        "computational-heterogeneous-catalysis",
        "materials-discovery-and-bulk-selection",
        "slab-construction-and-surface-modeling",
        "adsorption-site-screening",
        "thermo-free-energy-and-reporting",
        "transition-state-neb",
    ],
    "director": [
        "computational-heterogeneous-catalysis",
        "materials-discovery-and-bulk-selection",
        "slab-construction-and-surface-modeling",
        "adsorbate-and-intermediate-generation",
        "adsorption-site-screening",
        "mace-screening-and-relaxation",
        "vasp-input-preparation",
        "vasp-batch-execution",
        "transition-state-neb",
        "thermo-free-energy-and-reporting",
    ],
    "fast_director": [
        "computational-heterogeneous-catalysis",
        "materials-discovery-and-bulk-selection",
        "slab-construction-and-surface-modeling",
        "adsorbate-and-intermediate-generation",
        "adsorption-site-screening",
        "mace-screening-and-relaxation",
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
        "vasp-input-preparation",
        "vasp-batch-execution",
        "transition-state-neb",
        "thermo-free-energy-and-reporting",
    ],
}


def role_visible_skill_names(role: str) -> list[str]:
    return list(ROLE_SKILL_NAMES.get(str(role or "").strip(), []))


__all__ = ["ROLE_SKILL_NAMES", "role_visible_skill_names"]
