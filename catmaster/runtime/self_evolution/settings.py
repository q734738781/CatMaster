from __future__ import annotations

import os
from typing import Literal


SelfEvolutionMode = Literal["off", "observe", "auto"]
SELF_EVOLUTION_MODE_ENV = "CATMASTER_SELF_EVOLUTION_MODE"


def resolve_self_evolution_mode(value: str | None = None) -> SelfEvolutionMode:
    raw = str(value if value is not None else os.getenv(SELF_EVOLUTION_MODE_ENV, "auto")).strip().lower()
    if raw not in {"off", "observe", "auto"}:
        raise ValueError(f"{SELF_EVOLUTION_MODE_ENV} must be one of: off, observe, auto; got {raw!r}")
    return raw  # type: ignore[return-value]


def self_evolution_enqueue_enabled(mode: SelfEvolutionMode) -> bool:
    return mode != "off"


def self_evolution_promotion_enabled(mode: SelfEvolutionMode) -> bool:
    return mode == "auto"


__all__ = [
    "SELF_EVOLUTION_MODE_ENV",
    "SelfEvolutionMode",
    "resolve_self_evolution_mode",
    "self_evolution_enqueue_enabled",
    "self_evolution_promotion_enabled",
]
