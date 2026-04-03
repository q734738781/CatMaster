from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

LEGACY_GRAPH_DRIVER_KIND = "legacy_graph_deprecated"
LEGACY_GRAPH_DEPRECATION_NOTICE = (
    "build_graph_runner() is legacy and deprecated. "
    "The legacy graph entrypoint has been removed. "
    "Use build_specialist_runner()/SpecialistRunner instead."
)


@dataclass(frozen=True)
class BuiltGraphRunner:
    runner: Any
    run_context: Any


def build_graph_runner(
    *,
    workspace: Path,
    llm_profile: Any,
    reporter: Any | None,
    run_control: Any | None,
    project_id: str,
    run_dir: Path | None = None,
    run_policy: Any | None = None,
    bind_run_control_id: bool = True,
    stream_debug_console: bool = False,
) -> BuiltGraphRunner:
    raise RuntimeError(LEGACY_GRAPH_DEPRECATION_NOTICE)


__all__ = [
    "BuiltGraphRunner",
    "build_graph_runner",
    "LEGACY_GRAPH_DRIVER_KIND",
    "LEGACY_GRAPH_DEPRECATION_NOTICE",
]
