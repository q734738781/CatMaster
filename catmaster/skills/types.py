from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    id: str
    description: str
    tool_allowlist: set[str]
    prompt_snippet: str = ""
    keywords: tuple[str, ...] = ()
