from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SkillMeta:
    name: str
    description: str
    file_path: str
    abs_skill_dir: Path
    abs_skill_md: Path
    compatibility: str | None = None
    suggested_tools: list[str] = field(default_factory=list)


@dataclass
class SkillCatalogEntry:
    source_root: Path
    abs_skill_dir: Path
    abs_skill_md: Path
    frontmatter: dict[str, Any]
    meta: SkillMeta


__all__ = ["SkillMeta", "SkillCatalogEntry"]
