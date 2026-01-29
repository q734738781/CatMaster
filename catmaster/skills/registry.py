from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import logging

try:  # optional dependency
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from catmaster.skills.types import Skill


@dataclass
class SkillRegistry:
    skills: list[Skill]

    @classmethod
    def load_builtin_skills(cls, path: str | Path | None = None) -> "SkillRegistry":
        base = Path(path) if path is not None else Path(__file__).resolve().parent / "builtin"
        skills: list[Skill] = []
        if not base.exists():
            return cls(skills)
        for entry in sorted(base.glob("*.md")):
            skill = _load_skill_file(entry)
            if skill is not None:
                skills.append(skill)
        return cls(skills)

    def select_skills(self, task_goal: str) -> list[Skill]:
        goal = (task_goal or "").lower()
        selected: list[Skill] = []
        for skill in self.skills:
            if not skill.keywords:
                continue
            for keyword in skill.keywords:
                if keyword.lower() in goal:
                    selected.append(skill)
                    break
        return selected


def _load_skill_file(path: Path) -> Skill | None:
    logger = logging.getLogger(__name__)
    if yaml is None:
        logger.warning("PyYAML not available; skipping skill file %s", path)
        return None
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        logger.warning("Failed to parse skill file %s: %s", path, exc)
        return None
    if not isinstance(raw, dict):
        logger.warning("Skill file %s must be a mapping", path)
        return None
    skill_id = str(raw.get("id", "")).strip()
    if not skill_id:
        logger.warning("Skill file %s missing id", path)
        return None
    description = str(raw.get("description", "")).strip()
    keywords = tuple(str(item) for item in (raw.get("keywords") or []))
    tool_allowlist = set(str(item) for item in (raw.get("tools") or []))
    prompt_snippet = str(raw.get("prompt", ""))
    return Skill(
        id=skill_id,
        description=description,
        tool_allowlist=tool_allowlist,
        prompt_snippet=prompt_snippet,
        keywords=keywords,
    )


__all__ = ["SkillRegistry"]
