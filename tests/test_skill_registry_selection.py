from __future__ import annotations

import pytest

from catmaster.skills.registry import SkillRegistry


def test_skill_registry_selection() -> None:
    registry = SkillRegistry.load_builtin_skills()
    if not registry.skills:
        pytest.skip("No skills loaded")
    skills = registry.select_skills("run vasp batch")
    assert any(skill.id == "vasp_batch" for skill in skills)
