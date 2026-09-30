from pathlib import Path

from catmaster.runtime.skills.catalog import CatMasterSkillsRuntime
from catmaster.runtime.skills.role_skills import role_visible_skill_names
from catmaster.tools.geometry_inputs.slab_tools import SlabBuildInput


REPO_ROOT = Path(__file__).resolve().parents[1]


def _skill_text(name: str) -> str:
    return (REPO_ROOT / "skills" / "materials_worker" / name / "SKILL.md").read_text(encoding="utf-8")


def test_materials_task_runner_sees_surface_and_adsorption_workflow_skills():
    names = set(role_visible_skill_names("task_runner"))

    assert "surface-and-termination-screening" in names
    assert "adsorption-screening" in names


def test_materials_task_runner_catalog_resolves_workflow_skills():
    runtime = CatMasterSkillsRuntime.create_default(repo_root=REPO_ROOT)
    runtime.refresh_catalog()
    names = {skill.name for skill in runtime.visible_skills("task_runner")}

    assert "surface-and-termination-screening" in names
    assert "adsorption-screening" in names


def test_slab_adsorption_guidance_preserves_method_critical_defaults():
    slab_text = _skill_text("slab-construction-and-surface-modeling")
    surface_text = _skill_text("surface-and-termination-screening")
    adsorption_text = _skill_text("adsorption-screening")

    assert "orthogonal=true" in slab_text
    assert "Termination provenance review is mandatory" in slab_text
    assert "not a proof from one POSCAR" in slab_text
    assert "CN=1" in slab_text
    assert "CN=1" in surface_text
    assert "screening heuristic" in surface_text
    assert "termination-reviewed slab" in adsorption_text


def test_build_slab_schema_exposes_adsorption_orthogonal_preference():
    schema = SlabBuildInput.model_json_schema()
    description = schema["properties"]["orthogonal"]["description"]

    assert "adsorption-ready slabs" in description
    assert "compared terminations" in description
