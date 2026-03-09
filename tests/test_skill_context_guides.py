from __future__ import annotations

from pathlib import Path

from catmaster.runtime.skills.context_guides import (
    render_director_skill_guide,
    render_fast_director_skill_guide,
    render_proposal_skill_guide,
    render_research_lead_skill_guide,
    render_section_writer_skill_guide,
    render_write_director_skill_guide,
    render_write_reviewer_skill_guide,
)
from catmaster.runtime.skills.models import SkillMeta


def _skill(name: str, description: str, suggested_tools: list[str]) -> SkillMeta:
    root = Path("/tmp") / name
    return SkillMeta(
        name=name,
        description=description,
        file_path=f"skills/{name}/SKILL.md",
        abs_skill_dir=root,
        abs_skill_md=root / "SKILL.md",
        source_root_name="skills",
        mount_token="@skills",
        suggested_tools=suggested_tools,
    )


def test_proposal_skill_guide_stays_at_skill_level() -> None:
    text = render_proposal_skill_guide(
        [
            _skill(
                "slab-construction-and-surface-modeling",
                "Slab construction and fixing strategy.",
                ["build_slab", "fix_atoms_by_layers"],
            )
        ]
    )
    assert "role-visible skills" in text
    assert "slab-construction-and-surface-modeling" in text
    assert "Slab construction and fixing strategy." in text
    assert "Suggested tools:" not in text


def test_director_skill_guides_include_soft_tool_hints() -> None:
    skill = _skill(
        "mace-screening-and-relaxation",
        "Rapid MACE screening before DFT refinement.",
        ["mace_relax_batch", "mace_sp_batch"],
    )
    director_text = render_director_skill_guide([skill])
    fast_text = render_fast_director_skill_guide([skill])

    assert "Suggested tools: mace_relax_batch, mace_sp_batch" in director_text
    assert "Suggested tools: mace_relax_batch, mace_sp_batch" in fast_text
    assert "soft hints" in director_text
    assert "soft hints" in fast_text
    assert "task packet" in fast_text
    assert "Fast-lane execution should be framed" in fast_text


def test_research_lead_skill_guide_stays_planning_level() -> None:
    text = render_research_lead_skill_guide(
        [
            _skill(
                "literature-grounding",
                "Ground hypotheses against representative papers and benchmark conventions.",
                ["run_literature_research"],
            )
        ]
    )
    assert "Research-lane planning" in text
    assert "literature-grounding" in text
    assert "Suggested tools:" not in text


def test_writing_guides_reflect_role_purpose() -> None:
    skill = _skill(
        "scientific-section-synthesis",
        "Draft one evidence-grounded section at a time.",
        ["review_research_context", "read_research_pack"],
    )
    director_text = render_write_director_skill_guide([skill])
    section_text = render_section_writer_skill_guide([skill])
    reviewer_text = render_write_reviewer_skill_guide([skill])
    assert "section scope" in director_text
    assert "Suggested tools: review_research_context, read_research_pack" in section_text
    assert "unsupported claims" in reviewer_text
