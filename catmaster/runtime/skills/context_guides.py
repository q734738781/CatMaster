from __future__ import annotations

from .models import SkillMeta


def _dedupe_tools(tools: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in tools:
        item = str(raw or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _render_skill_lines(
    skills: list[SkillMeta],
    *,
    header: str,
    include_suggested_tools: bool,
) -> str:
    if not skills:
        return f"{header}\n\n- (no role-visible skills)"

    lines = [header, ""]
    for skill in skills:
        lines.append(f"- {skill.name}:")
        lines.append(f"  {skill.description}")
        if include_suggested_tools:
            tools = _dedupe_tools(list(skill.suggested_tools or []))
            if tools:
                lines.append(f"  Suggested tools: {', '.join(tools)}")
    return "\n".join(lines)


def render_proposal_skill_guide(skills: list[SkillMeta]) -> str:
    return _render_skill_lines(
        skills,
        header=(
            "Execution planning is organized around the following role-visible skills. "
            "Focus on stage-level capabilities, scientific scope, and evidence contracts rather than raw tool-by-tool micro-details."
        ),
        include_suggested_tools=False,
    )


def render_director_skill_guide(skills: list[SkillMeta]) -> str:
    return _render_skill_lines(
        skills,
        header=(
            "Task-runner execution should be delegated through the following role-visible skills. "
            "Use suggested tools only as soft hints; encode method-critical requirements in the task packet."
        ),
        include_suggested_tools=True,
    )


def render_fast_director_skill_guide(skills: list[SkillMeta]) -> str:
    return _render_skill_lines(
        skills,
        header=(
            "Fast-lane execution should be framed through the following role-visible skills. "
            "Prefer minimal stage-appropriate delegation over raw tool micro-planning. "
            "Use suggested tools only as soft hints; encode method-critical requirements in the task packet."
        ),
        include_suggested_tools=True,
    )


def render_research_lead_skill_guide(skills: list[SkillMeta]) -> str:
    return _render_skill_lines(
        skills,
        header=(
            "Research-lane planning can rely on the following role-visible skills. "
            "Use them as campaign-level scientific guidance rather than tool encyclopedias."
        ),
        include_suggested_tools=False,
    )


def render_write_director_skill_guide(skills: list[SkillMeta]) -> str:
    return _render_skill_lines(
        skills,
        header=(
            "Writing-plan decisions can rely on the following role-visible skills. "
            "Use them to shape section scope, figure intent, citation discipline, and manuscript structure."
        ),
        include_suggested_tools=False,
    )


def render_section_writer_skill_guide(skills: list[SkillMeta]) -> str:
    return _render_skill_lines(
        skills,
        header=(
            "Section drafting can rely on the following role-visible skills. "
            "Treat suggested tools as soft hints; keep claims grounded in retrieved evidence, artifacts, and figure outputs."
        ),
        include_suggested_tools=True,
    )


def render_write_reviewer_skill_guide(skills: list[SkillMeta]) -> str:
    return _render_skill_lines(
        skills,
        header=(
            "Writing review can rely on the following role-visible skills. "
            "Focus on unsupported claims, citation gaps, structure, and evidence discipline."
        ),
        include_suggested_tools=False,
    )


__all__ = [
    "render_proposal_skill_guide",
    "render_director_skill_guide",
    "render_fast_director_skill_guide",
    "render_research_lead_skill_guide",
    "render_write_director_skill_guide",
    "render_section_writer_skill_guide",
    "render_write_reviewer_skill_guide",
]
