from __future__ import annotations

from .models import SkillMeta


def _render_skill_line(skill: SkillMeta, *, mounted_skill_tokens: set[str]) -> list[str]:
    suggested = ", ".join(f"`{name}`" for name in skill.suggested_tools) if skill.suggested_tools else "(none specified)"
    lines = [
        f"- `{skill.name}`:",
        f"  {skill.description}",
        f"  Suggested tools: {suggested}",
    ]
    if skill.mount_token in mounted_skill_tokens:
        lines.append(f"  File: `{skill.mount_token}/{skill.name}/SKILL.md`")
    else:
        lines.append("  File: (skills mount unavailable in this invocation)")
    return lines


def render_skills_addendum(*, role: str, skills: list[SkillMeta], mounted_skill_tokens: set[str]) -> str:
    _ = role
    lines: list[str] = [
        "## Skills",
        "",
        "A skill is a set of local instructions stored in a `SKILL.md` file.",
        "The available skills list includes the skill name, description, suggested tools, and file-access status so the source can be accessed when that skill is used.",
        "",
        "## Available Skills",
        "",
    ]

    if skills:
        for skill in skills:
            lines.extend(_render_skill_line(skill, mounted_skill_tokens=mounted_skill_tokens))
    else:
        lines.append("- (no role-visible skills found)")

    discovery_lines = [
        "### Discovery",
        "- The list above is the skills available for this role in this invocation.",
    ]
    if mounted_skill_tokens:
        rendered_roots = ", ".join(f"`{token}/`" for token in sorted(mounted_skill_tokens))
        discovery_lines.append(f"- Skills are mounted read-only at {rendered_roots}.")
    else:
        discovery_lines.append("- Skills are not mounted in this invocation, so skill-mount paths are unavailable.")

    progressive_lines = [
        "### Progressive Disclosure Workflow",
    ]
    if mounted_skill_tokens:
        example_token = sorted(mounted_skill_tokens)[0]
        progressive_lines.extend(
            [
                f"1. After deciding to use a skill, read `{example_token}/<skill-name>/SKILL.md` with standard filesystem read tools.",
                "2. Resolve relative references against that skill directory first.",
                "3. For shell commands, remember the default cwd is the project root; resolve any skill-relative paths against the skill directory before running them.",
                "4. Load only needed assets under that skill directory (for example `references/` or `scripts/`).",
                "5. Prefer running or patching referenced scripts over retyping large blocks.",
                "6. Reuse templates/assets when available instead of recreating from scratch.",
            ]
        )
    else:
        progressive_lines.extend(
            [
                "1. If a skill file cannot be read because the skills mount is unavailable, rely on the summary above and continue with the best fallback.",
                "2. Do not claim to have opened skill-mount paths when the mount is unavailable.",
            ]
        )

    lines.extend(
        [
            "",
            "## How to Use Skills",
            "",
            *discovery_lines,
            "",
            "### Trigger Rules",
            "- If the user names a skill (with `$SkillName` or plain text), use that skill.",
            "- If the task clearly matches a listed skill description, use that skill.",
            "- If multiple skills clearly apply, use the minimal set that covers the request.",
            "- Do not assume a previously loaded skill is still the right one for the current step. If a skill is needed again, reload it explicitly.",
            "",
            "### Missing or Blocked Skill",
            "- If a named skill is not in the list or cannot be read, say so briefly and continue with the best fallback.",
            "",
            *progressive_lines,
            "",
            "### Coordination and Sequencing",
            "- If multiple skills apply, pick a minimal ordered subset and follow that order.",
            "- State which skill(s) you are using and why in one short line.",
            "",
            "### Context Hygiene",
            "- Keep context small: summarize long sections instead of pasting full bodies.",
            "- Avoid deep reference chasing unless directly needed to unblock progress.",
            "- When variants exist, load only the relevant reference files and note that choice.",
            "",
            "### Safety and Fallback",
            "- If a skill cannot be applied cleanly, state the issue, choose the next-best approach, and continue.",
        ]
    )
    return "\n".join(lines).strip()


__all__ = ["render_skills_addendum"]
