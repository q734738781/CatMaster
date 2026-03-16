from __future__ import annotations

from pathlib import Path
import re

from catmaster.runtime.skills import CatMasterSkillsRuntime, SkillCatalog


def _write_skill(
    *,
    root: Path,
    name: str,
    description: str,
    suggested_tools: str | None,
    body_lines: list[str] | None = None,
) -> None:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    frontmatter = [
        "---",
        f"name: {name}",
        f"description: {description}",
        "compatibility: local",
    ]
    if suggested_tools is not None:
        frontmatter.extend(
            [
                "metadata:",
                f'  catmaster-suggested-tools: "{suggested_tools}"',
            ]
        )
    frontmatter.extend(
        [
            "---",
            "",
            f"# {name}",
        ]
    )
    if body_lines:
        frontmatter.extend(["", *body_lines])
    (skill_dir / "SKILL.md").write_text(
        "\n".join(frontmatter),
        encoding="utf-8",
    )


def test_skill_catalog_parses_frontmatter_and_role_visibility(tmp_path: Path) -> None:
    _write_skill(
        root=tmp_path,
        name="skill-alpha",
        description="alpha skill",
        suggested_tools="tool_a tool_b tool_a",
    )

    mismatch_dir = tmp_path / "skills" / "mismatch"
    mismatch_dir.mkdir(parents=True, exist_ok=True)
    (mismatch_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: wrong-name",
                "description: should be skipped",
                "---",
            ]
        ),
        encoding="utf-8",
    )

    catalog = SkillCatalog(
        source_roots=[tmp_path / "skills"],
        repo_root=tmp_path,
    )
    metas = catalog.refresh()
    assert [item.name for item in metas] == ["skill-alpha"]
    meta = metas[0]
    assert meta.file_path == "skills/skill-alpha/SKILL.md"
    assert meta.mount_token == "@skills"
    assert meta.suggested_tools == ["tool_a", "tool_b"]

    runtime = CatMasterSkillsRuntime(
        catalog=catalog,
        role_skill_names={"proposal": ["skill-alpha"]},
    )
    assert [item.name for item in runtime.visible_skills("proposal", "standard")] == ["skill-alpha"]
    assert runtime.visible_skills("task_runner", "standard") == []


def test_default_catalog_discovers_starter_skills() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SkillCatalog.create_default(repo_root=repo_root)
    metas = catalog.refresh()
    names = {item.name for item in metas}

    assert len(metas) >= 8
    assert "slab-construction-and-surface-modeling" in names
    assert "adsorption-site-screening" in names
    assert "vasp-input-preparation" in names


def test_default_catalog_discovers_writing_skills() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SkillCatalog.create_default(repo_root=repo_root)
    metas = {item.name: item for item in catalog.refresh()}
    assert "scientific-section-synthesis" in metas
    assert metas["scientific-section-synthesis"].mount_token == "@writing_skills"
    assert metas["scientific-section-synthesis"].lanes == ["writing"]
    assert "section_writer" in metas["scientific-section-synthesis"].roles
    assert "achemso-latex-manuscript" in metas
    assert "write_director" in metas["achemso-latex-manuscript"].roles


def _body_suggested_tools(text: str) -> list[str]:
    match = re.search(
        r"^## Suggested tools\s*$\n(?P<body>.*?)(?:^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        return []
    tools: list[str] = []
    for raw in match.group("body").splitlines():
        line = raw.strip()
        if not line.startswith("- "):
            continue
        token = line[2:].strip().strip("`")
        if not token or token in {"(none specified)", "none specified"}:
            continue
        tools.append(token)
    return tools


def test_starter_skills_use_standard_sections_and_consistent_suggested_tools() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SkillCatalog.create_default(repo_root=repo_root)
    by_name = {item.name: item for item in catalog.refresh()}

    required_sections = [
        "## Overview",
        "## Quick Start",
        "## Suggested tools",
        "## Workflow",
        "## Output Contract",
        "## References",
    ]

    skill_root = repo_root / "skills"
    skill_paths = sorted(path for path in skill_root.glob("*/SKILL.md"))
    assert skill_paths, "expected starter skills under skills/*/SKILL.md"

    for path in skill_paths:
        name = path.parent.name
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"{path} should start with YAML frontmatter"
        assert f"# {name}" in text, f"{path} should use the skill directory name as the H1 title"
        for section in required_sections:
            assert section in text, f"{path} missing required section: {section}"

        body_tools = _body_suggested_tools(text)
        assert name in by_name, f"{path} missing from skill catalog"
        assert by_name[name].suggested_tools == body_tools, (
            f"{path} has mismatched suggested tools between catalog/frontmatter and body"
        )


def test_method_critical_defaults_sections_exist_for_targeted_skills() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    targeted = [
        "mace-screening-and-relaxation",
        "vasp-input-preparation",
        "thermo-free-energy-and-reporting",
        "transition-state-neb",
        "adsorption-site-screening",
    ]
    for name in targeted:
        path = repo_root / "skills" / name / "SKILL.md"
        text = path.read_text(encoding="utf-8")
        assert "## Method-critical defaults" in text, f"{path} missing method-critical defaults section"


def test_skill_catalog_falls_back_to_suggested_tools_section(tmp_path: Path) -> None:
    _write_skill(
        root=tmp_path,
        name="skill-beta",
        description="beta skill",
        suggested_tools=None,
        body_lines=[
            "## Overview",
            "Example",
            "",
            "## Suggested tools",
            "- `tool_x`",
            "- tool_y",
            "- (none specified)",
            "",
            "## Workflow",
            "1. Do the thing",
        ],
    )

    catalog = SkillCatalog(
        source_roots=[tmp_path / "skills"],
        repo_root=tmp_path,
    )
    metas = catalog.refresh()
    assert [item.name for item in metas] == ["skill-beta"]
    assert metas[0].suggested_tools == ["tool_x", "tool_y"]


def test_skill_catalog_normalizes_legacy_bash_exec_suggested_tool(tmp_path: Path) -> None:
    _write_skill(
        root=tmp_path,
        name="skill-gamma",
        description="gamma skill",
        suggested_tools="bash_exec tool_x bash_exec",
    )

    catalog = SkillCatalog(
        source_roots=[tmp_path / "skills"],
        repo_root=tmp_path,
    )
    metas = catalog.refresh()
    assert [item.name for item in metas] == ["skill-gamma"]
    assert metas[0].suggested_tools == ["bash", "tool_x"]
