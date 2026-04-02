from __future__ import annotations

from pathlib import Path
import re

from catmaster.runtime.skills import CatMasterSkillsRuntime, SkillCatalog


def _write_skill(
    *,
    root: Path,
    name: str,
    description: str,
    allowed_tools: str | None,
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
    if allowed_tools is not None:
        frontmatter.append(f'allowed-tools: "{allowed_tools}"')
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
        allowed_tools="tool_a tool_b tool_a",
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
    assert meta.allowed_tools == ["tool_a", "tool_b"]

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
    assert "scientific-writing" in metas
    assert metas["scientific-writing"].mount_token == "@writing"
    assert metas["scientific-writing"].lanes == ["writing"]
    assert "section_writer" in metas["scientific-writing"].roles
    assert "achemso-latex-manuscript" in metas
    assert "write_director" in metas["achemso-latex-manuscript"].roles


def test_default_catalog_assigns_nonwriting_visibility_for_experiment_and_ml_skills() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runtime = CatMasterSkillsRuntime.create_default(repo_root=repo_root)
    runtime.refresh_catalog()
    visible = {item.name for item in runtime.visible_skills("task_runner", "standard")}
    assert "bulk-relax-and-reference" in visible
    assert "mace-dataset-curation" in visible


def _body_allowed_tools(text: str) -> list[str]:
    match = re.search(
        r"^## Allowed tools\s*$\n(?P<body>.*?)(?:^## |\Z)",
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


def test_starter_skills_use_standard_sections_and_consistent_allowed_tools() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    catalog = SkillCatalog.create_default(repo_root=repo_root)
    by_name = {item.name: item for item in catalog.refresh()}

    required_sections = [
        "## Overview",
        "## Quick Start",
        "## Allowed tools",
        "## Workflow",
        "## Output Contract",
        "## References",
    ]

    skill_root = repo_root / "skills" / "experiment"
    skill_paths = sorted(path for path in skill_root.glob("*/SKILL.md"))
    assert skill_paths, "expected starter skills under skills/experiment/*/SKILL.md"

    for path in skill_paths:
        name = path.parent.name
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n"), f"{path} should start with YAML frontmatter"
        assert f"# {name}" in text, f"{path} should use the skill directory name as the H1 title"
        for section in required_sections:
            assert section in text, f"{path} missing required section: {section}"

        body_tools = _body_allowed_tools(text)
        assert name in by_name, f"{path} missing from skill catalog"
        assert by_name[name].allowed_tools == body_tools, (
            f"{path} has mismatched allowed tools between catalog/frontmatter and body"
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
        path = repo_root / "skills" / "experiment" / name / "SKILL.md"
        text = path.read_text(encoding="utf-8")
        assert "## Method-critical defaults" in text, f"{path} missing method-critical defaults section"


def test_skill_catalog_falls_back_to_allowed_tools_section(tmp_path: Path) -> None:
    _write_skill(
        root=tmp_path,
        name="skill-beta",
        description="beta skill",
        allowed_tools=None,
        body_lines=[
            "## Overview",
            "Example",
            "",
            "## Allowed tools",
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
    assert metas[0].allowed_tools == ["tool_x", "tool_y"]


def test_skill_catalog_prefers_allowed_tools_frontmatter(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "skill-delta"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "\n".join(
            [
                "---",
                "name: skill-delta",
                "description: delta skill",
                "allowed-tools: \"tool_a tool_b\"",
                "---",
                "",
                "# skill-delta",
                "",
                "## Allowed tools",
                "- tool_x",
                "- tool_y",
            ]
        ),
        encoding="utf-8",
    )

    catalog = SkillCatalog(
        source_roots=[tmp_path / "skills"],
        repo_root=tmp_path,
    )
    metas = catalog.refresh()
    assert [item.name for item in metas] == ["skill-delta"]
    assert metas[0].allowed_tools == ["tool_a", "tool_b"]


def test_repo_skills_use_allowed_tools_frontmatter_only() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for skill_md in sorted((repo_root / "skills").rglob("SKILL.md")):
        text = skill_md.read_text(encoding="utf-8")
        assert "catmaster-suggested-tools" not in text, f"{skill_md} still uses deprecated catmaster-suggested-tools"


def test_skill_catalog_normalizes_legacy_bash_exec_suggested_tool(tmp_path: Path) -> None:
    _write_skill(
        root=tmp_path,
        name="skill-gamma",
        description="gamma skill",
        allowed_tools="bash_exec tool_x bash_exec",
    )

    catalog = SkillCatalog(
        source_roots=[tmp_path / "skills"],
        repo_root=tmp_path,
    )
    metas = catalog.refresh()
    assert [item.name for item in metas] == ["skill-gamma"]
    assert metas[0].allowed_tools == ["execute", "tool_x"]
