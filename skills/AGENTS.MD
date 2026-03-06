# Skill Authoring Guide

This directory contains CatMaster starter skills. A skill here is prompt-level SOP, not a tool schema dump and not a hidden runtime contract.

## Required format

Each skill must live at `skills/<skill-name>/SKILL.md` and must satisfy all of the following:

- YAML frontmatter at the top.
- `name` must exactly match the directory name.
- `description` must say both what the skill is for and when to use it.
- Use this section order in the body:
  - `# <skill-name>`
  - `## Overview`
  - `## Quick Start`
  - `## Suggested tools`
  - `## Workflow`
  - `## Output Contract`
  - `## References`

## Frontmatter rules

Use this pattern:

```yaml
---
name: slab-construction-and-surface-modeling
description: Use this skill for slab construction, vacuum and layer choices, surface supercell setup, and atom-fixing strategy in heterogeneous catalysis workflows.
compatibility: Designed for CatMaster local tools and project-space relative-path execution.
metadata:
  catmaster-suggested-tools: "build_slab fix_atoms_by_layers fix_atoms_by_height supercell"
---
```

Rules:

- `metadata.catmaster-suggested-tools` should list tool names separated by spaces.
- The frontmatter suggested-tools list and the body `## Suggested tools` list must agree.
- If no tool should be suggested, use an empty frontmatter string and body text `- (none specified)`.

## Writing rules

Skills must be grounded in real tool behavior.

Do:

- Read the relevant tool source before editing the skill.
- Capture hard constraints, output files, batch summaries, and common failure modes.
- Mention exact returned keys or artifacts when they are important handoff contracts.
- Keep paths project-relative.
- Keep `Quick Start` to 3-5 lines.
- Keep `Workflow` to 3-5 subsections with concrete decisions.

Do not:

- Repeat the frontmatter description in multiple sections.
- Copy full tool schemas into the skill.
- Explain obvious tool names or restate every parameter.
- Use generic filler like "keep things consistent" unless you say what must stay consistent.
- Hide important caveats in `References`.

## Quality bar

A good skill should answer these questions quickly:

- Which tool should be used first?
- What invariants must stay fixed across the workflow?
- What files or metadata produced by the tool must be preserved?
- What common misuse would cause a bad run or misleading result?
- What must be returned at handoff time?

## Section intent

- `Overview`: one tight sentence on scope.
- `Quick Start`: the shortest correct execution path.
- `Suggested tools`: only the tools that should genuinely be surfaced for selection.
- `Workflow`: the real SOP, including non-obvious defaults and traps.
- `Output Contract`: the minimum handoff evidence expected from a good run.
- `References`: only extra detail or downstream handoff guidance.

## Maintenance

When changing a tool in a way that affects SOP, outputs, or constraints, update the corresponding skill in the same change.
