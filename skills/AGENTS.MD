# Skill Authoring Guide

This directory contains CatMaster starter skills grouped by agent-facing skill root. A skill here is prompt-level SOP, not a tool schema dump and not a hidden runtime contract.

The active runtime loads these skills through DeepAgent skill roots:

- `skills/materials_worker/*`
- `skills/dynamics_worker/*`
- `skills/ml_worker/*`
- `skills/orca_xtb_worker/*`
- `skills/research_specialist/*`
- `skills/litreview_agent/*`
- `skills/writing_specialist/*`
- `skills/writing_quality/*` as a shared prose-quality root used by report-, review-, and manuscript-producing agents.
- `skills/execution/*` as a shared managed-execution support root used by execution workers.

## Required format

Each skill must live under one agent-facing root, such as `skills/materials_worker/<skill-name>/SKILL.md`, `skills/dynamics_worker/<skill-name>/SKILL.md`, `skills/ml_worker/<skill-name>/SKILL.md`, `skills/orca_xtb_worker/<skill-name>/SKILL.md`, `skills/research_specialist/<skill-name>/SKILL.md`, `skills/litreview_agent/<skill-name>/SKILL.md`, or `skills/writing_specialist/<skill-name>/SKILL.md`. Shared prose-quality skills may live under `skills/writing_quality/<skill-name>/SKILL.md`, and shared managed-execution SOPs may live under `skills/execution/<skill-name>/SKILL.md`. Every skill must satisfy all of the following:

- YAML frontmatter at the top.
- For CatMaster-local skills, `name` should match the directory name.
- For third-party imported skills, preserve the upstream directory name and frontmatter `name` exactly, even if they differ.
- `description` must say both what the skill is for and when to use it.
- Use this section order in the body:
  - `# <skill-name>`
  - `## Overview`
  - `## Quick Start`
  - `## Allowed tools`
  - `## Workflow`
  - `## Method-critical defaults`
  - `## Output Contract`
  - `## References`

## Frontmatter rules

Use this pattern:

```yaml
---
name: slab-construction-and-surface-modeling
description: Use this skill for slab construction, vacuum and layer choices, surface supercell setup, and atom-fixing strategy in heterogeneous catalysis workflows.
license: project-local
compatibility: local
allowed-tools: "build_slab fix_atoms_by_layers fix_atoms_by_height supercell"
---
```

Rules:

- `name` and `description` are required.
- `license`, `compatibility`, and `allowed-tools` may be used when they add useful routing or invocation context.
- `allowed-tools` is optional. Do not add it to broad, route-like, imported, literature, writing, or policy skills just to satisfy a format convention.
- For CatMaster-local tool SOP skills that do declare `allowed-tools`, list concrete active-path tool names separated by spaces, not commas.
- If a skill declares `allowed-tools` and also has a body `## Allowed tools` list, keep the two in agreement.
- Declared tool names must match real active-path tool names or DeepAgent built-ins. Do not leave legacy names such as removed tools or stale aliases in new or updated skills.

## Writing rules

Skills must be grounded in real tool behavior.

Do:

- Read the relevant tool source before editing the skill.
- Read the current specialist tool surface when the skill is lane-specific, so the suggested tools reflect tools that are actually visible to the specialist or subagent.
- Capture hard constraints, output files, batch summaries, and common failure modes.
- Mention exact returned keys or artifacts when they are important handoff contracts.
- Write a `## Method-critical defaults` section whenever the workflow contains settings that affect comparability, ranking, interpretation, or scientific validity.
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
- Which method-critical settings must be explicit rather than silently inherited from tool defaults?
- What files or metadata produced by the tool must be preserved?
- What common misuse would cause a bad run or misleading result?
- What must be returned at handoff time?

## Section intent

- `Overview`: one tight sentence on scope.
- `Quick Start`: the shortest correct execution path.
- `Allowed tools`: optional. Include it only for skills that intentionally guide concrete tool selection; use only real active-path tool names or DeepAgent built-ins.
- `Workflow`: the real SOP, including non-obvious defaults and traps.
- `Method-critical defaults`: the settings that must be chosen and reported explicitly when they affect comparability, ranking, or interpretation.
- `Output Contract`: the minimum handoff evidence expected from a good run.
- `References`: only extra detail or downstream handoff guidance.

## Skill philosophy

### Skills carry SOP, tools carry capability interfaces

Runtime skills should carry:
- task-specific SOP
- parameter conventions
- evidence standards
- common pitfalls
- method-critical defaults

Tools should remain capability interfaces, not embedded SOP containers.

### Method-critical settings must be explicit

For quantitative scientific workflows, do not silently rely on tool defaults for method-critical settings when they affect comparability, ranking, or interpretation.

Such settings should be made explicit in task packets and/or skill-guided execution, and must remain consistent across reference and target calculations when required by the scientific question.

## Maintenance

When changing a tool in a way that affects SOP, outputs, or constraints, update the corresponding skill in the same change.

When changing a specialist or subagent tool surface, audit affected skill `allowed-tools` fields as part of the same change.
