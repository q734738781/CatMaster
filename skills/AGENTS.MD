# Skill Authoring Guide

This directory contains CatMaster starter skills grouped by agent-facing skill root. A skill here is prompt-level SOP, not a tool schema dump and not a hidden runtime contract.

The active runtime loads these skills through DeepAgent skill roots:

- `skills/materials_worker/*`
- `skills/dynamics_worker/*`
- `skills/ml_worker/*`
- `skills/orca_xtb_worker/*`
- `skills/atomistic/*` as the shared structure-assembly, geometry-validation,
  and reconstruction root for Experiment plus the coordinate-consuming workers.
- `skills/research_specialist/*`
- `skills/research_reasoning/*` as a shared scientific reasoning root for
  Result interpretation, hypothesis revision, direction recovery and independent stopping reconsideration.
  The reasoner can mutate scientific graph records while its filesystem remains read-only.
- `skills/litreview_agent/*`
- `skills/research_execution/*` as the shared post-result sampling and writeback SOP for Experiment and Literature Review agents, including direct entry outside Research. The host binds direct scientific entries to an existing selection or the newest unarchived workspace graph, including completed stages; a bound Experiment writes through an explicit Experiment focus, while Literature Review may add a sourced standalone Result. Unbound internal work returns an evidence packet.
- `skills/writing_specialist/*`
- `skills/presentation_worker/*` for editable presentation authoring and rendering.
- `skills/plot_worker/*` as the narrow direct-data plotting root for the Writing lane.
- `skills/writing_quality/*` as the shared scientific-communication and prose-quality root for Research, Experiment, Literature Review, Writing, review and presentation roles. Content guidance applies to narrative deliverables, not routine status or execution records. Scientific-communication is the shared prose entry; phrasing examples and presentation inspection guidance are read only when needed.
- `skills/execution/*` as a shared managed-execution support root used by execution workers.

A directory below one of these roots is not automatically a skill. Shared
definitions, examples, templates, fragments, and reference modules may remain
plain asset directories without `SKILL.md` or `manifest.yaml`; DeepAgents should
not discover them as standalone workflows. A real skill may read a specific
asset when useful without turning that asset package into another skill.

## Loader contract and suggested format

Each skill must live under one agent-facing root, such as `skills/materials_worker/<skill-name>/SKILL.md`, `skills/dynamics_worker/<skill-name>/SKILL.md`, `skills/ml_worker/<skill-name>/SKILL.md`, `skills/orca_xtb_worker/<skill-name>/SKILL.md`, `skills/atomistic/<skill-name>/SKILL.md`, `skills/research_specialist/<skill-name>/SKILL.md`, `skills/research_reasoning/<skill-name>/SKILL.md`, `skills/litreview_agent/<skill-name>/SKILL.md`, `skills/research_execution/<skill-name>/SKILL.md`, `skills/writing_specialist/<skill-name>/SKILL.md`, or `skills/plot_worker/<skill-name>/SKILL.md`. Shared prose-quality skills may live under `skills/writing_quality/<skill-name>/SKILL.md`, and shared managed-execution SOPs may live under `skills/execution/<skill-name>/SKILL.md`.

The active DeepAgents loader is the source of truth. Its functional minimum is
a readable `SKILL.md` with YAML frontmatter containing `name` and `description`.
If that loader reports an error, expose the exact diagnostic to the authoring
agent so it can repair the file. Do not replace loader behavior with a stricter
CatMaster format gate.

Within `skills/writing_specialist`, keep `publication-launch-writing` as the manuscript-construction entry. Use its on-demand reporting reference and the existing venue templates when required; do not add an overlapping manuscript workflow. Shared `scientific-communication` guidance distinguishes reports, literature syntheses and progress talks without fixing a chapter order or layout. Customer-validated Origin/NPG, prose-quality and rendered-artifact requirements remain authoritative. The presentation worker can read the plotting root for direct data-figure work; EasySlides handles deck production while layout remains task-specific.

- YAML frontmatter at the top with `name` and `description`.
- For CatMaster-local skills, matching `name` to the directory is a useful
  convention, not a loading or review requirement.
- For third-party imported skills, preserve upstream organization when useful.
- A concise `description` that says what the skill is for and when to use it
  usually improves discovery.
- The following body layout is a useful starting point, not a required order or
  completeness checklist:
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
---
```

Rules:

- `name` and `description` are required by the active loader.
- `license` and `compatibility` are optional descriptive metadata.
- `allowed-tools` is inert upstream-compatible metadata. CatMaster does not
  parse, audit, validate, or use it for discovery, loading, visibility, or tool
  authorization. Existing declarations may remain, but authors should not add
  them merely to imply access control.
- A body `## Allowed tools` section is ordinary SOP prose. It may help an agent
  choose a capability, but it is not reconciled with frontmatter or a registry.
- Actual tool access is determined only by the specialist binding and explicit
  runtime tool policy. Skill text cannot grant or revoke a tool.

## Progressive disclosure

Descriptions are for selection, not a startup checklist. Keep essential decisions,
scientific constraints and stopping conditions in `SKILL.md`. Put substantial
operation-specific syntax, schemas and examples in linked references and state
when each is needed. Reuse guidance already available in the active context.
Keep references and helper scripts fully reachable; do not hide skills or truncate
content to meet a token target. A skill may be short and self-contained.

Preserve internal knowledge that the model cannot infer from the visible tool
schema: file layouts, cross-field constraints, ownership, continuation and recovery
semantics. For task-specific contracts, link directly to the matching reference
and require it before the unfamiliar operation; do not make failed calls the
discovery mechanism. Keep a short shared invariant in the entrypoint when missing
it could cause the agent to choose the wrong operation before opening a reference.
Use live tool/spec queries for accepted fields and defaults when the recipient
can access them. Remove a duplicate only after checking that authoritative source
on the recipient's actual tool surface; retain needed Markdown documentation for
internal formats not described there. Judge simplification by correct completion and total
work, including discovery and retries, not entrypoint length alone.

Avoid duplicating the system prompt, tool schema, or another skill's procedure.
In particular, bibliography and template workflows must not launch unrelated
illustration, research or review work. Preserve scientific methods, evidence
standards, user-validated writing quality and real authorization boundaries.

## Executable resources

Bundled utilities in `scripts/`, `references/`, or another support directory may
be executed directly. Filesystem tools read `/.deepagents/skills/<relative-path>`;
the local shell uses `"$CATMASTER_SKILLS_ROOT/<relative-path>"`, bound to that run's
snapshot. Keep input/output paths in the workspace. Document the callable usage
and internal input contract; do not require source reading, copying, or packaging
before ordinary execution. Copy only when implementation changes are needed.

Use callable scripts to cover capabilities missing from registered tools. A
readable mount alone does not provide shell execution: document and verify the
real command path with workspace inputs and outputs. Importable modules may stay
modules; they need an import example, not a forced CLI or packaging step.

Diagnostic scripts should print status, actionable failure quantities and the
full report path. Keep pair tables, per-atom arrays and full batch metrics in the
report, reachable through normal reads and queries. Custom wrappers should not
embed those dictionaries in exception messages. Query/export scripts may return
requested data directly; use their output option or redirection for large batches.

Use `--verbose` for repetitive progress or an additional console copy of a saved
diagnostic report. Keep status, failure signals and retrieval paths visible by
default. Do not hide the only copy of a scientific result or a necessary error
behind a verbosity flag, or add flags to already concise tools without a concrete
output problem.

## Writing rules

Skills must be grounded in real tool behavior.

Explain only what the current caller needs to choose and perform the operation.
Keep method, input/output, ownership and recovery semantics; omit backend
frameworks, queue/checkpoint implementation and inaccessible agent hierarchy.
Name another role only when this recipient can actually call or inspect it.
Developer architecture explanations belong outside runtime skills.

Do:

- Read the relevant tool source before editing the skill.
- Read the current specialist tool surface when the skill is lane-specific, so the suggested tools reflect tools that are actually visible to the specialist or subagent.
- Capture hard constraints, output files, batch summaries, and common failure modes.
- Mention exact returned keys or artifacts when they are important handoff contracts.
- Make method-critical settings explicit whenever they affect comparability,
  ranking, interpretation, or scientific validity; use whatever organization
  makes the SOP clearest.
- Prefer deliberately incomplete, task-specific syntax fragments over complete scientific example inputs. Keep unrelated methods, structures, potentials, basis sets, solvents, and numerical values out of an operation example; place a concrete value only in method-critical guidance where choosing that value is the subject.
- Keep paths project-relative.
- Keep quick-start and workflow material compact enough to scan, while allowing
  the task to determine their structure and length.
- Specify one persisted format per logical scientific artifact. Skill examples
  must not default to equivalent BibTeX/JSON/Markdown, PNG/PDF/SVG/TIFF, or
  structure-format bundles; use a one-format example and reserve multiple
  formats for an explicit user request or a real downstream contract. Editable
  sources and requested renders may serve distinct authoring and delivery roles.

Do not:

- Repeat the frontmatter description in multiple sections.
- Copy full tool schemas into the skill.
- Explain obvious tool names or restate every parameter.
- Present a complete runnable scientific example as though its bundled settings were transferable defaults.
- Use generic filler like "keep things consistent" unless you say what must stay consistent.
- Hide selection-critical caveats behind references with no read-before-use trigger.

## Quality bar

A good skill should answer these questions quickly:

- When does this guidance apply, and what decision does it help the agent make?
- What invariants must stay fixed across the workflow?
- Which method-critical settings must be explicit rather than silently inherited from tool defaults?
- What files or metadata produced by the tool must be preserved?
- What common misuse would cause a bad run or misleading result?
- What must be returned at handoff time?

## Suggested section intent

- `Overview`: one tight sentence on scope.
- `Quick Start`: the shortest correct execution path.
- `Allowed tools`: optional. Include it only for skills that intentionally guide concrete tool selection; use only real active-path tool names or DeepAgent built-ins.
- `Workflow`: the real SOP, including non-obvious defaults and traps.
- `Method-critical defaults`: the settings that must be chosen and reported explicitly when they affect comparability, ranking, or interpretation.
- `Output Contract`: the minimum handoff evidence expected from a good run.
- `References`: task-specific contracts, syntax and examples with explicit reading triggers.

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

### Keep scientific provenance separate from operations

- Preserve scientific inputs, structures, model/checkpoint or parameter identity, physical conditions, method settings, analyses, evidence sources, and scientific results.
- Hardware/device identity, MPI/OpenMP or scheduler layout, software build, engine/framework/schema/manifest versions, runtime environment, executable/module/queue details, receipt IDs, access/license state, and performance telemetry belong to runtime records. Do not require or surface them in ordinary scientific QC, Result writeback, or successful-run handoffs. An explicit user request overrides this default; otherwise inspect only the smallest relevant field for a concrete failure or known result-changing compatibility issue.
- Do not calculate or compare hashes/checksums merely because work uses transfer, checkpoint, retry, an archive, or a remote task. Use a hash only when the user explicitly asks or an already-existing machine protocol actually requires file identity; keep it internal and do not add per-file digest inventories or parallel integrity gates.
- Do not treat ordinary workspace documents, structures, slabs, models, stages, or caches as hostile network inputs. Agent edits are visible workspace actions, not tampering events; do not add proactive SHA-256 validation or a secure-manifest pattern to skills or tools to police them.
- Existing tool/API/file-format contracts remain binding, but their internal schema/version fields need not enter agent context. Create an explicit contract artifact when the user requests one; do not create an ad hoc schema, secure manifest, baseline, lockfile, compatibility matrix, environment-equivalence gate, or repeated validation layer merely to formalize a one-off task.

### Asset and skill tests

Tests may verify that the runtime stages skill roots and leaves referenced asset
paths readable. They must not freeze optional asset imports, exact
`always_load` lists, or the complete names of editable reference files, and
must not require a shared asset directory to contain `SKILL.md` merely so a
test can discover it. A specific asset belongs in a behavioral test only when
its presence is an explicit stable product contract; otherwise let the agent
choose among the available assets at runtime.

## Maintenance

Every tool change requires reviewing its companion usage guidance in the same
change. Update all affected entrypoints, shared skills, linked references and
examples, not just one matching SKILL.md. Check defaults and override precedence,
source/output paths, batch selection, errors and continuation against the actual
implementation and final model-visible schema. Replace contradictory old text
instead of appending a competing explanation. For multi-tool work, record each
tool's reviewed guidance and whether it changed, already matched, or has no
dedicated skill for a stated reason. A tool change is incomplete while affected
skill instructions remain stale. See the repository-wide Agent-Facing
Explanations and Tool-Skill Maintenance rules in ../AGENTS.md.

When a specialist or subagent tool surface changes, update any SOP prose whose
instructions became stale. Do not audit `allowed-tools` metadata as a loading or
authorization contract.
