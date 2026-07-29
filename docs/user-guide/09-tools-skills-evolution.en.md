# 9. Project files, continuity, and reusable methods

[Previous](08-remote-execution.en.md) | [Contents](README.en.md) | [Next](10-deployment-operations.en.md)

CatMaster is most useful when a project accumulates structures, data, scripts, literature, manuscripts, and reviewable decisions across many sessions. The workspace holds that continuity. Users and agents share `files/`, while the system uses `metadata/` for threads, checkpoints, observability, and remote state.

## Organize around research, not code modules

Do not create directories named after tools or agents unless that matches the scientific project. A surface-catalysis workspace might evolve into:

```text
files/
  literature/
  structures/
    bulk/
    slabs/
    adsorption/
  calculations/
    bulk_reference/
    slab_screen/
    adsorption/
  data/
  scripts/
  notes/
  figures/
  writing/
```

Materials, Dynamics, and Writing can all use this layout. The same structure does not need to be copied into agent-specific folders. If an established project already has conventions, tell the agent to preserve them.

```text
This is an existing project. Read the files root, notes/project_conventions.md, and recent relevant results
to understand its layout, names, units, and versioning. Do not reorganize the project to match a CatMaster example.

In this turn, report your understanding of authoritative inputs, derived files, and ambiguities, and recommend
where later artifacts should go. Do not move, delete, or overwrite anything.
```

## Separate originals, derived results, and deliverables

Database downloads, instrument data, uploaded structures, and manuscript sources are originals. Preserve them with provenance. Standardized structures, filtered data, calculation stages, and generated figures are derivatives and should trace back to their inputs and methods. Final tables, figures, and reports should point to editable source and scripts.

An agent can maintain a manifest or local README with paths, dates, sources, parameters, and versions. It should not create a forest of empty directories and templates before the project contains work. Add documentation when there is something real to describe.

Keep an original structure and use names that express meaningful transformations, such as `ceo2_111_t0_raw.vasp`, `ceo2_111_t0_fixed.vasp`, and `ceo2_111_t0_pd_site03.vasp`. For large candidate sets, use a CSV or Markdown ledger rather than encoding every parameter in filenames.

## Reproducible scripts for project-specific work

Registered tools cover common operations, but research creates specialized analyses. A worker can use Python or shell for a bounded local step. Logic that will be reused, affects scientific conclusions, or handles a large batch should be saved under `scripts/` rather than hidden in one ephemeral command.

A reusable script should state its creation date, related agent, purpose, method, inputs, outputs, units, important parameters, and failure behavior. Reports should preserve the actual command or config used.

```text
Create a reusable script under scripts/ to analyze Pd-cluster connectivity in trajectories/run1.traj.
Parameterize input path, Pd-Pd cutoff, periodic boundaries, and frame stride rather than hard-coding this file.
Write per-frame components, largest-cluster size, and representative-frame indices.

Run a minimal validation on the current trajectory and document the command, cutoff rationale, outputs,
and limitations under notes/. Do not leave the implementation only inside one execute call.
```

## Artifacts connect conversation to project files

Files written by an agent can be registered as artifacts and appear as clickable cards in Chat. The inspector chooses a text, table, image, PDF, structure, or trajectory renderer from the file type. An artifact points to the real workspace file rather than duplicating it, so later moves or deletion affect the link.

Very long tool output is previewed in Chat and stored under `files/_tool_outputs/`. A final result should point to the full file or a clearer derived report rather than relying on a truncated preview.

Remote receipts are important artifacts as well. They connect a local stage, remote job, and transfer state. Do not treat all of `files/.deepagents/` as disposable cache when a project contains recoverable submissions.

## Project memory stores stable conventions

Workspace memory is for information that should influence future tasks: fixed energy references, naming rules, units, Selective Dynamics policy, or durable writing preferences. A temporary SSH failure, a one-off path, current progress, or an unverified mechanism belongs in the thread, log, or stage report instead.

The more memory resembles a concise project convention document, the more reliably later agents can use it. A transcript dump makes future decisions worse.

## Skill Evolution turns repeated methods into project capability

A skill is appropriate when a full workflow repeats. If a stepped CeO2 project has repeatedly validated one termination audit, atom naming rule, fixed-layer policy, and report structure, the system can propose a workspace skill containing a complete `SKILL.md` and, where needed, references or scripts.

The system does not turn every completed run into a skill, but it does send each terminal run with a user task through one semantic reflection. The reflection model reads the complete recorded model results, tool inputs, tool results, and final result. It distinguishes `no_change`, a failure to follow an adequate existing skill, and evidence that durable behavior should change. The first two outcomes create neither an observation nor a candidate. CatMaster does not use keywords, regular expressions, embeddings, or a fixed recurrence count to make that decision. One explicit durable correction may be sufficient; repeated wording is not sufficient by itself. Tool-schema defects, detailed scientific notes, and one-off facts remain in their authoritative stores instead of being disguised as skills. When an existing skill owns the behavior, CatMaster amends it rather than growing a duplicate.

The reflection model selects one exact target for an actionable signal. CatMaster combines only evidence with that exact target across threads and reopens the complete semantic trajectory for every related episode. There is no similarity cluster or minimum episode count. The proposer can still return ignore when the full evidence is insufficient.

Every candidate revision is immutable. CatMaster does not generate test prompts or start extra conversations to compare variants. Static validation covers mechanical contracts such as paths, bundle structure, frontmatter, references, code, the final tool registry, and hashes. The independent reviewer reads the same complete evidence and exact diff. Its `approve`, `reject`, or `needs_revision` value is advisory and cannot authorize release. Request revision creates a new revision without altering the reviewed one.

Candidate cards show the behavior change, evidence episodes and sources, applicability and non-applicability, static validation, reviewer counterexamples and concerns, and the available human actions. The lifecycle has seven states: `pending`, `review`, `revision`, `canary`, `stable`, `rejected`, and `inactive`. The reviewed diff is loaded only from the exact revision under Technical details; raw event JSON and internal payloads remain in gated Developer Diagnostics. Lists are newest first and support status filters and cursor pagination.

A logged-in user may request another revision, reject it, or start a skill canary on explicitly named threads or runs. The canary pins that exact revision while the stable version remains the default elsewhere. Starting it does not create a conversation, copy a prompt, or trigger another model call. Stable promotion is unavailable until the normal research run both reads or uses that revision and succeeds without false activation. A failure stops only that canary pointer. Stable revisions can later be quarantined, retired, or rolled back without deleting their immutable bundles. Workspace-memory preferences follow the same static-validation and human-review boundary but do not need a skill canary.

Good skill candidates include stable project-specific QC methods, directory and delivery contracts, a verified stage-and-result workflow for a remote task, or repeated writing and figure conventions. Temporary errors, one sample-specific threshold, fixed atom indices, incidental checksums, and unverified scientific conclusions should not become skills. A skill changes method guidance, not tool permissions or remote availability.

```text
Review the last three slab tasks and their audit reports in this workspace.
Identify conventions that genuinely repeated and were independently validated. Separate stable rules
from choices that belong only to one structure.

If a reusable workflow exists, judge its cause from the complete episodes and results first.
Prefer amending the owning skill. State applicability, non-applicability, and the exact expected decision change.
Do not turn one cutoff, atom index, or incidental checksum into a universal rule.
Keep every revision in human review; use an explicitly scoped canary before stable release.
```

## Resume facts before resuming a plan

When returning to an old thread, checkpoints provide context, but current files are authoritative. Ask the agent to reread core artifacts and determine which work really completed, which files are incomplete, which remote tasks still exist, and which decisions remain open.

```text
Continue this project. Reread notes/progress.md, calculations/summary.csv, recent receipts,
and relevant stages. Do not accept a chat statement of completion without checking the files.

Report confirmed completions, failed or incomplete items, remote jobs still active, and decisions I still own.
Preserve every successful result and forbid duplicate computation. Recommend the next stage only after restoring facts.
```

If the main objective changes, such as moving from surface computation to manuscript writing, create a Writing thread and pass a result contract, tables, figures, and bibliography. A clean evidence package is more reliable than a long transcript summary.

## What to back up

A complete workspace recovery requires both `files/` and `metadata/`. Backing up `files/` preserves scientific artifacts but loses thread checkpoints, approval state, observability, and some artifact indices. Login deployments also need `<PROJECT_SPACE_ROOT>/.webui_auth/auth.sqlite`.

Back up while the WebUI is stopped or no run is writing. Large trajectories and calculation outputs can use site-specific incremental policies, but retain receipts, manifests, reports, and critical config alongside the data. Deployment, permissions, and upgrade procedures are in the next chapter.
