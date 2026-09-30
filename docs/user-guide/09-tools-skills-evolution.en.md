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

Manifests, READMEs, and audit ledgers are not default outputs. Create one only when the user or an existing interface requires it, or when it carries decision-relevant information downstream. Put OCR text, conversions, exploratory snippets, one-off scripts, logs, and intermediate tables in `tmp/` (shown as `/tmp/` to file tools); downstream work ignores this scratch space unless it is explicitly referenced. Promote only results needed beyond the current task.

Keep an original structure and use names that express meaningful transformations, such as `ceo2_111_t0_raw.vasp`, `ceo2_111_t0_fixed.vasp`, and `ceo2_111_t0_pd_site03.vasp`. For large candidate sets, use a CSV or Markdown ledger rather than encoding every parameter in filenames.

## Reproducible scripts for project-specific work

Bundled skill utilities run directly without copying or packaging. File tools read
`/.deepagents/skills/<relative-path>`; shell commands use
`python "$CATMASTER_SKILLS_ROOT/<relative-path>"`, bound to the current run's
effective skill snapshot. Keep inputs and outputs in the workspace; copy source
only for implementation changes. Diagnostic scripts return status, actionable
violations and report paths, with complete scientific tables available in files.

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

Files written by an agent can be registered as artifacts and appear as clickable cards in Chat. The large preview chooses a text, table, image, PDF, structure, or trajectory renderer from the file type. An artifact points to the real workspace file rather than duplicating it, so later moves or deletion affect the link.

Very long tool output is previewed in Chat and stored under `files/_tool_outputs/`. A final result should point to the full file or a clearer derived report rather than relying on a truncated preview.

Remote receipts are important artifacts as well. They connect a local stage, remote job, and transfer state. Do not treat all of `files/.deepagents/` as disposable cache when a project contains recoverable submissions.

## Project memory stores stable conventions

Workspace memory is for information that should influence future tasks: fixed energy references, naming rules, units, Selective Dynamics policy, or durable writing preferences. A temporary SSH failure, a one-off path, current progress, or an unverified mechanism belongs in the thread, log, or stage report instead.

The more memory resembles a concise project convention document, the more reliably later agents can use it. A transcript dump makes future decisions worse.

## Skill Evolution turns repeated methods into project capability

After a local background run finishes, learning reads that turn's bound request,
final answer, and model/tool records. Starting a newer turn or selecting Learn on
an older answer does not change the selected evidence. Resumable interruptions
and errors wait for continuation to finish before automatic learning. Learning
notices appear in the originating chat without starting another research turn;
deferred, ignored, or failed outcomes remain available in Processing history.

A skill is appropriate when a full workflow repeats. If a stepped CeO2 project has repeatedly validated one termination audit, atom naming rule, fixed-layer policy, and report structure, the system can propose a workspace skill containing a complete `SKILL.md` and, where needed, references or scripts.

The system does not turn every completed run into a skill, but it does send each terminal user episode through one semantic reflection. If an episode is interrupted and resumed across several physical runs, the episode identity remains fixed and only its final terminal run becomes the reflection anchor; an explicit Learn action keeps the run the user actually selected. The reflection model receives a deterministic run index and queries the host-bound observability store for complete model responses, tool inputs, model-visible tool results, task boundaries, and the final outcome. Provider envelopes, encrypted transport, streaming deltas, and duplicate callback records are not part of this semantic view. It distinguishes `no_change`, an execution lapse, and evidence that durable behavior should change. When no durable SOP improvement is warranted, the job returns an explicit `no_change` with the reflector's stated reason; an execution lapse is recorded in the job outcome and creates neither an observation nor a candidate. Independently evidenced findings, including several findings that initially point to the same owner, can be returned in one batch and are processed separately, so one failed proposal or review does not discard the others. CatMaster does not use keywords, regular expressions, embeddings, or a fixed recurrence count to make that decision. One explicit durable correction may be sufficient; repeated wording is not sufficient by itself. The model judges whether a product, schema, scientific, or one-off issue implies a reusable SOP improvement; the host does not reject it by category. When an existing skill owns the behavior, CatMaster amends it rather than growing a duplicate.

Each actionable finding supplies an initial target anchor and cites complete event handles returned by queries. The proposer can inspect the complete mounted skill tree and authorized workspace history, then keep or correct that owner; the final resolved owner defines the unique candidate ID, lock, and revision chain. Pageable history includes observations, candidate revisions, reviews, jobs, exact-version read/helper/outcome contrasts, and authorized `run_ref` handles. A new pass starts with only the current open delta; absorbed history is not repeatedly serialized into evidence or parsed before agent startup. The proposer and independent reviewer receive a compact claim-and-handle index and can reopen authorized normalized or raw events through read-only SQL/JSON1 and explicit continuation. Both event views return handles that can be passed unchanged to the event reader; its default body matches the referenced view. Ordinary queries return complete rows. Large results return a complete JSON `result_path` readable with `read_file` and `grep`; long event fields also support `next_offset` continuation. Tool descriptions provide the queryable columns and two examples. Failures have an error status, a concrete reason and recovery guidance, without repeated schema dumps. There is no similarity cluster or minimum episode count: `defer` keeps an uncertain durable pattern open for later real use, while `ignore` records and consumes a finding that should not become guidance.

Reflection, proposal, and review submit decisions through their declared final-result tools. JSON in ordinary prose is not a submitted result. Explanations belong in text fields; the proposer changes skill or memory candidate files with normal filesystem tools, without a patch field. A prose-only ending receives one reminder in the same conversation, preserving evidence and file edits. If no final tool call follows, that item fails with full observations retained.

Every candidate revision is immutable. CatMaster does not generate test prompts or start extra conversations to compare variants. The host enforces only real path, authorization, immutable-revision, and selection-pointer boundaries. Candidate loadability is checked with the same DeepAgents skill middleware used by the active runtime. Headings, section order, prose style, optional metadata, file count, references, code layout, and `allowed-tools` never act as host-side quality gates. The independent reviewer owns semantic SOP review: `approve` advances under workspace policy, and an enabled Follow auto target is used by the next run in `auto` without a second human confirmation. `needs_revision` performs bounded automatic repair from the exact draft; `reject` terminates that branch. Ordinary new evidence starts from the currently effective version, while rejected draft files remain inspectable but are not silently copied into the next proposal.

Candidate and Processing history are diagnostic views, not an approval inbox. They show the reflected change, anchor and resolved targets, rationale, evidence handles, content parent, reviewer counterexamples and concerns, and an explicit terminal result for every job. Even without a candidate, history distinguishes `defer`, `ignore`, `no_change`, an existing-guidance execution lapse, and concrete errors while retaining every independent finding. The reviewed diff is loaded only from the exact revision under Technical details. Raw semantic event payloads remain available through the authorized raw trajectory view; transport internals remain in gated Developer Diagnostics. Candidate, observation, and Processing history lists are newest first and pageable.

Workspace mode controls what happens after review. When a workspace has no saved setting and the deployment does not override it, the default is `auto`. `off` stops new post-run evolution without disabling skills already selected. `observe` records approved revisions as an `auto_head` but leaves them dormant. `auto` selects an approved exact revision for the next run when the target is enabled and set to Follow auto. It never re-enables a disabled target or replaces a pinned selection. A real unresolved authorization, safety choice, or subjective preference is asked in the originating chat; the answer is recorded by exact user-message reference and resumes the held revision. Automatic activation and `no_change` produce concise non-blocking chat messages. Failed evolution jobs are not retried automatically; an authenticated user can retry one exact failed item while completed items remain untouched.

The **Manage skills** button inside the Skill Evolution page opens the optional exact-control dialog. It lists the complete effective set through pagination, including repository base versions, immutable workspace revisions, selected version, reviewer-approved `auto_head`, latest draft, enabled state, and Follow auto or Pinned policy. Choosing an eligible historical version pins it and acts as an exact rollback; returning to Follow auto resumes automatic selection. Rejected, unfinished, invalid, unreadable, and unresolved revisions remain inspectable but cannot be selected. Every automatic, chat, and GUI transition records its before/after state and source. All changes apply to subsequent runs.

Good skill candidates include stable project-specific QC methods, directory and delivery contracts, a verified stage-and-result workflow for a remote task, or repeated writing and figure conventions. Temporary errors, one sample-specific threshold, fixed atom indices, incidental checksums, and unverified scientific conclusions should not become skills. A skill changes method guidance, not tool permissions or remote availability.

```text
Review the last three slab tasks and their audit reports in this workspace.
Identify conventions that genuinely repeated and were independently validated. Separate stable rules
from choices that belong only to one structure.

If a reusable workflow exists, judge its cause from the complete episodes and results first.
Prefer amending the owning skill. State applicability, non-applicability, and the exact expected decision change.
Do not turn one cutoff, atom index, or incidental checksum into a universal rule.
Let the independent reviewer decide the revision. In `auto`, approved changes take effect on the next run unless the target is pinned, disabled, or truly needs my answer.
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
