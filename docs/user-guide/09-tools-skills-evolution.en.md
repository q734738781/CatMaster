# 9. Tools, skills, and evolution

[Previous](08-remote-execution.en.md) | [Contents](README.en.md) | [Next](10-deployment-operations.en.md)

Tools determine which actions an agent can perform. Skills define how to perform
work. Evolution determines whether a workspace can propose and activate a
project-level improvement. They work together but do not replace one another.

## 9.1 Base tools

The DeepAgents runtime provides basic task and file tools such as:

```text
write_todos
ls
read_file
write_file
edit_file
glob
grep
execute
read_document
```

`read_document` performs bounded parsing of PDF, DOCX, XLSX, and PPTX. `execute`
is for project preparation, checking, lightweight scripts, dependency probes,
and postprocessing. It is not a route around DPDispatcher for VASP, CP2K,
LAMMPS, ORCA, xTB, CREST, or managed MLFF.

## 9.2 Domain tools

Registered CatMaster tools broadly cover:

- Structure building, conversion, surfaces, adsorption, and geometry analysis.
- VASP, CP2K, LAMMPS, ORCA, xTB/CREST input preparation and result analysis.
- Trajectories, vibrations, thermodynamics, bands, DOS, elasticity, and data
  analysis.
- ML data, active learning, symbolic regression, MACE train/eval, and MLFF task
  schemas.
- Materials Project, literature search, local corpora, web access, and citation
  management.
- Figures, document reading, Markdown PDF, TeX compilation, and image generation.
- DPDispatcher task catalogs, single-stage submission, batch submission, and
  receipts.

Not every agent sees every tool. Runtime allowlists and task `audiences` define
the actual specialist or worker surface. A prompt normally does not need to
force a tool name unless it is querying a task schema or reproducing a known
call.

## 9.3 Tool schema as an interface

The agent constructs arguments from tool descriptions and JSON schemas. An
optional field should usually be omitted or passed as an empty object or array,
not guessed as `null`. Remote task parameters should be queried from the catalog
because backend status, models, defaults, and override keys depend on deployment.

If an agent repeatedly sends an empty critical parameter:

1. Expand the tool card in Chat and inspect final arguments.
2. Query the catalog's full spec.
3. Correct the action in a Review card, or reject and request a rebuild.
4. Put a reusable correct contract in a project skill instead of repeating it
   in every prompt.

## 9.4 What a skill is

A skill is a task SOP with `SKILL.md` and optional references, scripts,
templates, and acceptance rules. Main skill groups currently cover:

| Group | Typical scope |
|---|---|
| Materials | Bulk, slab, termination, adsorption, defects, VASP, CP2K, NEB, phonons, and MLFF |
| Dynamics | CP2K AIMD, LAMMPS, MLFF MD, restart, and trajectory analysis |
| ML | Datasets, MACE training/evaluation, and active learning |
| ORCA/xTB | Conformers, xTB, CREST, ORCA opt/freq/TS/IRC/TDDFT/NMR |
| Research | Planning, state, evidence, and cross-specialist coordination |
| Literature | Search, reading, corpora, citations, and reports |
| Writing | Manuscripts, responses, figures, LaTeX, PDF, and language editing |
| Execution | Remote stage layouts and DPDispatcher receipt recovery |
| Writing quality | Natural prose while preserving technical facts |

A placeholder directory without a valid `SKILL.md` is not an available
capability.

## 9.5 Skill load locations

Each run stages applicable built-in skills under:

```text
files/.deepagents/skills/<group>/
```

Workspace self-development overlays can replace a built-in skill of the same
name. Staged content is a runtime snapshot. Editing it casually during a task
does not update the repository skill.

A skill supplies method, not authorization. A remote stage-layout skill can
describe VASP directories, but an agent without `remote_submission` in its
allowlist still cannot submit.

## 9.6 Project scripts

When an agent writes a reusable lightweight operation, place it in `scripts/`
and record at least:

```text
creation date
agent or source
purpose and scientific principle
inputs and outputs
units and critical parameters
dependencies
failure modes
minimal example
```

One-off shell snippets are useful for exploration. Logic that will be reused,
affect scientific results, or be called by later threads should become an
auditable script.

## 9.7 Sequential work in a shared workspace

Specialists and workers share one workspace. The runtime delegates one at a time
and waits for the result, preventing several agents from writing the same
directory concurrently. Provider support for `parallel_tool_calls` does not make
project writes safe to parallelize.

Truly independent remote stages can use `remote_submission_batch` inside one
managed call. Do not replace that contract with several agents submitting in
parallel.

## 9.8 Review-mode boundary

Review interrupts only before `write_file`, `edit_file`, `remote_submission`, and
`remote_submission_batch`. It does not replace:

- External project backup.
- SSH and queue access control.
- Remote task/resource audiences.
- Human review of scientific settings and cost.
- Scheduler management of an existing remote job.

A stricter deployment needs controls at the network, account, file-permission,
cluster-queue, and secret-manager layers.

## 9.9 Evolution modes

Environment variable:

```bash
export CATMASTER_SELF_EVOLUTION_MODE=observe
```

Values:

| Mode | Behavior |
|---|---|
| `off` | Do not create candidates |
| `observe` | Default; propose and review, then wait for human Promote or Reject |
| `auto` | Promote automatically after gate and reviewer approval |

WebUI evolution is fully disabled in `--no-login` mode. `auto` changes future
run behavior and should be enabled only after an administrator validates
candidate quality, rollback, and monitoring.

## 9.10 Candidate lifecycle

```text
terminal run trace
  -> proposer: ignore / memory / skill
  -> static gate
  -> independent reviewer
  -> observe: human decision
  -> auto: automatic promotion
  -> active from the next run
```

A candidate can propose a complete memory file or one complete skill bundle. It
cannot directly edit repository-built-in skills. The static gate checks path,
size, symlinks, frontmatter, sections, references, and Python or Shell syntax.
The reviewer is read-only.

Promotion uses content hashes, a target hash, and locking. If the target changed
after proposal, promotion conflicts instead of overwriting it. Rollback likewise
requires the target to remain at the promoted version.

## 9.11 Handle candidates in the UI

In Skill Evolution:

1. Read the source run and reason for the candidate.
2. Compare target, old content, and candidate content.
3. Confirm it is project-specific and should not affect another workspace.
4. Check that one accidental failure was not generalized into a permanent rule.
5. After `Promote`, create a minimal validation run.
6. `Reject` an unhelpful candidate and `Rollback` a regression.

Candidates are shared across threads in the same workspace and load on the next
run. Promotion does not change a context that is already running.

## 9.12 Good and bad material for evolution

Good candidates:

- Stable project directory, naming, unit, and delivery contracts.
- Repeatedly verified structure checks or analysis steps.
- Stable stage preparation for a specific remote task.
- Writing or reporting rules the user explicitly wants to retain.

Bad candidates:

- One network error, temporary filename, or single failed example.
- SSH keys, tokens, accounts, or private host details.
- Unvalidated scientific defaults.
- Methods to bypass workers, Review, or remote-execution controls.
- Instructions relevant only to the current turn.
