---
name: vasp-batch-execution
description: Use this skill for dispatching prepared VASP jobs with remote_submission or remote_submission_batch, choosing valid stage layouts, and collecting clean failure evidence.
---

# vasp-batch-execution

## Overview
Use this skill to submit prepared VASP jobs without corrupting the input tree or losing failure evidence.

## Quick Start
1. Query `get_avail_remote_task`, then validate `vasp_execute` with `get_remote_task_spec`; `execution_binding.status=configured` is sufficient platform preflight.
2. For a single calculation, set `work_dir` to one folder containing `INCAR`, `POTCAR`, `POSCAR`, and `KPOINTS`.
3. For a batch, set `work_dir` to the common root. Omit `stage_paths` for all first-level calculation folders, or select explicit relative paths, including nested stages.
4. For NEB or dimer-style VASP work, use `task_name="vasp_execute_neb"` when the larger default resource preset fits.
5. After success, inspect stage-local outputs and logs. Use receipt recovery only after a returned failure or ambiguous transport error.

## Allowed tools
- `get_avail_remote_task`
- `get_remote_task_spec`
- `remote_submission`
- `remote_submission_batch`
- `execute`

## Workflow

### 1. Validate the execution layout
- A calc folder is identified by `INCAR`, `POTCAR`, `POSCAR`, and `KPOINTS`.
- `remote_submission` runs exactly one prepared calc directory.
- `remote_submission_batch` selects all first-level children by default. With `stage_paths`, it submits only those relative directories; nested stages are allowed, but selected stages must not overlap and must share task/config. Any preparation failure reports all failing paths and submits nothing.

### 2. Avoid illegal path layouts
- Without `stage_paths`, use a dedicated batch root containing only intended stages. With explicit selection, unrelated unselected directories are not submitted.
- Do not include unrelated nested calculation folders inside a stage child.
- Before retrying a returned failure, resolve uncertain remote state using the recovery guidance. Keep independent calculations in separate stages.

### 3. Check task availability
- When the task spec reports `execution_binding.status=configured`, proceed with the prepared inputs. Investigate execution setup only after a concrete submission error or an explicit user request.
- `get_avail_resources` discovers environments for handwritten scripts. VASP task availability comes from its task spec; it does not require a separate resource query.

### 4. Submit and collect
- Every selected stage must contain valid VASP inputs. `stage_paths=["branch_a/relax", "branch_b/static"]` selects two directories relative to `work_dir`; omit it to use the default first-level layout.
- Outputs are downloaded back into the same stage directory.
- Use returned recovery handles after a failure; ordinary successful handoffs need the scientific outputs.

### 5. Triage failures minimally
- Only after the tool returns a failure, inspect its receipt before resubmitting; the remote job may still be live.
- Inspect only the focused `status.json`, `stdout.log`, `stderr.log`, VASP stdout, or scheduler evidence for failed stages.
- After accounting for the previous remote work, rerun the failed subset using explicit `stage_paths` or a dedicated batch root. Prepare a fresh stage when inputs or restart lineage change; selection alone does not require copying unchanged stages.

### 6. Hand off to structured analysis
- After collection, prefer a task-specific VASP analysis skill/tool when available, or write a narrow `pymatgen` parser (`Vasprun`, `Outcar`, `Structure`) for the requested evidence. Do not substitute a generic VASP result-summary tool for geometry optimization, DOS/band, NEB, frequency, or force-collection work.
- Use `grep` only as a fallback for missing/corrupted structured outputs or quick log triage.
- When comparing ordinary VASP total energies from structured parsing, use `E0` as the default reference energy unless the workflow explicitly requires another convention.
- Treat this skill as execution-only; dispatch success is not the same as usable scientific output.

## Method-critical defaults
- Keep the staged calculation tree focused and auditable.
- Do not use launch success as a scientific result; the post-run evidence files are part of the contract.

## Output Contract
Return:
- whether the run used `remote_submission` or `remote_submission_batch`
- submitted calc-directory count
- `work_dir_rel`
- representative output path
- whether every required VASP output was returned

## References
- Use `execute` only for focused follow-up reads after the receipt or stage-local status files point to a concrete failure target.
- Hand off finished NEB or MD batches to `neb-analysis` or `md-diffusion-analysis` rather than reusing this skill as an analysis layer.
