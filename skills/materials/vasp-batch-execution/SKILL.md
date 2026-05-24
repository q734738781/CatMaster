---
name: vasp-batch-execution
description: Use this skill for dispatching prepared VASP jobs with remote_submission or remote_submission_batch, choosing valid stage layouts, and collecting clean failure evidence.
---

# vasp-batch-execution

## Overview
Use this skill to submit prepared VASP jobs without corrupting the input tree or losing failure evidence.

## Quick Start
1. Prepare a low-level VASP stage layout, then use `remote_submission` or `remote_submission_batch` with `task_name="vasp_execute"`.
2. Keep `input_dir` and `output_dir` as separate trees.
3. Do not make the input root both a calc folder and a parent of nested calc folders.
4. After submission or failure, check `_BATCH_STATE.json` first.

## Allowed tools
- `get_avail_remote_task`
- `remote_submission`
- `remote_submission_batch`
- `analyze_vasp_results`
- `execute`

## Workflow

### 1. Validate the execution layout
- A calc folder is identified by `INCAR` and `POTCAR`.
- Single-folder mode runs one calc directory.
- Batch mode recursively finds descendant calc folders under `input_dir`.

### 2. Avoid illegal path layouts
- `output_dir` must not be inside `input_dir`.
- Output mapping must not overlap any source calc directory.
- Nested calc folders under an already-valid calc root are rejected.

### 3. Submit and collect
- The remote submission tool expects the stage directory to already be a valid VASP calculation folder, or a batch root whose first-level children are valid VASP calculation folders.
- Treat the final output tree as a collected snapshot, not an in-place mutation of the input tree.

### 4. Triage failures minimally
- Read `_BATCH_STATE.json` first.
- If needed, inspect only the focused scheduler/stdout/stderr evidence for failed jobs.
- Rerun only the failed subset, again into a fresh output root.

### 5. Hand off to structured analysis
- After collection, prefer `analyze_vasp_results` over ad hoc manual parsing when the next step needs convergence, energy, or bandgap summaries.
- When comparing ordinary VASP total energies from `analyze_vasp_results`, use `E0` as the default reference energy unless the workflow explicitly requires another convention.
- Treat this skill as execution-only; dispatch success is not the same as usable scientific output.

## Method-critical defaults
- Keep the input tree and output tree separate so the collected snapshot stays auditable.
- Do not use launch success as a scientific result; the post-run evidence files are part of the contract.

## Output Contract
Return:
- whether the run used single-folder or batch mode
- submitted calc-directory count
- `output_root_rel`
- `batch_state_rel`
- representative output path
- whether the batch finished as `collected_complete` or only `collected_partial`

## References
- Use `execute` only for focused follow-up reads after the batch state points to a concrete failure target.
- Hand off finished NEB or MD batches to `neb-analysis` or `md-diffusion-analysis` rather than reusing this skill as an analysis layer.
