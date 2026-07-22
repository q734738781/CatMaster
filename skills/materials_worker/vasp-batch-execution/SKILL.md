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
3. For a batch, set `work_dir` to a batch root whose first-level children are complete VASP calculation folders.
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
- `remote_submission_batch` submits each first-level child directory under `work_dir`; nested discovery is not performed.

### 2. Avoid illegal path layouts
- Do not point `work_dir` at a mixed project tree.
- Do not include unrelated nested calculation folders inside a stage child.
- Use a fresh, task-specific stage directory when retrying after a failed or interrupted remote submission.

### 3. Trust the registered execution boundary
- When the catalog lists `vasp_execute` or `vasp_execute_neb` and its task spec reports `execution_binding.status=configured`, treat the task/resource/machine wiring as deployment-authoritative and continue with stage validation and submission.
- Do not inspect active DPDispatcher YAML, local shell executables, unrelated receipts, or historical jobs to re-prove that binding. Do not ask the user for preset/config IDs or revisions, queue/account/node/MPI/walltime fields, VASP module/license identifiers, gamma-binary confirmation, or a comparable prior receipt.
- The registered VASP boot contract owns executable selection, including `--auto_gamma`. If the environment, license, executable, queue, or gamma path is actually broken, the managed submission must return that concrete error; only then treat it as an infrastructure blocker.
- `get_avail_resources` is for general custom-boot cards. The absence of `vasp_cpu` there is expected and is not evidence of a missing VASP binding.

### 4. Submit and collect
- The remote submission tool expects the stage directory to already be a valid VASP calculation folder, or a batch root whose first-level children are valid VASP calculation folders.
- Outputs are downloaded back into the same stage directory.
- The tool returns `remote_context_id`, `submission_hash`, `receipt_rel`, and `task_state_counts`; keep those fields in the handoff summary.

### 5. Triage failures minimally
- Only after the tool returns a failure, inspect its receipt before resubmitting; the remote job may still be live.
- Inspect only the focused `status.json`, `stdout.log`, `stderr.log`, VASP stdout, or scheduler evidence for failed stages.
- Rerun only the failed subset from a clean stage directory or a batch root containing only those failed first-level children.

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
- `remote_context_id`, `submission_hash`, and `receipt_rel` when present
- representative output path
- whether every required VASP output was returned

## References
- Use `execute` only for focused follow-up reads after the receipt or stage-local status files point to a concrete failure target.
- Hand off finished NEB or MD batches to `neb-analysis` or `md-diffusion-analysis` rather than reusing this skill as an analysis layer.
