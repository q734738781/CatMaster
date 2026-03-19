---
name: vasp-batch-execution
description: Use this skill for dispatching prepared VASP jobs with vasp_execute_batch, choosing valid input/output layouts, avoiding nested or overlapping calc trees, and collecting clean failure evidence.
license: project-local
compatibility: local
allowed-tools: "vasp_execute_batch analyze_vasp_results execute"
metadata:
  catmaster-suggested-tools: "vasp_execute_batch analyze_vasp_results execute"
---

# vasp-batch-execution

## Overview
Use this skill to submit prepared VASP jobs without corrupting the input tree or losing failure evidence.

## Quick Start
1. Use `vasp_execute_batch`; do not use deprecated `vasp_execute`.
2. Keep `input_dir` and `output_dir` as separate trees.
3. Do not make the input root both a calc folder and a parent of nested calc folders.
4. After submission or failure, check `_BATCH_STATE.json` first.

## Suggested tools
- `vasp_execute_batch`
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
- The tool stages work under the output tree, injects the bootstrap script, dispatches via DPDispatcher, then collects outputs back into the final output tree.
- Treat the final output tree as a collected snapshot, not an in-place mutation of the input tree.

### 4. Triage failures minimally
- Read `_BATCH_STATE.json` first.
- If needed, inspect only the focused scheduler/stdout/stderr evidence for failed jobs.
- Rerun only the failed subset, again into a fresh output root.

### 5. Hand off to structured analysis
- After collection, prefer `analyze_vasp_results` over ad hoc manual parsing when the next step needs convergence, energy, or bandgap summaries.
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
- Use `bash` only for focused follow-up reads after the batch state points to a concrete failure target.
- Hand off finished NEB or MD batches to `reaction-neb-analysis` or `md-diffusion-analysis` rather than reusing this skill as an analysis layer.
