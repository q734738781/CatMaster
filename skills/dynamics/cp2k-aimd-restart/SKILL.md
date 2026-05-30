---
name: cp2k-aimd-restart
description: Use this skill to continue CP2K AIMD from existing result directories without losing restart context or overwriting previous outputs.
allowed-tools: "cp2k_aimd_prepare cp2k_output_summary md_trajectory_summary remote_submission remote_submission_batch get_avail_remote_task execute"
---

# cp2k-aimd-restart

## Overview
Use this skill when a CP2K AIMD calculation needs continuation from a prior stage. Restart setup belongs to `dynamics_worker`.

## Quick Start
1. Inspect the prior result directory and identify the intended structure and restart file.
2. Use `cp2k_output_summary` and `md_trajectory_summary` to confirm the prior stage state.
3. Prepare the continuation with `cp2k_aimd_prepare(recipe="restart")`.
4. Submit the new stage with `remote_submission(task_name="cp2k_execute")`.
5. Preserve both old and new receipt/context IDs in the result report.

## Allowed tools
- `cp2k_aimd_prepare`
- `cp2k_output_summary`
- `md_trajectory_summary`
- `remote_submission`
- `remote_submission_batch`
- `get_avail_remote_task`
- `execute`

## Workflow

### 1. Verify restart inputs
- Locate the restart file and the structure used for continuation.
- Do not assume the newest file is correct if the user specified a particular restart.
- Keep the continuation in a new output directory.

### 2. Prepare controlled continuation
- Use `settings.restart_file` and `settings.structure_file` when auto-discovery would be ambiguous.
- Preserve or intentionally change `ensemble`, `temperature`, `pressure`, `timestep_fs`, `trajectory_stride`, `energy_stride`, and `restart_stride`.

### 3. Submit and inspect
- Submit only the new prepared stage.
- After completion, inspect `cp2k_summary.json`, `job.out`, `.ener`, trajectory, and restart files before reporting.

## Method-critical defaults
- Project policy: remote receipt/context fields are part of the restart record.
- Do not silently create a fresh AIMD run when the task is continuation.
- Do not infer equilibration from frame count alone.

## Output Contract
Return:
- prior result path inspected
- restart file path
- new AIMD stage path
- submitted receipt/context
- CP2K output and trajectory summary paths
- any ambiguity in selected restart files

## References
- Local source note: `references/cp2k_aimd_restart_reference.md`
- CP2K EXT_RESTART: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/EXT_RESTART.html
- CP2K MOTION/PRINT/RESTART: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/PRINT/RESTART.html
- CP2K MOTION/MD: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/MD.html
- CP2K manual index: https://manual.cp2k.org/
- CP2K PLUMED integration: https://www.cp2k.org/howto%3Ainstall_with_plumed
