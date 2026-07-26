---
name: lammps-restart
description: Use this skill to continue LAMMPS stages from restart files while preserving prior stage context.
allowed-tools: "lammps_forcefield_validate lammps_prepare remote_submission remote_submission_batch get_avail_remote_task lammps_log_summary md_trajectory_summary execute"
---

# lammps-restart

## Overview
Use this skill when a LAMMPS stage must continue from an existing restart file.

## Quick Start
1. Inspect the prior result directory and identify the intended restart file.
2. Run `lammps_log_summary` and `md_trajectory_summary` on the prior stage.
3. Prepare with `lammps_prepare(recipe="restart")`.
4. Query enabled tasks and submit with CPU `lammps_execute` or compatible strict GPU `lammps_execute_kokkos`.
5. Keep old and new receipt/context IDs in the report.

## Allowed tools
- `lammps_forcefield_validate`
- `lammps_prepare`
- `remote_submission`
- `remote_submission_batch`
- `get_avail_remote_task`
- `lammps_log_summary`
- `md_trajectory_summary`
- `execute`

## Workflow

### 1. Verify restart source
- Prefer an explicit `settings.restart_file` when multiple restart files exist.
- Keep continuation outputs in a new stage directory.
- Reuse the force-field card deliberately; do not regenerate or alter parameters silently.

### 2. Prepare continuation
- `lammps_prepare(recipe="restart")` copies the restart file and writes `read_restart`.
- Preserve or intentionally change ensemble, timestep, temperature, pressure, thermo/dump/restart strides.

### 3. Preserve execution compatibility
- Prefer the same CPU/KOKKOS execution mode as the source stage unless the restart and every active style are verified against the other deployment build.
- Use `lammps_execute` when KOKKOS compatibility is incomplete; use `lammps_execute_kokkos` only for a fully compatible stage.

### 4. Analyze continuation
- Use `lammps_log_summary` to confirm completion and thermo behavior.
- Use `md_trajectory_summary` to confirm frame output and new restart files.

## Method-critical defaults
- LAMMPS restart files are binary and tied to compatible LAMMPS builds/settings; use `read_restart` intentionally.
- Do not use a restart stage as a stateless fresh submission.

## Output Contract
Return:
- prior result path inspected
- restart file path
- new stage path
- submitted receipt/context
- log and trajectory summary paths
- warnings/errors and restart ambiguity

## References
- Local source note: `references/lammps_restart_reference.md`
- LAMMPS read_restart: https://docs.lammps.org/read_restart.html
- LAMMPS write_restart: https://docs.lammps.org/write_restart.html
- LAMMPS restart how-to: https://docs.lammps.org/Howto_restart.html
