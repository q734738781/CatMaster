---
name: cp2k-aimd-preparation
description: Use this skill for source-grounded CP2K AIMD preparation, restart staging, generic execution handoff through cp2k_execute, and run-health inspection.
allowed-tools: "cp2k_aimd_prepare cp2k_output_summary remote_submission remote_submission_batch get_avail_remote_task md_trajectory_summary analyze_trajectory execute"
---

# cp2k-aimd-preparation

## Overview
Use this skill for CP2K AIMD tasks in `dynamics_worker`: NVE, NVT, NPT, restart, and user-provided PLUMED metadynamics.

## Quick Start
1. Verify the starting structure or restart source.
2. Use `cp2k_aimd_prepare` with `recipe="nve"`, `nvt`, `npt`, `restart`, or `metadynamics_user_plumed`.
3. Verify `job.inp`, `manifest.json`, and any restart or `plumed.dat` files.
4. Submit with `remote_submission(task_name="cp2k_execute")`.
5. Run `cp2k_output_summary` and `md_trajectory_summary` after outputs are collected.

## Allowed tools
- `cp2k_aimd_prepare`
- `cp2k_output_summary`
- `remote_submission`
- `remote_submission_batch`
- `get_avail_remote_task`
- `md_trajectory_summary`
- `analyze_trajectory`
- `execute`

## Workflow

### 1. Keep AIMD in dynamics_worker
- Conventional CP2K `sp`, `geo_opt`, and `cell_opt` belong to `materials_worker`.
- AIMD owns ensemble, timestep, thermostat/barostat, restart, and trajectory output decisions.

### 2. Prepare controlled AIMD input
- Set `timestep_fs`, `steps`, `temperature`, `trajectory_stride`, `energy_stride`, and `restart_stride` explicitly when they matter.
- For `recipe="restart"`, verify the restart file and source structure before preparing the new stage.
- For `metadynamics_user_plumed`, require `settings.plumed_input_path`; do not invent collective variables.

### 3. Submit and inspect
- Use only `task_name="cp2k_execute"` for CP2K execution.
- After execution, inspect `cp2k_summary.json`, `job.out`, CP2K energy files, trajectory files, and restart files before judging success.
- Use `cp2k_output_summary` for reusable run evidence: completion, return code, energy lines, SCF evidence, optimization/frequency markers, and CP2K `.ener` files.

## Method-critical defaults
- Parameter priority: honor explicit user requirements first; otherwise choose CP2K AIMD `settings` overrides from the system class and sampling objective; if that judgment remains uncertain, run a narrow literature or official documentation check before finalizing the override.
- Do not add CP2K `settings` overrides just to restate the tool baseline; only override when the user, system class, task objective, or a checked source justifies it.
- Project policy: prefer output completion over performance when safe, but keep receipt/context if remote execution fails.
- Report ensemble, timestep, total simulated time, temperature/pressure controls, trajectory stride, and restart stride.
- Do not infer free-energy barriers or mechanisms from a generic trajectory summary.

## Output Contract
Return:
- AIMD stage path
- selected recipe and ensemble
- `manifest.json` path
- submitted receipt/context
- `cp2k_output_summary` and trajectory summary artifact paths

## References
- Local source note: `references/cp2k_aimd_reference.md`
- CP2K MOTION/MD: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/MD.html
- CP2K MOTION/PRINT/RESTART: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/PRINT/RESTART.html
- CP2K EXT_RESTART: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/EXT_RESTART.html
- CP2K manual index: https://manual.cp2k.org/
- CP2K PLUMED integration: https://www.cp2k.org/howto%3Ainstall_with_plumed
