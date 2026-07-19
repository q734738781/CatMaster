---
name: mlff-md-sampling
description: Use this skill for managed MLFF molecular dynamics, restart-safe trajectory continuation, ensemble selection, and trajectory-health analysis when mlff_md is available.
license: project-local
allowed-tools: "ls read_file write_file edit_file execute get_avail_remote_task get_remote_task_spec remote_submission remote_submission_batch md_trajectory_summary analyze_trajectory"
---

# mlff-md-sampling

## Overview

Run one trajectory lineage per stage with typed MD controls and preserve restart evidence.

## Quick Start

1. Put exactly one start structure or restart trajectory directly under `input/`.
2. Query the selected backend directly, using `get_remote_task_spec(task_name="mlff_md", template_overrides={"backend": "<enabled-backend>"}, detail="full")`; use its resolved defaults and concrete nested backend/task schema.
3. Submit one lineage with `remote_submission`; use `remote_submission_batch` only for independent replicas. Leave `submission_config.resources` and `.machine` unset; the selected backend owns them.
4. Inspect `output/batch_summary.json`, per-trajectory `summary.json`, logs, trajectories, and restart files.
5. Use `analyze_trajectory` for MLFF ASE `.traj` output. `md_trajectory_summary` is only for CP2K/LAMMPS trajectory formats.

## Allowed tools

- `ls`
- `read_file`
- `write_file`
- `edit_file`
- `execute`
- `get_avail_remote_task`
- `get_remote_task_spec`
- `remote_submission`
- `remote_submission_batch`
- `md_trajectory_summary`
- `analyze_trajectory`

## Workflow

### 1. Keep one lineage per stage

- A new stage contains exactly one ASE-readable file under `input/`; no params file is needed.
- Independent replicas are first-level child stages under a batch root. Continuation segments are dependent and must wait for the previous restart artifact.
- The runner reads the last frame, preserves compatible momenta, and replaces velocities only when `dynamics.reinitialize_velocities=true`.

### 2. Set typed controls

- `backend_config` selects provider-specific model, precision, acceleration, and device controls. MACE is the registered default, while every deployment-enabled backend returned by `get_remote_task_spec` can run the same MD task contract.
- `task_config.dynamics` selects ensemble, temperatures, timestep, steps, seed, and velocity behavior.
- `thermostat`, `barostat`, and `output` remain separate nested groups. NPT requires a non-none barostat; non-NPT requires `barostat.type=none`.
- `dynamics.temperature_K` is the constant target or schedule start. Leave `temperature_end_K=0` (the default) or equal to `temperature_K` for constant temperature; set a different positive end value for a per-step linear schedule.
- Before scheduling temperature, use the task-spec constraints: variable-temperature NVT accepts Langevin or Berendsen, and variable-temperature NPT accepts the Berendsen barostat. NVE, Bussi, NHC, and MTK schedules are invalid; change the method or return the validation error instead of patching integrator internals.

### 3. Preserve restart semantics

- Prefer the previous `restart.traj` for segmented sampling because it contains the true final frame and CatMaster restart metadata.
- A compatible Bussi restart restores its random and integrator state. Langevin restores its random stream but has no additional thermostat state.
- NHC/MTK extended states are not checkpointed; do not claim exact segmented continuation for those methods.

### 4. Analyze before claiming convergence

- Verify actual device, elapsed/startup timing, errors, final frame, energy/temperature behavior, and restart sources.
- Short runs are equilibration or exploratory evidence unless the requested observable has a credible production window and uncertainty analysis.

## Method-critical defaults

- Default to NVT, 300 K, 1 fs, 1000 steps, Bussi thermostat, seed 2026, and trajectory/log intervals of 10.
- MACE MD defaults to `float32`; `enable_cueq=false` and compilation disabled remain conservative until a comparable GPU/model/system benchmark justifies them.
- For fixed-composition UMA trajectories, `inference_settings=turbo` is the speed preset. For the pinned MatterSim 1.2.5 stack, keep `direct_graph=false` and `compile=false`: both accelerated paths failed finite-output regression checks, while the standard graph path passed. ORB-v3 normally uses `precision=float32-high`, `compile_mode=auto`, `edge_method=knn_alchemi`, and `half_supercell=auto`.
- Use NVE for energy-conservation studies and NPT only with a real three-dimensional periodic cell.
- Set Berendsen compressibility explicitly. Keep timestep, ensemble, constant target or schedule endpoints, thermostat/barostat, model/head, precision, and dispersion visible in the result.

## Output Contract

Return:

- stage/replica identity, backend/model, ensemble, timestep, step count, temperature/pressure controls, and seed;
- `work_dir_rel` plus receipt/context identifiers;
- `output/batch_summary.json`, trajectory/log/restart paths, and restart-source fields;
- whether the trajectory is exploratory, equilibration, or production evidence.

## References

- MACE acceleration and restart details: `references/mace.md`
- Canonical stage tree: `skills/execution/remote-stage-layouts/SKILL.md#mlff_md`
