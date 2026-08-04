---
name: mlff-md-sampling
description: Use this skill for managed MLFF molecular dynamics, restart-safe trajectory continuation, ensemble selection, and trajectory-health analysis when mlff_md is available.
license: project-local
allowed-tools: "ls read_file write_file edit_file execute get_avail_remote_task get_remote_task_spec remote_submission remote_submission_batch analyze_trajectory"
---

# mlff-md-sampling

## Overview

Run one trajectory lineage per stage with typed MD controls and preserve restart evidence.

## Quick Start

1. Put exactly one start structure or restart trajectory directly under `input/`.
2. Query the selected backend directly, using `get_remote_task_spec(task_name="mlff_md", template_overrides={"backend": "<enabled-backend>"}, detail="full")`; use its resolved defaults and concrete nested backend/task schema.
3. Submit one segment, or batch independent lineages at the same segment. Leave `submission_config.resources` and `.machine` unset; the selected backend owns them.
4. Inspect `output/batch_summary.json`, per-trajectory `summary.json`, logs, trajectories, and restart files.

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
- `analyze_trajectory`

## Workflow

### 1. Keep one lineage per stage

- A new stage contains exactly one ASE-readable file under `input/`; no params file is needed.
- One trajectory lineage may span multiple dependent stages; each hold or ramp is a continuation segment that waits for and consumes the previous `restart.traj`.
- At one segment, submit two or more independent same-config lineages together with one `remote_submission_batch`; dependent segments wait for the preceding `restart.traj`.
- The runner reads the last frame, preserves compatible momenta, and replaces velocities only when `dynamics.reinitialize_velocities=true`.

### 2. Set typed controls

- `backend_config` selects the backend's model and precision plus any execution controls required by the concrete task spec. Use that spec instead of copying another provider's fields.
- `task_config.dynamics` selects ensemble, temperatures, timestep, steps, seed, and velocity behavior.
- `thermostat`, `barostat`, and `output` remain separate nested groups. NPT requires a non-none barostat; non-NPT requires `barostat.type=none`.
- `dynamics.temperature_K` is the constant target or schedule start. Leave `temperature_end_K=0` (the default) or equal to `temperature_K` for constant temperature; set a different positive end value for a per-step linear schedule.
- Before scheduling temperature, use the task-spec constraints: variable-temperature NVT accepts Langevin or Berendsen, and variable-temperature NPT accepts the Berendsen barostat. NVE, Bussi, NHC, and MTK schedules are invalid; change the method or return the validation error instead of patching integrator internals.

### 3. Preserve restart semantics

- Prefer the previous `restart.traj` for segmented sampling because it contains the true final frame and CatMaster restart metadata.
- A compatible Bussi restart restores its random and integrator state. Langevin restores its random stream but has no additional thermostat state.
- NHC/MTK extended states are not checkpointed; do not claim exact segmented continuation for those methods.

### 4. Analyze before claiming convergence

- Verify errors, final frame, energy/temperature behavior, and restart sources. Inspect actual device and elapsed/startup timing only for a concrete compatibility/performance problem or when the user asks for them.
- Short runs are equilibration or exploratory evidence unless the requested observable has a credible production window and uncertainty analysis.

## Method-critical defaults

- Default to NVT, 300 K, 1 fs, 1000 steps, Bussi thermostat, seed 2026, and trajectory/log intervals of 10.
- Resolve model and precision from the selected backend spec and report them. Acceleration/device choices remain runtime metadata unless they materially change the scientific result.
- Use NVE for energy-conservation studies and NPT only with a real three-dimensional periodic cell.
- Set Berendsen compressibility explicitly. Keep timestep, ensemble, constant target or schedule endpoints, thermostat/barostat, model/head, precision, and dispersion visible in the result.

## Output Contract

Return:

- stage/replica identity, backend/model, ensemble, timestep, step count, temperature/pressure controls, and seed;
- `work_dir_rel`;
- `output/batch_summary.json`, trajectory/log/restart paths, and restart-source fields;
- whether the trajectory is exploratory, equilibration, or production evidence.

Keep receipt/context identifiers, hardware/device identity, launcher layout, and performance telemetry in runtime records unless a concrete failure or compatibility issue makes them relevant. If the user explicitly asks to inspect, compare, record, or report any of these fields, follow that request directly.

## References

- MACE acceleration and restart details: `references/mace.md`
- Canonical stage tree: `skills/execution/remote-stage-layouts/SKILL.md#mlff_md`
