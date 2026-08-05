---
name: lammps-preparation
description: Use this skill for source-grounded LAMMPS force-field validation and preparation of minimization, MD, and restart stages.
allowed-tools: "lammps_forcefield_validate lammps_prepare remote_submission remote_submission_batch get_avail_remote_task lammps_log_summary md_trajectory_summary execute"
---

# lammps-preparation

## Overview
Use this skill for LAMMPS stages in `dynamics_worker`. LAMMPS requires an explicit validated force-field card before input generation.

## Quick Start
1. Validate the force-field card with `lammps_forcefield_validate`.
2. Prepare a stage with `lammps_prepare(recipe="minimize" | "nve" | "nvt" | "npt" | "anneal" | "restart")`.
3. Verify `in.lammps`, `system.data` or restart file, `manifest.json`, and potential files.
4. Query the enabled LAMMPS tasks, select one compatible execution path, and submit it; do not create a CPU/GPU comparison gate.
5. Summarize with `lammps_log_summary` and `md_trajectory_summary` when outputs are collected.

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

### 1. Validate before prepare
- The force-field card must declare `units`, `atom_style`, `pair_style`, and `pair_coeff`.
- Referenced potential files must exist in the workspace.
- Do not choose or invent a force field from chemistry alone.

### 2. Prepare one stage per simulation
- `lammps_prepare` writes `in.lammps`, `system.data` when starting from a structure, and `manifest.json`.
- Use `recipe="minimize"` for force-field minimization.
- Use `nve`, `nvt`, `npt`, or `anneal` for MD stages.
- Use `restart` only when the intended restart file is present and verified.

### 3. Select one explicit execution task
- Use `task_name="lammps_execute_kokkos"` when it is listed and the input has no known pair, fix, compute, or related style incompatibility with the registered KOKKOS path.
- Use `task_name="lammps_execute"` when a method-critical style is known to be unsupported by that KOKKOS path or when the CPU task is otherwise the selected registered route.
- The KOKKOS task requests a GPU and does not fall back to CPU. The CPU task never enables GPU acceleration.
- The registered task owns launcher, rank-layout, build, and accelerator checks. Treat a concrete incompatibility as an execution failure; do not turn these platform details into routine scientific QC.
- Select and run one path. A CPU baseline, same-coordinate CPU/KOKKOS comparison, or alternate-backend smoke run is not required before production. Use cross-backend comparison only when the user requests it or an observed accelerator-specific result may affect the scientific conclusion.
- Do not pass `submission_config.resources` or `submission_config.machine`; each registered task owns its deployment binding.

## Method-critical defaults
- Report `units`, `atom_style`, `pair_style`, thermostat/barostat recipe, timestep, steps, thermo stride, dump stride, and restart stride.
- Do not hide force-field assumptions; they are part of the scientific model.
- RDF/MSD are generic only when species or group selections are explicit. Use task-specific scripts for residence time, adsorption events, or reaction analysis.

## Output Contract
Return:
- normalized force-field card path
- LAMMPS stage path
- `lammps_log_summary` and trajectory summary paths when generated
- any force-field or parser limitation

Receipt IDs, launcher/rank evidence, build details, and hardware identity remain in runtime records. Surface them for a concrete execution failure or whenever the user explicitly asks to inspect, compare, record, or report them.

## References
- Local source note: `references/lammps_input_reference.md`
- LAMMPS units: https://docs.lammps.org/units.html
- LAMMPS read_data: https://docs.lammps.org/read_data.html
- LAMMPS pair_style: https://docs.lammps.org/pair_style.html
- LAMMPS fix nvt/npt: https://docs.lammps.org/fix_nh.html
- LAMMPS minimize: https://docs.lammps.org/minimize.html
- LAMMPS thermo_style: https://docs.lammps.org/thermo_style.html
- LAMMPS dump: https://docs.lammps.org/dump.html
- LAMMPS write_restart: https://docs.lammps.org/write_restart.html
- LAMMPS read_restart: https://docs.lammps.org/read_restart.html
- LAMMPS compute rdf: https://docs.lammps.org/compute_rdf.html
- LAMMPS compute msd: https://docs.lammps.org/compute_msd.html
- pymatgen LAMMPS module: https://pymatgen.org/pymatgen.io.lammps.html
- lammpsio docs: https://lammpsio.readthedocs.io/
