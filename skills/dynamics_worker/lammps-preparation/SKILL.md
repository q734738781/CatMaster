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
4. Query the enabled LAMMPS tasks, then submit with CPU `lammps_execute` or strict GPU `lammps_execute_kokkos`.
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
- Use `task_name="lammps_execute"` for the CPU resource, including inputs whose pair, fix, compute, or other styles do not have complete KOKKOS support.
- Use `task_name="lammps_execute_kokkos"` only when it is listed and every method-critical input style is compatible with the deployment's KOKKOS build.
- The KOKKOS task requests a GPU and does not fall back to CPU. The CPU task never enables GPU acceleration.
- The CPU boot path maps `SLURM_NTASKS` to MPI ranks and probes the launcher before LAMMPS starts. On single-node Slurm allocations where Intel MPI is installed but `srun` is absent, it uses Hydra `fork` locally and still verifies the exact rank count. Treat a serial/stub build, missing launcher, or rank mismatch as a resource failure rather than running with fewer ranks.
- Do not pass `submission_config.resources` or `submission_config.machine`; each registered task owns its deployment binding.

## Method-critical defaults
- Report `units`, `atom_style`, `pair_style`, thermostat/barostat recipe, timestep, steps, thermo stride, dump stride, and restart stride.
- Do not hide force-field assumptions; they are part of the scientific model.
- RDF/MSD are generic only when species or group selections are explicit. Use task-specific scripts for residence time, adsorption events, or reaction analysis.

## Output Contract
Return:
- normalized force-field card path
- LAMMPS stage path
- submitted receipt/context
- `lammps_log_summary` and trajectory summary paths when generated
- `lammps_summary.json` MPI rank, launcher, and probe evidence
- any force-field or parser limitation

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
