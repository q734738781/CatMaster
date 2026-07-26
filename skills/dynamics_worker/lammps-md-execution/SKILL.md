---
name: lammps-md-execution
description: Use this skill for LAMMPS NVE/NVT/NPT/annealing preparation, execution, restart output, and generic MD health analysis.
allowed-tools: "lammps_forcefield_validate lammps_prepare remote_submission remote_submission_batch get_avail_remote_task lammps_log_summary md_trajectory_summary analyze_trajectory execute"
---

# lammps-md-execution

## Overview
Use this skill for LAMMPS equilibration, annealing, and production MD stages in `dynamics_worker`.

## Quick Start
1. Validate or reuse an explicit force-field card.
2. Prepare with `lammps_prepare(recipe="nve" | "nvt" | "npt" | "anneal")`.
3. Query enabled tasks and submit with CPU `lammps_execute` or compatible strict GPU `lammps_execute_kokkos`.
4. Summarize with `lammps_log_summary` and `md_trajectory_summary`.

## Allowed tools
- `lammps_forcefield_validate`
- `lammps_prepare`
- `remote_submission`
- `remote_submission_batch`
- `get_avail_remote_task`
- `lammps_log_summary`
- `md_trajectory_summary`
- `analyze_trajectory`
- `execute`

## Workflow

### 1. Choose the ensemble deliberately
- Use `nve` for microcanonical production only after appropriate preparation.
- Use `nvt`/`anneal` for temperature-controlled stages.
- Use `npt` only when pressure control is physically meaningful for the model.

### 2. Preserve output controls
- Set `timestep`, `steps`, `thermo`, `dump_stride`, and `restart_stride` explicitly for production runs.
- Enable in-run `rdf` or `msd` only when the requested observable is generic enough for all atoms/groups represented by the stage.

### 3. Select CPU or KOKKOS explicitly
- Prefer `lammps_execute_kokkos` only after checking that the stage's pair, fix, compute, and related styles support the enabled KOKKOS build.
- Use `lammps_execute` for unsupported styles. Do not rely on GPU failure followed by an implicit CPU retry.

### 4. Analyze health before interpretation
- Use `lammps_log_summary` for thermo segments, drift, warnings/errors, and run completion.
- Use `md_trajectory_summary` for frame count, final frame export, restart files, and RDF/MSD table presence.
- Write focused scripts for residence time, adsorption/desorption, reactions, or region-specific observables.

## Method-critical defaults
- Do not hide thermostat/barostat settings in prose; they are method choices.
- Do not infer equilibrated production data from a completed run without drift and sampling checks.

## Output Contract
Return:
- force-field card and stage paths
- submitted receipt/context
- log summary and trajectory summary paths
- ensemble, timestep, steps, thermo/dump/restart strides
- warnings, errors, and observable limitations

## References
- Local source note: `references/lammps_md_reference.md`
- LAMMPS fix nvt/npt: https://docs.lammps.org/fix_nh.html
- LAMMPS fix: https://docs.lammps.org/fix.html
- LAMMPS thermo_style: https://docs.lammps.org/thermo_style.html
- LAMMPS dump: https://docs.lammps.org/dump.html
- LAMMPS write_restart: https://docs.lammps.org/write_restart.html
- LAMMPS compute rdf: https://docs.lammps.org/compute_rdf.html
- LAMMPS compute msd: https://docs.lammps.org/compute_msd.html
