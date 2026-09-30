---
name: trajectory-analysis
description: Use this skill for generic CP2K/LAMMPS/ASE trajectory and run-health checks and for deciding when task-specific trajectory parsing is required.
allowed-tools: "md_trajectory_summary analyze_trajectory lammps_log_summary cp2k_output_summary execute"
---

# trajectory-analysis

## Overview
Use this skill for generic MD health summaries. It is not a replacement for task-specific mechanistic analysis.

## Quick Start
1. Locate the result directory or trajectory file.
2. Use `md_trajectory_summary` on one LAMMPS dump, XYZ, or ASE `.traj` for frame counts, final frame export, restart-file presence, generic thermo/log evidence, CP2K `.ener` files, and LAMMPS RDF/MSD tables.
3. Select MSD and/or RDF with `analyze_trajectory`. Supply the stored-frame interval; an explicit fit window is required only with MSD. For periodic generic XYZ/ASE files, state whether coordinates are wrapped or unwrapped.
4. Write a focused parser under `scripts/` for system-specific residence, reaction, adsorption, or free-energy questions.

## Allowed tools
- `md_trajectory_summary`
- `analyze_trajectory`
- `lammps_log_summary`
- `cp2k_output_summary`
- `execute`

## Workflow

### 1. Start with health, not interpretation
- Check frame count, atom count, final frame export, time span if recoverable, log completion, energy/thermo files, and restart files.
- Inspect temperature, energy, and pressure drift before interpreting a trajectory.

### 2. Keep observables explicit
- RDF and MSD require explicit species or group choices; if they were produced by in-run LAMMPS computes, summarize file presence and numeric rows before interpretation.
- Mechanistic labels such as diffusion path, desorption event, residence time, or reaction coordinate usually require a task-specific script.

## Method-critical defaults
- Generic summaries may prove that a run produced data; they do not prove equilibration or scientific convergence.
- Do not infer barriers or free energies from raw MD without the appropriate method and parser.

## Output Contract
Return:
- trajectory file path
- summary JSON path
- log or CP2K output summary path if available
- any task-specific parser path
- explicit limitations

## References
- Local source note: `references/md_trajectory_outputs_reference.md`
- LAMMPS output how-to: https://docs.lammps.org/Howto_output.html
- LAMMPS compute rdf: https://docs.lammps.org/compute_rdf.html
- LAMMPS compute msd: https://docs.lammps.org/compute_msd.html
- CP2K MOTION/MD: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/MD.html
- lammpsio docs: https://lammpsio.readthedocs.io/

## Tool scope and examples

The CP2K/LAMMPS log summarizers accept result_root as a file or recursive directory, or result_files for explicit logs; multiple scientific logs in one directory require selection. `md_trajectory_summary` marks a truncated final frame partial, returns last_complete_frame separately and does not call it a successful final frame. Use observable_files for custom numeric tables. For quantitative analysis, select compute_msd/compute_rdf/make_plots and frame_start/stop/stride as needed. Wrapped source frames are unwrapped before thinning for MSD; fit indices refer to selected frames. Example: `analyze_trajectory(path="run/md.extxyz", frame_interval_fs=10, coordinate_semantics="wrapped", compute_msd=false, make_plots=false, frame_stride=5)` computes RDF only. Selected frames remain in memory.
