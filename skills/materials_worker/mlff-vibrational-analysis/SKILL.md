---
name: mlff-vibrational-analysis
description: Use this skill for general constrained MLFF normal-mode analysis of minima, transition states, adsorbates, molecules, and material structures after a geometry has been accepted.
license: project-local
allowed-tools: "ls read_file write_file edit_file execute get_avail_remote_task get_remote_task_spec remote_submission remote_submission_batch"
---

# mlff-vibrational-analysis

## Overview

Compute a complete constrained harmonic spectrum without turning the task into transition-state optimization or phonon-dispersion analysis.

## Quick Start

1. Put accepted structures directly under `stage/input/` with constraints encoded in POSCAR/VASP or extxyz.
2. Query `get_remote_task_spec` for `mlff_vib` and the selected backend.
3. Submit one stage with `remote_submission`; use a remote batch for independent configurations that should run concurrently.
4. Inspect projected force, resolved Hessian method, frequencies, mode vectors, and stationary-point classification.

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

## Workflow

### 1. Define the physical mode space

- Use the structure-file constraints as the only free-coordinate definition.
- For adsorbate-only thermochemistry, freeze every slab atom and leave only the adsorbate coordinates mobile.
- `FixAtoms`, Cartesian-component `FixCartesian`, and POSCAR/VASP scaled-coordinate `FixScaled` constraints are projected exactly.

### 2. Select the Hessian path

- Keep `hessian_method=auto` unless comparing methods.
- Auto uses a calculator's public analytic Hessian when available and otherwise evaluates the full constrained Hessian by finite differences.
- Use `nfree=2` normally. Use `nfree=4` only for a deliberate stencil comparison because it doubles the force-evaluation count.

### 3. Interpret without assuming TS semantics

- `stationary_point=false` means the projected force exceeds the chosen threshold; frequencies may still be diagnostic but are not a validated harmonic analysis.
- Zero significant imaginary modes at a stationary geometry indicates a local minimum in the analyzed subspace.
- One indicates a first-order saddle; more than one indicates a higher-order saddle.
- For a periodic structure, this task is a finite-supercell Gamma-point normal-mode analysis, not a phonon dispersion calculation.

### 4. Preserve compact scientific artifacts

- `vibrations.npz` is the canonical result and contains geometry, masses, constraint basis, reduced Hessian, frequencies, and mass-normalized full-atom modes.
- `frequencies.csv` is the human-readable mode table.
- `modes.extxyz` is one multi-frame viewer/interchange file with one mode vector field per frame.
- Do not create ASE `Vibrations.run()` displacement caches or one text file per mode.

## Method-critical defaults

- Keep the same backend model, head/task, charge, spin, and precision used for the associated energy workflow when quantitative comparison matters.
- `hessian_delta=0.01` Angstrom and `nfree=2` are the normal finite-difference starting point.
- Compare displacement sizes when force noise or very soft modes affect the conclusion.
- The imaginary-frequency threshold classifies modes; it does not change the Hessian.
- Do not feed translational, rotational, or constrained near-zero modes blindly into thermochemistry.

## Output Contract

Return:

- input stage and backend/model metadata;
- free and constrained degrees of freedom;
- projected-force stationarity check and stationary-point class;
- resolved Hessian method, displacement settings, force-evaluation count, and Hessian asymmetry diagnostic;
- significant imaginary-mode count and frequency range;
- `vibrations.npz`, `frequencies.csv`, `modes.extxyz`, `summary.json`, and the remote receipt/context.

## References

- Use `thermo-free-energy-and-reporting` after mode selection for thermochemical corrections.
- Use `mlff-transition-state-refinement` when the geometry itself still requires constrained RS-pRFO refinement.
- Use `phonon-displacement-workflow` for q-dependent periodic phonons rather than this finite-supercell Gamma-point analysis.
