---
name: cp2k-dft-preparation
description: "Use this skill for source-grounded CP2K conventional DFT preparation in materials workflows: single-point, fixed-cell geometry optimization, cell optimization, frequency, and DOS-style stages."
allowed-tools: "cp2k_prepare remote_submission remote_submission_batch get_avail_remote_task"
---

# cp2k-dft-preparation

## Overview
Use this skill when the task is a conventional CP2K DFT calculation stage owned by `materials_worker`. AIMD belongs to `dynamics_worker`.

## Quick Start
1. Verify the source structure path.
2. Use `cp2k_prepare` with `recipe="sp"`, `geo_opt`, `cell_opt`, `freq`, or `dos`.
3. Verify the prepared stage contains `job.inp` and `manifest.json`.
4. Submit with `remote_submission(task_name="cp2k_execute")` or `remote_submission_batch` for first-level batch children.
5. Parse accepted results with task-specific scripts or mature libraries; do not assume a universal CP2K analyzer.

## Allowed tools
- `cp2k_prepare`
- `remote_submission`
- `remote_submission_batch`
- `get_avail_remote_task`

## Workflow

### 1. Keep CP2K conventional DFT in materials_worker
- Use this skill for single point, geometry optimization, cell optimization, frequency, DOS/PDOS, and population-analysis setup.
- Do not use it for AIMD thermostat, barostat, trajectory, or restart workflows.

### 2. Prepare one clean stage per calculation
- `cp2k_prepare` writes `job.inp`, `input.xyz`, and `manifest.json`.
- `recipe="sp"` prepares an `ENERGY_FORCE` style calculation.
- `recipe="geo_opt"` prepares a fixed-cell geometry optimization.
- `recipe="cell_opt"` prepares a cell optimization and requests stress handling.
- `recipe="freq"` prepares a vibrational-analysis stage.
- `recipe="dos"` prepares a DOS-oriented energy stage with property output enabled.

### 3. Keep settings controlled
- Use the `settings` object only for supported knobs such as `xc`, `basis_set`, `potential`, `charge`, `multiplicity`, `cutoff`, `rel_cutoff`, `eps_scf`, `max_scf`, `kpoints`, `periodic`, `cell_abc`, `dispersion`, convergence tolerances, and `properties`.
- Project policy: do not insert arbitrary CP2K blocks through the LLM. If an unsupported section is required, write a reviewed local recipe or ask for confirmation.

### 4. Submit through the generic execute task
- Use only `task_name="cp2k_execute"` for CP2K execution.
- Do not invent per-recipe task names such as `cp2k_geo_opt_execute`.
- For a batch root, every first-level child must already be a complete CP2K stage.

## Method-critical defaults
- Report the chosen XC functional, basis, potential, cutoff, charge, multiplicity, periodicity, and k-points when they affect comparisons.
- Treat frequency, DOS/PDOS, and charge analysis as follow-up stages from an accepted structure, not as automatic analysis of every CP2K run.
- For surfaces or low-dimensional systems, make `periodic` explicit rather than silently relying on automatic inference.

## Output Contract
Return:
- prepared CP2K stage path
- selected recipe
- `manifest.json` path
- `cp2k_execute` receipt/context when submitted
- any unsupported setting or parser limitation

## References
- Local source note: `references/cp2k_dft_reference.md`
- CP2K manual index: https://manual.cp2k.org/
- CP2K geometry and cell optimization: https://manual.cp2k.org/trunk/methods/optimization/geometry_and_cell_opt.html
- CP2K FORCE_EVAL/PROPERTIES: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/PROPERTIES.html
- CP2K VIBRATIONAL_ANALYSIS: https://manual.cp2k.org/cp2k-2024_1-branch/CP2K_INPUT/VIBRATIONAL_ANALYSIS.html
- pymatgen CP2K module: https://pymatgen.org/pymatgen.io.cp2k.html
