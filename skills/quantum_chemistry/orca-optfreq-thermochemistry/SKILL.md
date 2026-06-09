---
name: orca-optfreq-thermochemistry
description: Use this skill for bounded ORCA molecular optimization, frequency, opt+freq thermochemistry, TDDFT, or NMR preparation/execution/analysis after the molecular structure set is already chosen.
---

# orca-optfreq-thermochemistry

## Overview
Use this skill when the task is one ORCA molecular batch rooted in a known structure set.

## Quick Start
1. Prepare the batch with `orca_prepare`.
2. Submit the prepared ORCA stage with `remote_submission` or `remote_submission_batch` using `task_name="orca_execute"`.
3. Close the loop with `analyze_orca_results`.
4. If a downstream stage needs only the accepted optimized geometries, collect them with `extract_optimized_molecules`.

## Allowed tools
- `orca_prepare`
- `remote_submission`
- `remote_submission_batch`
- `analyze_orca_results`
- `extract_optimized_molecules`

## Workflow

### 1. Keep the ORCA task explicit
- Use `task="sp"` for single-point refinement only.
- Use `task="opt"` for geometry optimization.
- Use `task="freq"` or `task="optfreq"` when thermochemistry or vibrational validation matters.
- Use `task="td"` or `task="nmr"` only after the geometry footing is clear.

### 2. Do not mix preparation and acceptance
- Submission success is not the same thing as a chemically acceptable result.
- Always report convergence state, final structure path, and any imaginary-frequency count from `analyze_orca_results`.

## Method-critical defaults
- Parameter priority: honor explicit user requirements first; otherwise choose ORCA method, basis, dispersion, solvation, charge, spin, and tightness from the molecule class and task objective; if that judgment remains uncertain, run a narrow literature or official documentation check before finalizing the override.
- Do not add ORCA overrides just to restate the tool baseline; only override when the user, molecule class, task objective, or a checked source justifies it.
- The `orca_prepare` auto level is workflow-layered: `opt`, `freq`, and `optfreq` resolve to `r2SCAN-3c` with no separate basis keyword; `sp`, `td`, and `nmr` resolve to `WB97X-D4/def2-TZVP`.
- Prefer `r2SCAN-3c` for large-molecule initial optimization, conformer screening/refinement, frequency or thermal-correction stages tied to the optimized geometry, and noncovalent-complex structures unless the user or checked source indicates a different level.
- For final reaction barriers, comparison-sensitive relative energies, charge-transfer/long-range interaction cases, and excited states, add a higher-level hybrid/TZ-or-larger single-point/property stage rather than treating `r2SCAN-3c` as the final evidence by default.
- For publication-facing final energies, consider a hybrid/TZ-or-larger single point or a DLPNO-CCSD(T)-style calibration stage when system size and task objective justify it.
- For transition-metal complexes, `r2SCAN-3c` is an acceptable quick optimization trial, but benchmark or cross-check the level before using it as final evidence.

## Output Contract
Return:
- ORCA batch root
- ORCA summary path
- extracted optimized-structure directory when generated
