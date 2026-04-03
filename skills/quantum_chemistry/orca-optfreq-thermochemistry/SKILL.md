---
name: orca-optfreq-thermochemistry
description: Use this skill for bounded ORCA molecular optimization, frequency, opt+freq thermochemistry, TDDFT, or NMR preparation/execution/analysis after the molecular structure set is already chosen.
license: project-local
compatibility: local
allowed-tools: "orca_prepare orca_execute_batch analyze_orca_results extract_optimized_molecules"
---

# orca-optfreq-thermochemistry

## Overview
Use this skill when the task is one ORCA molecular batch rooted in a known structure set.

## Quick Start
1. Prepare the batch with `orca_prepare`.
2. Submit with `orca_execute_batch`.
3. Close the loop with `analyze_orca_results`.
4. If a downstream stage needs only the accepted optimized geometries, collect them with `extract_optimized_molecules`.

## Allowed tools
- `orca_prepare`
- `orca_execute_batch`
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

## Output Contract
Return:
- ORCA batch root
- ORCA summary path
- extracted optimized-structure directory when generated

