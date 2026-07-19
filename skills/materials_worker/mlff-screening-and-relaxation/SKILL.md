---
name: mlff-screening-and-relaxation
description: Use this skill for MLFF single-point screening, ranking, and geometry relaxation when choosing among enabled MACE, FairChem UMA, MatterSim, or ORB-v3 backends.
license: project-local
allowed-tools: "ls read_file write_file edit_file execute get_avail_remote_task get_remote_task_spec remote_submission remote_submission_batch"
---

# mlff-screening-and-relaxation

## Overview

Prepare deterministic SP/relax stages, choose an enabled backend, and keep backend settings separate from operation settings.

## Quick Start

1. Read `remote-stage-layouts`, then put uniquely named structures directly under a clean `input/`.
2. Call `get_avail_remote_task`, then query the intended backend directly with `detail="full"`, for example `get_remote_task_spec(task_name="mlff_sp", template_overrides={"backend": "mattersim"}, detail="full")`; do not query `{}` first when selecting a non-default backend.
3. Pass only nested `backend`, `backend_config`, and `task_config` overrides returned by that query.
4. Submit one stage with `remote_submission`, or a parent of complete first-level stages with `remote_submission_batch`. Leave `submission_config.resources` and `.machine` unset; the selected backend owns them.
5. Inspect `output/batch_summary.json`, per-input summaries, and remote receipt/context fields.

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

### 1. Select operation before backend

- Use `mlff_sp` for energies/forces on unchanged geometries and `mlff_relax` when positions or the cell must be optimized.
- Query the concrete backend schema before overrides. Do not reuse MACE keys after switching to UMA, MatterSim, or ORB-v3.
- Empty overrides use the administrator-enabled default backend. If no default is effective, choose one of the returned `available_backends` explicitly.

### 2. Build one deterministic stage

- Put files directly under `input/`; do not submit recursive project trees.
- Preserve atom constraints and source provenance. Reject filename-stem collisions such as `case.xyz` plus `case.vasp` in one stage.
- For UMA, set shared physical metadata in `backend_config.defaults` and exceptions in `backend_config.items`, keyed by exact filenames relative to `input/`.

### 3. Group for model reuse

- One stage initializes one selected model and processes its structures sequentially.
- For similarly sized short relaxations, 30-50 structures per stage is an empirical starting point, not a probe requirement or hard limit. Use smaller groups for heterogeneous costs.
- SP can use substantially larger groups. Use `remote_submission_batch` only when the parent contains several independent complete stages with the same overrides.

### 4. Interpret as MLFF evidence

- Verify backend, model/checkpoint, provider version, precision/device, method settings, per-input errors, and actual outputs.
- Use a consistent backend/model/domain task across structures being ranked unless a difference is scientifically intentional and reported.
- Treat MLFF screening as triage unless the user explicitly accepts ML-potential accuracy for the conclusion.

## Method-critical defaults

- MACE is the initial deployment default: `model=mh-1`, `head=omat_pbe`, `dispersion=false`, and `float64` for SP/relax. State any change that affects comparability.
- Choose dispersion explicitly for adsorption-energy comparisons and apply the same choice to clean slab, adsorbate, and references.
- Keep `relax_cell=false` unless periodic-cell optimization is intended; cell relaxation requires a valid fully periodic cell.
- For UMA `omol`, set charge and multiplicity-style spin explicitly. Non-`omol` UMA tasks require charge and spin zero.
- MatterSim-v1 is a bulk-material model; do not present surface, interface, or long-range-interaction results as quantitative without validation.
- ORB-v3 defaults to `precision=float32-high`; change precision only as a declared numerical choice.

## Output Contract

Return:

- operation, backend, model/checkpoint, and method-critical overrides;
- stage path or batch root with first-level stage count;
- `work_dir_rel`, `remote_context_id`, `submission_hash`, and `receipt_rel` when present;
- `output/batch_summary.json`, per-input error count, and shortlist/keep-drop rule;
- the boundary between MLFF screening and any downstream reference calculation.

## References

- MACE fields and model semantics: `references/mace.md`
- FairChem UMA domain and charge/spin semantics: `references/fairchem_uma.md`
- MatterSim scope and model identity: `references/mattersim.md`
- ORB-v3 precision and confidence metadata: `references/orb_v3.md`
