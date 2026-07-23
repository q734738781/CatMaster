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
2. Call `get_avail_remote_task`, then query the intended backend and exact official model name directly with `detail="full"`, for example `get_remote_task_spec(task_name="mlff_sp", template_overrides={"backend": "mace", "backend_config": {"model": "omol-0"}}, detail="full")` or `template_overrides={"backend": "mattersim", "backend_config": {"model": "MatterSim-v1.0.0-1M"}}`; do not invent a model, head, or task name.
3. Pass only nested `backend`, `backend_config`, and `task_config` overrides returned by that query.
4. Submit one stage with `remote_submission`; submit two or more independent same-config stages with one `remote_submission_batch`. Leave `submission_config.resources` and `.machine` unset.
5. Inspect `output/batch_summary.json` and per-input summaries. Use receipt recovery only after a returned failure.

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
- Query the concrete backend/model schema before overrides. Do not reuse MACE keys after switching to UMA, MatterSim, or ORB-v3, and do not treat a MACE-MH-1 head as a separate model.
- Use only `enabled_models` and the selected entry in `model_capabilities`. The catalog uses exact provider model names for UMA, MatterSim, and ORB-v3; do not abbreviate or normalize their capitalization.
- Empty overrides use the administrator-enabled default backend. If no default is effective, choose one of the returned `available_backends` explicitly.

### 2. Build one deterministic stage

- Put files directly under `input/`; do not submit recursive project trees.
- Preserve atom constraints and source provenance. Reject filename-stem collisions such as `case.xyz` plus `case.vasp` in one stage.
- For registered MACE `omol-0`, set shared charge and multiplicity-style spin in `backend_config.defaults` and exceptions in `backend_config.items`, keyed by exact filenames relative to `input/`. UMA uses the same item pattern plus `uma_task`.

### 3. Group for model reuse

- One stage initializes one selected model and processes its structures sequentially.
- For similarly sized short relaxations, 30-50 structures per stage is an empirical starting point, not a probe requirement or hard limit. Use smaller groups for heterogeneous costs.
- SP can use substantially larger groups. Split only into complete, independently runnable stages.

### 4. Interpret as MLFF evidence

- Verify backend, model/checkpoint, provider version, precision/device, method settings, per-input errors, and actual outputs.
- Use a consistent backend/model/domain task across structures being ranked unless a difference is scientifically intentional and reported.
- Treat MLFF screening as triage unless the user explicitly accepts ML-potential accuracy for the conclusion.

## Method-critical defaults

- MACE is the initial deployment default: `model=mh-1`, `head=omat_pbe`, `dispersion=false`, and `float64` for SP/relax. MACE-MH-1 heads are model-specific choices, not model aliases. Use registered `model=omol-0`, `head=omol`, and explicit charge/spin when the standalone charge/spin-aware MACE-OMOL model is required.
- Choose dispersion explicitly for adsorption-energy comparisons and apply the same choice to clean slab, adsorbate, and references.
- Keep `relax_cell=false` unless periodic-cell optimization is intended; cell relaxation requires a valid fully periodic cell.
- UMA model and task support is model-specific: `uma-s-1p2` exposes seven official tasks, while `uma-s-1p1` and `uma-m-1p1` do not expose `oc22` or `oc25`. `auto` is not an official UMA task. For `omol`, set charge and multiplicity-style spin explicitly; non-`omol` tasks require both values to be zero.
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
