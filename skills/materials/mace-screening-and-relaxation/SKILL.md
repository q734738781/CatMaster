---
name: mace-screening-and-relaxation
description: Use this skill for MACE-based rapid screening and relaxation loops before DFT, including candidate pruning and handoff criteria.
---

# mace-screening-and-relaxation

## Overview
Use this skill to run cheap MACE screening on a structure batch before spending VASP resources.

## Quick Start
1. Prepare a clean structure batch under `input_dir`.
2. Choose `mace_relax_batch` for geometry cleanup or `mace_sp_batch` for static ranking.
3. Keep `output_root` outside `input_dir`.
4. Use the collected outputs and batch-state files to decide which candidates advance to VASP.

## Allowed tools
- `mace_relax_batch`
- `mace_sp_batch`

## Workflow

### 1. Choose relax vs single-point deliberately
- `mace_relax_batch` needs a `model`; it can also toggle `head`, `dispersion`, and `relax_lattice`.
- `mace_sp_batch` is for energy evaluation only and does not relax geometry.
- Do not compare relax and SP outputs as if they were the same screening stage.
- For geometry optimization with `mace_relax_batch`, keep `default_dtype=float64` by default. Only switch to `float32` when the user explicitly wants a cheaper, lower-rigor screening pass and the numerical looseness is acceptable.

### 2. Keep input and output trees separate
- Both tools reject `output_root` inside `input_dir`.
- The runtime stages a temporary batch tree under `output_root`, dispatches remotely, collects outputs, then removes the staging tree.

### 3. Use collected evidence, not launch success alone
- Returned metadata includes `batch_state_rel`, collected stdout/stderr/status files, and any `batch_summary_rel`.
- On dispatch failure, the tool still tries to collect partial outputs; inspect those before deciding to rerun.

### 4. Use this skill while the workflow artifact is still materials-side
- Use this skill for structure batches, candidate ranking, geometry cleanup, and materials-side post-analysis before expensive reference calculations.
- If the screening run produces a shortlist that should become a training dataset or an active-learning update, hand off that artifact to the ML skills as the next step.

## Method-critical defaults
- For adsorption-energy screening on slabs, do not silently inherit the tool default for `dispersion`; choose it explicitly.
- Keep the dispersion setting consistent across clean slab, gas-phase reference, and adsorbed structures when the comparison depends on relative adsorption energies.
- Unless the user explicitly asks for a no-dispersion baseline, prefer enabling dispersion when surface-adsorbate interactions or ranking sensitivity may depend on it.
- Always report whether dispersion was enabled.
- If a screening stage is intended only as a cheap geometry triage rather than an energy-ranking stage, say so explicitly.
- Treat `default_dtype=float64` as the conservative default for geometry relaxation. If you deliberately downgrade to `float32` for speed, say so explicitly in the run summary.

## Output Contract
Return:
- chosen MACE stage (`relax` or `sp`)
- `output_root_rel`
- `batch_state_rel`
- shortlist or keep/drop rule for downstream VASP handoff

## References
- Use `vasp-input-preparation` only after a MACE shortlist exists; do not send the whole raw candidate pool forward by default.
- When the loop is ready for dataset building or retraining, hand off to `mace-dataset-curation` and `active-learning-relabel-loop`.
