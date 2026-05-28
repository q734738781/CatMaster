---
name: mace-screening-and-relaxation
description: Use this skill for MACE-based rapid screening, relaxation, single-point ranking, and short MD sampling before DFT, including candidate pruning and handoff criteria.
---

# mace-screening-and-relaxation

## Overview
Use this skill to run cheap MACE screening or short MACE MD sampling on a structure batch before spending VASP resources.

## Quick Start
1. Prepare a clean stage directory containing an `input/` folder with the structures to evaluate.
2. Choose `task_name="mace_relax_dir"` for geometry cleanup, `task_name="mace_sp_dir"` for static ranking, or `task_name="mace_md_dir"` for ASE-backed MD sampling.
3. Pass calculator and run controls through `params`; for MD, write the grouped MD controls to a staged JSON file and pass its path as `params.params_path`.
4. Submit with `remote_submission` for one stage, or `remote_submission_batch` when the batch root has one first-level child stage per task.
5. Use stage-local outputs plus receipt/context fields to decide which candidates advance to VASP.

## Allowed tools
- `get_avail_remote_task`
- `remote_submission`
- `remote_submission_batch`

## Workflow

### 1. Choose relax vs single-point deliberately
- Use lightweight local filesystem/Python checks for paths and batch shape before launching managed MACE. Submit one intentional managed batch rather than probing remote execution for setup questions local inspection can answer.
- `mace_relax_dir` needs `input/`; it can toggle `model`, `head`, `dispersion`, `relax_lattice`, and `device` through `params`.
- `mace_sp_dir` is for energy evaluation only and does not relax geometry.
- `mace_md_dir` needs `input/` plus a staged params JSON file, defaulting to `params/md_params.json`; it is for trajectory generation and thermal sampling, not a replacement for converged relaxation.
- Do not compare relax and SP outputs as if they were the same screening stage.
- For geometry optimization with `mace_relax_dir`, keep `default_dtype=float64` by default. Only switch to `float32` when the user explicitly wants a cheaper, lower-rigor screening pass and the numerical looseness is acceptable.
- Managed MACE GPU tasks default to `device="auto"` so small or exploratory jobs can still produce results when the remote CUDA environment is unavailable. After completion, inspect `status.json` or `output/batch_summary.json` and report the actual device used; if it fell back to CPU, call it a performance downgrade. Use `device="cuda"` only when the user explicitly wants GPU validation or a GPU-required production run.

### 2. Configure MACE MD through a staged params JSON
- Write MD controls into a stage-local JSON, normally `params/md_params.json`, and pass `params={"params_path": "params/md_params.json"}`.
- Use only needed groups: `calculator`, `dynamics`, `thermostat`, `barostat`, and `output`.
- Omitted defaults are NVT, 300 K, 1 fs, 1000 steps, Bussi thermostat, and trajectory/log every 10 steps.
- Set `dynamics.ensemble` to `nve`, `nvt`, or `npt`; choose thermostat/barostat keys only when the method needs them. For NPT Berendsen, set `barostat.compressibility_bar_inv` explicitly.

### 3. Use the stage layout expected by the remote task
- For `mace_sp_dir` and `mace_relax_dir`, the stage contains `input/` and the downloaded `output/` directory after completion.
- For `mace_md_dir`, the stage contains `input/`, `params/md_params.json`, and the downloaded `output/` directory after completion.
- For batch submission, every first-level child under `work_dir` must be one complete MACE stage; nested discovery is not performed.

### 4. Use collected evidence, not launch success alone
- Returned metadata includes `work_dir_rel`, `remote_context_id`, `submission_hash`, `receipt_rel`, and `task_state_counts`.
- Stage-local evidence includes `status.json`, `stdout.log`, `stderr.log`, and remote runner outputs such as `output/batch_summary.json`.
- On dispatch failure, inspect receipt/context fields and stage-local evidence before deciding to resubmit; the remote work may still be live.

### 5. Use this skill while the workflow artifact is still materials-side
- Use this skill for structure batches, candidate ranking, geometry cleanup, and materials-side post-analysis before expensive reference calculations.
- If the screening run produces a shortlist that should become a training dataset or an active-learning update, hand off that artifact to the ML skills as the next step.

## Method-critical defaults
- For adsorption-energy screening on slabs, choose `dispersion` explicitly and keep it consistent across clean slab, gas reference, and adsorbed structures. Prefer enabling dispersion unless the user asked for a no-dispersion baseline.
- Always report whether dispersion was enabled.
- If a screening stage is intended only as a cheap geometry triage rather than an energy-ranking stage, say so explicitly.
- Treat `default_dtype=float64` as the conservative default for geometry relaxation. If you deliberately downgrade to `float32` for speed, say so explicitly in the run summary.
- For MACE MD, default to `dynamics.ensemble="nvt"` with `thermostat.type="bussi"` for generic sampling unless the scientific question requires energy conservation (`nve`) or pressure control (`npt`).
- For MACE MD, keep `default_dtype=float32` by default for throughput. Use `float64` only when explicitly checking numerical sensitivity.
- For NPT, use only structures with a real 3D periodic cell; prefer `barostat.type="isotropic_mtk"` unless anisotropic cell fluctuations are part of the question.
- Keep timestep, steps, ensemble, thermostat/barostat, targets, total time, dtype, device, and dispersion visible in summaries.

## Output Contract
Return:
- chosen MACE stage (`relax` or `sp`)
- for MD, chosen ensemble plus staged params JSON path and grouped `calculator`, `dynamics`, `thermostat`, `barostat`, and `output` controls
- `work_dir_rel`
- `remote_context_id`, `submission_hash`, and `receipt_rel` when present
- shortlist or keep/drop rule for downstream VASP handoff

## References
- Use `vasp-input-preparation` only after a MACE shortlist exists; do not send the whole raw candidate pool forward by default.
- When the loop is ready for dataset building or retraining, hand off to `mace-dataset-curation` and `active-learning-relabel-loop`.
