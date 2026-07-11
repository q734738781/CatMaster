---
name: mace-screening-and-relaxation
description: Use this skill for MACE-based screening, relaxation, single-point ranking, path work, and MD within a connected materials workflow.
---

# mace-screening-and-relaxation

## Overview
Use this skill to keep MACE inference calculations together when they operate on the same materials structures, including relaxation, single-point screening, path work, and MD.

## Quick Start
1. Prepare a clean stage directory containing an `input/` folder with the structures to evaluate.
2. Choose `mace_relax_dir` for geometry cleanup, `mace_sp_dir` for static ranking, or `mace_md_dir` for MD.
3. Pass calculator and run controls through `template_overrides` (`params` is the legacy alias).
4. Submit with `remote_submission` for one stage, or `remote_submission_batch` when the batch root has one first-level child stage per task.
5. Use stage-local outputs plus receipt/context fields to decide which candidates advance to VASP.

## Allowed tools
- `get_avail_remote_task`
- `remote_submission`
- `remote_submission_batch`

## Workflow

### 1. Choose the MACE task deliberately
- Use lightweight local filesystem/Python checks for paths and batch shape before launching managed MACE. Submit one intentional managed batch rather than probing remote execution for setup questions local inspection can answer.
- `mace_relax_dir` needs `input/`; it can toggle `model`, `head`, `dispersion`, `relax_lattice`, and `device` through `template_overrides`.
- Do not edit copied `task_script/` files or use `sitecustomize.py` to force MACE arguments. If the task's default `head`/`model`/relax controls are wrong for the request, set them in `template_overrides`; if that cannot express the required run, report the template gap.
- `mace_sp_dir` is for energy evaluation only and does not relax geometry.
- `mace_md_dir` uses grouped controls in `params/md_params.json`; keep the model, head, precision, ensemble, thermostat, timestep, temperature, and output intervals explicit.
- Do not compare relax and SP outputs as if they were the same screening stage.
- For geometry optimization with `mace_relax_dir`, keep `default_dtype=float64` by default. Only switch to `float32` when the user explicitly wants a cheaper, lower-rigor screening pass and the numerical looseness is acceptable.
- Managed MACE GPU tasks default to `device="auto"` so small or exploratory jobs can still produce results when the remote CUDA environment is unavailable. After completion, inspect `status.json` or `output/batch_summary.json` and report the actual device used; if it fell back to CPU, call it a performance downgrade. Use `device="cuda"` only when the user explicitly wants GPU validation or a GPU-required production run.

### 2. Keep connected MACE work together
- A materials workflow may continue from MACE relaxation into `mace_md_dir` without changing workers; preserve structure provenance and atom constraints across the stage transition.
- Use `dynamics_worker` instead when the primary work is a standalone dynamics protocol, restart, or trajectory-health investigation.

### 3. Use the stage layout expected by the remote task
- For `mace_sp_dir` and `mace_relax_dir`, the stage contains `input/`; `mace_md_dir` additionally contains `params/md_params.json`. Each writes downloaded results under `output/`.
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
- When a user asks for a specific MACE head such as OMOL or OMAT, set and later verify the rendered `head` explicitly. For MACE-mh-1, OMAT maps to the task's `omat_pbe` head string; OMOL should be submitted as `head=omol`, not silently substituted with the default.
- Treat `default_dtype=float64` as the conservative default for geometry relaxation. If you deliberately downgrade to `float32` for speed, say so explicitly in the run summary.
- Do not silently convert a relaxation or single-point request into MD sampling. MD controls and finite-time limitations must be explicit.

## Output Contract
Return:
- chosen MACE stage (`relax`, `sp`, `path`, or `md`)
- `work_dir_rel`
- `remote_context_id`, `submission_hash`, and `receipt_rel` when present
- shortlist or keep/drop rule for downstream VASP handoff

## References
- Dynamics-first MACE MD and deeper trajectory-health work may use `mace-md-sampling` in `dynamics_worker`.
- Use `vasp-input-preparation` only after a MACE shortlist exists; do not send the whole raw candidate pool forward by default.
- When the loop is ready for dataset building or retraining, hand off to `mace-dataset-curation` and `active-learning-relabel-loop`.
