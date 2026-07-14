---
name: mace-md-sampling
description: Use this skill for MACE-backed ASE MD sampling, thermal stability checks, trajectory generation, and trajectory-health analysis through the managed mace_md_dir remote task.
allowed-tools: "get_avail_remote_task remote_submission remote_submission_batch md_trajectory_summary analyze_trajectory execute"
---

# mace-md-sampling

## Overview
Use this skill for MACE MD in `dynamics_worker`. MACE relaxation, single-point ranking, and NEB/path optimization stay in `materials_worker`; MACE dataset construction, training, and evaluation stay in `ml_worker`.

## Quick Start
1. Verify the starting structure or structure batch path from the materials workflow.
2. Prepare a clean MACE MD stage with `input/` and grouped controls in `params/md_params.json`.
3. Submit with `remote_submission(task_name="mace_md_dir")` for one stage, or `remote_submission_batch` when each first-level child is one complete MACE MD stage.
4. Inspect `status.json`, `stdout.log`, `stderr.log`, and `output/batch_summary.json` after execution.
5. Summarize trajectories with `md_trajectory_summary` or `analyze_trajectory`, and write a focused script for system-specific observables.

## Allowed tools
- `get_avail_remote_task`
- `remote_submission`
- `remote_submission_batch`
- `md_trajectory_summary`
- `analyze_trajectory`
- `execute`

## Workflow

### 1. Prepare the stage layout
- Stage directory must contain `input/` with ASE-readable structures.
- Write grouped MD controls to `params/md_params.json`; pass `template_overrides={"params": "params/md_params.json"}` to `remote_submission`.
- Use the task catalog entry `mace_md_dir`. Do not submit MACE MD through `mace_relax_dir` or `mace_sp_dir`.

```text
stage/
  input/
    POSCAR or *.vasp/*.cif/*.xyz/*.traj
  params/
    md_params.json
```

### 2. Keep MD controls grouped
- Use one `md_config` object with optional `calculator`, `dynamics`, `thermostat`, `barostat`, and `output` groups.
- Acceleration is opt-in. The validated default for the current `mace_gpu` route is `calculator.enable_cueq=false` with compilation disabled. Accepted compile modes are `default`, `reduce-overhead`, and `max-autotune` when an explicitly validated workload needs one.
- Omitted defaults are NVT, 300 K, 1 fs, 1000 steps, Bussi thermostat, and trajectory/log every 10 steps.
- The runner always reads the last input frame. It preserves input momenta when present and generates Maxwell-Boltzmann velocities only when they are absent. Set `dynamics.reinitialize_velocities=true` only when the workflow explicitly requires replacing existing velocities.
- Set `dynamics.seed` explicitly for reproducible velocity generation and stochastic Bussi/Langevin sampling. The default is `2026`; batch inputs use `seed + sorted_input_index`, and each structure records its actual `rng_seed` in `summary.json`. A compatible CatMaster `restart.traj` restores its embedded RNG state instead of restarting from this seed.
- Set `dynamics.ensemble` to `nve`, `nvt`, or `npt`; choose thermostat/barostat keys only when the method needs them.
- For NPT Berendsen, set `barostat.compressibility_bar_inv` explicitly.

### 3. Parallelize independent MD trajectories
- Use only `task_name="mace_md_dir"` for MACE MD execution.
- Prepare one complete first-level stage per independent long MD trajectory and submit those stages with `remote_submission_batch`; do not place multiple independent production trajectories in one stage merely to reuse one model load, because the runner advances inputs in that stage sequentially. Let the remote scheduler determine how many trajectory stages run concurrently.
- Continuation segments of the same trajectory are dependent work, not parallel replicas: submit the next segment only after the preceding restart artifact is available.
- Managed MACE GPU tasks default to `device="auto"` so exploratory jobs can still produce results when CUDA is unavailable. After completion, inspect `status.json` or `output/batch_summary.json` and report the actual device used.
- Use `device="cuda"` only when the user explicitly asks for GPU validation or a GPU-required production run.

### 4. Analyze before interpreting
- Start from run-health evidence: completion, errors, final structure, trajectory files, log files, total time, and actual device.
- Use `md_trajectory_summary` or `analyze_trajectory` for generic frame/RDF/MSD artifacts when they match the requested observable.
- Write a focused script for residence time, adsorption/desorption, reaction labels, or region-specific observables.

## Method-critical defaults
- Default to `dynamics.ensemble="nvt"` with `thermostat.type="bussi"` for generic thermal sampling unless the scientific question requires energy conservation (`nve`) or pressure control (`npt`).
- Keep `calculator.default_dtype="float32"` by default for MD throughput. Use `float64` only when explicitly checking numerical sensitivity.
- For the current `mace_gpu` route (MACE-MH-1, RTX 4090), use `calculator.enable_cueq=false` and leave `calculator.compile_mode` disabled by default. A 32-atom Cu test remained fastest with this baseline at 500 and 2000 steps; cuEq was slower, while `reduce-overhead` compilation became much slower at 2000 steps despite matching energies and forces.
- Do not add a per-run benchmark stage or select acceleration from step count alone. Enable cuEq or compilation only when a recorded benchmark for the same model family, comparable system size, GPU class, and runtime stack supports it, or when the user explicitly requests an acceleration experiment.
- Do not combine cuEq with `reduce-overhead` on the current validated stack; that combination failed CUDA graph capture in the recorded smoke test.
- Select the cuEquivariance ops wheel from `torch.version.cuda`: CUDA 12.x uses `cuequivariance-ops-torch-cu12`, while CUDA 13.x uses `cuequivariance-ops-torch-cu13`. Do not infer the wheel from the NVIDIA driver version alone.
- For NPT, use only structures with a real 3D periodic cell; prefer `barostat.type="isotropic_mtk"` unless anisotropic cell fluctuations are part of the question.
- Keep timestep, steps, ensemble, thermostat/barostat, targets, total simulated time, dtype, device, dispersion, cuEq state, compile mode, total elapsed time, pre-step startup overhead, steady-state steps/s, and the step-timing CSV visible in summaries.
- For segmented runs, prefer the previous `restart.traj` because it always contains the true final step. New CatMaster `md.traj` files also embed matching RNG/integrator checkpoint metadata in every saved frame, so their last frame can be resumed exactly; legacy or external trajectories without that metadata preserve positions and velocities but start stochastic sampling from `dynamics.seed`. Check `velocity_source`, `rng_source`, and `integrator_state_source` in `summary.json`.
- Exact segmented continuation is currently supported for unchanged Bussi/CSVR runs; Langevin restores its random stream but has no additional evolving thermostat state. NHC/MTK extended chain and barostat states are not checkpointed, so do not claim exact continuation for those methods.
- Do not treat a completed short MACE MD run as converged diffusion or mechanistic evidence without a credible production window.

## Output Contract
Return:
- MACE MD stage path
- staged `params/md_params.json` path
- selected ensemble and grouped `calculator`, `dynamics`, `thermostat`, `barostat`, and `output` controls
- `work_dir_rel`
- `remote_context_id`, `submission_hash`, and `receipt_rel` when present
- `status.json`, `output/batch_summary.json`, and trajectory/log summary paths
- whether the result is production evidence or exploratory sampling

## References
- Local source note: `references/mace_md_sampling_reference.md`
- Project runner: `catmaster/remote/gpu/mace_md.py`
- Stage layout: `skills/execution/remote-stage-layouts/SKILL.md#mace_md_dir`
