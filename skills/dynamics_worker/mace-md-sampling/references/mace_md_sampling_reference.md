# MACE MD sampling reference notes

Checked: 2026-07-11

This file is a local, source-grounded note for `mace-md-sampling`.

## Project sources

- Remote runner: `catmaster/remote/gpu/mace_md.py`
- DPDispatcher task: `configs/dpdispatcher/tasks.yaml#mace_md_dir`
- Remote layout skill: `skills/execution/remote-stage-layouts/SKILL.md#mace_md_dir`

## Practical notes

- `mace_md_dir` runs the project remote script `mace_md.py` over structures under `input/` and writes outputs under `output/`.
- The remote script loads grouped controls from a params JSON. The compact payload shape is `{"md_config": ...}` plus calculator overrides such as `model`, `head`, `dispersion`, and `default_dtype`.
- Supported dynamics ensembles are `nve`, `nvt`, and `npt`.
- Supported thermostat types are `bussi`, `nhc`, `langevin`, and `berendsen`.
- Supported NPT barostat types are `isotropic_mtk`, `full_mtk`, and `berendsen`; Berendsen NPT requires `compressibility_bar_inv`.
- Defaults in the runner are NVT, 300 K, 1 fs, 1000 steps, Bussi thermostat, `float32`, and trajectory/log output every 10 steps.
- MACE 0.3.16 exposes `enable_cueq` and `compile_mode` on the ASE calculator. CatMaster accepts `default`, `reduce-overhead`, and `max-autotune` compile modes and records elapsed time plus steps/s.
- cuEquivariance ops wheels follow the CUDA ABI reported by `torch.version.cuda`, not the maximum CUDA generation supported by the installed driver.
- The runner resolves `device="auto"` to CUDA when available and CPU otherwise; if `device="cuda"` is requested while CUDA is unavailable, it raises an environment error.

## Recorded RTX 4090 benchmark

Validated on 2026-07-11 through the real `mace_gpu` DPDispatcher route with MACE 0.3.16, MACE-MH-1/`omat_pbe`, a 32-atom Cu cell, float32, NVE at 300 K, and a 1 fs timestep. All variants agreed in final energy within about `2.3e-5 eV` and in maximum force within about `4.8e-6 eV/A`.

| Steps | Baseline | cuEq | `reduce-overhead` |
| ---: | ---: | ---: | ---: |
| 500 | 21.3 s | 33.5 s | 46.5 s |
| 2000 | 73.7 s | 99.5 s | 467.5 s |

Use baseline inference (`enable_cueq=false`, compilation disabled) as the empirical default for this route. The 2000-step compiled run alternated roughly 0.01 s and 0.43 s steps, so its slower total was not merely one-time startup cost. Do not infer that compilation will win from a short-run median or from step count alone. Revisit the default only with a recorded benchmark for a materially different model, system-size regime, GPU class, or runtime stack; do not make every normal MD job run its own benchmark.

## SOP implications for the skill body

- Keep MACE MD in `dynamics_worker`, not `materials_worker`.
- Prepare `input/` and `params/md_params.json` before submission.
- Submit through `remote_submission(task_name="mace_md_dir")`; do not use the legacy `mace_md_batch` wrapper.
- Report actual device, dtype, total simulated time, ensemble, thermostat/barostat, and trajectory/log artifacts before scientific interpretation.
