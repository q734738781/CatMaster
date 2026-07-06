# MACE MD sampling reference notes

Checked: 2026-05-29

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
- The runner resolves `device="auto"` to CUDA when available and CPU otherwise; if `device="cuda"` is requested while CUDA is unavailable, it raises an environment error.

## SOP implications for the skill body

- Keep MACE MD in `dynamics_worker`, not `materials_worker`.
- Prepare `input/` and `params/md_params.json` before submission.
- Submit through `remote_submission(task_name="mace_md_dir")`; do not use the legacy `mace_md_batch` wrapper.
- Report actual device, dtype, total simulated time, ensemble, thermostat/barostat, and trajectory/log artifacts before scientific interpretation.
