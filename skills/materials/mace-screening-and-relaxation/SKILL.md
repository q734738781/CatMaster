---
name: mace-screening-and-relaxation
description: Use this skill for MACE-based rapid screening, relaxation, single-point ranking, and short MD sampling before DFT, including candidate pruning and handoff criteria.
---

# mace-screening-and-relaxation

## Overview
Use this skill to run cheap MACE screening or short MACE MD sampling on a structure batch before spending VASP resources.

## Quick Start
1. Prepare a clean structure batch under `input_dir`.
2. Choose `mace_relax_batch` for geometry cleanup, `mace_sp_batch` for static ranking, or `mace_md_batch` for ASE-backed MD sampling.
3. Choose `model`, `default_dtype`, dispersion policy, and any shortlist criterion needed for the next VASP or ML step before dispatch.
4. Keep `output_root` outside `input_dir`.
5. Use the collected outputs and batch-state files to decide which candidates advance to VASP.

## Allowed tools
- `mace_relax_batch`
- `mace_sp_batch`
- `mace_md_batch`

## Workflow

### 1. Choose relax vs single-point deliberately
- Use lightweight local filesystem/Python checks for paths and batch shape before launching managed MACE. Submit one intentional managed batch rather than probing remote execution for setup questions local inspection can answer.
- `mace_relax_batch` needs a `model`; it can also toggle `head`, `dispersion`, and `relax_lattice`.
- `mace_sp_batch` is for energy evaluation only and does not relax geometry.
- `mace_md_batch` is for trajectory generation and thermal sampling, not a replacement for a converged relaxation.
- Do not compare relax and SP outputs as if they were the same screening stage.
- For geometry optimization with `mace_relax_batch`, keep `default_dtype=float64` by default. Only switch to `float32` when the user explicitly wants a cheaper, lower-rigor screening pass and the numerical looseness is acceptable.

### 2. Configure MACE MD through `md_config`
- The tool schema intentionally exposes only one free-form `md_config` object for MD controls. Do not expect the tool schema to list every ASE parameter.
- Keep calculator-level basics (`model`, `head`, `dispersion`, `default_dtype`) in the top-level tool arguments unless you need advanced calculator switches.
- Inside `md_config`, use optional groups named `calculator`, `dynamics`, `thermostat`, `barostat`, and `output`.
- Use only the keys needed by the chosen method; do not fill thermostat/barostat keys just because they exist in a template.
- The remote runner supplies defaults for omitted keys: NVT, 300 K, 1 fs, 1000 steps, Bussi thermostat, trajectory/log every 10 steps.

Common `md_config` templates:

```json
{
  "dynamics": {"ensemble": "nve", "temperature_K": 300, "timestep_fs": 1.0, "steps": 1000}
}
```

```json
{
  "dynamics": {"ensemble": "nvt", "temperature_K": 300, "timestep_fs": 1.0, "steps": 1000},
  "thermostat": {"type": "langevin", "friction_per_fs": 0.01}
}
```

```json
{
  "dynamics": {"ensemble": "nvt", "temperature_K": 300, "timestep_fs": 1.0, "steps": 1000},
  "thermostat": {"type": "nhc", "tau_fs": 100, "tchain": 3, "tloop": 1}
}
```

```json
{
  "dynamics": {"ensemble": "npt", "temperature_K": 300, "timestep_fs": 1.0, "steps": 1000},
  "thermostat": {"tau_fs": 100},
  "barostat": {"type": "isotropic_mtk", "pressure_bar": 1.01325, "pdamp_fs": 1000}
}
```

ASE mapping:
- `dynamics.ensemble="nve"` maps to `VelocityVerlet`.
- NVT `thermostat.type` maps as `bussi` -> `Bussi`, `nhc` -> `NoseHooverChainNVT`, `langevin` -> `Langevin`, and `berendsen` -> `NVTBerendsen`.
- NPT `barostat.type` maps as `isotropic_mtk` -> `IsotropicMTKNPT`, `full_mtk` -> `MTKNPT`, and `berendsen` -> `NPTBerendsen`.
- For NPT Berendsen, include `barostat.compressibility_bar_inv`; it is system-specific.
- For output cadence, set `md_config.output` with `traj_interval`, `log_interval`, `log_stress`, or `overwrite` only when defaults are unsuitable.

### 3. Keep input and output trees separate
- Both tools reject `output_root` inside `input_dir`.
- The runtime stages a temporary batch tree under `output_root`, dispatches remotely, collects outputs, then removes the staging tree.

### 4. Use collected evidence, not launch success alone
- Returned metadata includes `batch_state_rel`, collected stdout/stderr/status files, and any `batch_summary_rel`.
- On dispatch failure, the tool still tries to collect partial outputs; inspect those before deciding to rerun.

### 5. Use this skill while the workflow artifact is still materials-side
- Use this skill for structure batches, candidate ranking, geometry cleanup, and materials-side post-analysis before expensive reference calculations.
- If the screening run produces a shortlist that should become a training dataset or an active-learning update, hand off that artifact to the ML skills as the next step.

## Method-critical defaults
- For adsorption-energy screening on slabs, do not silently inherit the tool default for `dispersion`; choose it explicitly.
- Keep the dispersion setting consistent across clean slab, gas-phase reference, and adsorbed structures when the comparison depends on relative adsorption energies.
- Unless the user explicitly asks for a no-dispersion baseline, prefer enabling dispersion when surface-adsorbate interactions or ranking sensitivity may depend on it.
- Always report whether dispersion was enabled.
- If a screening stage is intended only as a cheap geometry triage rather than an energy-ranking stage, say so explicitly.
- Treat `default_dtype=float64` as the conservative default for geometry relaxation. If you deliberately downgrade to `float32` for speed, say so explicitly in the run summary.
- For MACE MD, default to `dynamics.ensemble="nvt"` with `thermostat.type="bussi"` for generic sampling unless the scientific question requires energy conservation (`nve`) or pressure control (`npt`).
- For MACE MD, keep `default_dtype=float32` by default for throughput. Use `float64` only when explicitly checking numerical sensitivity.
- For NPT, only use structures with a real periodic cell and periodic boundary conditions in all three directions.
- For NPT Berendsen, set `barostat.compressibility_bar_inv` explicitly because it is system-specific; do not invent it silently for solids.
- Prefer `barostat.type="isotropic_mtk"` for generic NPT unless anisotropic cell fluctuations are part of the question. Use `full_mtk` deliberately and report it.
- Keep `timestep_fs`, `steps`, ensemble, thermostat, barostat, target pressure, target temperature, and total time visible in summaries.

## Output Contract
Return:
- chosen MACE stage (`relax` or `sp`)
- for MD, chosen ensemble plus grouped `calculator`, `dynamics`, `thermostat`, `barostat`, and `output` controls
- `output_root_rel`
- `batch_state_rel`
- shortlist or keep/drop rule for downstream VASP handoff

## References
- Use `vasp-input-preparation` only after a MACE shortlist exists; do not send the whole raw candidate pool forward by default.
- When the loop is ready for dataset building or retraining, hand off to `mace-dataset-curation` and `active-learning-relabel-loop`.
