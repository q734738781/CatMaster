---
name: cp2k-aimd-preparation
description: Use this skill to author, stage, execute, and inspect complete native CP2K AIMD inputs for NVE, NVT, NPT, restart, or explicit PLUMED workflows.
allowed-tools: "cp2k_prepare cp2k_output_summary remote_submission remote_submission_batch get_avail_remote_task md_trajectory_summary analyze_trajectory execute"
---

# CP2K AIMD preparation

## Workflow

1. Choose the ensemble, timestep, length, thermostat/barostat parameters, initial velocities, and output/restart strides from the sampling objective.
2. Author a complete native CP2K input. Take only the requested MD, thermostat, output, or restart syntax from [cp2k_aimd_native_examples.md](references/cp2k_aimd_native_examples.md). Select the ensemble, electronic method, system, and numerical settings independently with [the CP2K numerical-settings reference](../../materials_worker/cp2k-dft-preparation/references/cp2k_numerical_settings.md) when needed.
3. Call `cp2k_prepare` with the input path and explicit mappings for structures, `EXT_RESTART`, wavefunctions, PLUMED input/state, or other referenced files.
4. Submit with `remote_submission(task_name="cp2k_execute")` or batch independent stages.
5. Use `cp2k_output_summary` for execution/SCF evidence and `md_trajectory_summary` for trajectory and native-observable inventory. For `analyze_trajectory`, supply the stored-frame interval and explicit wrapped/unwrapped semantics for periodic generic XYZ/ASE files; provide a fit window when computing MSD. RDF-only can use `compute_msd=false`.

## Scientific choices

- Keep NVE, NVT, NPT, thermostat, barostat, timestep, and stride choices visible in `job.inp`.
- Do not invent PLUMED collective variables. Stage the user- or method-authored PLUMED files explicitly.
- A frame count is not equilibration evidence. Interpret drift and sampling against the intended ensemble and observable.

## Handoff

Return the AIMD stage, ensemble and sampling settings, CP2K summary, trajectory inventory, and any requested quantitative analysis.

## References

- [Native CP2K AIMD task fragments](references/cp2k_aimd_native_examples.md)
- [Choosing CP2K numerical settings](../../materials_worker/cp2k-dft-preparation/references/cp2k_numerical_settings.md)
- [CP2K MOTION/MD](https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/MD.html)
- [CP2K EXT_RESTART](https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/EXT_RESTART.html)
