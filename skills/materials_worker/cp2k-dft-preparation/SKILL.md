---
name: cp2k-dft-preparation
description: Use this skill to author and stage complete native CP2K inputs for materials single points, geometry or cell optimization, and related DFT calculations.
allowed-tools: "cp2k_prepare cp2k_output_summary remote_submission remote_submission_batch get_avail_remote_task execute"
---

# CP2K DFT preparation

## Workflow

1. Decide the scientific method from the structure and requested result: periodicity, charge/spin, XC, basis/potential, cutoff, SCF algorithm, k-points, dispersion, and run type.
2. Author a complete native CP2K input file. Take only the requested task or algorithm syntax from [cp2k_native_examples.md](references/cp2k_native_examples.md); select the electronic method, system, and numerical settings independently. When numerical settings are not already validated for the same method family and target property, consult [cp2k_numerical_settings.md](references/cp2k_numerical_settings.md).
3. Call `cp2k_prepare` with the input path, a fresh `output_root`, and explicit `asset_mappings` for every referenced restart, coordinate, basis, potential, include, or other file. The tool copies the input unchanged to `job.inp`.
4. Submit one stage with `remote_submission(task_name="cp2k_execute")`; use `remote_submission_batch` for independent same-config stages. It selects first-level children by default; use relative `stage_paths` for a selected subset or nested stages.
5. Use `cp2k_output_summary` for process, SCF, optimization, energy, and requested-frequency states. Parse specialized property files with a focused script when needed.

Before analyzing custom-named logs or a mixed result tree, read the [CP2K output-selection guidance](../../dynamics_worker/cp2k-run-analysis/SKILL.md). It explains explicit `result_files`, ambiguous sources and per-run failures.

## Scientific choices

- Use `RUN_TYPE ENERGY_FORCE`, `GEO_OPT`, `CELL_OPT`, `VIBRATIONAL_ANALYSIS`, `BAND`, or another documented value only when it matches the task.
- For isolated systems, define an adequate cell and the intended nonperiodic electrostatics. For periodic systems, make the cell and k-point sampling explicit.
- Do not combine OT with k-point sampling; use diagonalization/mixing for k-point calculations. Gamma-point OT and k-point diagonalization are separate algorithm fragments in the reference.
- Set `UKS` and `MULTIPLICITY` explicitly for open-shell calculations. Keep basis, potential, XC, and periodicity consistent across compared stages unless the comparison intentionally changes them.
- Treat convergence thresholds and print/property sections as scientific input. Do not infer or add them from the preparation tool.
- Start from method- and system-appropriate ordinary settings; no CP2K cutoff or threshold is a universal standard. Use the task- and observable-dependent starting choices in the numerical reference rather than treating an official example, a larger number, or a smaller tolerance as inherently better. Do not lower `EPS_SCF`/`EPS_DEFAULT`, raise `CUTOFF`/`REL_CUTOFF`, add outer SCF, or enlarge `MAX_SCF` merely because a result is described as reliable, reproducible, production, final, or QC. Tighten the relevant control only for an explicit requirement, a source-backed method need, or observed convergence/grid-sensitivity evidence; `MAX_SCF` is a recovery ceiling, not an accuracy setting.

## Handoff

Return the prepared stage, the scientific settings that matter to the comparison, and the CP2K result summary or specialized analysis artifacts.

## References

- [Native CP2K task and algorithm fragments](references/cp2k_native_examples.md)
- [Choosing CP2K numerical settings](references/cp2k_numerical_settings.md)
- [CP2K manual](https://manual.cp2k.org/)
- [Geometry and cell optimization](https://manual.cp2k.org/trunk/methods/optimization/geometry_and_cell_opt.html)
- [FORCE_EVAL/PROPERTIES](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/PROPERTIES.html)
