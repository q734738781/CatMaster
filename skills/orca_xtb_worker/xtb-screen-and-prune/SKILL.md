---
name: xtb-screen-and-prune
description: Use this skill to compose native xTB argv stages for molecular single points, optimization, Hessian, dynamics, constraints, and comparable screening.
allowed-tools: "xtb_prepare remote_submission remote_submission_batch analyze_xtb_results filter_conformer_ensemble extract_optimized_molecules execute"
---

# xTB screen and prune

## Workflow

1. Compose the complete ordered native xTB tokens after the executable. Take only the requested operation or option tokens from [xtb_native_examples.md](references/xtb_native_examples.md); choose the method, molecular state, solvent, and numerical controls independently.
2. Stage every coordinate, detailed input, parameter file, or other referenced file through explicit `asset_mappings` in `xtb_prepare`.
3. Submit with `task_name="xtb_execute"`; the execute task accepts no scientific overrides.
4. Use `analyze_xtb_results`, then prune or extract accepted structures for the next method level.

Before filtering or extracting an ensemble, read the [conformer tool scope](../conformer-search-and-preopt/SKILL.md#tool-scope-and-examples). It defines energy/missing-data policy, frame selection, optimization acceptance and fresh output directories. Process success alone does not establish an optimized geometry.

## Scientific choices

- A coordinate-only argv retains xTB's native single-point behavior. Add `--opt`, `--hess`, `--md`, solvation, charge, spin, or detailed-input options only when intended.
- Keep native accuracy and ordinary optimization settings unless the user, a method-specific source, or observed numerical behavior requires a change. Do not add `--acc`, choose tighter `--opt` levels, or add optimization/Hessian work merely because the result is called reliable, final, reproducible, or QC.
- Keep GFN family, charge, unpaired-electron count, solvation, and comparison protocol consistent across ranked structures.
- Put complete `$constrain`, `$fix`, metadynamics, or other detailed input in a staged file and reference it through native argv.
- Missing frequency output is not zero imaginary modes; a `NOT_CONVERGED` marker remains a nonconverged task even if the process returned normally.

## Handoff

Return the native stage, result summary, and retained structure directory when pruning was applied.

## References

- [Native xTB argv fragments](references/xtb_native_examples.md)
- [xTB command-line documentation](https://xtb-docs.readthedocs.io/en/latest/commandline.html)
