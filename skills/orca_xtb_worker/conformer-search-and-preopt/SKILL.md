---
name: conformer-search-and-preopt
description: Use this skill for a bounded RDKit, CREST, and xTB conformer-search and preoptimization episode before higher-level ORCA work.
allowed-tools: "create_molecule_from_smiles enumerate_molecular_conformers crest_prepare xtb_prepare filter_conformer_ensemble remote_submission remote_submission_batch analyze_xtb_results extract_optimized_molecules execute"
---

# Conformer search and preoptimization

## Workflow

1. Build an initial 3D seed or RDKit ensemble when needed.
2. For CREST exploration, compose complete native argv from only the relevant [CREST fragments](references/crest_native_examples.md), choose the model and molecular state for the actual ensemble, then call `crest_prepare` with explicit asset mappings and submit `task_name="crest_execute"`.
3. Analyze and prune the ensemble with one stated energy/RMSD policy.
4. For xTB preoptimization, create one native xTB stage per independent structure with `xtb_prepare`, then batch independent stages with `xtb_execute`; use the default first-level layout or select relative `stage_paths`.
5. Extract accepted optimized XYZ files into a clean directory for ORCA.

## Scientific choices

- Keep charge, unpaired-electron count, GFN family, solvation, temperature, and pruning criteria explicit and consistent across compared conformers.
- Native CREST/xTB options remain fully available; the prepare tools do not add search modes, RMSD flags, or optimization defaults.
- Do not automatically repeat xTB optimization after CREST at the same model and conditions; use it only for a deliberate model/solvent change, output normalization, or a concrete acceptance failure.
- Keep ensemble generation, energy ranking, and higher-level refinement as distinct method levels.

## References

- [Native CREST argv fragments](references/crest_native_examples.md)
- [Native xTB argv fragments](../xtb-screen-and-prune/references/xtb_native_examples.md)
- [CREST documentation](https://crest-lab.github.io/crest-docs/)

## Tool scope and examples

SMILES preparation honors `name` when `output_path` is omitted and exposes `optimize`, `random_seed` and `max_iterations`. Conformer generation reports failed or unconverged optimization per conformer; an embedded geometry alone is not an optimized one. Generation, filtering and extraction use fresh output directories. Filtering defaults to all frames and requires energies for the energy window; choose `missing_energy="exclude"` or `"keep"` explicitly, or use `apply_energy_window=false` for geometry-only deduplication. Absolute energies are rebased to their ensemble minimum.

`extract_optimized_molecules` accepts a result tree or explicit `source_files`, selects the last frame by default and requires optimization convergence unless explicitly disabled; inspect its skipped reasons. Analyzers discover logs recursively and report ambiguity and per-source failures. Select custom ORCA/xTB logs with `result_files`, for example `["runs/custom_name.out"]`.
