---
name: nmr-ensemble-workup
description: Use this skill for conformer-aware xTB or CREST cleanup followed by ORCA NMR shielding calculations and optional referenced shift conversion.
allowed-tools: "enumerate_molecular_conformers crest_prepare xtb_prepare filter_conformer_ensemble remote_submission remote_submission_batch analyze_xtb_results extract_optimized_molecules orca_prepare analyze_orca_results execute"
---

# NMR ensemble workup

## Workflow

1. Generate or collect the conformers and keep charge, spin, solvation, and ranking level consistent across the ensemble.
2. Use native CREST/xTB argv stages for exploration or cleanup, then prune and extract the retained geometries.
3. Prepare ORCA NMR stages with complete method/basis/solvation tokens, `NMR`, any detailed native blocks, and explicit charge/multiplicity.
4. Analyze per-conformer isotropic shieldings. Perform Boltzmann weighting only from a stated energy/free-energy model and temperature.
5. Convert shieldings to shifts only with an explicit reference/calibration; retain both quantities and labels.

Before filtering or extracting an ensemble, read the [conformer tool scope](../conformer-search-and-preopt/SKILL.md#tool-scope-and-examples). It defines energy/missing-data policy, frame selection, optimization acceptance and fresh output directories. Process success alone does not establish an optimized geometry.

## Scientific choices

- Select the NMR functional, basis, solvent treatment, conformer energy model, and reference from the spectral objective or a checked source.
- Do not label raw ORCA shielding as chemical shift.

## References

- [ORCA NMR tutorial](https://www.faccts.de/docs/orca/6.1/tutorials/spec/NMR.html)
