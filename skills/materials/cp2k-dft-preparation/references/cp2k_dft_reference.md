# CP2K conventional DFT reference notes

Checked: 2026-05-29

This file is a local, source-grounded note for `cp2k-dft-preparation`. It is not a full copy of the upstream documentation.

## Upstream sources

- CP2K manual index: https://manual.cp2k.org/
- CP2K geometry and cell optimization: https://manual.cp2k.org/trunk/methods/optimization/geometry_and_cell_opt.html
- CP2K FORCE_EVAL/PROPERTIES: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/PROPERTIES.html
- CP2K VIBRATIONAL_ANALYSIS: https://manual.cp2k.org/cp2k-2024_1-branch/CP2K_INPUT/VIBRATIONAL_ANALYSIS.html
- pymatgen CP2K module: https://pymatgen.org/pymatgen.io.cp2k.html

## Practical notes

- CP2K calculation intent is encoded mainly through the input file, especially `GLOBAL/RUN_TYPE`, `FORCE_EVAL`, `SUBSYS`, and `MOTION` sections.
- Conventional materials recipes covered by the skill map to prepared CP2K input stages: single-point energy/force, fixed-cell geometry optimization, cell optimization, vibrational analysis, and property-oriented follow-up stages.
- Geometry and cell optimization require explicit convergence and optimizer choices when they matter for comparing structures or energies.
- `FORCE_EVAL/PROPERTIES` is where many electronic-property requests are configured, but property output is not universal and must be requested in the input.
- `VIBRATIONAL_ANALYSIS` is a separate CP2K input section and should be treated as a deliberate follow-up stage from an accepted stationary structure.
- `pymatgen.io.cp2k` provides structured CP2K input helpers and input-set support, so generated input should prefer structured APIs over free-form LLM text when possible.

## SOP implications for the skill body

- Use `cp2k_prepare`; do not let the agent write arbitrary CP2K blocks unless the workflow is explicitly outside the supported recipe.
- Report XC, basis, potential, cutoff, charge, multiplicity, periodicity, and k-point choices when they affect conclusions.
- Submit all CP2K conventional stages through `remote_submission(task_name="cp2k_execute")` or the batch equivalent.
- Use task-specific parsing for frequency, DOS/PDOS, charge, and band-style outputs.
