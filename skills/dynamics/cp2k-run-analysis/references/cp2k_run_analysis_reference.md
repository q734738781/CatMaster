# CP2K run-analysis reference notes

Checked: 2026-05-29

This file is a local, source-grounded note for `cp2k-run-analysis`. It is not a full copy of the upstream documentation.

## Upstream sources

- CP2K manual index: https://manual.cp2k.org/
- CP2K GLOBAL/RUN_TYPE: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/GLOBAL.html
- CP2K FORCE_EVAL/PROPERTIES: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/PROPERTIES.html
- CP2K VIBRATIONAL_ANALYSIS: https://manual.cp2k.org/cp2k-2024_1-branch/CP2K_INPUT/VIBRATIONAL_ANALYSIS.html
- CP2K MOTION/MD: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/MD.html

## Practical notes

- CP2K output layout depends on `RUN_TYPE`, print settings, and property sections. A generic run summary can confirm execution health but cannot replace property-specific analysis.
- Common reusable evidence includes wrapper status, completion markers, return code, energy lines, SCF markers, optimization/frequency markers, `.ener` files, trajectories, restart files, and produced property files.
- DOS, charge, band, PLUMED/free-energy, and pathway-barrier analysis require task-specific parsers because their file layout and interpretation depend on the requested CP2K sections.

## SOP implications for the skill body

- Start with `cp2k_output_summary` for reusable run evidence.
- Escalate to a focused parser only after identifying exact output files.
- Never present generic completion or energy extraction as scientific convergence.
