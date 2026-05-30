# CP2K pathway-calculation reference notes

Checked: 2026-05-29

This file is a local, source-grounded note for `cp2k-pathway-calculations`. It is not a full copy of the upstream documentation.

## Upstream sources

- CP2K NEB exercise: https://www.cp2k.org/exercises%3Acommon%3Aneb
- CP2K MOTION/BAND manual: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/BAND.html
- CP2K transition-state section: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/GEO_OPT/TRANSITION_STATE.html
- CP2K manual index: https://manual.cp2k.org/

## Practical notes

- CP2K path refinement is configured in the input file, especially `MOTION/BAND` for band/NEB-style calculations and `MOTION/GEO_OPT/TRANSITION_STATE` for transition-state searches.
- NEB-style work requires a coherent image path. Images must describe the same atom list in the same order; otherwise the path is not physically interpretable.
- `MOTION/BAND` includes replica, optimizer, spring, and CI-NEB related configuration. These are method choices, not incidental settings.
- CP2K dimer/transition-state workflows require a transition-state guess and careful mode/vector handling when supplied.

## SOP implications for the skill body

- Use the single materials-side `cp2k_prepare` tool with `recipe="neb"` or `recipe="dimer"`; do not add separate path-specific remote task names.
- Validate atom count and ordering before staging.
- Report whether endpoints were fixed or optimized and parse image energies/barriers with a focused script after execution.
