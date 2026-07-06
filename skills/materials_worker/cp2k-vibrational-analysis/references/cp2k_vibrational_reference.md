# CP2K vibrational-analysis reference notes

Checked: 2026-05-29

This file is a local, source-grounded note for `cp2k-vibrational-analysis`. It is not a full copy of the upstream documentation.

## Upstream sources

- CP2K VIBRATIONAL_ANALYSIS: https://manual.cp2k.org/cp2k-2024_1-branch/CP2K_INPUT/VIBRATIONAL_ANALYSIS.html
- CP2K manual index: https://manual.cp2k.org/

## Practical notes

- CP2K exposes vibrational analysis as a dedicated input section, not as an automatic result of geometry optimization.
- Vibrational calculations should normally start from an accepted stationary structure and comparable electronic settings.
- Frequency output, Hessian output, Molden-style vibration output, and thermochemistry evidence depend on the exact print settings and output files produced.
- Imaginary modes require explicit reporting and often indicate that the structure is not the intended minimum, unless the task is transition-state validation.

## SOP implications for the skill body

- Prepare with `cp2k_prepare(recipe="freq")` only after the geometry is accepted or the user explicitly requests diagnostic frequencies.
- Keep the frequency stage separate from the optimization stage for traceable files.
- Parse frequencies with a focused script and report imaginary-mode count plus the exact output file inspected.
