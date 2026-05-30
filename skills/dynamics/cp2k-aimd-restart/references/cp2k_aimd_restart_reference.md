# CP2K AIMD restart reference notes

Checked: 2026-05-29

This file is a local, source-grounded note for `cp2k-aimd-restart`. It is not a full copy of the upstream documentation.

## Upstream sources

- CP2K EXT_RESTART: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/EXT_RESTART.html
- CP2K MOTION/PRINT/RESTART: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/PRINT/RESTART.html
- CP2K MOTION/MD: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/MD.html
- CP2K PLUMED integration: https://www.cp2k.org/howto%3Ainstall_with_plumed

## Practical notes

- CP2K restart continuation uses restart files and restart-related input sections; selecting the wrong restart can silently change the simulated trajectory.
- Continuation should preserve or intentionally change ensemble, thermostat/barostat, timestep, and print strides.
- A restart run should write into a new stage directory so prior output and receipt context remain auditable.
- PLUMED continuation requires preserving compatible PLUMED input/state expectations when metadynamics is involved.

## SOP implications for the skill body

- Inspect the prior run before preparing continuation.
- Use explicit `settings.restart_file` and `settings.structure_file` when multiple candidates exist.
- Preserve both old and new remote receipt/context IDs in the report.
