# CP2K AIMD preparation reference notes

Checked: 2026-05-29

This file is a local, source-grounded note for `cp2k-aimd-preparation`. It is not a full copy of the upstream documentation.

## Upstream sources

- CP2K MOTION/MD: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/MD.html
- CP2K MOTION/PRINT/RESTART: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/PRINT/RESTART.html
- CP2K EXT_RESTART: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/EXT_RESTART.html
- CP2K PLUMED integration: https://www.cp2k.org/howto%3Ainstall_with_plumed
- CP2K manual index: https://manual.cp2k.org/

## Practical notes

- CP2K AIMD is controlled through `MOTION/MD` plus print/restart sections. Ensemble, timestep, number of steps, temperature, pressure, thermostat, and barostat choices are method choices.
- Trajectory, energy, and restart output must be requested intentionally through print settings and stride choices.
- `EXT_RESTART` and restart print sections are relevant when continuing from prior CP2K state; restart inputs should be selected explicitly when more than one candidate exists.
- CP2K + PLUMED support depends on the CP2K build and a user-provided `plumed.dat`; collective variables should not be invented by the agent.

## SOP implications for the skill body

- Prepare AIMD with `cp2k_aimd_prepare`; conventional `sp`, `geo_opt`, and `cell_opt` stay in materials workflows.
- Submit with `task_name="cp2k_execute"`; do not create AIMD-specific remote task names.
- After execution, inspect wrapper status, `job.out`, `.ener`, trajectory, and restart files before interpreting a trajectory.
