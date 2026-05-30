# LAMMPS restart reference notes

Checked: 2026-05-29

This file is a local, source-grounded note for `lammps-restart`. It is not a full copy of the upstream documentation.

## Upstream sources

- LAMMPS read_restart: https://docs.lammps.org/read_restart.html
- LAMMPS write_restart: https://docs.lammps.org/write_restart.html
- LAMMPS restart how-to: https://docs.lammps.org/Howto_restart.html

## Practical notes

- LAMMPS restart files are binary state files intended for continuation.
- `write_restart` and related restart controls produce restart files during or after a run.
- `read_restart` starts from a restart file instead of a normal data file; input compatibility and LAMMPS build compatibility matter.
- A restarted workflow should preserve prior context and write into a new stage directory.

## SOP implications for the skill body

- Select the intended restart file explicitly when multiple candidates exist.
- Preserve or intentionally change force-field card, ensemble, timestep, temperature, pressure, and output strides.
- Report old and new receipt/context IDs; a restart is not a stateless fresh run.
