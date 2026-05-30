# LAMMPS MD execution reference notes

Checked: 2026-05-29

This file is a local, source-grounded note for `lammps-md-execution`. It is not a full copy of the upstream documentation.

## Upstream sources

- LAMMPS fix nvt/npt: https://docs.lammps.org/fix_nh.html
- LAMMPS fix: https://docs.lammps.org/fix.html
- LAMMPS thermo_style: https://docs.lammps.org/thermo_style.html
- LAMMPS dump: https://docs.lammps.org/dump.html
- LAMMPS write_restart: https://docs.lammps.org/write_restart.html
- LAMMPS compute rdf: https://docs.lammps.org/compute_rdf.html
- LAMMPS compute msd: https://docs.lammps.org/compute_msd.html

## Practical notes

- NVE, NVT, NPT, and annealing are different method choices and should not be swapped silently.
- Nose-Hoover thermostat/barostat settings for `nvt` and `npt` affect dynamics and sampling; they must be reported for production runs.
- `thermo_style` controls the global values printed to screen/log, and dump commands control per-atom trajectory snapshots.
- Restart output is configured explicitly and should be present for long or resumable runs.
- RDF and MSD require explicit compute definitions and meaningful atom/group selections.

## SOP implications for the skill body

- Preserve timestep, step count, thermo stride, dump stride, restart stride, ensemble, temperature, and pressure settings.
- Analyze log health and trajectory inventory before claiming physical interpretation.
- Use focused scripts for residence time, adsorption/desorption, reaction events, or region-specific observables.
