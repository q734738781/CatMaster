# LAMMPS input preparation reference notes

Checked: 2026-05-29

This file is a local, source-grounded note for `lammps-preparation`. It is not a full copy of the upstream documentation.

## Upstream sources

- LAMMPS units: https://docs.lammps.org/units.html
- LAMMPS read_data: https://docs.lammps.org/read_data.html
- LAMMPS pair_style: https://docs.lammps.org/pair_style.html
- LAMMPS fix nvt/npt: https://docs.lammps.org/fix_nh.html
- LAMMPS minimize: https://docs.lammps.org/minimize.html
- LAMMPS thermo_style: https://docs.lammps.org/thermo_style.html
- LAMMPS dump: https://docs.lammps.org/dump.html
- LAMMPS write_restart: https://docs.lammps.org/write_restart.html
- LAMMPS read_restart: https://docs.lammps.org/read_restart.html
- LAMMPS compute rdf: https://docs.lammps.org/compute_rdf.html
- LAMMPS compute msd: https://docs.lammps.org/compute_msd.html
- pymatgen LAMMPS module: https://pymatgen.org/pymatgen.io.lammps.html
- lammpsio docs: https://lammpsio.readthedocs.io/

## Practical notes

- `units` sets the unit system for the script and data/potential files and must be set before the simulation box is defined.
- `read_data` reads the atom topology/box data file for structure-started runs; restart-started runs use `read_restart`.
- `pair_style` and `pair_coeff` define the force-field model. They must not be guessed from chemistry alone.
- Thermostat/barostat behavior is implemented through fix commands such as the Nose-Hoover family (`nvt`, `npt`, etc.).
- Thermo, dump, restart, RDF, and MSD outputs are all explicitly requested in the input script.
- pymatgen and lammpsio can help with structured data handling, but they do not validate scientific force-field choice by themselves.

## SOP implications for the skill body

- Validate `units`, `atom_style`, `pair_style`, `pair_coeff`, masses/type mapping, and potential-file paths before preparation.
- Use one prepared stage directory per simulation. Submit through CPU `lammps_execute`, or through `lammps_execute_kokkos` only when every active style supports the enabled KOKKOS build.
- Report output strides and ensemble/force-field choices with results.
