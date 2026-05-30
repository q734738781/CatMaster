# MD trajectory and output reference notes

Checked: 2026-05-29

This file is a local, source-grounded note for `trajectory-analysis`. It is not a full copy of the upstream documentation.

## Upstream sources

- LAMMPS output how-to: https://docs.lammps.org/Howto_output.html
- LAMMPS dump: https://docs.lammps.org/dump.html
- LAMMPS compute rdf: https://docs.lammps.org/compute_rdf.html
- LAMMPS compute msd: https://docs.lammps.org/compute_msd.html
- CP2K MOTION/MD: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/MD.html
- lammpsio docs: https://lammpsio.readthedocs.io/

## Practical notes

- LAMMPS has multiple output classes: thermo/log output, dump files, fix output files, and restart files.
- Dump files store snapshots at configured intervals; custom dumps need stable atom IDs or sorting for frame-to-frame analysis.
- RDF and MSD are compute outputs and require meaningful atom/group selections.
- CP2K AIMD output depends on MD print sections and may include `.ener`, trajectory, velocity, force, and restart artifacts depending on settings.
- Generic trajectory inventory is run-health evidence; mechanistic interpretation needs system-specific parsing.

## SOP implications for the skill body

- First check frame count, atom count, time span, final-frame export, restart files, and thermo/energy drift.
- Report which trajectory/log/energy files were parsed.
- Write focused scripts for residence, reaction, adsorption, or free-energy analysis rather than overusing generic summaries.
