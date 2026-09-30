---
name: lammps-preparation
description: Use this skill to author and stage complete native LAMMPS scripts, data/restart files, and potentials for minimization or dynamics.
allowed-tools: "lammps_prepare remote_submission remote_submission_batch get_avail_remote_task lammps_log_summary md_trajectory_summary analyze_trajectory execute"
---

# LAMMPS preparation

## Workflow

1. Select the force field and its units, atom style, type mapping, boundary conditions, and all referenced parameter files from an appropriate scientific source.
2. Author a complete native LAMMPS script. Take only the requested operation syntax from [lammps_native_examples.md](references/lammps_native_examples.md); choose the force field, system setup, and numerical settings independently rather than turning a fragment into a method template. For minimization, choose the unit-aware force norm and stopping values with [lammps_minimization_criteria.md](../lammps-minimization/references/lammps_minimization_criteria.md).
3. Call `lammps_prepare` with the script path, a fresh `output_root`, and explicit `asset_mappings` for every data, restart, potential, table, molecule, or include file. The tool copies the script unchanged to `in.lammps`.
4. Submit one stage through an available compatible `lammps_execute` or `lammps_execute_kokkos` task; batch independent same-config stages, using first-level children by default or explicit relative `stage_paths`.
5. Use `lammps_log_summary` for execution, minimization, and thermo states. Use `md_trajectory_summary` for inventory, or `analyze_trajectory` with an explicit stored-frame interval for quantitative MSD/RDF work; a fit window is required only when computing MSD.

## Scientific choices

- Keep `units`, `atom_style`, `boundary`, `read_data`/`read_restart`, force-field commands, timestep, ensemble fixes, run length, and output strides visible in the script.
- Do not infer a potential or type mapping from elemental composition alone.
- Use native LAMMPS variables, groups, fixes, computes, and hybrid styles directly when required; the preparation tool does not reduce or reinterpret them.
- For diffusion, emit unwrapped coordinates (`xu yu zu` or `xsu ysu zsu`) or image flags with wrapped coordinates, and state the physical time between stored frames. For periodic generic XYZ/ASE trajectories, pass the wrapping semantics explicitly to `analyze_trajectory` because the file format does not establish it.
- A same-coordinate CPU/KOKKOS comparison is not required before production. Use one only when a result discrepancy gives a concrete scientific reason to diagnose backend equivalence.

## Handoff

Return the prepared stage, the scientific model and sampling settings, and the requested log/trajectory analysis artifacts.

## References

- [Native LAMMPS task fragments](references/lammps_native_examples.md)
- [Choosing LAMMPS minimization criteria](../lammps-minimization/references/lammps_minimization_criteria.md)
- [LAMMPS units](https://docs.lammps.org/units.html)
- [LAMMPS input commands](https://docs.lammps.org/Commands_input.html)
- [LAMMPS dump](https://docs.lammps.org/dump.html)
