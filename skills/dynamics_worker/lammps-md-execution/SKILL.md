---
name: lammps-md-execution
description: Use this skill to author, run, and analyze native LAMMPS NVE, NVT, NPT, annealing, or production dynamics stages.
allowed-tools: "lammps_prepare remote_submission remote_submission_batch get_avail_remote_task lammps_log_summary md_trajectory_summary analyze_trajectory execute"
---

# LAMMPS MD execution

## Workflow

1. Author a complete LAMMPS script with the intended ensemble, initialization/restart behavior, timestep, run length, thermostat/barostat parameters, and output definitions.
2. Stage it and its explicit assets with `lammps_prepare`, then submit through one compatible registered LAMMPS task.
3. Use `lammps_log_summary` for process and thermo behavior, and `md_trajectory_summary` to locate trajectories, native MSD/RDF tables, and restarts.
4. For quantitative trajectory analysis, pass the actual stored-frame interval to `analyze_trajectory`; select the fit interval when computing MSD. RDF-only uses `compute_msd=false` and needs no diffusion fit. Native LAMMPS columns determine wrapping; periodic generic XYZ/ASE files require an explicit `coordinate_semantics`. Use species/group-specific analysis when the scientific question requires it.

## Scientific choices

- NVE, NVT, NPT, and annealing answer different sampling questions; do not exchange them implicitly.
- State the force field, ensemble, timestep, total time, temperature/pressure controls, thermo stride, dump stride, and restart stride.
- A completed run or stable frame count is not by itself equilibration evidence.
- Prefer unwrapped dump coordinates or image flags for diffusion. RDF requires periodic geometry and a meaningful species/group selection.
- Cross-backend diagnosis is exceptional: do not add CPU/KOKKOS comparisons unless an observed scientific discrepancy makes them relevant.

## References

- [Native LAMMPS task fragments](../lammps-preparation/references/lammps_native_examples.md)
- [Nose-Hoover fixes](https://docs.lammps.org/fix_nh.html)
- [LAMMPS compute MSD](https://docs.lammps.org/compute_msd.html)
- [LAMMPS compute RDF](https://docs.lammps.org/compute_rdf.html)
