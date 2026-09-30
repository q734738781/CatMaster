---
name: lammps-restart
description: Use this skill to continue a LAMMPS simulation from an explicitly selected binary restart with a complete native continuation script.
allowed-tools: "lammps_prepare remote_submission remote_submission_batch get_avail_remote_task lammps_log_summary md_trajectory_summary analyze_trajectory execute"
---

# LAMMPS restart

## Workflow

1. Inspect the prior scientific result and select the intended restart file.
2. Author a complete continuation script using `read_restart`, then restate or change the required force-field/fix/output commands according to LAMMPS restart semantics.
3. Stage the script and restart/potential assets into a fresh directory with `lammps_prepare`.
4. Submit the continuation and analyze its new log, trajectory, and restart outputs separately from the source stage.

## Scientific choices

- Preserve or deliberately change the ensemble, timestep, temperature/pressure controls, run length, and output strides.
- Do not silently replace continuation with a fresh velocity initialization.
- Treat the selected binary restart and the new script as the scientific lineage of the continuation.

## References

- [Native restart fragment](../lammps-preparation/references/lammps_native_examples.md)
- [LAMMPS read_restart](https://docs.lammps.org/read_restart.html)
- [LAMMPS restart workflow](https://docs.lammps.org/Howto_restart.html)
