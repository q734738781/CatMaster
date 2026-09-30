---
name: lammps-minimization
description: Use this skill to author, execute, and interpret a native LAMMPS minimization with an explicit force field and stopping criteria.
allowed-tools: "lammps_prepare remote_submission remote_submission_batch get_avail_remote_task lammps_log_summary md_trajectory_summary execute"
---

# LAMMPS minimization

## Workflow

1. Author the complete `in.lammps`, including the sourced force field, type mapping, boundary conditions, any frozen-atom groups/fixes, `min_style`, force norm, and `minimize etol ftol maxiter maxeval`. Read [lammps_minimization_criteria.md](references/lammps_minimization_criteria.md) before selecting stopping values.
2. Stage the script and every referenced file with `lammps_prepare`.
3. Submit through one compatible registered LAMMPS task.
4. Use `lammps_log_summary` to distinguish process completion from the minimizer stopping state and to retrieve final thermo/force evidence.

## Scientific choices

- Frozen atoms and force-zeroing fixes change the physical optimization space and must be intentional.
- Reaching a minimizer stop criterion does not validate the force field; interpret the final structure and forces for the stated model.
- Choose the unit style and force norm before choosing `ftol`. LAMMPS defaults to the 2-norm of the complete `3N` force vector; with a supported minimizer such as `cg`, `sd`, or `fire`, use `min_modify norm max` when the scientific criterion is maximum per-atom force rather than copying a tolerance across different system sizes. Choose `etol` and `ftol` for the physical resolution, potential class, and downstream observable. Do not drive them toward machine precision, or inflate `maxiter`/`maxeval`, merely because the result is final, reproducible, or checked against an analytic toy model. The iteration and evaluation values are recovery ceilings, not accuracy targets; tighten only for an explicit requirement or observed result sensitivity.
- Report the final energy/force metrics and whether the stop was converged, iteration-limited, or otherwise unresolved.

## References

- [Native LAMMPS minimization fragment](../lammps-preparation/references/lammps_native_examples.md)
- [Choosing LAMMPS minimization criteria](references/lammps_minimization_criteria.md)
- [LAMMPS minimize](https://docs.lammps.org/minimize.html)
- [LAMMPS min_style](https://docs.lammps.org/min_style.html)
