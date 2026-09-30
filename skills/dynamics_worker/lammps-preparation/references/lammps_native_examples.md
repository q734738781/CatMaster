# Native LAMMPS task fragments

These fragments are deliberately incomplete. They show where a requested operation is expressed, not a force field, unit system, atom representation, boundary condition, type mapping, numerical standard, or complete runnable script. Compose the final script from the actual model and use only the relevant fragment.

## Choose one starting state

For a data file:

```text
read_data DATA_FILE
```

For a binary continuation:

```text
read_restart RESTART_FILE
```

Do not combine these as interchangeable setup lines. A restart restores some state that a data file does not; after `read_restart`, reissue only the fixes, computes, dumps, output controls, and model data that the selected styles and continuation actually require.

## Minimization fragment

```text
min_style MINIMIZER
min_modify norm FORCE_NORM
minimize ETOL FTOL MAXITER MAXEVAL
```

Select `FORCE_NORM`, energy/force criteria, and recovery ceilings from the active `units`, potential, system, and downstream use. See [the minimization criteria reference](../../lammps-minimization/references/lammps_minimization_criteria.md); the placeholders above are not values to copy.

## NVT operation fragment

Use this only when NVT is the requested ensemble:

```text
fix ENSEMBLE_FIX GROUP_ID nvt temp TSTART TSTOP TDAMP
run NSTEPS
unfix ENSEMBLE_FIX
```

The timestep, temperatures, damping time, velocity initialization or restart behavior, and run length are separate scientific choices. Other ensembles require their own documented fixes.

## Trajectory and restart-output fragments

```text
thermo THERMO_EVERY
dump TRAJECTORY GROUP_ID custom DUMP_EVERY TRAJECTORY_FILE DUMP_FIELDS
restart RESTART_EVERY RESTART_PATTERN
write_restart FINAL_RESTART
```

Choose `DUMP_FIELDS` and all output strides for the requested analysis. For example, unwrapped coordinates are useful for periodic displacement and diffusion analysis, but that is an analysis requirement rather than a default for every trajectory.
