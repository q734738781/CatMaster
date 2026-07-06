# LAMMPS minimization reference notes

Checked: 2026-05-29

This file is a local, source-grounded note for `lammps-minimization`. It is not a full copy of the upstream documentation.

## Upstream sources

- LAMMPS minimize: https://docs.lammps.org/minimize.html
- LAMMPS min_style: https://docs.lammps.org/min_style.html
- LAMMPS fix setforce: https://docs.lammps.org/fix_setforce.html
- LAMMPS thermo_style: https://docs.lammps.org/thermo_style.html

## Practical notes

- `minimize` takes energy tolerance, force tolerance, maximum iterations, and maximum force/energy evaluations.
- Minimization stopping criteria include energy change, global force norm, line-search behavior, iteration limit, and evaluation limit.
- The chosen minimizer is controlled by `min_style`; damping-style minimizers may depend on timestep choices.
- `fix setforce` can be used to hold selected atoms fixed by setting forces to zero, but frozen-atom choices are part of the physical model and should be reviewed.
- Thermo output during minimization is still triggered by the timestep-like outer iteration count, so log parsing should inspect minimization stats and thermo rows.

## SOP implications for the skill body

- Do not invent force-field cards for minimization.
- Report `etol`, `ftol`, `maxiter`, `maxeval`, `min_style`, stopping criterion, and warning/error lines.
- Treat minimization completion as run-health evidence, not as proof that the force field is valid for the chemistry.
