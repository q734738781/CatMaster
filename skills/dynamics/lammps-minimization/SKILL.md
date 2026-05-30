---
name: lammps-minimization
description: Use this skill for LAMMPS force-field minimization stages and generic minimization log inspection.
allowed-tools: "lammps_forcefield_validate lammps_prepare remote_submission remote_submission_batch get_avail_remote_task lammps_log_summary md_trajectory_summary execute"
---

# lammps-minimization

## Overview
Use this skill for LAMMPS minimization from an explicit force-field card or prebuilt `system.data`.

## Quick Start
1. Validate the force-field card with `lammps_forcefield_validate`.
2. Prepare `lammps_prepare(recipe="minimize")`.
3. Submit with `remote_submission(task_name="lammps_execute")`.
4. Run `lammps_log_summary` and inspect minimization stopping criterion.

## Allowed tools
- `lammps_forcefield_validate`
- `lammps_prepare`
- `remote_submission`
- `remote_submission_batch`
- `get_avail_remote_task`
- `lammps_log_summary`
- `md_trajectory_summary`
- `execute`

## Workflow

### 1. Keep force-field assumptions explicit
- Do not invent `pair_style` or `pair_coeff`.
- Preserve `units`, `atom_style`, masses, type mapping, and potential files in the normalized card.

### 2. Prepare minimization
- Use `settings.etol`, `settings.ftol`, `settings.maxiter`, `settings.maxeval`, and `settings.min_style` only as intentional overrides.
- Use frozen atoms only through reviewed LAMMPS fixes in a custom script or curated recipe.

### 3. Analyze generic evidence
- Use `lammps_log_summary` for thermo rows, warnings/errors, minimization stopping criterion, final energy, and force evidence.
- Use task-specific scripts for adsorption distances, reconstruction labels, or chemical interpretation.

## Method-critical defaults
- The minimization stopping criterion is run health evidence, not a guarantee that the force field is valid.
- Report force-field card path and stage path with the result.

## Output Contract
Return:
- normalized force-field card path
- prepared stage path
- submitted receipt/context
- `lammps_log_summary` path
- minimization stopping criterion and warnings/errors

## References
- Local source note: `references/lammps_minimization_reference.md`
- LAMMPS minimize: https://docs.lammps.org/minimize.html
- LAMMPS min_style: https://docs.lammps.org/min_style.html
- LAMMPS fix setforce: https://docs.lammps.org/fix_setforce.html
- LAMMPS thermo_style: https://docs.lammps.org/thermo_style.html
