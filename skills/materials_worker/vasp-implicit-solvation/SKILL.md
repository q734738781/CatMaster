---
name: vasp-implicit-solvation
description: Use this skill for VASPsol or LSOL calculations that need vacuum-seeded SCF continuation, manual state-specific WAVECAR staging, solvent INCAR setup, and removal of dipole-correction tags.
---

# vasp-implicit-solvation

## Overview
Use this SOP when LSOL preparation requires work beyond `vasp_prepare`.

## Quick Start
1. Converge or reuse the exact state's vacuum SCF with `LWAVE=True`.
2. Prepare LSOL with the same electronic setup and `LSOL=True`, `ISTART=1`, `ICHARG=0`.
3. Manually copy the compatible vacuum `WAVECAR` into the prepared LSOL directory.
4. Remove `IDIPOL`, `LDIPOL`, and `DIPOL`; dispatch, then confirm restart reading and SCF convergence.

## Allowed tools
- `vasp_prepare`
- `remote_submission`
- `remote_submission_batch`
- `execute`

## Workflow

### 1. Prepare the vacuum seed
- Strongly prefer this two-stage route for new or previously difficult LSOL calculations; reuse an already converged compatible seed instead of repeating it.
- Match structure and atom order, cell, KPOINTS, POTCAR, ENCUT, charge, and spin setup. Each clean or adsorbate state uses its own `WAVECAR` and intended `MAGMOM`; do not derive startup moments from OUTCAR.

### 2. Build the LSOL continuation
- Use `vasp_prepare` with `enable_dipole=false` and `user_incar_patch={"LSOL": true, "ISTART": 1, "ICHARG": 0, "IDIPOL": null, "LDIPOL": null, "DIPOL": null}`; keep other comparison-sensitive settings unchanged.
- `vasp_prepare` does not copy a general `WAVECAR`; the agent must copy it afterward and inspect the final directory. `CHGCAR` does not replace this step, and this is not an `ICHARG=11` branch.

### 3. Audit and accept
- Require `IDIPOL`, `LDIPOL`, and `DIPOL` to be absent. `LDIPOL=False` does not make a retained `IDIPOL` inert. Allow a combined setup only when explicitly requested and independently justified.
- Confirm from VASP output that `WAVECAR` was read and SCF converged. If the restart is rejected, regenerate a compatible vacuum seed rather than borrowing another state.

## Method-critical defaults
- Keep the solvent model and all energy-comparison settings consistent across clean and adsorbate states.
- Do not tighten EDIFF, ENCUT, k-point density, or iteration ceilings merely because LSOL is enabled.
- A validated direct-LSOL start remains allowed when explicitly requested or already known to converge.

## Output Contract
Return the vacuum-seed path, LSOL path, `WAVECAR` source, restart settings, three-tag dipole audit, and whether restart reading and SCF convergence succeeded.

## References
- Read `../vasp-input-preparation/SKILL.md` for canonical input preparation.
- Read `../vasp-batch-execution/SKILL.md` before managed execution or recovery.
