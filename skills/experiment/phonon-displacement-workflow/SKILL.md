---
name: phonon-displacement-workflow
description: Use this skill for finite-displacement phonon force-collection setup when the immediate task is to generate supercell displacements, prepare force jobs, and collect consistent VASP outputs for later phonon fitting.
license: project-local
compatibility: local
allowed-tools: "generate_phonon_displacements vasp_prepare vasp_execute_batch analyze_vasp_results"
---

# phonon-displacement-workflow

## Overview
Use this skill to generate finite-displacement supercells and collect force calculations without pretending that force-constant fitting or phonon spectra are already done. Do not use it when the real deliverable is a fitted phonon DOS/band structure rather than a validated force-job set.

## Quick Start
1. Start from one accepted relaxed bulk reference.
2. Generate displaced supercells with `generate_phonon_displacements`.
3. Prepare one consistent force-calculation stage for all displacements.
4. Keep the displacement metadata and calculation-directory mapping explicit.
5. Dispatch the displacement batch and summarize completion with `analyze_vasp_results`.

## Allowed tools
- `generate_phonon_displacements`
- `vasp_prepare`
- `vasp_execute_batch`
- `analyze_vasp_results`

## Workflow

### 1. Generate the displacement set explicitly
- Keep the supercell size and displacement amplitude visible in the workflow record.
- If the backend falls back to a manual generator, report that instead of assuming symmetry reduction happened.

### 2. Run force jobs, not generic relax jobs
- Prepare the displacement set for force evaluation under one method contract.
- Use static/force semantics for the displaced structures; do not quietly relax the displaced cells or ionic positions.
- If the phonon workflow requires symmetry to stay off for the displaced structures, surface `ISYM=0` explicitly rather than assuming the wrapper will infer it.
- Do not mix displacement generation and phonon fitting logic in one opaque step.

### 3. Collect only accepted force runs
- Use `analyze_vasp_results` to identify failed or unconverged force points before any later phonon fitting.
- Keep a displacement-to-calculation ledger so later fitting can map every force set back to its displacement ID.

### 4. Hand off only a complete force dataset
- If any displacement member failed, report the collection as partial instead of implying phonon readiness.
- Keep the generation backend note (`phonopy` vs fallback) with the force-job summary.

## Method-critical defaults
- Keep ENCUT, k-point density, spin treatment, and any `DFT+U`/dispersion toggles fixed across every displacement.
- Do not relax displaced structures during force collection; the target observable is the force response of the displaced geometry.
- Keep symmetry handling explicit for displaced jobs when that matters to the force-set interpretation.
- Report supercell and displacement amplitude explicitly because both affect the eventual phonon interpretation.

## Output Contract
Return:
- displacement metadata path
- prepared/executed batch root
- VASP analysis summary path
- any note about phonopy vs fallback generation
- displacement-to-calculation mapping ledger

## References
- This skill stops at force-collection readiness; phonon DOS/band fitting should be handled as a separate analysis stage.
