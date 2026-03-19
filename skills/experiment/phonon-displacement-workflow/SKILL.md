---
name: phonon-displacement-workflow
description: Use this skill for finite-displacement phonon force-collection setup when the immediate task is to generate supercell displacements, prepare force jobs, and collect consistent VASP outputs for later phonon fitting.
license: project-local
compatibility: local
allowed-tools: "generate_phonon_displacements vasp_prepare vasp_execute_batch analyze_vasp_results"
metadata:
  catmaster-suggested-tools: "generate_phonon_displacements vasp_prepare vasp_execute_batch analyze_vasp_results"
---

# phonon-displacement-workflow

## Overview
Use this skill to generate finite-displacement supercells and collect force calculations without pretending that force-constant fitting or phonon spectra are already done.

## Quick Start
1. Start from one accepted relaxed bulk reference.
2. Generate displaced supercells with `generate_phonon_displacements`.
3. Prepare one consistent force-calculation stage for all displacements.
4. Dispatch the displacement batch and summarize completion with `analyze_vasp_results`.

## Suggested tools
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
- Do not mix displacement generation and phonon fitting logic in one opaque step.

### 3. Collect only accepted force runs
- Use `analyze_vasp_results` to identify failed or unconverged force points before any later phonon fitting.

## Method-critical defaults
- Keep ENCUT, k-point density, spin treatment, and any `DFT+U`/dispersion toggles fixed across every displacement.
- Report supercell and displacement amplitude explicitly because both affect the eventual phonon interpretation.

## Output Contract
Return:
- displacement metadata path
- prepared/executed batch root
- VASP analysis summary path
- any note about phonopy vs fallback generation

## References
- This skill stops at force-collection readiness; phonon DOS/band fitting should be handled as a separate analysis stage.
