---
name: surface-and-termination-screening
description: Use this skill for turning one relaxed bulk reference into a controlled slab/termination screening set, including slab generation, freezing policy, optional lateral expansion, and standardized VASP ranking runs.
---

# surface-and-termination-screening

## Overview
Use this skill to screen surface terminations without quietly changing slab geometry, fixing policy, or execution settings between candidates.

## Quick Start
1. Start from one accepted relaxed bulk reference.
2. Generate all relevant slab terminations with one fixed thickness/vacuum contract.
3. Apply one explicit freezing rule across the compared slabs.
4. Prepare one uniform slab relax or static batch.
5. Rank with a task-specific `pymatgen` parser; only then hand off winning slabs to adsorption work.

## Allowed tools
- `build_slab`
- `fix_atoms_by_layers`
- `fix_atoms_by_height`
- `fix_atoms_by_indices`
- `supercell`
- `vasp_prepare`
- `remote_submission`
- `remote_submission_batch`
- `execute`

## Workflow

### 1. Enumerate the termination family
- Use `build_slab` once per Miller index and keep slab thickness, vacuum, orthogonality, and reduction policy fixed during the screen.
- If the task is surface-energy oriented, surface `get_symmetry_slab=true` explicitly.

### 2. Apply one freezing strategy
- Use layer-based freezing by default for simple slabs.
- Switch to height- or index-based fixing only when the slab geometry or later adsorbate bookkeeping requires it.
- If lateral coverage needs more spacing, apply `supercell` after the fixing policy is understood.

### 3. Inspect before paying for DFT
- Use the `structure-visual-inspection` skill script when termination identity or vacuum/fixing mistakes are visually ambiguous.
- Do not send obviously malformed or duplicated slabs into the VASP batch.

### 4. Run one controlled ranking stage
- Use one consistent `vasp_prepare` preset/regime across the termination set.
- Dispatch as one batch and parse convergence/energy with a focused `pymatgen` parser before selecting survivors.
- When ranking by total energy, use `E0` as the default energy field from structured parser output.

## Method-critical defaults
- Keep thickness, vacuum, k-point density, spin treatment, and dispersion policy fixed across the termination screen unless the workflow explicitly studies those variables.
- Do not compare slabs if some candidates were relaxed with cell degrees of freedom and others were not.
- Preserve selective dynamics through any supercell expansion and report that the mask survived.

## Output Contract
Return:
- termination set or retained termination IDs
- representative slab artifact path(s)
- freezing policy used
- batch ranking/parser summary path

## References
- Hand off accepted slabs to `adsorption-screening` rather than combining slab enumeration and adsorption placement in one campaign.
