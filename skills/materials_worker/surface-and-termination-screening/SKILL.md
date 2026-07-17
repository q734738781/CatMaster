---
name: surface-and-termination-screening
description: Use this skill for turning one relaxed bulk reference into a controlled slab/termination screening set, including slab generation, freezing policy, optional lateral expansion, and standardized VASP ranking runs.
---

# surface-and-termination-screening

## Overview
Use this skill to screen surface terminations without quietly changing slab geometry, fixing policy, or execution settings between candidates.

## Quick Start
1. Start from one accepted relaxed bulk reference.
2. Generate all relevant slab terminations with one fixed thickness/vacuum and orthogonality contract.
3. Audit the exposed layers and surface-atom coordination of every termination before selecting survivors.
4. For adsorption handoff, prefer `orthogonal=true` unless a non-orthogonal native surface cell is explicitly required.
5. Apply one explicit freezing rule, then rank the survivors with one uniform slab relax or static batch.

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
- Do not promote `termination_index=0` just because it is first. Check each emitted termination for exposed species/layer sequence, stoichiometry, polarity or dipole concerns, duplicate geometry, and whether it matches the intended surface chemistry.
- After cutting the slab, identify the top and bottom surface layers with one consistent geometric rule and numerically compare their coordination distributions across the termination family. Unless the user requests a particular reactive, defective, or under-coordinated termination, prefer candidates whose surface atoms retain relatively higher coordination and fewer severely under-coordinated sites.
- Treat a large population of coordination-number-one (`CN=1`) surface atoms as a dangling-atom warning. Do not promote that termination without checking the neighbor criterion, slab thickness, cut/orientation, polarity, and visual geometry; normally reject or deprioritize it because such surfaces are empirically associated with high surface energy. This is a screening heuristic, not a substitute for a calculated surface energy.
- When the survivor will be used for adsorption placement, set `orthogonal=true` during slab generation by default. Use `orthogonal=false` only for a stated scientific reason and keep that setting fixed across the termination family.
- If the task is surface-energy oriented, surface `get_symmetry_slab=true` explicitly.

### 2. Apply one freezing strategy
- Use layer-based freezing by default for simple slabs.
- Switch to height- or index-based fixing only when the slab geometry or later adsorbate bookkeeping requires it.
- If lateral coverage needs more spacing, apply `supercell` after the fixing policy is understood.

### 3. Inspect before paying for DFT
- Use the `structure-visual-inspection` skill script when termination identity or vacuum/fixing mistakes are visually ambiguous.
- Inspect enough representative terminations to verify the intended exposed layer, vacuum direction, and any low-coordination warning before preparing relax inputs. Use visual inspection only as corroboration; preserve the numerical atom indices and coordination evidence behind any dangling-atom claim.
- Do not send obviously malformed or duplicated slabs into the VASP batch.

### 4. Run one controlled ranking stage
- Use one consistent `vasp_prepare` preset/regime across the termination set.
- Dispatch as one batch and parse convergence/energy with a focused `pymatgen` parser before selecting survivors.
- When ranking by total energy, use `E0` as the default energy field from structured parser output.

## Method-critical defaults
- Keep thickness, vacuum, k-point density, spin treatment, and dispersion policy fixed across the termination screen unless the workflow explicitly studies those variables.
- Use one stated neighbor algorithm/cutoff and one stated surface-layer definition for every candidate. Report coordination distributions by element and surface side; do not compare coordination numbers produced by different heuristics as if they were equivalent.
- Termination provenance review is not optional for slab modeling; report what terminations were generated, which were retained or rejected, and why. Treat the review as an evidence-backed selection record, not an absolute proof of surface identity from one structure file.
- For adsorption-ready survivors, report the `orthogonal` setting and prefer `orthogonal=true`.
- Do not compare slabs if some candidates were relaxed with cell degrees of freedom and others were not.
- Preserve selective dynamics through any supercell expansion and report that the mask survived.

## Output Contract
Return:
- termination set or retained termination IDs
- termination provenance summary, including retained/rejected rationale and remaining uncertainty
- surface-coordination audit, including the neighbor criterion, surface-layer definition, per-element coordination summary, and count/indices of any `CN=1` surface atoms
- `orthogonal` setting used for the family
- representative slab artifact path(s)
- freezing policy used
- batch ranking/parser summary path

## References
- Hand off accepted slabs to `adsorption-screening` rather than combining slab enumeration and adsorption placement in one campaign.
