---
name: adsorption-screening
description: Use this skill for adsorption-candidate generation, execution, and first-pass ranking when screening one adsorbate across one slab family with reproducible site provenance and thermochemistry-ready metadata.
---

# adsorption-screening

## Overview
Use this skill to build, run, and rank a controlled adsorption screen while preserving site labels and adsorbate atom indices for later thermochemistry.

## Quick Start
1. Start from one termination-reviewed slab and one canonical adsorbate file.
2. Prefer an orthogonal adsorption slab; if the slab is non-orthogonal, record the reason before placement.
3. Enumerate sites first; keep the site JSON as the ledger.
4. Generate only the adsorption candidates you actually plan to relax.
5. Prepare and dispatch one consistent adsorption batch.
6. Rank with a task-specific `pymatgen` parser and keep `ads_indices` metadata for any later frequency correction work.

## Allowed tools
- `enumerate_adsorption_sites`
- `place_adsorbate`
- `generate_batch_adsorption_structures`
- `vasp_prepare`
- `remote_submission`
- `remote_submission_batch`
- `vaspkit_adsorbate_thermo_correction`
- `execute`

## Workflow

### 1. Preserve adsorption provenance
- Always enumerate sites before broad placement.
- Verify slab provenance before site enumeration: selected termination, termination provenance summary, and `orthogonal` setting should be known. If the task only provides an unlabeled slab, state that limitation and inspect or regenerate the slab family before quantitative screening.
- Treat `generate_batch_adsorption_structures` as a convenience wrapper over site enumeration plus placement, not as a separate scientific abstraction.

### 2. Build a controlled candidate set
- Use one adsorbate geometry source across the whole screen.
- Use an orthogonal, c-oriented slab for new adsorption model generation unless the user explicitly requests native-cell adsorption. Non-orthogonal adsorption cells are allowed only as an intentional exception, not as an unnoticed tool default.
- If only one site is under study, use `place_adsorbate` directly and keep the chosen site label explicit.
- Use the `structure-visual-inspection` skill script when collisions or orientation ambiguity need a visual check before DFT.

### 3. Run one comparable relaxation stage
- Prepare adsorption candidates with one consistent slab regime and method policy.
- Keep clean slab, gas reference, and adsorbed systems method-aligned if adsorption energies will be compared later.

### 4. Rank, then branch
- Use a focused `pymatgen` parser to detect failed or unconverged candidates before ranking.
- For adsorption-energy ranking, read `E0` from structured parser output by default rather than other OUTCAR energy fields.
- Run `vaspkit_adsorbate_thermo_correction` only on the retained adsorption states that genuinely need thermal corrections.

## Method-critical defaults
- Do not begin adsorption ranking from a slab whose termination provenance has not been reviewed or documented.
- For newly built adsorption slabs, the project preference is `orthogonal=true`; keep that choice consistent across clean slab, adsorbed structures, and any reference calculations derived from the slab.
- Do not compare adsorption energies if the clean slab or gas reference was prepared under a different spin, dispersion, or `DFT+U` contract.
- Preserve and report `ads_indices`; they are part of the downstream thermochemistry contract.
- Do not silently accept truncated candidate batches from the wrapper if the ranking depends on full site coverage.

## Output Contract
Return:
- slab provenance: selected termination, provenance-review status or uncertainty, and `orthogonal` setting
- site JSON path
- generated adsorption structure root
- retained candidate list
- adsorption ranking/parser summary path
- adsorbate thermochemistry artifact path when used

## References
- Use the primitive skills `adsorption-site-screening` and `adsorbate-and-intermediate-generation` when only the front half of the workflow is needed.
