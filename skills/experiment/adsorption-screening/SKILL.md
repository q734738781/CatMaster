---
name: adsorption-screening
description: Use this skill for adsorption-candidate generation, execution, and first-pass ranking when screening one adsorbate across one slab family with reproducible site provenance and thermochemistry-ready metadata.
---

# adsorption-screening

## Overview
Use this skill to build, run, and rank a controlled adsorption screen while preserving site labels and adsorbate atom indices for later thermochemistry.

## Quick Start
1. Start from one accepted slab and one canonical adsorbate file.
2. Enumerate sites first; keep the site JSON as the ledger.
3. Generate only the adsorption candidates you actually plan to relax.
4. Prepare and dispatch one consistent adsorption batch.
5. Use `analyze_vasp_results` for ranking and keep `ads_indices` metadata for any later frequency correction work.

## Allowed tools
- `enumerate_adsorption_sites`
- `place_adsorbate`
- `generate_batch_adsorption_structures`
- `vasp_prepare`
- `vasp_execute_batch`
- `analyze_vasp_results`
- `vaspkit_adsorbate_thermo_correction`
- `render_structure_views`

## Workflow

### 1. Preserve adsorption provenance
- Always enumerate sites before broad placement.
- Treat `generate_batch_adsorption_structures` as a convenience wrapper over site enumeration plus placement, not as a separate scientific abstraction.

### 2. Build a controlled candidate set
- Use one adsorbate geometry source across the whole screen.
- If only one site is under study, use `place_adsorbate` directly and keep the chosen site label explicit.
- Use `render_structure_views` when collisions or orientation ambiguity need a visual check before DFT.

### 3. Run one comparable relaxation stage
- Prepare adsorption candidates with one consistent slab regime and method policy.
- Keep clean slab, gas reference, and adsorbed systems method-aligned if adsorption energies will be compared later.

### 4. Rank, then branch
- Use `analyze_vasp_results` to detect failed or unconverged candidates before ranking.
- For adsorption-energy ranking, read `E0` from `analyze_vasp_results` by default rather than other OUTCAR energy fields.
- Run `vaspkit_adsorbate_thermo_correction` only on the retained adsorption states that genuinely need thermal corrections.

## Method-critical defaults
- Do not compare adsorption energies if the clean slab or gas reference was prepared under a different spin, dispersion, or `DFT+U` contract.
- Preserve and report `ads_indices`; they are part of the downstream thermochemistry contract.
- Do not silently accept truncated candidate batches from the wrapper if the ranking depends on full site coverage.

## Output Contract
Return:
- site JSON path
- generated adsorption structure root
- retained candidate list
- VASP analysis summary path
- adsorbate thermochemistry artifact path when used

## References
- Use the primitive skills `adsorption-site-screening` and `adsorbate-and-intermediate-generation` when only the front half of the workflow is needed.
