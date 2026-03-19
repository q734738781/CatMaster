---
name: thermo-free-energy-and-reporting
description: Use this skill for thermodynamic and free-energy post-processing, result normalization, explicit adsorbate-only frequency-job SOP, and concise reporting standards for catalyst comparison.
license: project-local
compatibility: local
allowed-tools: "fix_atoms_by_indices identify_structure_fragments vaspkit_adsorbate_thermo_correction vaspkit_gas_thermo_correction"
metadata:
  catmaster-suggested-tools: "fix_atoms_by_indices identify_structure_fragments vaspkit_adsorbate_thermo_correction vaspkit_gas_thermo_correction"
---

# thermo-free-energy-and-reporting

## Overview
Use this skill to convert raw electronic-structure results into comparable thermodynamic conclusions.

## Quick Start
1. Gather only validated energies and corrections.
2. Fix one reference convention before computing any comparison table.
3. Keep units and correction assumptions explicit.
4. Report both the final ranking and the assumptions that could change it.

## Suggested tools
- `fix_atoms_by_indices`
- `identify_structure_fragments`
- `vaspkit_adsorbate_thermo_correction`
- `vaspkit_gas_thermo_correction`

## Workflow

### 1. Gather the required inputs
- Collect clean slab, adsorbed, gas, and pathway energies as needed.
- Exclude unconverged or obviously pathological structures from the final table.
- For vibrational thermochemistry, ensure the frequency job was prepared explicitly rather than inferred from a generic relax or SP setup.

### 2. Apply one thermodynamic convention
- Use consistent reference states and correction assumptions across the compared entries.
- Do not mix adsorption-energy and free-energy conventions without labeling them.

### 3. Frequency-job preparation requirements
- For VASP finite-difference frequency jobs, use `IBRION=5`, `POTIM=0.015`, `NFREE=2`, and `ISYM=0`.
- Treat `ISYM=0` as mandatory for these jobs; leaving symmetry on can conflict with parallel settings such as `NCORE` and fail before frequencies are evaluated.
- For slab adsorbate thermochemistry, freeze the slab and keep only the adsorbate degrees of freedom active unless the task explicitly calls for a broader vibrational model.
- Do not assume the relaxed adsorbate-slab `CONTCAR` already has the correct selective-dynamics mask for adsorbate-only thermochemistry. A relax-stage mask that still leaves surface atoms mobile is not acceptable for adsorbate-only vibrational treatment.
- If the objective is adsorbate-only slab thermochemistry and `ads_indices` are known, explicitly refreeze the relaxed structure before writing the frequency job. Use `fix_atoms_by_indices(indices=ads_indices, reverse=true)` on the relaxed adsorbate-slab structure so only the adsorbate atoms remain `T T T` and every slab atom is `F F F`.
- If the adsorbate may have detached into a weakly bound molecular fragment, probe the relaxed structure first with `identify_structure_fragments`. Prefer the default `jmolnn` probe and compare the fragment/reference-index overlap against existing `ads_indices`. Use this as a physisorption sanity check only; do not treat it as a robust chemisorption detector.
- Treat this explicit refixing step as mandatory unless the task explicitly requests a broader vibrational model. Do not skip it merely because the bottom slab layers were already frozen during relaxation.
- Preserve the same electronic-structure settings used for the associated energy campaign whenever they materially affect the comparison: `ENCUT`, `EDIFF`, smearing, spin treatment, `DFT+U`, `D3`, dipole correction, and related reference-sensitive toggles.
- Keep the thermochemistry directory traceable to the relaxed structure and the exact reference-state convention used for the final table.

### 4. Report comparison-ready outputs
- Produce a table with values, units, and the exact convention used.
- Keep the supporting artifact paths with the summary so the ranking is auditable.

## Method-critical defaults
- Do not mix raw electronic energies, adsorption energies, and free-energy values without labeling the convention explicitly.
- Keep reference states and correction conventions fixed within one comparison table.
- Report the exact convention used and any assumptions that could change the ranking.
- For adsorbate calculation's vibrational corrections, use adsorbate-only vibrational treatment rather than whole-slab thermochemistry unless the task explicitly justifies another convention.
- If ASE fallback is used instead of VASPKIT, keep the backend label in the result summary and treat the correction as an approximation rather than silently presenting it as native VASPKIT output.

## Output Contract
Return:
- final table or summary path
- value definitions and units
- reference-state convention
- any caveats that materially affect comparison

## References
- If correction details become nontrivial, load the relevant stage outputs and compute them explicitly rather than summarizing from memory.
- For band/DOS, NEB, and MD-derived tables, use `band-and-dos-analysis`, `reaction-neb-analysis`, or `md-diffusion-analysis` as the upstream workflow skill before final thermochemistry reporting.
