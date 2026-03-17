---
name: vasp-input-preparation
description: Use this skill for preparing VASP relaxation and single-point input sets, choosing calc_type and key defaults correctly, handling INCAR override edge cases, explicit slab-frequency setup, and producing execution-ready folder layouts before dispatch.
license: project-local
compatibility: local
allowed-tools: "fix_atoms_by_indices vasp_relax_prepare vasp_sp_prepare"
metadata:
  catmaster-suggested-tools: "fix_atoms_by_indices vasp_relax_prepare vasp_sp_prepare"
---

# vasp-input-preparation

## Overview
Use this skill to produce execution-ready VASP input trees without fighting tool-enforced defaults.

## Quick Start
1. Choose `vasp_relax_prepare` for ionic relaxation and `vasp_sp_prepare` for static calculations.
2. Set `calc_type` correctly: `bulk`, `slab`, or `gas`.
3. Put one preparation campaign under one clean `output_root`.
4. Override INCAR only where the tool actually allows it.

## Suggested tools
- `fix_atoms_by_indices`
- vasp_relax_prepare
- vasp_sp_prepare

## Workflow

### 1. Pick the right preparation tool
- `vasp_relax_prepare` is for relax jobs.
- `vasp_sp_prepare` is for static jobs and already enforces `NSW=1`, `IBRION=-1`.

### 2. Choose the correct regime
- `bulk`: periodic bulk models; `relax_cell=True` is allowed and switches `ISIF=3`.
- `slab`: surface slabs; `ISIF=2`, KPOINTS z forced to 1, `relax_cell=True` is invalid.
- `gas`: molecules or gas references; forced `1x1x1` KPOINTS and `relax_cell=True` is invalid.

### 3. Respect built-in defaults
- The tool already owns the main preset, including pseudopotential family, `ENCUT`, `EDIFF`, smearing defaults, and relax-vs-SP mode settings.
- `compute_dos`, `use_d3`, `use_dft_plus_u`, and `enable_dipole` are the toggles worth surfacing when they matter.
- Project house default for `k_product` is `35` for production slab work unless the task explicitly provides a convergence-backed value.

### 4. Override carefully
- `user_incar_settings` is for targeted overrides, not replacing the preset.
- `MAGMOM`, `LDAUU`, and `LDAUJ` must be element maps.
- `null` can remove optional keys, but tool-required regime keys still win.

### 5. Keep output layout clean
- Single structure input writes to `output_root/<stem>/`.
- Directory input preserves relative layout and appends `<stem>/` per structure.
- Do not mix unrelated relax and SP campaigns in the same ambiguous tree.

### 6. Prepare frequency jobs explicitly if requested
- Do not treat vibrational thermochemistry jobs as ordinary relax or SP jobs. Prepare them as a separate stage after the relevant relaxed structure is finalized.
- For finite-difference frequency runs, use `IBRION=5`, `POTIM=0.015`, `NFREE=2`, and `ISYM=0`.
- For slab adsorbate frequency jobs, do not blindly preserve the relax-stage selective-dynamics mask. A relaxed adsorbate-slab structure may still have mobile surface atoms, which is not acceptable for adsorbate-only vibrational thermochemistry. If the target is adsorbate-only slab thermochemistry and `ads_indices` are available, explicitly run `fix_atoms_by_indices(indices=ads_indices, reverse=true)` on the relaxed adsorbate-slab structure before writing the frequency input.
- Treat this explicit refixing step as mandatory unless the task explicitly requests a broader vibrational model.
- Keep the frequency job on the same scientifically relevant electronic-structure footing as the corresponding energy campaign: same `ENCUT`, `EDIFF`, smearing policy, spin treatment, `DFT+U`, `D3`, dipole correction, and other reference-sensitive toggles.

## Method-critical defaults
- Do not silently rely on defaults for `use_d3`, `use_dft_plus_u`, `enable_dipole`, spin treatment, or reference-state-sensitive INCAR toggles when they affect comparison.
- Keep clean slab, gas-phase reference, adsorbed structures, and downstream static calculations scientifically comparable.
- For slab adsorption studies, do not set `relax_cell=true` unless the task explicitly requires a variable-cell study and the system is not in slab mode.
- For slab work, use `k*a ~= 35 Å` as the initial project default unless convergence evidence justifies something else.
- Do not force bulk references to obey the slab default. Bulk `k_product` should be chosen from the bulk convergence requirement, not copied mechanically from slab policy.
- For DOS / projected-DOS / finer electronic-structure jobs, it is acceptable to increase `k_product` to around `50` when the extra sampling is part of the stated objective.
- Always report any non-default toggles that materially affect interpretation.
- When frequency-derived thermochemistry is the target, do not silently reuse generic relax or SP INCARs; surface the frequency-specific overrides explicitly.

## Output Contract
Return:
- `output_root_rel`
- representative prepared directory
- processed structure count
- any non-default method toggles or overrides that materially affect execution

## References
- Inspect the tool schema/source when edge cases around `k_product`, DFT+U, or INCAR normalization matter.
