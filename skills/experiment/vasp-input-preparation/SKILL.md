---
name: vasp-input-preparation
description: Use this skill for preparing canonical VASP relax/static/frequency/DOS/MD input sets, choosing the correct regime and preset, handling INCAR patch-policy edge cases, explicit slab-frequency setup, and producing execution-ready folder layouts before dispatch.
license: project-local
compatibility: local
allowed-tools: "fix_atoms_by_indices vasp_prepare"
metadata:
  catmaster-suggested-tools: "fix_atoms_by_indices vasp_prepare"
---

# vasp-input-preparation

## Overview
Use this skill to produce execution-ready VASP input trees without fighting tool-enforced defaults.

## Quick Start
1. Use `vasp_prepare` with the right `preset`: `relax`, `static`, `freq`, `dos`, or `md`.
2. Set `regime` correctly: `bulk`, `slab`, or `gas`.
3. Put one preparation campaign under one clean `output_root`.
4. Use `patch_policy="safe"` by default.
5. For `dos` and `md`, treat the preset as a starter template and do job-specific tuning through `user_incar_patch` in the same call.

## Suggested tools
- `fix_atoms_by_indices`
- `vasp_prepare`

## Workflow

### 1. Pick the right canonical preset
- `vasp_prepare(preset="relax", ...)` is for ionic relaxation jobs.
- `vasp_prepare(preset="static", ...)` is for static jobs and enforces `NSW=1`, `IBRION=-1`.
- `vasp_prepare(preset="freq", ...)` is for finite-difference vibrational jobs and enforces the frequency-specific overrides.
- `vasp_prepare(preset="dos", ...)` is for DOS/PDOS-style jobs. The tool starts from a tetrahedron-style template with `ISMEAR=-5`, `NEDOS=2001`, `LORBIT=11`, and if `dos_charge_density_path` is provided it copies `CHGCAR` and sets `ICHARG=11`.
- `vasp_prepare(preset="md", ...)` is for molecular dynamics. The tool starts from a Nose-Hoover NVT-style template with `IBRION=0`, `MDALGO=2`, `NSW=1000`, `POTIM=1.0`, `TEBEG=300`, `TEEND=300`, and `SMASS=0`.

### 2. Choose the correct regime
- `bulk`: periodic bulk models; `relax_cell=True` is allowed only with `preset="relax"` and switches `ISIF=3`.
- `slab`: surface slabs; `ISIF=2`, KPOINTS z forced to 1, `relax_cell=True` is invalid.
- `gas`: molecules or gas references; forced `1x1x1` KPOINTS and `relax_cell=True` is invalid.

### 3. Respect built-in defaults
- The tool already owns the main preset, including pseudopotential family, `ENCUT`, `EDIFF`, smearing defaults, and relax-vs-SP mode settings.
- `compute_dos`, `use_d3`, `use_dft_plus_u`, and `enable_dipole` are the toggles worth surfacing when they matter.
- Project house default for `k_product` is `35` for production slab work unless the task explicitly provides a convergence-backed value.
- For DOS jobs, the preset is a recommended starting point, not a claim that one DOS INCAR works for every material class.
- For MD jobs, the preset is a recommended thermostat/template start, not a full replacement for system-specific choices of timestep, temperature schedule, thermostat mass, or ensemble controls.

### 4. Override carefully
- `user_incar_patch` is for targeted overrides. For `dos` and `md`, it is also the preferred path for one-call tuning of method knobs.
- `MAGMOM`, `LDAUU`, and `LDAUJ` must be element maps.
- `patch_policy="safe"` rejects overrides only for the small protected set of preset/regime-bound keys. For `dos` and `md`, safe intentionally leaves most method controls overrideable so the model can change `NEDOS`, `ISMEAR`, `NSW`, `POTIM`, `TEBEG`, `TEEND`, `SMASS`, `MDALGO`, and similar knobs without switching to `force`.
- `patch_policy="force"` applies the patch after canonical defaults, including explicit removal with `null`.
- Only use `force` when you are intentionally breaking the preset identity itself, for example replacing `IBRION` with a non-matching job type or tearing down other protected invariants.

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
- For DOS, keep in mind the VASP-recommended pattern: relax first, then do a DOS-oriented static stage; if reusing a converged `CHGCAR`, surface that explicitly with `dos_charge_density_path`.
- For MD, explicitly state whether you are keeping the default Nose-Hoover template or overriding thermostat/integration controls through `user_incar_patch`.
- Always report any non-default toggles that materially affect interpretation.
- When frequency-derived thermochemistry is the target, do not silently reuse generic relax or SP INCARs; surface the frequency-specific overrides explicitly.

## Output Contract
Return:
- `output_root_rel`
- representative prepared directory
- selected `preset` and `regime`
- generated `k_grid`
- any non-default method toggles or overrides that materially affect execution

## References
- Inspect the tool schema/source when edge cases around `k_product`, DFT+U, or INCAR normalization matter.
