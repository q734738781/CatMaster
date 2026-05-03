---
name: vasp-input-preparation
description: Use this skill for preparing canonical VASP relax/static/frequency/DOS/MD input sets, choosing the correct regime and preset, handling INCAR patch-policy edge cases, explicit slab-frequency setup, and producing execution-ready folder layouts before dispatch.
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
6. If a later DOS/band branch should reuse `CHGCAR`, make that an explicit upstream requirement and preserve the file in the self-consistent stage.
7. If comparison-sensitive toggles matter (`use_d3`, `use_dft_plus_u`, `enable_dipole`, spin/`MAGMOM`, smearing, or reference-state choices), set them explicitly in the first preparation call and report them.
8. Use `vasp_band_prepare` instead of `vasp_prepare` when the job is a line-mode band-structure calculation with an explicit band `KPOINTS`.

## Allowed tools
- `fix_atoms_by_indices`
- `vasp_prepare`
- `vasp_band_prepare`
- `generate_kpath`

## Workflow

### INCAR patching
- Use `user_incar_patch` for targeted job-specific INCAR controls.
- For DOS jobs, `dos_charge_density_path` copies a CHGCAR into the output root, enables `ICHARG=11`, and applies the recommended `LMAXMIX` baseline when absent.
- For DOS or MD presets, `user_incar_patch` is the main hook for controls such as `NEDOS`, `ISMEAR`, `TEBEG`, `TEEND`, `POTIM`, `NSW`, `SMASS`, or `MDALGO`.
- `MAGMOM`, `LDAUU`, and `LDAUJ` must be element-value maps such as `{"Fe": 2.2}`.
- Use `null` in `user_incar_patch` to request removal of a key from the final INCAR.
- Keep `patch_policy="safe"` by default; use `force` only when intentionally overriding canonical defaults.

### 1. Pick the right canonical preset
- Confirm the input path exists before preparing a VASP tree.
- `vasp_prepare(preset="relax", ...)` is for ionic relaxation jobs.
- `vasp_prepare(preset="static", ...)` is for static jobs and enforces `NSW=1`, `IBRION=-1`.
- `vasp_prepare(preset="freq", ...)` is for finite-difference vibrational jobs and enforces the frequency-specific overrides.
- `vasp_prepare(preset="dos", ...)` is for DOS/PDOS-style jobs. The tool starts from a bulk-leaning tetrahedron-style template with `ISMEAR=-5`, `NEDOS=2001`, `LORBIT=11`, and if `dos_charge_density_path` is provided it copies `CHGCAR`, sets `ICHARG=11`, and auto-fills a recommended `LMAXMIX` baseline when absent.
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

### 4. Add a system-aware starting point
- The tool's canonical defaults are safe starters, but VASP Wiki guidance is more system-aware than a single global default. If the material class is known, surface the system-specific occupancy choice explicitly through `user_incar_patch`.
- Good start for unknown / high-throughput / gap-uncertain systems: `ISMEAR=0`, `SIGMA=0.05-0.1`; for clearly gapped or unknown-gap cases, consider `EFERMI=MIDGAP`.
- Good start for metallic relaxations or metallic force-sensitive work: `ISMEAR=1`, `SIGMA=0.2`; check that the entropy term is acceptably small before treating the setup as converged.
- Good start for adsorbate slab calculations: `ISMEAR=0`, `SIGMA=0.1`.
- Good start for semiconductors or insulators in bulk static or DOS-like post-processing: `ISMEAR=-5` on a sufficiently dense `Gamma`-centered mesh.
- Good start for slab/gas/low-dimensional or tetrahedron-ineligible DOS work: do not assume tetrahedron is safe; fall back to `ISMEAR=0` with small `SIGMA`, typically `0.03-0.1`.
- Good start for gas references: keep `ISYM=0`, `ISMEAR=0`, `SIGMA=0.01`, and `1x1x1` KPOINTS.
- Treat these as explicit starting recipes, not convergence proofs. Once the system class is known, the starting recipe should be stated in the task packet instead of left implicit.

### 5. Override carefully
- `user_incar_patch` is for targeted overrides. For `dos` and `md`, it is also the preferred path for one-call tuning of method knobs.
- For DOS tuning, the common explicit overrides are `ISMEAR`, `SIGMA`, `EFERMI`, `NEDOS`, `EMIN`, `EMAX`, and occasionally `LMAXMIX`.
- `MAGMOM`, `LDAUU`, and `LDAUJ` must be element maps.
- `patch_policy="safe"` rejects overrides only for the small protected set of preset/regime-bound keys. For `dos` and `md`, safe intentionally leaves most method controls overrideable so the model can change `NEDOS`, `ISMEAR`, `NSW`, `POTIM`, `TEBEG`, `TEEND`, `SMASS`, `MDALGO`, and similar knobs without switching to `force`.
- `patch_policy="force"` applies the patch after canonical defaults, including explicit removal with `null`.
- Only use `force` when you are intentionally breaking the preset identity itself, for example replacing `IBRION` with a non-matching job type or tearing down other protected invariants.

### 6. Keep output layout clean
- Single structure input writes to `output_root/<stem>/`.
- Directory input preserves relative layout and appends `<stem>/` per structure.
- Do not mix unrelated relax and SP campaigns in the same ambiguous tree.

### 7. Add explicit band-path or phonon handoffs
- Use `generate_kpath` only after a relaxed bulk baseline is accepted; it emits the band-path `KPOINTS`, not a full calc directory.
- Use `vasp_band_prepare` when that line-mode `KPOINTS` should become a dedicated band-structure job root.
- For phonon workflows, keep displacement generation outside `vasp_prepare`; the prepare tool should only own the force-job input deck after the displacement set already exists.

### 8. Prepare frequency jobs explicitly if requested
- Do not treat vibrational thermochemistry jobs as ordinary relax or SP jobs. Prepare them as a separate stage after the relevant relaxed structure is finalized.
- For finite-difference frequency runs, use `IBRION=5`, `POTIM=0.015`, `NFREE=2`, and `ISYM=0`.
- For slab adsorbate frequency jobs, do not blindly preserve the relax-stage selective-dynamics mask. A relaxed adsorbate-slab structure may still have mobile surface atoms, which is not acceptable for adsorbate-only vibrational thermochemistry. If the target is adsorbate-only slab thermochemistry and `ads_indices` are available, explicitly run `fix_atoms_by_indices(indices=ads_indices, reverse=true)` on the relaxed adsorbate-slab structure before writing the frequency input.
- Treat this explicit refixing step as mandatory unless the task explicitly requests a broader vibrational model.
- Keep the frequency job on the same scientifically relevant electronic-structure footing as the corresponding energy campaign: same `ENCUT`, `EDIFF`, smearing policy, spin treatment, `DFT+U`, `D3`, dipole correction, and other reference-sensitive toggles.

## Method-critical defaults
- Do not silently rely on defaults for `use_d3`, `use_dft_plus_u`, `enable_dipole`, spin treatment, or reference-state-sensitive INCAR toggles when they affect comparison.
- Keep clean slab, gas-phase reference, adsorbed structures, and downstream static calculations scientifically comparable.
- When the system class is known, prefer a VASP-Wiki-style explicit smearing recipe over silent inheritance of the tool baseline.
- For unknown systems, `ISMEAR=0` with small `SIGMA` is the safer generic starting point than jumping directly to Methfessel-Paxton.
- For metallic relaxations, a practical first explicit recipe is `ISMEAR=1`, `SIGMA=0.2`, then check the entropy term before treating it as production-worthy.
- For adsorbate slab calculations, `ISMEAR=0`, `SIGMA=0.1` is the project-default explicit starting recipe.
- For slab adsorption studies, do not set `relax_cell=true` unless the task explicitly requires a variable-cell study and the system is not in slab mode.
- For slab work, use `k*a ~= 35 Å` as the initial project default unless convergence evidence justifies something else.
- Do not force bulk references to obey the slab default. Bulk `k_product` should be chosen from the bulk convergence requirement, not copied mechanically from slab policy.
- For DOS / projected-DOS / finer electronic-structure jobs, it is acceptable to increase `k_product` to around `50` when the extra sampling is part of the stated objective.
- For DOS, keep in mind the VASP-recommended pattern: relax first, then do a DOS-oriented static stage; do not rely on relax-stage `DOSCAR` as the final result.
- The default DOS preset is VASP-Wiki-aligned for bulk tetrahedron DOS, not a universal recommendation for slab/gas/low-dimensional meshes. If the mesh cannot support tetrahedra or the system class is uncertain, override to `ISMEAR=0` with small `SIGMA` (`0.03-0.1`).
- For semiconducting or insulating bulk post-processing, `ISMEAR=-5` is a good explicit start only when the mesh is dense enough and `Gamma`-centered.
- For clearly gapped or unknown-gap systems using Gaussian fallback, consider surfacing `EFERMI=MIDGAP`.
- For tetrahedron DOS, keep the mesh Gamma-centered and dense enough to justify interpolation.
- If reusing a converged `CHGCAR`, surface that explicitly with `dos_charge_density_path`, and make sure the upstream self-consistent stage actually kept `CHGCAR` (for example by setting `LCHARG=True` there).
- For fixed-density DOS/band reuse (`ICHARG=11`), VASP Wiki recommends `LMAXMIX=2/4/6` for s,p / d / f dominated systems; the tool now auto-fills a baseline when absent, but deliberate overrides should still be reported.
- Tune `NEDOS` together with `EMIN/EMAX` when narrow peaks or edge resolution matter; VASP Wiki recommends checking the integrated DOS to confirm the grid is fine enough.
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
- Use `band-and-dos-analysis` or `phonon-displacement-workflow` when the task goes beyond pure input preparation.
- Hand off band/DOS follow-up to `band-and-dos-analysis`, strain grids to `elastic-property-workup`, displacement sets to `phonon-displacement-workflow`, and MD trajectories to `md-diffusion-analysis`.
