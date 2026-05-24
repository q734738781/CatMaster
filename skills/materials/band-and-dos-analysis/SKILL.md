---
name: band-and-dos-analysis
description: Use this skill for post-relax bulk electronic-structure workflows when the goal is a clean DOS or band-path-ready calculation sequence with explicit KPOINTS provenance and result summaries.
---

# band-and-dos-analysis

## Overview
Use this skill to build a controlled relax-to-static-to-band/DOS sequence instead of improvising electronic-structure jobs from an unconverged structure. Follow the VASP Wiki pattern: finalize the structure first, then run a dedicated static/DOS stage with explicit k-point and charge-density provenance.

## Quick Start
1. Start from one accepted relaxed bulk structure.
2. If you may reuse fixed charge density, first prepare a self-consistent static stage that keeps `CHGCAR` available, e.g. surface `LCHARG=True` through `user_incar_patch`.
3. For bulk DOS with a suitable 3D Gamma-centered mesh, prepare a DOS stage with `vasp_prepare(preset="dos", ...)`; add `dos_charge_density_path` only when you are intentionally doing the VASP Wiki `ICHARG=11` non-self-consistent branch.
4. For slab, gas, unknown-gap, or tetrahedron-unsuitable meshes, override the DOS starter template with `user_incar_patch`, typically `ISMEAR=0`, `SIGMA=0.03-0.1`, and optionally `EFERMI=MIDGAP` for clearly gapped or unknown-gap systems.
5. For bands, generate a recommended line-mode `KPOINTS` with `generate_kpath` and assemble the band job with `vasp_band_prepare`.
6. Dispatch the resulting VASP stages with `remote_submission` or `remote_submission_batch` using `task_name="vasp_execute"`.
7. Use `analyze_vasp_results` to summarize convergence and parsed bandgap evidence.
8. If you also need a single total-energy value from that summary, use `E0` by default.

## Allowed tools
- `generate_kpath`
- `vasp_prepare`
- `vasp_band_prepare`
- `remote_submission`
- `remote_submission_batch`
- `analyze_vasp_results`
- `execute`

## Workflow

### 1. Accept the relaxed bulk first
- Do not start DOS or band work from a bulk structure that has not passed the base convergence screen.
- Do not treat the DOS written during an ionic relaxation as the final DOS artifact; VASP Wiki recommends a separate static/DOS stage for the converged structure.

### 2. Build DOS as a dedicated post-relax stage
- DOS should be prepared as a separate static-like branch after the structure is finalized.
- For more accurate DOS, increase the k-point density relative to the relaxation/static baseline when that is the stated goal.
- If you want a non-self-consistent DOS (`ICHARG=11`), the preceding self-consistent stage must have produced a reusable `CHGCAR`.
- State whether the DOS branch is self-consistent or fixed-charge-density.
- Band-path generation is a separate primitive: `generate_kpath` proposes the line-mode `KPOINTS`, and `vasp_band_prepare` owns the dedicated band input deck.

### 3. Choose the DOS occupation method explicitly
- For bulk DOS and very accurate static energies, VASP Wiki recommends tetrahedron integration with Blöchl corrections (`ISMEAR=-5`) on a Gamma-centered mesh that can actually form tetrahedra.
- Do not blindly keep the tetrahedron starter for slab, gas, line-mode, or otherwise low-dimensional / too-sparse meshes. In those cases use `user_incar_patch` to switch to Gaussian smearing (`ISMEAR=0`) with small `SIGMA`, typically `0.03-0.1`.
- When the system class or gap status is uncertain, prefer the Gaussian fallback over Methfessel-Paxton. For clearly gapped or unknown-gap cases, `EFERMI=MIDGAP` is a reasonable explicit stabilizer to surface.

### 4. Keep projection and fixed-density caveats visible
- `LORBIT` controls PDOS output, but fixed-density reuse also needs the charge-density provenance to be scientifically traceable.
- For `ICHARG=11/12`, VASP Wiki recommends `LMAXMIX` matched to the highest angular momentum in the PAW data (`2/4/6` for s,p / d / f dominated cases). The tool now auto-fills a baseline when absent; surface any deliberate override.

### 5. Keep provenance visible
- Report the exact `KPOINTS` artifact used for the band job.
- Report the DOS smearing choice, whether the run is self-consistent or `ICHARG=11`, and the source of any reused `CHGCAR`.
- If the upstream static stage was created specifically to feed DOS/bands, say so and keep the `CHGCAR`-writing choice explicit.
- Use the `structure-visual-inspection` skill script when the bulk cell standardization or orientation is visually ambiguous.

## Method-critical defaults
- Do not compare DOS or bandgaps across jobs that changed spin, `DFT+U`, smearing, or k-point density silently.
- For DOS, VASP Wiki recommends a dedicated post-relax static branch; do not interpret relax-stage `DOSCAR` as the final electronic-structure result.
- Keep the tetrahedron default for the cases where it is justified; do not quietly apply it to slab/gas or tetrahedron-ineligible meshes.
- When the electronic character is uncertain, prefer `ISMEAR=0` with small `SIGMA` over `ISMEAR>0`.
- Surface any `user_incar_patch` changes that alter DOS/band interpretation.
- For DOS reuse with `CHGCAR`, report the charge-density source explicitly and make sure the upstream stage actually wrote that `CHGCAR`.
- For line-mode band runs, keep the generated `KPOINTS` artifact and any reused `CHGCAR` source explicit.

## Output Contract
Return:
- DOS or band job root
- generated `KPOINTS` path when relevant
- DOS mode and smearing choice when DOS is in scope
- VASP analysis summary path
- any parsed bandgap value or convergence caveat

## References
- Use `bulk-relax-and-reference` to establish the accepted bulk baseline before this skill.
