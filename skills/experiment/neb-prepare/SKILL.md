---
name: neb-prepare
description: "Use this skill for NEB and dimer preparation work: validate endpoint pairs, choose image counts, build image trees, prepare VASP NEB roots, and prepare dimer-ready inputs or raw mode guesses."
license: project-local
compatibility: local
allowed-tools: "estimate_neb_image_count make_neb_geometry vasp_neb_prepare vasp_dimer_prepare make_dimer_mode_from_neb make_dimer_mode_from_mace"
---

# neb-prepare

## Overview
Use this skill when the job is to prepare a pathway calculation rather than execute or analyze it. It covers endpoint validation, interpolation count selection, NEB input-tree assembly, and dimer-ready setup. For the actual run strategy, use `neb-calculation`. For post-run interpretation and pitfalls, use `neb-analysis`.

## Quick Start
1. Confirm the endpoint pair is one local elementary step and preserves atom order.
2. Estimate `n_images` with `estimate_neb_image_count` before interpolating.
3. Build the image tree with `make_neb_geometry`.
4. Prepare the VASP NEB root with `vasp_neb_prepare`.
5. If a dimer refinement branch is needed, prepare the raw mode with `make_dimer_mode_from_neb` or `make_dimer_mode_from_mace`, then build the dimer input with `vasp_dimer_prepare`.
6. If you need the detailed run protocol for `plain-NEB -> CI-NEB` or `NEB/frequency/dimer`, switch to `neb-calculation`.

## Allowed tools
- `estimate_neb_image_count`
- `make_neb_geometry`
- `vasp_neb_prepare`
- `vasp_dimer_prepare`
- `make_dimer_mode_from_neb`
- `make_dimer_mode_from_mace`

## Workflow

### 1. Validate the endpoint contract before interpolation
- The initial and final structures must keep the same atom ordering. Do not interpolate reordered atoms.
- A single NEB should describe one local elementary event, not a long migration across several equivalent sites.
- If the displacement is really a long hop, remodel the final state onto the nearest equivalent site and prepare that primitive hop first.

### 2. Choose the image count from the periodic displacement, not by feel
- Prefer `estimate_neb_image_count` before choosing `n_images`.
- Under periodic boundary conditions, use the same minimum-image convention (`mic=true`) for the estimate that you intend to use for interpolation.
- When no better prior exists, use
  `n_images ≈ ceil(sqrt(sum_i ||Δr_i||^2) / 0.8 Å)`.
- For routine local events, prefer about 4-8 intermediate images.
- If the periodic root-sum-squared displacement exceeds about 6 Å, or if the estimate asks for more than about 8 intermediate images, treat that as a warning that the path is probably too long for one primitive NEB.

### 3. Build the shared image tree
- Use `make_neb_geometry` to generate the flat numbered image tree (`00.vasp`, `01.vasp`, ...).
- Prefer this flat image-tree layout as the common handoff format.
- If the output directory already exists, require `overwrite=true` rather than silently mixing trees.

### 4. Prepare the NEB-ready VASP root
- Use `vasp_neb_prepare` only to assemble the VASP NEB root and protected NEB INCAR settings.
- In image-tree mode, the tool will warn that endpoint `OUTCAR` files still need to be copied in later for barrier analysis. That reminder is expected; preparation should still continue.
- Keep `patch_policy="safe"` unless you intentionally need to override NEB-critical keys.
- `iopt` must stay one of `7`, `2`, or `1`.

### 5. Prepare the dimer branch when needed
- `vasp_dimer_prepare` is for the official VASP improved dimer method (`IBRION=44`).
- Use `make_dimer_mode_from_neb` when a NEB path already localizes the TS region well enough to extract a reaction-direction guess.
- Use `make_dimer_mode_from_mace` when you need a finite-difference mode guess instead.
- `vasp_dimer_prepare` applies the required per-atom `1/sqrt(mass)` transformation internally. Pass it the raw mode, not an already mass-divided mode.
- If you copy a vibrational mode from a VASP `OUTCAR`, use the ordinary raw `dx dy dz` block, not the `Eigenvectors after division by SQRT(mass)` block.

### 6. Keep the execution branches conceptually separate
- The common robust NEB route is `plain-NEB -> CI-NEB`.
- A common TS-refinement route is `plain-NEB or TS guess -> frequency/mode guess -> dimer`.
- This skill only prepares those branches. Use `neb-calculation` if you need the detailed execution sequence.

## Method-critical defaults
- Keep endpoint preparation, image generation, and later execution settings scientifically aligned.
- Do not treat image-count choice as cosmetic; it is part of the pathway model.
- Prefer odd intermediate-image counts for `CI-NEB` refinement so there is a natural central climbing image.
- Prefer even intermediate-image counts for plain `NEB` only when the path is otherwise symmetric and no unique climbing image is needed yet.
- For dimer preparation, never skip the mass weighting performed by `vasp_dimer_prepare`.
- If several imaginary modes exist, the correct dimer direction is the chemically meaningful one, not automatically the largest-magnitude imaginary mode.

## Output Contract
Return:
- validated endpoint pair or the reason it is not NEB-ready
- chosen image-count rationale
- image-tree root
- prepared NEB root plus `INCAR` and `neb_incar_patch.json`
- if dimer preparation was requested, the raw mode path and dimer-ready `POSCAR` path

## References
- Use `neb-calculation` for the detailed execution protocol after preparation is complete.
- Use `neb-analysis` once images have been collected and you need barrier interpretation or QC.
