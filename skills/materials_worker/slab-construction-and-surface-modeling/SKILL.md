---
name: slab-construction-and-surface-modeling
description: Use this skill for slab construction, vacuum and layer choices, surface supercell setup, and atom-fixing strategy in heterogeneous catalysis workflows.
---

# slab-construction-and-surface-modeling

## Overview
Use this skill to build slab models, choose a freezing strategy, and resize the surface cell without corrupting later adsorption workflows.

## Quick Start
1. Decide whether the slab is for adsorption, surface-energy ranking, or another surface model before calling `build_slab`.
2. For adsorption-ready slabs, prefer `orthogonal=true` unless the user explicitly needs the native non-orthogonal surface cell.
3. Build and review the termination family before choosing a slab; do not silently use the first emitted termination.
4. Apply layer-based, height-based, or explicit-index fixing deliberately.
5. Use `supercell` only when coverage or lateral separation requires it.

## Allowed tools
- `build_slab`
- `fix_atoms_by_layers`
- `fix_atoms_by_height`
- `fix_atoms_by_indices`
- `supercell`

## Workflow

### 1. Build slabs with explicit termination policy
- `build_slab` can run on one bulk structure or a whole `bulk_dir`.
- It emits every termination for the chosen Miller index, not just one surface.
- For adsorption calculations, pass `orthogonal=true` by default so adsorbate placement, lateral separation, and downstream cell interpretation use a c-oriented orthogonal slab. Use `orthogonal=false` only when preserving the native surface cell is intentional, and report that exception.
- Keep `slab_thickness`, `vacuum_thickness`, `supercell`, `orthogonal`, and `lll_reduce` fixed across compared slabs unless there is a reason to change them.
- Treat `termination_index=0` as a label, not a scientific default. Compare the emitted terminations by stoichiometry, exposed species/layers, polarity, symmetry, and visual sanity before selecting one for fixing or adsorption.

### 2. Freeze atoms with one clear rule
- `fix_atoms_by_layers` bins atoms by z using `layer_tol`; `freeze_layers` must not exceed the detected layer count.
- `fix_atoms_by_height` freezes atoms inside explicit `z_ranges` and rejects invalid ranges.
- `fix_atoms_by_indices` uses explicit `0-based` atom indices and is the safest choice when downstream metadata such as `ads_indices` already identifies the target atoms.
- All three fixing tools support `reverse=true`, which inverts the usual meaning and keeps the selected atoms free while freezing everything else.
- `centralize=true` is a real geometry transform; use it intentionally, not by habit.

### 3. Expand cells only after fixing policy is understood
- `supercell` supports single-file and batch directory mode.
- `supercell` preserves `selective_dynamics` in POSCAR/VASP outputs, so fixed/free masks can survive lateral expansion when the input already carries them.

## Method-critical defaults
- Keep slab thickness, vacuum thickness, termination choice, and freezing policy fixed across a comparison set.
- For adsorption-ready slab construction, the project preference is `orthogonal=true`; any non-orthogonal adsorption slab must be intentional and stated.
- Termination provenance review is mandatory before adsorption or surface ranking. This is not a proof from one POSCAR; record which terminations were generated or supplied, the visible exposed-layer/stoichiometry/polarity evidence used, and any uncertainty. If the task provides only one slab with no provenance, state that the termination has not been reviewed and either inspect it or regenerate the termination family.
- Do not compare surface calculations if the mask or cell expansion strategy changed silently.

## Output Contract
Return:
- slab structure path(s)
- chosen termination or termination set
- termination provenance evidence or the reason a single provided termination was accepted despite limited certainty
- `orthogonal` setting and any exception to the adsorption default
- fixing strategy and key parameters
- whether inherited `selective_dynamics` was preserved through any supercell expansion

## References
- If the task needs adsorption-ready slabs, hand off the post-fix structures to `adsorption-site-screening` instead of mixing slab generation and adsorption placement in one step.
- If you are screening terminations as a workflow, hand off to `surface-and-termination-screening` once the slab family is ready.
