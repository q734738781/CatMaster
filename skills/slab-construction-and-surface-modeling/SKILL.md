---
name: slab-construction-and-surface-modeling
description: Use this skill for slab construction, vacuum and layer choices, surface supercell setup, and atom-fixing strategy in heterogeneous catalysis workflows.
compatibility: Designed for CatMaster local tools and project-space relative-path execution.
metadata:
  catmaster-suggested-tools: "build_slab fix_atoms_by_layers fix_atoms_by_height supercell"
---

# slab-construction-and-surface-modeling

## Overview
Use this skill to build slab models, choose a freezing strategy, and resize the surface cell without corrupting later adsorption workflows.

## Quick Start
1. Build slabs from a bulk reference before applying any fixing policy.
2. Treat termination choice, vacuum, and slab thickness as part of the comparison contract.
3. Apply either layer-based or height-based fixing deliberately.
4. Use `supercell` only when coverage or lateral separation requires it.

## Suggested tools
- build_slab
- fix_atoms_by_layers
- fix_atoms_by_height
- supercell

## Workflow

### 1. Build slabs with explicit termination policy
- `build_slab` can run on one bulk structure or a whole `bulk_dir`.
- It emits every termination for the chosen Miller index, not just one surface.
- Keep `slab_thickness`, `vacuum_thickness`, `supercell`, `orthogonal`, and `lll_reduce` fixed across compared slabs unless there is a reason to change them.

### 2. Freeze atoms with one clear rule
- `fix_atoms_by_layers` bins atoms by z using `layer_tol`; `freeze_layers` must not exceed the detected layer count.
- `fix_atoms_by_height` freezes atoms inside explicit `z_ranges` and rejects invalid ranges.
- `centralize=true` is a real geometry transform; use it intentionally, not by habit.

### 3. Expand cells only after fixing policy is understood
- `supercell` supports single-file and batch directory mode.
- It drops all selective-dynamics information, so do not call it after you have already finalized atom constraints unless you plan to rebuild them.

## Output Contract
Return:
- slab structure path(s)
- chosen termination or termination set
- fixing strategy and key parameters
- whether `supercell` invalidated previous selective-dynamics flags

## References
- If the task needs adsorption-ready slabs, hand off the post-fix structures to `adsorption-site-screening` instead of mixing slab generation and adsorption placement in one step.
