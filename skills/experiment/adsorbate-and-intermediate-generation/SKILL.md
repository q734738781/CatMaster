---
name: adsorbate-and-intermediate-generation
description: Use this skill for generating adsorbates and reaction intermediates, standardizing molecular inputs, and preparing structures for adsorption placement.
license: project-local
compatibility: local
allowed-tools: "create_molecule_from_smiles"
metadata:
  catmaster-suggested-tools: "create_molecule_from_smiles"
---

# adsorbate-and-intermediate-generation

## Overview
Use this skill to turn SMILES-level adsorbate requests into stable 3D molecule files for later slab placement.

## Quick Start
1. Normalize the requested species name and canonical SMILES before generation.
2. Choose `fmt` intentionally: `poscar` for slab workflows, `xyz` for inspection, `both` when both are useful.
3. Set `output_path` as a path prefix, not a final filename.
4. Carry forward the returned `xyz_file_rel` or `poscar_file_rel` instead of reconstructing paths by hand.

## Suggested tools
- `create_molecule_from_smiles`

## Workflow

### 1. Normalize chemistry first
- Resolve ambiguous protonation, charge, or radical assumptions before calling the tool.
- If the SMILES is chemically wrong, the generated structure will still be wrong.

### 2. Generate one canonical molecule file per species
- `create_molecule_from_smiles` builds a 3D conformer with deterministic embedding.
- `fmt=poscar` writes a boxed `.vasp`; `fmt=xyz` writes `.xyz`; `fmt=both` writes both.
- `box_padding` controls the cubic POSCAR box size for isolated-molecule references.

### 3. Use returned paths as the handoff contract
- The tool returns `formula`, `natoms`, `xyz_file_rel`, `poscar_file_rel`, and `box_size`.
- Downstream adsorption workflows should use the returned molecule path directly.

## Method-critical defaults
- Keep protonation, charge, and radical assumptions explicit because they change the generated molecule, not just the file name.
- Use the same `fmt` convention across a comparison set so the downstream placement workflow sees a consistent input type.

## Output Contract
Return:
- canonical molecule file path(s)
- chosen `fmt`
- any charge or species assumptions that remain unresolved

## References
- For slab placement, hand off the returned molecule file to `adsorption-site-screening` first, then promote the screening set to `adsorption-screening`.
