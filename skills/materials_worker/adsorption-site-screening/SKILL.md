---
name: adsorption-site-screening
description: Use this skill for adsorption-site enumeration and adsorbate placement workflows, including candidate screening setup and batch structure generation.
---

# adsorption-site-screening

## Overview
Use this skill to enumerate adsorption sites, place adsorbates reproducibly, and emit batch-ready adsorption structures with metadata.

## Quick Start
1. Start from a validated, termination-reviewed slab and a canonical adsorbate file.
2. Prefer an orthogonal adsorption slab; if the slab is non-orthogonal, keep the exception explicit.
3. For site-label selection or broad screening, run `enumerate_adsorption_sites` and keep `sites_json_rel`. An explicitly supplied placement coordinate does not require enumeration.
4. Use `place_adsorbate` for one chosen site or `generate_batch_adsorption_structures` for a screening set.
5. For a screening batch, choose the slab source, adsorbate source, site family, distance, and `max_structures` from the task intent before calling the tool.
6. Preserve the returned `ads_indices` metadata for downstream relaxations and thermochemistry.

## Allowed tools
- `enumerate_adsorption_sites`
- `place_adsorbate`
- `generate_batch_adsorption_structures`

## Workflow

### 1. Choose an enumerated site or explicit coordinate
- Do not call placement tools on guessed paths. First create or locate the exact slab and adsorbate files under the workspace, then reuse returned paths.
- Before enumeration, confirm the slab has a known selected termination and recorded `orthogonal` setting. If not, use `slab-construction-and-surface-modeling` or `surface-and-termination-screening` first instead of treating an arbitrary POSCAR as adsorption-ready.
- `enumerate_adsorption_sites` writes a JSON site list and returns `default_site_label`.
- Each enumerated site row includes Cartesian `cart_coords`; the `ontop_0` / `bridge_1` / `hollow_2` labels used by `place_adsorbate` come from this enumeration.
- In `mode=all`, the candidate families are `ontop`, `bridge`, and `hollow`.
- For single-structure placement, do not guess the site label if the JSON has already been generated.

### 2. Place one structure intentionally
- `place_adsorbate` accepts `site_label` values like `ontop_0`; `site_label=auto` prefers the first available `ontop`, then `bridge`, then `hollow`.
- `place_adsorbate` also accepts `site_cart_coords=[x, y, z]` in Cartesian Angstrom for direct placement without enumerating ASF sites. `site_label` and `site_cart_coords` are mutually exclusive.
- XYZ/internal molecular geometry is preserved during placement; the tool does not automatically reorient the molecule.
- The placement point is the adsorption-site coordinate returned by ASF at the requested `distance`; the molecule is translated so the center of mass of its lowest-z atom layer lands on that site coordinate.
- The tool preserves slab selective dynamics and marks newly added adsorbate atoms as movable.
- Returned metadata includes `ads_indices_added`, merged `ads_indices`, `metadata_rel`, `ads_indices_json_rel`, and the chosen site coordinates.

### 3. Batch only the candidates you want to screen
- `generate_batch_adsorption_structures` supports either `slab_file` or `slab_dir`, not both.
- When `slab_dir` is used, each slab gets its own subdirectory under `output_dir`. Colliding flattened names receive numeric suffixes; use the returned source/output mapping rather than deriving names. Keep the source set unchanged when continuing the same site enumeration.
- `max_structures` caps generated structures **per slab per call**, not the whole multi-slab batch. Keep `site_manifest_ref`; when `has_more_sites=true`, continue with the returned `next_site_offset` so stable site IDs are neither repeated nor skipped.
- Each slab writes `all_sites.json` before bounded structure generation. It records every ordered site, coordinates, current selection, and generated state.
- The batch output writes `batch_structures.json` plus `ads_indices.json` under `output_dir`, and per-structure `<output>.meta.json` sidecars with `ads_indices`.
- `ads_indices.json` rows include the slab source, slab id, site label, generated POSCAR path, newly added adsorbate indices, and merged adsorbate indices.

## Method-critical defaults
- The slab termination provenance must be reviewed before adsorption placement. If only one unknown slab is supplied, report the missing termination provenance and remaining uncertainty.
- For new adsorption-ready slabs, prefer `orthogonal=true` upstream and preserve that choice across all generated candidates.
- If the screening is intended for quantitative ranking, preserve metadata and reference-state traceability needed for downstream consistent energy evaluation.
- Do not generate candidate structures without carrying forward the adsorbate indices and site provenance required for later interpretation.

## Output Contract
Return:
- slab termination provenance and `orthogonal` setting when known
- site source (`sites_json_rel`, its label, or explicit Cartesian coordinates)
- generated structure path or `output_dir_rel`
- `ads_indices` metadata path(s)
- whether the batch was truncated
- `site_manifest_ref`, `has_more_sites`, and `next_site_offset` for every slab

## References
- When the slab already carries adsorbate metadata, rely on the merged `ads_indices` returned by the tool instead of recomputing adsorbate atom indices.
- For the full screening workflow, hand off to `adsorption-screening` after the primitive site list is established.
