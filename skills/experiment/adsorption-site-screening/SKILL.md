---
name: adsorption-site-screening
description: Use this skill for adsorption-site enumeration and adsorbate placement workflows, including candidate screening setup and batch structure generation.
---

# adsorption-site-screening

## Overview
Use this skill to enumerate adsorption sites, place adsorbates reproducibly, and emit batch-ready adsorption structures with metadata.

## Quick Start
1. Start from a validated slab and a canonical adsorbate file.
2. Run `enumerate_adsorption_sites` first and keep the returned `sites_json_rel`.
3. Use `place_adsorbate` for one chosen site or `generate_batch_adsorption_structures` for a screening set.
4. Preserve the returned `ads_indices` metadata for downstream relaxations and thermochemistry.

## Allowed tools
- `enumerate_adsorption_sites`
- `place_adsorbate`
- `generate_batch_adsorption_structures`

## Workflow

### 1. Enumerate before placing
- `enumerate_adsorption_sites` writes a JSON site list and returns `default_site_label`.
- Each enumerated site row includes Cartesian `cart_coords`; the `ontop_0` / `bridge_1` / `hollow_2` labels used by `place_adsorbate` come from this enumeration.
- In `mode=all`, the candidate families are `ontop`, `bridge`, and `hollow`.
- For single-structure placement, do not guess the site label if the JSON has already been generated.

### 2. Place one structure intentionally
- `place_adsorbate` accepts `site_label` values like `ontop_0`; `site_label=auto` prefers the first available `ontop`, then `bridge`, then `hollow`.
- `place_adsorbate` also accepts `site_cart_coords=[x, y, z]` in Cartesian Angstrom for direct placement. `site_label` and `site_cart_coords` are mutually exclusive.
- XYZ/internal molecular geometry is preserved during placement; the tool does not automatically reorient the molecule.
- The placement point is the adsorption-site coordinate returned by ASF at the requested `distance`; the molecule is translated so the center of mass of its lowest-z atom layer lands on that site coordinate.
- The tool preserves slab selective dynamics and marks newly added adsorbate atoms as movable.
- Returned metadata includes `ads_indices_added`, merged `ads_indices`, `metadata_rel`, `ads_indices_json_rel`, and the chosen site coordinates.

### 3. Batch only the candidates you want to screen
- `generate_batch_adsorption_structures` supports either `slab_file` or `slab_dir`, not both.
- `max_structures` is a real cap; if the site count exceeds it, the batch is truncated.
- The batch output writes `batch_structures.json` plus `ads_indices.json` under `output_dir`.

## Method-critical defaults
- If the screening is intended for quantitative ranking, preserve metadata and reference-state traceability needed for downstream consistent energy evaluation.
- Do not generate candidate structures without carrying forward the adsorbate indices and site provenance required for later interpretation.

## Output Contract
Return:
- site source (`sites_json_rel` or explicit label)
- generated structure path or `output_dir_rel`
- `ads_indices` metadata path(s)
- whether the batch was truncated

## References
- When the slab already carries adsorbate metadata, rely on the merged `ads_indices` returned by the tool instead of recomputing adsorbate atom indices.
- For the full screening workflow, hand off to `adsorption-screening` after the primitive site list is established.
