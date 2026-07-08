---
name: materials-discovery-and-bulk-selection
description: Use this skill for materials discovery and bulk structure selection before slab construction, including database query strategy, candidate filtering, and export readiness.
---

# materials-discovery-and-bulk-selection

## Overview
Use this skill to turn an open-ended catalyst search into a shortlist of downloaded bulk structures with traceable Materials Project provenance.

ExperimentSpecialist may use the MP lookup tools directly for lightweight database retrieval. When this skill is invoked inside a delegated materials workflow, perform the lookup here and report precise API-key, client-package, criteria, or field blockers rather than saying materials discovery is generally unavailable.

## Quick Start
1. Define search criteria and requested fields before querying Materials Project.
2. Choose a non-empty `criteria`, non-empty `fields`, and an `output_csv_rel` ledger path before calling `mp_search_materials`.
3. Use `mp_search_materials` to write a CSV candidate table, not just a chat summary.
4. Prune the candidate table before downloading structures.
5. Use `mp_download_structure` only for the shortlisted `mp_id` values.

## Allowed tools
- `mp_search_materials`
- `mp_download_structure`

## Workflow

### 1. Search with explicit criteria
- `mp_search_materials` fails on empty criteria or empty fields.
- The tool writes the full result table to `output_csv_rel` and returns preview rows for quick inspection.
- Keep the requested fields aligned with the intended downstream ranking so the CSV is directly usable.
- Common `criteria` keys include `material_ids`, `elements`, `chemsys`, `formula`, `is_stable`, `energy_above_hull`, `formation_energy`, `band_gap`, `crystal_system`, `spacegroup_number`, `spacegroup_symbol`, `num_sites`, `density`, `volume`, and `num_elements`.
- Range criteria should be passed as `[min, max]` pairs.
- Common output fields include `material_id`, `formula_pretty`, `energy_above_hull`, `formation_energy_per_atom`, `band_gap`, `crystal_system`, `spacegroup_number`, `spacegroup_symbol`, `nsites`, `density`, and `volume`.

### 2. Prune before download
- Use the CSV output as the candidate ledger instead of retyping IDs from memory.
- Keep the filter logic explicit so later bulk or slab comparisons remain auditable.

### 3. Download only retained candidates
- `mp_download_structure` writes conventional standard structures under `output_dir`.
- It supports partial success: some `mp_id` values can fail while others download cleanly.
- Use the returned `results` and `errors` rather than assuming every requested ID resolved.

### 4. Hand off the shortlist
- Once the bulk shortlist is stable, hand it to `bulk-relax-and-reference` for relaxation and reference-state cleanup.

## Method-critical defaults
- Keep the search criteria, requested fields, and shortlist filter fixed while you compare candidates.
- Do not treat a candidate table built with a different field set as interchangeable with the current shortlist.

## Output Contract
Return:
- search CSV path
- retained `mp_id` list
- downloaded structure path(s)
- any partial-download failures that still need follow-up

## References
- If a task needs advanced MP field selection, inspect the tool schema before broadening the search request.
- Use `bulk-relax-and-reference` when the discovery phase is finished and you need a real reference structure.
