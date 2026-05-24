---
name: bulk-relax-and-reference
description: Use this skill for bulk-reference preparation and analysis when a workflow needs a relaxed bulk baseline, symmetry-inequivalent site ledger, and optional band/DOS follow-up from one consistent starting structure.
---

# bulk-relax-and-reference

## Overview
Use this skill to turn a bulk structure into a traceable relaxed reference before any slab, defect, or band/DOS branch.

## Quick Start
1. Start from one explicit bulk structure file.
2. Run `enumerate_unique_sites` if later defect or dopant work needs group IDs.
3. Prepare a canonical bulk relax with `vasp_prepare`.
4. Dispatch the prepared stage with `remote_submission` or `remote_submission_batch` using `task_name="vasp_execute"`, then summarize with `analyze_vasp_results`.
5. Only after the relaxed bulk is accepted should you branch into `generate_kpath` or downstream surface/defect workflows.

## Allowed tools
- `enumerate_unique_sites`
- `vasp_prepare`
- `remote_submission`
- `remote_submission_batch`
- `analyze_vasp_results`
- `generate_kpath`

## Workflow

### 1. Lock the bulk baseline first
- Keep one clean bulk reference per composition and magnetic assumption.
- If later vacancy or substitution work is likely, emit the unique-site JSON before any supercell or defect step.

### 2. Relax under one explicit contract
- Use `vasp_prepare(preset="relax", regime="bulk", ...)`.
- Only set `relax_cell=true` when the bulk reference itself is meant to be variable-cell.
- Carry the same spin, `DFT+U`, dispersion, and k-point policy into every comparison built from this reference.

### 3. Separate execution from acceptance
- Submit the prepared root after it matches the VASP remote stage layout.
- Accept the relaxed bulk only after `analyze_vasp_results` confirms convergence and reports the final structure path.
- When the summary includes total energies, use `E0` as the default comparison energy.

### 4. Branch cleanly after acceptance
- For band structures, generate a line-mode `KPOINTS` recommendation with `generate_kpath`.
- For surfaces, defects, or phonons, hand off the accepted bulk structure rather than the raw starting file.

## Method-critical defaults
- Do not compare surface, defect, or electronic-structure branches that start from different unconverged bulk baselines.
- If the workflow is ranking-sensitive, surface the exact magnetic and `DFT+U` assumptions in the handoff.
- Keep the bulk k-point density explicit; do not silently copy slab defaults backward into bulk work.

## Output Contract
Return:
- accepted relaxed bulk result path
- unique-site JSON path when generated
- convergence summary path
- any generated band-path artifact path

## References
- Hand off to `surface-and-termination-screening`, `defect-and-dopant-screening`, or `band-and-dos-analysis` instead of mixing those branches into the same run root.
