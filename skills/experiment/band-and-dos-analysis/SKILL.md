---
name: band-and-dos-analysis
description: Use this skill for post-relax bulk electronic-structure workflows when the goal is a clean DOS or band-path-ready calculation sequence with explicit KPOINTS provenance and result summaries.
license: project-local
compatibility: local
allowed-tools: "generate_kpath vasp_prepare vasp_band_prepare vasp_execute_batch analyze_vasp_results render_structure_views"
---

# band-and-dos-analysis

## Overview
Use this skill to build a controlled relax-to-static-to-band/DOS sequence instead of improvising electronic-structure jobs from an unconverged structure.

## Quick Start
1. Start from one accepted relaxed bulk structure.
2. For DOS, prepare a DOS-oriented static stage with `vasp_prepare(preset="dos", ...)`.
3. For bands, generate a recommended line-mode `KPOINTS` with `generate_kpath` and assemble the band job with `vasp_band_prepare`.
4. Dispatch the resulting VASP jobs with `vasp_execute_batch`.
5. Use `analyze_vasp_results` to summarize convergence and parsed bandgap evidence.

## Allowed tools
- `generate_kpath`
- `vasp_prepare`
- `vasp_band_prepare`
- `vasp_execute_batch`
- `analyze_vasp_results`
- `render_structure_views`

## Workflow

### 1. Accept the relaxed bulk first
- Do not start DOS or band work from a bulk structure that has not passed the base convergence screen.

### 2. Split DOS and band setup cleanly
- DOS should be prepared as a dedicated static-like stage, optionally reusing a converged `CHGCAR`.
- Band-path generation is a separate primitive: `generate_kpath` proposes the line-mode `KPOINTS`, and `vasp_band_prepare` owns the dedicated band input deck.

### 3. Keep provenance visible
- Report the exact `KPOINTS` artifact used for the band job.
- Use `render_structure_views` when the bulk cell standardization or orientation is visually ambiguous.

## Method-critical defaults
- Do not compare DOS or bandgaps across jobs that changed spin, `DFT+U`, smearing, or k-point density silently.
- Surface any `user_incar_patch` changes that alter DOS/band interpretation.
- For DOS reuse with `CHGCAR`, report the charge-density source explicitly.
- For line-mode band runs, keep the generated `KPOINTS` artifact and any reused `CHGCAR` source explicit.

## Output Contract
Return:
- DOS or band job root
- generated `KPOINTS` path when relevant
- VASP analysis summary path
- any parsed bandgap value or convergence caveat

## References
- Use `bulk-relax-and-reference` to establish the accepted bulk baseline before this skill.
