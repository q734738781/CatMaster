---
name: elastic-property-workup
description: Use this skill for finite-strain elastic workflows when the immediate goal is to generate a controlled strain set, run the corresponding VASP calculations, and collect the stress/energy outputs needed for later fitting.
license: project-local
compatibility: local
allowed-tools: "generate_strained_structures vasp_prepare vasp_execute_batch analyze_vasp_results"
---

# elastic-property-workup

## Overview
Use this skill to build a finite-strain screening set and collect the calculation evidence needed for later elastic fitting.

## Quick Start
1. Start from one accepted relaxed bulk structure.
2. Generate the strain family explicitly with `generate_strained_structures`.
3. Prepare one consistent VASP stage across the whole strain set.
4. Dispatch as a batch and summarize with `analyze_vasp_results`.

## Allowed tools
- `generate_strained_structures`
- `vasp_prepare`
- `vasp_execute_batch`
- `analyze_vasp_results`

## Workflow

### 1. Define the strain family explicitly
- Use explicit deformation matrices when the study has a published protocol.
- Use mode/value grids only when the intended strain amplitudes are already justified.

### 2. Keep execution uniform
- Apply one method contract across the whole strain set.
- Do not mix relax/static semantics across different strain members without stating why.

### 3. Collect, do not overclaim
- `analyze_vasp_results` separates failed strain points from usable ones.
- This skill stops at collected energies/forces/stresses; it does not fit elastic constants itself.

## Method-critical defaults
- Keep the unstrained reference and every strained member on the same magnetic and electronic-structure footing.
- Report the actual strain matrices used; “small strain” is not specific enough for later fitting or reproduction.

## Output Contract
Return:
- strained structure batch root
- execution root
- VASP analysis summary path
- explicit strain-matrix ledger

## References
- Treat tensor fitting and uncertainty analysis as a downstream analysis stage, not as an implicit promise of this skill.
