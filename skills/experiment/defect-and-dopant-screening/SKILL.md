---
name: defect-and-dopant-screening
description: Use this skill for first-pass vacancy, substitution, and explicit interstitial candidate generation plus standardized VASP screening, without pretending to solve formation energies or charge corrections in one step.
license: project-local
compatibility: local
allowed-tools: "enumerate_unique_sites create_vacancy substitute_species insert_interstitial_at_coords supercell vasp_prepare vasp_execute_batch analyze_vasp_results"
---

# defect-and-dopant-screening

## Overview
Use this skill to generate primitive defect/dopant candidates and screen them consistently before any deeper formation-energy analysis.

## Quick Start
1. Start from one accepted bulk reference.
2. Enumerate symmetry-inequivalent sites first and keep the group ledger.
3. Build the intended vacancy/substitution/interstitial candidate set.
4. Expand to a supercell only when the defect model actually requires it.
5. Run one standardized VASP screen and summarize with `analyze_vasp_results`.

## Allowed tools
- `enumerate_unique_sites`
- `create_vacancy`
- `substitute_species`
- `insert_interstitial_at_coords`
- `supercell`
- `vasp_prepare`
- `vasp_execute_batch`
- `analyze_vasp_results`

## Workflow

### 1. Use group IDs as the defect ledger
- `enumerate_unique_sites` is the bridge between a clean bulk reference and a reproducible defect screen.
- Keep the emitted `group_id` mapping with the generated candidates.

### 2. Generate only primitive defect candidates
- `create_vacancy` and `substitute_species` operate on one explicit site or one representative site per group.
- `insert_interstitial_at_coords` only inserts at user-provided coordinates; it does not search candidate interstitial sites for you.

### 3. Add cell size deliberately
- Use `supercell` when interaction range or concentration control requires it.
- Do not pretend the primitive candidate generator already solved finite-size effects.

### 4. Run one screening stage
- Prepare and dispatch the retained candidate set under one method contract.
- Use `analyze_vasp_results` to separate geometry/execution failure from physically interesting candidates.

## Method-critical defaults
- Do not report formation energies, charge corrections, or chemical-potential conclusions from this skill alone.
- Keep the bulk reference, defect supercell size, spin treatment, and `DFT+U` policy explicit in the screening summary.
- If interstitial coordinates are heuristic, say so explicitly.

## Output Contract
Return:
- unique-site JSON path
- generated candidate root
- any supercell expansion path
- VASP analysis summary path

## References
- This skill builds candidate structures; deeper charged-defect methodology should be treated as a separate workflow.
