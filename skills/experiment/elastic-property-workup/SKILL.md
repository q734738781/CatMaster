---
name: elastic-property-workup
description: Use this skill for finite-strain elastic workflows when the immediate goal is to generate a controlled strain set, run the corresponding VASP calculations, and collect the stress/energy outputs needed for later fitting.
---

# elastic-property-workup

## Overview
Use this skill to build a finite-strain screening set and collect the calculation evidence needed for later elastic fitting. Do not use it when the real task is elastic-tensor fitting, uncertainty quantification, or equation-of-state work rather than controlled strain-job collection.

## Quick Start
1. Start from one accepted relaxed bulk structure.
2. Decide up front whether the strain members will be evaluated as fixed-cell stress jobs or with an explicitly stated relaxation policy.
3. Generate the strain family explicitly with `generate_strained_structures`.
4. Prepare one consistent VASP stage across the whole strain set.
5. Dispatch as a batch and summarize with `analyze_vasp_results`.

## Allowed tools
- `generate_strained_structures`
- `vasp_prepare`
- `vasp_execute_batch`
- `analyze_vasp_results`

## Workflow

### 1. Define the strain family explicitly
- Use explicit deformation matrices when the study has a published protocol.
- Use mode/value grids only when the intended strain amplitudes are already justified.
- Record the actual strain amplitudes and matrix convention; “small strain” is not a reproducible input.

### 2. Choose the evaluation semantics before execution
- For ordinary finite-strain elastic screening, default to strained-structure single-point/stress collection rather than quietly relaxing each strained cell.
- If the scientific protocol requires internal-coordinate relaxation or another non-default treatment, state that explicitly and keep it uniform across the whole set.
- Do not mix fixed-cell stress jobs and relax-style jobs inside one comparison set without separating the results.

### 3. Keep execution uniform
- Apply one method contract across the whole strain set.
- Keep the same k-point density, ENCUT, spin treatment, `DFT+U`, and dispersion policy across every strain member.

### 4. Collect, do not overclaim
- `analyze_vasp_results` separates failed strain points from usable ones.
- This skill stops at collected energies/forces/stresses; it does not fit elastic constants itself.
- Keep a ledger of which strained member produced usable stress/energy output and which failed.
- When a single total-energy scalar is needed from `analyze_vasp_results`, use `E0` by default.

## Method-critical defaults
- Keep the unstrained reference and every strained member on the same magnetic and electronic-structure footing.
- Report the actual strain matrices used; “small strain” is not specific enough for later fitting or reproduction.
- Keep the stress-vs-relax semantics explicit in the final handoff; elastic fitting cannot be audited without that distinction.
- Report the stress/energy units and the reference strain convention together.

## Output Contract
Return:
- strained structure batch root
- execution root
- VASP analysis summary path
- explicit strain-matrix ledger
- whether the stage used fixed-cell stress evaluation or an explicitly relaxed protocol

## References
- Treat tensor fitting and uncertainty analysis as a downstream analysis stage, not as an implicit promise of this skill.
