---
name: thermo-free-energy-and-reporting
description: Use this skill for thermodynamic and free-energy post-processing, result normalization, and concise reporting standards for catalyst comparison.
compatibility: Designed for CatMaster local tools and project-space relative-path execution.
metadata:
  catmaster-suggested-tools: ""
---

# thermo-free-energy-and-reporting

## Overview
Use this skill to convert raw electronic-structure results into comparable thermodynamic conclusions.

## Quick Start
1. Gather only validated energies and corrections.
2. Fix one reference convention before computing any comparison table.
3. Keep units and correction assumptions explicit.
4. Report both the final ranking and the assumptions that could change it.

## Suggested tools
- (none specified)

## Workflow

### 1. Gather the required inputs
- Collect clean slab, adsorbed, gas, and pathway energies as needed.
- Exclude unconverged or obviously pathological structures from the final table.

### 2. Apply one thermodynamic convention
- Use consistent reference states and correction assumptions across the compared entries.
- Do not mix adsorption-energy and free-energy conventions without labeling them.

### 3. Report comparison-ready outputs
- Produce a table with values, units, and the exact convention used.
- Keep the supporting artifact paths with the summary so the ranking is auditable.

## Method-critical defaults
- Do not mix raw electronic energies, adsorption energies, and free-energy values without labeling the convention explicitly.
- Keep reference states and correction conventions fixed within one comparison table.
- Report the exact convention used and any assumptions that could change the ranking.

## Output Contract
Return:
- final table or summary path
- value definitions and units
- reference-state convention
- any caveats that materially affect comparison

## References
- If correction details become nontrivial, load the relevant stage outputs and compute them explicitly rather than summarizing from memory.
