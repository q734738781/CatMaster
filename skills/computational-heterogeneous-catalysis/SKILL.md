---
name: computational-heterogeneous-catalysis
description: Use this skill for end-to-end heterogeneous catalysis workflows, including proposal framing, slab and adsorbate strategy, reaction/energy evaluation, and evidence standards for comparing catalyst candidates.
compatibility: Designed for CatMaster local tools and project-space relative-path execution.
metadata:
  catmaster-suggested-tools: "write_note"
---

# computational-heterogeneous-catalysis

## Overview
Use this skill to keep a heterogeneous-catalysis study coherent from catalyst selection to final comparison.

## Quick Start
1. Fix the catalyst set, reaction target, and comparison metric before creating work packages.
2. Separate the study into bulk, slab, adsorbate, screening, execute, and reporting stages.
3. Keep reference states, parameter defaults, and evidence requirements consistent across all stages.
4. Record the campaign plan early so later execution and reporting stay aligned.

## Suggested tools
- write_note

## Workflow

### 1. Define scope
- Capture catalyst family, facet policy, adsorbates or intermediates, and target elementary steps.
- Lock the ranking metric up front: adsorption energy, reaction energy, barrier, or free energy.
- State operating assumptions that matter later, especially temperature, pressure, and reference states.

### 2. Build a staged study plan
- Split work into bulk selection, slab construction, adsorbate generation, screening, refinement, and reporting.
- Keep the screening tier and the final-evidence tier separate.
- Decide what qualifies as enough evidence to stop or escalate.

### 3. Enforce consistency across stages
- Use one naming scheme for catalysts, facets, adsorbates, and run folders.
- Keep structure provenance and path evidence project-relative.
- Reuse the same clean-slab and gas references inside one comparison campaign.
- Keep one explicit k-point-density policy for the whole comparison campaign; for slab production runs, the project house default is `k*a ~= 35 Å` unless a convergence-backed reason says otherwise.

### 4. Report like a campaign, not isolated jobs
- Summarize what was compared, with which assumptions, and why the final ranking is credible.
- Report both the result table and the caveats that could change the ranking.

## Output Contract
Return a compact campaign summary with:
- compared catalysts or surfaces
- chosen metric and reference convention
- stage-by-stage artifact paths
- final ranking or next-step recommendation

## References
- Load the downstream domain skill for the active stage instead of keeping all details here.
