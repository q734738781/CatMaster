---
name: neb-analysis
description: "Use this skill for NEB post-analysis: barrier extraction, profile interpretation, endpoint-energy pitfalls, and common QC checks after a pathway run finishes."
license: project-local
compatibility: local
allowed-tools: "analyze_vasp_neb_results"
---

# neb-analysis

## Overview
Use this skill after a NEB run has been collected and the task is to extract or interpret the barrier. It focuses on common post-analysis checks, missing endpoint-energy pitfalls, and how to read suspicious profiles. For setup, use `neb-prepare`. For the run protocol itself, use `neb-calculation`.

## Quick Start
1. Run `analyze_vasp_neb_results` on the collected NEB result root.
2. If image energies are incomplete, stop and diagnose the missing images instead of inferring a barrier.
3. Keep the barrier convention explicit: forward, reverse, or both.
4. If the profile is wave-like rather than single-saddle, look for hidden intermediates rather than forcing one long pathway interpretation.

## Allowed tools
- `analyze_vasp_neb_results`

## Workflow

### 1. Require a complete energy profile
- The workflow is not complete until `analyze_vasp_neb_results` exports the barrier summary, CSV profile, and plot.
- If image energies are missing or partial, report that explicitly instead of inventing a barrier.
- Do not collapse a partial profile into a single barrier number.

### 2. Handle endpoint `OUTCAR` correctly
- VASP NEB endpoint images do not generate fresh endpoint energies for barrier analysis.
- If `analyze_vasp_neb_results` reports missing `OUTCAR` energy for image `00` or the last image, copy the original endpoint relax `OUTCAR` files into those endpoint image directories and rerun the analysis.
- Treat that copy as a normal post-processing repair step, not as a scientific change to the pathway.

### 3. Keep the energy reference explicit
- State whether the reported barrier is forward, reverse, or both.
- Unless the task says otherwise, take the forward barrier relative to the initial endpoint and the reverse barrier relative to the final endpoint.
- If endpoint energies come from copied relax `OUTCAR` files while the interior images come from the NEB run itself, say that plainly.

### 4. Interpret suspicious profiles as modeling clues
- If the profile shows wave-like repeated rises and falls rather than one localized saddle region, suspect hidden metastable intermediates.
- Use those local minima as hints for splitting the path into shorter primitive hops.
- If the original setup needed too many images or had a periodic displacement above about 6 Å, connect that observation to the possibility that the path was overextended from the start.

### 5. Keep convergence claims narrow
- A collected run is not automatically a trustworthy barrier.
- If the climbing-image stage was skipped or unclear, say so.
- If the profile is incomplete, oscillatory, or based on a questionable endpoint contract, report those QC limitations directly.

## Method-critical defaults
- Never report a final barrier from an incomplete image-energy profile.
- Endpoint validation and endpoint-energy provenance are part of the barrier contract.
- Wave-like multi-bump profiles are often a modeling warning, not merely an ugly plot.
- Keep the barrier convention explicit in the final answer.

## Output Contract
Return:
- result root
- summary JSON, CSV, and plot path
- stated barrier convention
- any QC flags or reasons the barrier should be treated as provisional

## References
- Use `neb-prepare` if the pathway itself needs to be rebuilt.
- Use `neb-calculation` if the next step is a rerun or a refinement stage rather than interpretation.
