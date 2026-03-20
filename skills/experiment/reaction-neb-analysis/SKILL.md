---
name: reaction-neb-analysis
description: Use this skill for end-to-end NEB pathway setup, execution, and post-analysis when a reaction coordinate needs a barrier estimate and image-profile artifacts rather than just an image tree.
license: project-local
compatibility: local
allowed-tools: "make_neb_geometry vasp_neb_prepare vasp_execute_batch analyze_neb_results"
---

# reaction-neb-analysis

## Overview
Use this skill to convert a validated endpoint pair into a dispatched NEB campaign and a barrier/profile summary.

## Quick Start
1. Start from explicitly accepted initial and final endpoint structures.
2. Generate the image tree with `make_neb_geometry`.
3. Prepare the NEB root with `vasp_neb_prepare`.
4. Dispatch with `vasp_execute_batch`.
5. Summarize the finished run with `analyze_neb_results`.

## Allowed tools
- `make_neb_geometry`
- `vasp_neb_prepare`
- `vasp_execute_batch`
- `analyze_neb_results`

## Workflow

### 1. Keep geometry and input assembly separate
- Use `make_neb_geometry` only for the interpolation/image tree.
- Use `vasp_neb_prepare` only for NEB-ready VASP support files and protected INCAR controls.

### 2. Run one pathway contract
- Keep endpoint method settings compatible with the pathway run.
- Do not mix image-count changes and INCAR changes inside the same comparison.

### 3. Require barrier evidence
- After dispatch, the workflow is not complete until `analyze_neb_results` exports the barrier summary, CSV profile, and profile plot.
- If image energies are missing or partial, report that explicitly instead of inventing a barrier.

## Method-critical defaults
- Treat endpoint validation as part of the barrier contract; a bad endpoint pair gives a bad pathway no matter how clean the dispatch is.
- NEB-critical INCAR keys should stay under the wrapper unless the task explicitly needs a controlled override.

## Output Contract
Return:
- NEB root path
- image count
- execution state path
- NEB summary JSON, CSV, and plot path

## References
- Use `transition-state-neb` when only the primitive NEB setup side is needed.
