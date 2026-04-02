---
name: reaction-neb-analysis
description: Use this skill for end-to-end NEB pathway setup, execution, and post-analysis when a reaction coordinate needs a barrier estimate and image-profile artifacts rather than just an image tree.
license: project-local
compatibility: local
allowed-tools: "estimate_neb_image_count make_neb_geometry vasp_neb_prepare vasp_execute_batch analyze_vasp_neb_results"
---

# reaction-neb-analysis

## Overview
Use this skill to convert a validated endpoint pair into a dispatched NEB campaign and a barrier/profile summary. Do not use it when the task is really about dimer-mode preparation, TS refinement heuristics, or custom NEB/dimer branching; those belong to `transition-state-neb`.

## Quick Start
1. Start from explicitly accepted initial and final endpoint structures.
2. Confirm that the two endpoints preserve atom order and represent one local elementary step rather than a long migration.
3. Estimate the intermediate-image count before interpolation instead of choosing it arbitrarily, preferably with `estimate_neb_image_count`.
4. Generate the image tree with `make_neb_geometry`.
5. Prepare the NEB root with `vasp_neb_prepare`.
6. Dispatch with `vasp_execute_batch`.
7. Summarize the finished run with `analyze_vasp_neb_results`.
8. State the barrier convention explicitly, normally forward barrier relative to the initial endpoint.

## Allowed tools
- `estimate_neb_image_count`
- `make_neb_geometry`
- `vasp_neb_prepare`
- `vasp_execute_batch`
- `analyze_vasp_neb_results`

## Workflow

### 1. Keep geometry and input assembly separate
- Use `make_neb_geometry` only for the interpolation/image tree.
- Prefer `estimate_neb_image_count` before setting `n_images` so the count follows the actual periodic endpoint displacement rather than intuition.
- Use `vasp_neb_prepare` only for NEB-ready VASP support files and protected INCAR controls.
- The initial and final endpoints must keep the same atom ordering; fix the structures first instead of interpolating between reordered atoms.
- A single NEB should describe one local elementary event. Do not use one band for long-distance diffusion across several equivalent sites.
- If the apparent reaction is really a long migration, remodel the final state onto the nearest equivalent site and compute that primitive hop instead.
- When no prior image-count estimate exists, use the practical first guess
  `n_images ≈ ceil(sqrt(sum_i ||Δr_i||^2) / 0.8 Å)` for the intermediate-image count, after the endpoint matching/PBC convention is fixed.
- Under periodic boundary conditions, compute `Δr_i` with the same minimum-image convention (`mic=true`) that will be used for interpolation. Do not infer the path length from wrapped endpoint coordinates by eye.
- For routine local events, prefer about 4-8 intermediate images unless the displacement heuristic points elsewhere.
- If that estimate exceeds about 8 intermediate images, treat it as a warning that the chosen endpoints may encode an unrelated long-range path rather than one local barrier event.
- If the periodic root-sum-squared displacement exceeds about 6 Å, treat that as a warning that the band is probably too long and should be remodeled into shorter primitive hops.

### 2. Run one pathway contract
- Keep endpoint method settings compatible with the pathway run.
- Do not mix image-count changes and INCAR changes inside the same comparison.
- Keep the chosen image count and climbing-image policy explicit in the run record.
- Prefer a two-stage NEB contract by default: first run plain `NEB` with climbing image disabled for coarse convergence, then rerun from the coarse-converged images with `CI-NEB` enabled for barrier refinement.
- If the workflow intentionally skips the coarse plain-`NEB` stage, say so explicitly instead of implying the standard two-stage pattern was followed.

### 3. Require barrier evidence
- After dispatch, the workflow is not complete until `analyze_vasp_neb_results` exports the barrier summary, CSV profile, and profile plot.
- If image energies are missing or partial, report that explicitly instead of inventing a barrier.
- If the profile shows wave-like repeated rises and falls instead of one dominant saddle region, treat that as evidence that the chosen path may pass through hidden metastable intermediates.
- Use those suspect local minima images as clues for rebuilding the problem into shorter primitive hops instead of insisting on one long NEB.

### 4. Keep the energy reference explicit
- State whether the reported barrier is forward, reverse, or both.
- Unless the task says otherwise, treat the forward barrier as referenced to the initial endpoint energy and the reverse barrier as referenced to the final endpoint.
- If the endpoints were evaluated under a different footing than the NEB images, say so explicitly instead of implying a fully self-consistent barrier contract.

## Method-critical defaults
- Treat endpoint validation as part of the barrier contract; a bad endpoint pair gives a bad pathway no matter how clean the dispatch is.
- NEB-critical INCAR keys should stay under the wrapper unless the task explicitly needs a controlled override.
- Default to `plain-NEB -> CI-NEB` rather than starting directly from `CI-NEB`; direct climbing-image starts need an explicit reason because they are often harder to converge robustly.
- If the image count is not obvious, prefer the displacement-norm heuristic above over a blind fixed default.
- If the heuristic asks for more than about 8 intermediate images, reconsider whether the endpoints should be decomposed into a shorter primitive step before running NEB.
- If the converged or partially converged profile shows a wave-like multi-bump pattern, treat that as a strong hint that undiscovered intermediate states may exist and that the path should be split.
- Do not present a barrier as final if image energies are partial, the climbing-image logic is unclear, or the image set never produced a credible saddle region.
- Keep the barrier convention explicit in the final answer; “the barrier” is ambiguous otherwise.

## Output Contract
Return:
- NEB root path
- image count
- execution state path
- NEB summary JSON, CSV, and plot path
- stated barrier convention (forward/reverse reference)

## References
- Use `transition-state-neb` when only the primitive NEB setup side is needed.
