---
name: cp2k-pathway-calculations
description: Use this skill for source-grounded CP2K NEB and dimer path-refinement preparation through the single cp2k_prepare tool.
allowed-tools: "cp2k_prepare remote_submission remote_submission_batch get_avail_remote_task execute"
---

# cp2k-pathway-calculations

## Overview
Use this skill for CP2K pathway refinement after endpoints, images, or transition-state guesses already exist. The only bound preparation tool is `cp2k_prepare`.

## Quick Start
1. Verify image count, atom count, and atom ordering before preparation.
2. Use `cp2k_prepare(recipe="neb")` for an existing image directory, or `recipe="dimer"` for an existing transition-state guess.
3. Verify `job.inp`, `manifest.json`, and copied replica files.
4. Submit with `remote_submission(task_name="cp2k_execute")`.
5. Parse barrier or mode evidence with a focused script; do not infer barriers from preparation alone.

## Allowed tools
- `cp2k_prepare`
- `remote_submission`
- `remote_submission_batch`
- `get_avail_remote_task`
- `execute`

## Workflow

### 1. Validate pathway structures
- NEB images must have identical atom counts and symbol ordering.
- `cp2k_prepare(recipe="neb")` does not generate endpoints or interpolate images.
- `cp2k_prepare(recipe="dimer")` starts from a user-provided or previously generated transition-state guess.

### 2. Prepare controlled CP2K path stages
- For NEB, keep `band_type`, `k_spring`, `nproc_rep`, endpoint optimization, and max iterations explicit when they affect results.
- For dimer, keep optimizer, max force, max iterations, and any provided dimer vector traceable.
- Keep all pathway work in `materials_worker`; AIMD and trajectory work belong to `dynamics_worker`.

### 3. Analyze narrowly
- Use a focused script to parse image energies, barrier, endpoint relaxation, and convergence.
- Report exact files parsed and whether endpoints were fixed or optimized.

## Method-critical defaults
- Pathway preparation is not evidence of a barrier.
- Atom ordering mismatches invalidate the path.
- Project policy: do not create a separate pathway-preparation tool; use `cp2k_prepare` for CP2K conventional and pathway preparation.

## Output Contract
Return:
- image or TS-guess source path
- prepared CP2K path stage path
- `manifest.json` path
- submitted receipt/context if executed
- parser artifact path and limitations after analysis

## References
- Local source note: `references/cp2k_pathway_reference.md`
- CP2K NEB exercise: https://www.cp2k.org/exercises%3Acommon%3Aneb
- CP2K MOTION/BAND manual: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/BAND.html
- CP2K transition-state section: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/GEO_OPT/TRANSITION_STATE.html
- CP2K manual index: https://manual.cp2k.org/
