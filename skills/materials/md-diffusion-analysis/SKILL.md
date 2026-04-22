---
name: md-diffusion-analysis
description: Use this skill for MD execution and post-analysis when the goal is trajectory-based MSD/RDF/diffusion evidence rather than a generic VASP run log.
---

# md-diffusion-analysis

## Overview
Use this skill to prepare an MD stage, dispatch it, and summarize the resulting trajectory with MSD, RDF, and diffusion-fit artifacts. Do not use it for a generic VASP run log, one-off thermalization checks, or when the trajectory is too short to support diffusion claims.

## Quick Start
1. Prepare MD inputs with `vasp_prepare(preset="md", ...)`.
2. Make the intended ensemble, timestep, target temperature schedule, and total run length explicit through `user_incar_patch`.
3. Decide what part of the trajectory is equilibration and what part is production.
4. Dispatch with `vasp_execute_batch`.
5. Analyze the collected trajectory with `analyze_trajectory`.

## Allowed tools
- `vasp_prepare`
- `vasp_execute_batch`
- `analyze_trajectory`

## Workflow

### 1. Use the MD preset as a starter template
- Keep the default Nose-Hoover/NVT starter only when it matches the scientific question.
- Use `user_incar_patch` in the same `vasp_prepare` call to set the actual timestep, thermostat, or temperature schedule needed for the run.
- Surface the intended ensemble semantics explicitly; do not leave the reader guessing whether the run is being interpreted as NVT-like sampling, annealing, or another protocol.

### 2. Separate equilibration from production
- Report how much trajectory is being discarded as warmup before any diffusion fit.
- If no equilibration window is removed, say that explicitly instead of implying a production-only trajectory.

### 3. Keep trajectory provenance visible
- Report the exact MD control overrides, not just “ran MD”.
- Keep the output root clean so the trajectory analyzer sees one clear result directory.

### 4. Analyze before interpreting
- `analyze_trajectory` emits MSD, RDF, and any temperature/energy time-series it can recover.
- Do not claim diffusion coefficients if the fit window or sampled trajectory length is obviously inadequate.
- If diffusion is species-specific or anisotropic, set the species filter and diffusion dimension explicitly.

### 5. Report finite-time limitations
- If the MSD never reaches a clear diffusive regime, report that the run is qualitative only.
- Do not compare diffusion coefficients across runs that changed ensemble, temperature, composition, or cell size silently.

## Method-critical defaults
- MD controls are not one-size-fits-all; always surface the patched thermostat and timestep knobs when they affect interpretation.
- If diffusion is species-specific, pass the species filter explicitly at analysis time and keep `rdf_species` / `diffusion_dimension` aligned with the scientific question.
- Do not report a diffusion coefficient just because the analyzer can fit one numerically; the fit still needs a physically credible production window.
- Keep ensemble, temperature schedule, equilibration cut, and production length visible in the final summary.

## Output Contract
Return:
- MD input/output root
- key MD control overrides
- chosen ensemble interpretation and equilibration cut
- trajectory summary JSON
- MSD/RDF artifact paths
- whether a diffusion coefficient is being reported or withheld as unreliable

## References
- The canonical one-call customization path is `user_incar_patch`, not a later patching pass after preset generation.
