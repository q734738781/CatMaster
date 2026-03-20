---
name: md-diffusion-analysis
description: Use this skill for MD execution and post-analysis when the goal is trajectory-based MSD/RDF/diffusion evidence rather than a generic VASP run log.
license: project-local
compatibility: local
allowed-tools: "vasp_prepare vasp_execute_batch analyze_trajectory"
---

# md-diffusion-analysis

## Overview
Use this skill to prepare an MD stage, dispatch it, and summarize the resulting trajectory with MSD, RDF, and diffusion-fit artifacts.

## Quick Start
1. Prepare MD inputs with `vasp_prepare(preset="md", ...)`.
2. Override system-specific control knobs through `user_incar_patch` in the same call.
3. Dispatch with `vasp_execute_batch`.
4. Analyze the collected trajectory with `analyze_trajectory`.

## Allowed tools
- `vasp_prepare`
- `vasp_execute_batch`
- `analyze_trajectory`

## Workflow

### 1. Use the MD preset as a starter template
- Keep the default Nose-Hoover/NVT starter only when it matches the scientific question.
- Use `user_incar_patch` in the same `vasp_prepare` call to set the actual timestep, thermostat, or temperature schedule needed for the run.

### 2. Keep trajectory provenance visible
- Report the exact MD control overrides, not just “ran MD”.
- Keep the output root clean so the trajectory analyzer sees one clear result directory.

### 3. Analyze before interpreting
- `analyze_trajectory` emits MSD, RDF, and any temperature/energy time-series it can recover.
- Do not claim diffusion coefficients if the fit window or sampled trajectory length is obviously inadequate.

## Method-critical defaults
- MD controls are not one-size-fits-all; always surface the patched thermostat and timestep knobs when they affect interpretation.
- If diffusion is species-specific, pass the species filter explicitly at analysis time and keep `rdf_species` / `diffusion_dimension` aligned with the scientific question.

## Output Contract
Return:
- MD input/output root
- key MD control overrides
- trajectory summary JSON
- MSD/RDF artifact paths

## References
- The canonical one-call customization path is `user_incar_patch`, not a later patching pass after preset generation.
