---
name: transition-state-neb
description: Use this skill for transition-state and NEB workflows, including image generation, INCAR setup, and execution/evidence checks for pathway calculations.
compatibility: Designed for CatMaster local tools and project-space relative-path execution.
metadata:
  catmaster-suggested-tools: "make_neb_geometry make_neb_incar vasp_execute_batch"
---

# transition-state-neb

## Overview
Use this skill to generate NEB image directories, prepare NEB-specific INCAR settings, and hand off a valid pathway batch for execution.

## Quick Start
1. Validate the initial and final structures before generating images.
2. Use `make_neb_geometry` to create the image directory tree.
3. Use `make_neb_incar` from a template INCAR instead of editing a relax INCAR by hand.
4. Run the resulting NEB folders through the standard VASP batch execution path.

## Suggested tools
- make_neb_geometry
- make_neb_incar
- vasp_execute_batch

## Workflow

### 1. Build the image set from a valid endpoint pair
- `make_neb_geometry` validates the endpoint pair before interpolation.
- It writes the standard image directory tree (`00`, `01`, ...) under `output_dir`.
- If `output_dir` already exists, `overwrite=true` is required to replace it.

### 2. Generate NEB INCAR from a template
- `make_neb_incar` enforces the core NEB settings; `iopt` must be one of `7`, `2`, or `1`.
- It writes the resulting INCAR plus `neb_incar_patch.json`, which is the authoritative diff from the template.
- Use `additional_overrides` only for targeted changes, not to fight the NEB core settings.

### 3. Hand off to execution as a VASP batch
- Treat the NEB image tree as a prepared VASP input set.
- Report image count, INCAR patch path, and execution status together; launch status alone is not enough.

## Method-critical defaults
- Keep endpoint preparation, image generation, and execution settings scientifically consistent across the whole pathway calculation.
- Do not treat launch success as pathway validity; evidence must include image count, INCAR patch, and outcome diagnostics.

## Output Contract
Return:
- NEB image root
- image count
- INCAR path and patch JSON path
- execution evidence path(s) if the run was dispatched

## References
- Pair this skill with `vasp-batch-execution` for dispatch and rerun handling instead of inventing a separate NEB execution path.
