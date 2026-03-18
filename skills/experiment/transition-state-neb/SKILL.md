---
name: transition-state-neb
description: Use this skill for transition-state and NEB workflows, including image generation, NEB VASP input setup, and execution/evidence checks for pathway calculations.
license: project-local
compatibility: local
allowed-tools: "make_neb_geometry vasp_neb_prepare vasp_execute_batch"
---

# transition-state-neb

## Overview
Use this skill to generate NEB image directories, prepare a NEB-ready VASP input root, and hand off a valid pathway batch for execution.

## Quick Start
1. Validate the initial and final structures before generating images.
2. Use `make_neb_geometry` to create the image directory tree.
3. Use `vasp_neb_prepare` to assemble the NEB root with canonical support files and NEB-critical INCAR settings.
4. Run the resulting NEB folders through the standard VASP batch execution path.

## Allowed tools
- make_neb_geometry
- vasp_neb_prepare
- vasp_execute_batch

## Workflow

### 1. Build the image set from a valid endpoint pair
- `make_neb_geometry` validates the endpoint pair before interpolation.
- It writes the standard image directory tree (`00`, `01`, ...) under `output_dir`.
- If `output_dir` already exists, `overwrite=true` is required to replace it.

### 2. Prepare the NEB VASP root
- `vasp_neb_prepare` keeps geometry as a separate primitive: it can either consume an endpoint pair or reuse an existing image tree.
- It enforces the core NEB settings; `iopt` must be one of `7`, `2`, or `1`.
- It writes the resulting support files plus `neb_incar_patch.json`, which is the authoritative diff from the canonical support-file baseline.
- In `patch_policy="safe"`, NEB-critical keys remain protected; use `force` only for intentional overrides.

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
