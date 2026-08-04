---
name: mlff-path-optimization
description: Use this skill for managed MLFF NEB optimization after a complete locally interpolated fixed-image path has been validated.
license: project-local
allowed-tools: "ls read_file write_file edit_file execute get_avail_remote_task get_remote_task_spec remote_submission remote_submission_batch"
---

# mlff-path-optimization

## Overview

Execute one validated fixed-image MLFF path per stage without remote interpolation.

## Quick Start

1. Finish endpoint validation, atom remapping, image-count selection, interpolation, and overlap QC locally.
2. Copy one contiguous `00.vasp` through `NN.vasp` tree into `stage/input/path/`.
3. Query the selected backend directly with `get_remote_task_spec(task_name="mlff_neb", template_overrides={"backend": "<enabled-backend>"}, detail="full")`, then use its resolved defaults and concrete convergence schema.
4. Submit one path with `remote_submission`; submit two or more independent same-config paths with one `remote_submission_batch`.
5. Inspect the batch summary, per-path summary, energy CSV/profile, and final images. Use receipt recovery only after a returned failure.

## Allowed tools

- `ls`
- `read_file`
- `write_file`
- `edit_file`
- `execute`
- `get_avail_remote_task`
- `get_remote_task_spec`
- `remote_submission`
- `remote_submission_batch`

## Workflow

### 1. Require a complete local path

- Endpoint-only input is invalid. The tree must include at least one intermediate image.
- Numbering starts at `00`, is contiguous and consistently zero-padded, and every image has identical atom order, cell, PBC, and constraints.
- Rework any overlap/short-distance warning before submission; remote launch cannot repair interpolation.

### 2. Keep one path per stage

- The only canonical tree is `stage/input/path/*.vasp`.
- Several independent paths become several first-level stages under a batch root. Do not put multiple path directories in one stage.
- Remote AutoNEB insertion is intentionally unsupported; interpolation stays in local preparation.

### 3. Choose the optimization episode

- Start with fixed-image plain NEB. Enable climbing only as an explicit refinement decision when the band already localizes the saddle sufficiently.
- Keep coarse and climbing-image refinement in separate stage copies if their results must be compared or audited.

### 4. Validate collected evidence

- Check convergence, maximum projected NEB force, barrier, endpoint energy difference, highest-energy image, profile shape, and per-task errors.
- Preserve final image files and use pathway analysis before treating the highest image as a transition state.

## Method-critical defaults

- MACE is the registered default with `float64`; every deployment-enabled backend returned by `get_remote_task_spec` uses the same default `fmax=0.05 eV/Angstrom`, 300-step plain-mode contract with climbing disabled.
- Keep model/head/dispersion/precision fixed across all images and comparison paths.
- A climbing-image run is refinement, not the default first rough optimization.

## Output Contract

Return:

- local image-tree/QC provenance, stage path, backend/model, fmax, steps, and climb choice;
- `work_dir_rel`;
- `output/batch_summary.json`, per-path `summary.json`, energy CSV/profile, and final-image paths;
- convergence status and the required downstream barrier/TS validation.

Keep receipt/context identifiers and platform details in runtime records unless failure recovery needs them; provide them whenever the user explicitly asks to inspect, compare, record, or report them.

## References

- MACE path-specific notes: `references/mace.md`
- Use `neb-prepare` before this skill and `neb-analysis` after collection.
