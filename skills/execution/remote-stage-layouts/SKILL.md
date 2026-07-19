---
name: remote-stage-layouts
description: Use this skill before remote_submission or remote_submission_batch to build and verify canonical stage directories for registered DPDispatcher tasks, including deterministic MLFF SP/relax, MD, and NEB input layouts.
license: project-local
allowed-tools: "ls read_file write_file edit_file execute get_avail_remote_task get_remote_task_spec get_avail_resources remote_submission remote_submission_batch"
---

# remote-stage-layouts

## Overview

Build a clean, task-specific stage copy before low-level remote submission. One stage is the working directory of one DPDispatcher Task.

## Quick Start

1. Call `get_avail_remote_task` to select a task. For a non-default MLFF backend, make the first schema query concrete, for example `get_remote_task_spec(task_name="mlff_sp", template_overrides={"backend": "mattersim"}, detail="full")`; use that response's `resolved_template_defaults`, accepted paths, and concrete schema.
2. Copy the required inputs into a new workspace-relative stage that matches the layout below; do not rearrange the source tree in place.
3. Verify the prepared tree with `ls` or a focused `find` command.
4. Use `remote_submission` for one stage.
5. For multiple independent stages with the same task and configuration, place them under one parent and call `remote_submission_batch` once.

## Allowed tools

- `ls`
- `read_file`
- `write_file`
- `edit_file`
- `execute`
- `get_avail_remote_task`
- `get_remote_task_spec`
- `get_avail_resources`
- `remote_submission`
- `remote_submission_batch`

## Workflow

### 1. Keep submission boundaries literal

- `remote_submission(work_dir=...)` treats `work_dir` itself as one stage and creates one DPDispatcher Task.
- `remote_submission_batch(work_dir=...)` treats every first-level child directory as one complete stage and creates one Task per child.
- Batch discovery is not recursive. Nested directories inside a stage are scientific inputs only when the selected layout explicitly requires them.
- One batch call applies the same `task_name`, `template_overrides`, and `submission_config` to every child. Use separate calls when those values differ.
- A stage may contain several input structures only when its task layout permits internal sequential processing. Multiple files do not by themselves require `remote_submission_batch`.

```text
single stage -> remote_submission
stage/
  task inputs

multiple independent stages -> remote_submission_batch
batch_root/
  stage_000/
    task inputs
  stage_001/
    task inputs
```

### 2. Build a clean submission copy

- Copy required inputs into a fresh stage; preserve the original structures, restart files, endpoint trees, and provenance.
- Do not submit a raw project root merely because the runner can recursively discover files.
- Keep prior outputs, nested batches, receipts, and unrelated calculation directories out of the new stage.
- Use stable, unique case names when flattening SP/relax inputs. Preserve a source-to-stage mapping in the surrounding workflow when files are renamed.
- Do not edit copied `task_script/` files or add `sitecustomize.py`. Built-in scripts are staged by the submission tool, and registered controls belong in catalog-declared overrides/configuration.

### 3. Use the canonical registered-task layout

#### vasp_execute

Prepare one complete VASP calculation per stage:

```text
stage/
  INCAR
  POTCAR
  POSCAR
  KPOINTS
  optional required VASP inputs
```

For several calculations, put one complete VASP stage in each first-level batch child.

#### vasp_execute_neb

Prepare the complete VASP NEB/dimer root locally before submission:

```text
stage/
  INCAR
  POTCAR
  KPOINTS
  00/POSCAR
  01/POSCAR
  ...
  NN/POSCAR
```

Use `vasp_neb_prepare` or an equivalent checked preparation path. Do not submit endpoint-only input and do not ask the remote boot script to interpolate. Reject atom-order/cell mismatches and inspect any `short_distance_count > 0` warning before submission.

#### cp2k_execute

```text
stage/
  job.inp
  manifest.json
  optional referenced structure, restart, basis, potential, or include files
```

Scientific recipe selection belongs in `job.inp`. Use one prepared CP2K stage per first-level batch child.

#### lammps_execute

```text
stage/
  in.lammps
  manifest.json
  system.data or referenced restart file
  optional potential files
```

#### orca_execute

```text
stage/
  job.inp
  optional referenced files
```

#### xtb_run

Place the molecular input named by `template_overrides.input` directly in the stage; the default is `input.xyz`.

#### crest_run

Place the molecular input named by `template_overrides.input` directly in the stage; the default is `input.xyz`. Add the explicitly referenced constraint file for constrained runs.

#### mlff_sp and mlff_relax

Use flat, uniquely named structure files directly under `input/`:

```text
stage/
  input/
    case_a.vasp
    case_b.vasp
```

One stage may contain one or many compatible structures. The selected backend is initialized once and files are processed sequentially. Do not use subdirectories or recursive project-tree discovery. Put MACE model artifacts under optional `models/` and refer to them through `backend_config.checkpoint_artifact`. UMA per-item task/charge/spin belongs in nested `backend_config.items`, keyed by the exact filename relative to `input/`; do not create a second metadata file.

For short, similarly sized relaxations, roughly 30-50 structures per stage is an empirical starting point. Use smaller groups for large or heterogeneous structures and substantially larger groups for cheap SP screening. The agent may create these stage copies directly; no generic automatic partitioner is required.

#### mlff_md

Prepare exactly one independent trajectory source per stage:

```text
stage/
  input/
    start.vasp or start.xyz or restart.traj
```

One stage means one trajectory lineage. Set grouped dynamics, thermostat, barostat, and output controls through `template_overrides.task_config`, not a duplicate params file. Use one first-level batch child per independent replica. Submit continuation segments sequentially after the preceding `restart.traj` is available.

#### mlff_neb

Prepare exactly one locally interpolated and checked path per stage:

```text
stage/
  input/
    path/
      00.vasp
      01.vasp
      ...
      NN.vasp
```

The numbered files must be contiguous from `00`, contain both endpoints and at least one intermediate image, and have identical atom count/order, cell, PBC, and constraints. Build them locally with `make_neb_geometry` after endpoint validation/remapping. Endpoint-only stages are invalid.

Do not place several path directories under one stage. Use one stage per path and `remote_submission_batch` for independent paths. Only fixed-image `plain` mode is accepted. Remote AutoNEB insertion is not part of this contract.

#### mace_train

```text
training_stage/
  dataset/
  params/train_params.json
```

#### mace_eval

```text
evaluation_stage/
  dataset/
  params/eval_params.json
```

### 4. Verify before and after submission

- Before submission, verify the exact first-level children and task-required files. Do not assume a copied root has the intended depth.
- After submission, inspect the stage-local `status.json`, `stdout.log`, `stderr.log`, and declared task outputs before retrying.
- On failure, use receipt/context fields to determine whether remote work may still be live. Retry only failed stages from fresh stage copies.

## Method-critical defaults

- Registered tasks use their task-bound resource card. Do not select machine/resource internals through agent-authored task configuration.
- Use only catalog-declared registered-task overrides. Keep scientific/backend settings identical across children of one batch call.
- Treat SP/relax structures-per-stage as an approximate throughput choice, not a scientific parameter or a probe requirement.
- Treat one MD trajectory and one NEB path per stage as hard canonical layout rules for new MLFF stages.
- Perform NEB endpoint validation and interpolation locally. A successful remote launch cannot repair a malformed image tree.

## Output Contract

Before dispatch, retain or report:

- selected `task_name` and the validated layout above;
- prepared stage path, or batch root plus first-level stage count;
- focused verification of required files and forbidden extra nesting;
- the shared configuration applied to the stage or batch;
- for NEB, the local image-tree/QC artifact;
- for MD, the single start/restart file and trajectory lineage.

After dispatch, retain the returned `work_dir_rel`, receipt/context identifiers, task-state counts, and stage-local output/log paths.

## References

- Use `mlff-screening-and-relaxation` for SP/relax operation choices and its backend references only when needed.
- Use `mlff-md-sampling` for MD method and restart semantics.
- Use `neb-prepare` before packaging a path and `mlff-path-optimization` for MLFF NEB execution choices.
- Use `dpdispatcher-remote-receipts` for receipt-driven failure triage.
