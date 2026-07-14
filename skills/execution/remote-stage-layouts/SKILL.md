---
name: remote-stage-layouts
description: Use this skill before calling low-level remote_submission or remote_submission_batch; it defines the stage-directory layout for registered DPDispatcher task_name templates.
license: project-local
compatibility: local
allowed-tools: "ls read_file write_file edit_file execute get_avail_remote_task get_avail_resources remote_submission remote_submission_batch"
---

# remote-stage-layouts

## Overview
`remote_submission` and `remote_submission_batch` are low-level execution tools. They do not discover scientific inputs, build VASP/ORCA/MACE directories, or mirror arbitrary result trees. Prepare the stage directory first, verify it, then submit.

## Required workflow
1. Call `get_avail_remote_task` when task names or layout refs are uncertain.
2. Read the matching layout section below.
3. Create a clean workspace-relative stage directory.
4. Verify the exact files with `ls` or `find`.
5. Submit with `remote_submission` for one stage. When there are two or more independent prepared stages for the same `task_name`, `template_overrides`, and `config`, make a parent root and prefer one `remote_submission_batch` call.

Never pass a raw project input tree to low-level remote submission unless it already matches the declared layout.
Do not use `execute` to import, wrap, or call CatMaster managed tool implementations; call the exposed `remote_submission` or `remote_submission_batch` tool directly.

## Submission Decision
- Use `remote_submission` when `work_dir` itself is one prepared stage matching the selected `task_name` layout. The boot script runs directly in `work_dir`.
- Prefer `remote_submission_batch` when there are two or more independent prepared stages for the same `task_name`, `template_overrides`, and `config`; do not issue multiple parallel `remote_submission` calls for that case.
- Use `remote_submission_batch` only when `work_dir` is a parent batch root and each first-level child directory is a complete independent stage for the same `task_name`. The parent directory is not the task cwd; the boot script runs once inside each first-level child directory.
- A single stage may contain many scientific inputs if the task layout says so. This is still `remote_submission`, not `remote_submission_batch`. Common example: `mace_sp_dir`, `mace_relax_dir`, `uma_sp_dir`, or `uma_relax_dir` with many structures under one `input/`.
- `remote_submission_batch` does not recursively discover nested jobs. It submits only first-level child directories.
- `remote_submission_batch` applies the same `task_name`, `template_overrides`, and `config` to every first-level child. Do not use it when children need different task templates or incompatible overrides.

```text
single stage -> remote_submission
stage/
  INCAR
  POSCAR
  KPOINTS
  POTCAR

parent with independent stages -> remote_submission_batch
batch_root/
  job_a/
    INCAR
    POSCAR
    KPOINTS
    POTCAR
  job_b/
    INCAR
    POSCAR
    KPOINTS
    POTCAR

one MACE stage with internal input batch -> remote_submission
mace_stage/
  input/
    a.vasp
    b.vasp
```

## General layout rules
- The remote cwd is the submitted stage directory, and outputs are downloaded back into that same stage.
- Multiple same-template independent jobs should be grouped under one parent and submitted with `remote_submission_batch`; this keeps them in one DPDispatcher submission and one result-download lifecycle.
- Built-in boot scripts are copied automatically from the task catalog.
- `get_avail_resources` lists general custom-boot resource cards. Use `general_cpu` for custom pure-Python/ASE CPU scripts and `general_gpu` for custom GPU scripts when visible; otherwise rely on the card marked `default_for_custom_boot`.
- Domain task resource defaults are shown through `get_avail_remote_task(return_resource=true)`. For registered tasks, do not pass `submission_config.resources`; the task resource card owns machine/environment initialization. Override only exposed sizing fields such as `cpu_per_node` or `gpu_per_node` when intentionally requested.
- For worker tools, do not pass `submission_config.machine`; machine-level selection is a backend/admin detail.
- For batch submission, every first-level child of `work_dir` is submitted as one task; nested discovery is not performed.
- Use `template_overrides` for registered task command-template values. Only use keys returned by `get_avail_remote_task`; unknown keys fail instead of being ignored. Use this for method-critical defaults such as MACE `head`/`fmax` or UMA `uma_task`/`spin`.
- Use `submission_config` only for custom-boot resource-card selection, allowed sizing overrides, and submission controls.
- Do not edit copied `task_script/` files or add `sitecustomize.py` to change built-in template defaults. Built-in boot scripts are copied by the submission tool, and template values belong in `template_overrides`.

## vasp_execute
Stage directory must be one complete VASP calculation folder:

```text
stage/
  INCAR
  POTCAR
  POSCAR
  KPOINTS
  optional other VASP inputs
```

For batch submission, make each first-level child one complete calculation folder. Use `vasp_execute_neb` instead when the calculation is a NEB/dimer-style VASP path that needs the larger resource preset.

## vasp_execute_neb
Same as `vasp_execute`, but the stage is a VASP NEB/dimer-style folder with the image subdirectories and root inputs expected by VASP.
Before submission, verify that the stage came from `vasp_neb_prepare` or an equivalent checked image tree. Do not submit if endpoint ordering/cell validation failed. If the preparation artifact reports `short_distance_count > 0` (default warning threshold: minimum interatomic distance below `0.8 Å`), treat it as strong evidence of potentially abnormal interpolation and verify or remediate the image tree before deciding to submit.

## cp2k_execute
Stage directory must contain a prepared CP2K input named `job.inp`. The task runs the generic CP2K boot script with `cp2k.psmp`, `OMP_NUM_THREADS=1`, and MPI ranks from the scheduler allocation.

```text
stage/
  job.inp
  manifest.json
  optional structure / restart / basis / potential / include files
```

For batch submission, make each first-level child one complete CP2K stage directory. Scientific task type is encoded in `job.inp`; do not use per-recipe remote task names.

## lammps_execute
Stage directory must contain a prepared LAMMPS input named `in.lammps`. Fresh structure runs should also contain `system.data`; restart runs should contain the referenced restart file.

```text
stage/
  in.lammps
  manifest.json
  system.data or restart file
  optional potential files
```

The generic boot script runs `lmp` and detects GPU/KOKKOS/GPU-package availability when possible. If acceleration fails, it may fall back to CPU execution so the calculation can still produce output.

## orca_execute
Stage directory must contain a prepared ORCA input named `job.inp`:

```text
stage/
  job.inp
  optional referenced files
```

## xtb_run
Stage directory must contain the molecular input file named by `template_overrides.input`; default is `input.xyz`.

Common template overrides: `input`, `mode`, `gfn`, `solvent_model`, `solvent`, `charge`, `uhf`, `opt_level`.

## crest_run
Stage directory must contain the molecular input file named by `template_overrides.input`; default is `input.xyz`.

Optional constrained runs may include a constraint file and set `template_overrides.constraint_file`.

## mace_sp_dir
Stage directory must contain `input/` with periodic structures. Outputs are written to `output/`.

```text
stage/
  input/
    POSCAR or *.vasp/*.cif/*.poscar
```

Common template overrides: `model`, `head`, `dispersion`, `default_dtype`, `device`. Managed GPU MACE tasks default to `device=auto`, which may fall back to CPU; pass `device=cuda` only when CUDA execution is required.

## mace_relax_dir
Same layout as `mace_sp_dir`, with relaxation template overrides such as `fmax`, `steps`, `relax_lattice`, and `enable_cueq`. Common overrides also include `model`, `head`, `dispersion`, `default_dtype`, and `device`.

`enable_cueq=false` remains the default. When enabling it, also pass `device=cuda`; cuEquivariance is CUDA-only and the runner fails explicitly rather than falling back to CPU. Use the registered override instead of creating a custom boot script.

## uma_sp_dir
Stage directory must contain `input/` with structures for FairChem UMA single-point inference. Outputs are written to `output/`.

```text
stage/
  input/
    POSCAR or *.vasp/*.cif/*.poscar/*.xyz/*.extxyz
  optional params/
    uma_metadata.json
```

Common template overrides: `model`, `uma_task`, `charge`, `spin`, `metadata`, `device`.

- `model` defaults to `uma-s-1p2`.
- `uma_task=auto` makes a conservative molecule-vs-periodic choice: periodic structures with valid cells use `omat`; nonperiodic structures use `omol`.
- For catalysis, oxide catalysis, electrocatalysis, DAC/MOF, or molecular crystals, set `uma_task` explicitly (`oc20`, `oc22`, `oc25`, `odac`, or `omc`) instead of relying on `auto`.
- `omol` requires correct `charge` and `spin` values. FairChem examples use the `spin` field as the spin-state value expected by OMOL, for example singlet `spin=1` and triplet `spin=3`.
- Non-`omol` tasks should normally use `charge=0` and `spin=0`; CatMaster's UMA boot scripts reject nonzero values for those tasks to avoid mixing molecular spin metadata into materials/catalysis tasks.
- The optional `params/uma_metadata.json` can override `uma_task`, `charge`, and `spin` per input file. If used, pass `template_overrides={"metadata": "params/uma_metadata.json"}`.

Use `remote_submission` for one UMA stage even when `input/` contains many structures. Use `remote_submission_batch` only when the parent directory contains multiple independent UMA stage directories.

## uma_relax_dir
Same layout as `uma_sp_dir`, with relaxation template overrides such as `fmax`, `steps`, `optimizer`, and `relax_cell`.

Common template overrides: `model`, `uma_task`, `charge`, `spin`, `metadata`, `device`, `fmax`, `steps`, `optimizer`, `relax_cell`.

- `relax_cell=false` is the default and should remain the default for first-pass screening.
- `relax_cell=true` is only enabled for `uma_task=omat` in CatMaster. Other UMA tasks either target nonperiodic systems or may not have stress-label supervision.
- UMA relaxations are screening/preoptimization steps. For molecular quantum-chemistry claims, use ORCA/xTB follow-up validation rather than treating UMA as final quantum evidence.

## mace_md_dir
Materials- or dynamics-worker task. Stage directory must contain `input/` and a params JSON file. Default `params` is `params/md_params.json`. ASE `.traj` inputs are supported; the runner reads the last frame and preserves its momenta unless explicit velocity reinitialization is requested. For segmented CSVR/Bussi or Langevin runs, stage the prior `restart.traj` so its final frame and RNG checkpoint are restored.

Common template overrides: `params`, `device`. Inside `md_params.json`, use `md_config.dynamics.seed` to control velocity initialization and stochastic Bussi/Langevin sampling.

## mace_neb_dir
Stage directory must contain `input/` with one prepared path task directory per NEB job. Outputs are written to `output/`.
Each path task should be generated or checked by the NEB preparation workflow before submission. Do not submit a tree with atom-order mismatches. If the preparation artifact reports `short_distance_count > 0`, treat it as strong evidence of potentially abnormal interpolation and verify or remediate the image tree before deciding to submit.

Common template overrides: `mode`, `fmax`, `steps`, `climb`, `model`, `head`, `dispersion`, `default_dtype`, `device`. Managed GPU MACE tasks default to `device=auto`, which may fall back to CPU.

## mace_train_dir
Stage directory must contain a dataset directory and training params JSON:

```text
stage/
  dataset/
  params/
    train_params.json
```

Defaults are `dataset_root=dataset`, `params=params/train_params.json`, and `output_root=output`. Training JSON uses exact MACE CLI names such as `name`, `E0s`, and `lr`; additional flags belong in `cli_args` under their exact long-option name without leading `--`.

## mace_eval_dir
Stage directory must contain a dataset directory and evaluation params JSON:

```text
stage/
  dataset/
  params/
    eval_params.json
```

Defaults are `dataset_root=dataset`, `params=params/eval_params.json`, and `output_root=output`.
