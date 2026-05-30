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
5. Submit with `remote_submission` for one stage, or `remote_submission_batch` when `work_dir` contains one first-level child directory per task.

Never pass a raw project input tree to low-level remote submission unless it already matches the declared layout.
Do not use `execute` to import, wrap, or call CatMaster managed tool implementations; call the exposed `remote_submission` or `remote_submission_batch` tool directly.

## General layout rules
- The remote cwd is the submitted stage directory, and outputs are downloaded back into that same stage.
- Built-in boot scripts are copied automatically from the task catalog.
- `get_avail_resources` lists general custom-boot resource cards. Use `general_cpu` for custom pure-Python/ASE CPU scripts and `general_gpu` for custom GPU scripts when visible; otherwise rely on the card marked `default_for_custom_boot`.
- Domain task resource defaults are shown through `get_avail_remote_task(return_resource=true)`. For registered tasks, do not pass `config.resources`; the task resource card owns machine/environment initialization. Override only exposed sizing fields such as `cpu_per_node` or `gpu_per_node` when intentionally requested.
- For worker tools, do not pass `config.machine`; machine-level selection is a backend/admin detail.
- For batch submission, every first-level child of `work_dir` is submitted as one task; nested discovery is not performed.
- Use `params` only for command-template values. Use `config` for custom-boot resource-card selection, allowed sizing overrides, and submission controls.

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
Stage directory must contain the molecular input file named by `params.input_name`; default is `input.xyz`.

Common params: `mode`, `gfn`, `solvent_model`, `solvent`, `charge`, `uhf`, `opt_level`.

## crest_run
Stage directory must contain the molecular input file named by `params.input_name`; default is `input.xyz`.

Optional constrained runs may include a constraint file and set `params.constraint_file`.

## mace_sp_dir
Stage directory must contain `input/` with periodic structures. Outputs are written to `output/`.

```text
stage/
  input/
    POSCAR or *.vasp/*.cif/*.poscar
```

Common params: `model`, `head`, `dispersion`, `default_dtype`, `device`. Managed GPU MACE tasks default to `device=auto`, which may fall back to CPU; pass `device=cuda` only when CUDA execution is required.

## mace_relax_dir
Same layout as `mace_sp_dir`, with relaxation params such as `fmax`, `maxsteps`, and `relax_lattice`. Common params also include `device`; managed GPU tasks default to `device=auto`, which prioritizes completion and may fall back to CPU. Pass `device=cuda` only for hard GPU validation.

## mace_md_dir
Dynamics-worker task. Stage directory must contain `input/` and a params JSON file. Default `params_path` is `params/md_params.json`.

Common params: `params_path`, `device`.

## mace_neb_dir
Stage directory must contain `input/` with one prepared path task directory per NEB job. Outputs are written to `output/`.

Common params: `mode`, `fmax`, `steps`, `climb`, `model`, `head`, `dispersion`, `default_dtype`, `device`. Managed GPU MACE tasks default to `device=auto`, which may fall back to CPU.

## mace_train_dir
Stage directory must contain a dataset directory and training params JSON:

```text
stage/
  dataset/
  params/
    train_params.json
```

Defaults are `dataset_root=dataset`, `params_path=params/train_params.json`, and `output_root=output`.

## mace_eval_dir
Stage directory must contain a dataset directory and evaluation params JSON:

```text
stage/
  dataset/
  params/
    eval_params.json
```

Defaults are `dataset_root=dataset`, `params_path=params/eval_params.json`, and `output_root=output`.
