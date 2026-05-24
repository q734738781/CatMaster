---
name: remote-stage-layouts
description: Use this skill before calling low-level remote_submission or remote_submission_batch; it defines the stage-directory layout for registered DPDispatcher task_name templates.
license: project-local
compatibility: local
allowed-tools: "ls, read_file, write_file, edit_file, execute, get_avail_remote_task, get_avail_resources, remote_submission, remote_submission_batch"
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

## General layout rules
- The remote cwd is the submitted stage directory.
- Built-in boot scripts are copied automatically from the task catalog.
- Custom `boot_script` submission requires either `config.resources` for an existing preset or `config.machine` plus any needed resource overrides such as `cpu_per_node`, `queue_name`, or scheduler flags.
- For registered tasks, `config.machine` and resource fields override the task's default resource template for that submission only.
- Outputs are downloaded back into the same stage directory.
- For batch submission, every first-level child of `work_dir` is submitted as one task; nested discovery is not performed.
- Use `params` only for command-template values, and `config` only for resources/submission controls.

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
Stage directory must contain an `input/` directory of periodic structures. Outputs are written to `output/`.

```text
stage/
  input/
    POSCAR or *.vasp/*.cif/*.poscar
```

Common params: `model`, `head`, `dispersion`, `default_dtype`.

## mace_relax_dir
Same layout as `mace_sp_dir`, with relaxation params such as `fmax`, `maxsteps`, and `relax_lattice`.

## mace_md_dir
Stage directory must contain `input/` and a params JSON file. Default `params_path` is `params/md_params.json`.

## mace_neb_dir
Stage directory must contain `input/` with one prepared path task directory per NEB job. Outputs are written to `output/`.

Common params: `mode`, `fmax`, `steps`, `climb`, `model`, `head`, `dispersion`, `default_dtype`.

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
