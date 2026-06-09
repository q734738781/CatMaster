# Remote Setup

Read this chapter only when you need to submit calculations to a remote machine or cluster. If you only need the local WebUI, writing, literature review, or analysis of existing files, you can skip it.

CatMaster uses DPDispatcher for remote submission. You configure three things:

- `machine`: how to connect, such as SSH host, username, key, and work directories.
- `resource`: which machine, queue, CPU/GPU count, and environment setup to use.
- `task`: what command to run, which files to upload, and which results to download.

## 1. Configure Machines

CatMaster reads:

```text
configs/dpdispatcher/machines.yaml
```

For a first setup, copy all three templates:

```bash
cp configs/dpdispatcher/machines_template.yaml configs/dpdispatcher/machines.yaml
cp configs/dpdispatcher/resources_template.yaml configs/dpdispatcher/resources.yaml
cp configs/dpdispatcher/tasks_template.yaml configs/dpdispatcher/tasks.yaml
```

`machines.yaml`, `resources.yaml`, and `tasks.yaml` are active deployment config files. They contain login hosts, SSH keys, local/remote work roots, queues, `source_list`, and task commands, so they are ignored by Git and by deployment packaging. Public releases include only `*_template.yaml`.

The template uses two default machine keys:

- `cpu_server_2`
- `gpu_server`

These names match the resource cards in `configs/dpdispatcher/resources.yaml`. You may rename them, but then you must update each resource card's `machine` field.

Simplified example:

```yaml
cpu_server_2:
  batch_type: Slurm
  context_type: SSHContext
  local_root: /path/to/local/dpdispatcher_work
  remote_root: /path/to/remote/dpdispatcher_work
  retry_count: 0
  remote_profile:
    hostname: login.cluster.edu
    port: 22
    username: your_user
    key_filename: /home/your_user/.ssh/id_ed25519
  env_setup: |
    ulimit -s unlimited
    module load python/3.10
```

First verify SSH outside CatMaster:

```bash
ssh -i /home/your_user/.ssh/id_ed25519 your_user@login.cluster.edu
```

## 2. Understand Resource Cards

Active resource cards live in:

```text
configs/dpdispatcher/resources.yaml
```

Common keys:

- `vasp_cpu`: VASP single-stage CPU jobs.
- `vasp_cpu_neb`: VASP NEB / dimer path jobs.
- `cp2k_cpu`: CP2K CPU MPI jobs.
- `lammps_cpu`: LAMMPS CPU jobs.
- `mace_gpu`: MACE GPU jobs.
- `general_cpu`: custom CPU boot scripts.
- `general_gpu`: custom GPU boot scripts.
- `xtb_cpu`, `crest_cpu`, `orca_cpu`: molecular quantum chemistry and conformer jobs.

Check these fields first:

- `machine` points to a key in `machines.yaml`.
- `queue_name` matches your scheduler queue.
- `cpu_per_node` / `gpu_per_node` follow queue limits.
- `custom_flags` use the right Slurm/PBS syntax.
- `source_list` or `prepend_script` loads VASP, CP2K, MACE, or other required programs.

## 3. Edit Active Resource Cards

If `resources.yaml` does not match your cluster, edit that private file directly. It is not committed or packaged. Check that `source_list` and `prepend_script` load the remote task environment. Example for `vasp_cpu`:

```yaml
vasp_cpu:
  kind: domain
  capabilities: [vasp]
  description: "My VASP CPU queue."
  audiences: [materials_worker]
  machine: cpu_server_2
  number_node: 1
  cpu_per_node: 64
  queue_name: normal
  group_size: 1
  custom_flags:
    - "#SBATCH -t 2-00:00:00"
    - "#SBATCH --export=ALL"
  source_list:
    - /path/to/remote/vasp_env.sh
```

Existing task templates can keep referencing `vasp_cpu`.

## 4. Understand Task Config

Active tasks live in:

```text
configs/dpdispatcher/tasks.yaml
```

Task keys include:

- `vasp_execute`
- `vasp_execute_neb`
- `cp2k_execute`
- `lammps_execute`
- `mace_relax_dir`
- `mace_sp_dir`
- `mace_md_dir`
- `mace_neb_dir`
- `mace_train_dir`
- `mace_eval_dir`
- `xtb_run`

Each task defines:

- `resources`: default resource key.
- `boot_script`: script copied into the remote stage.
- `command`: remote execution command.
- `defaults`: default values for command placeholders.
- `forward_files`: files to upload.
- `backward_files`: files to download.
- `task_work_path`: stage subdirectory where the command runs.

## 5. Add Custom Tasks

For custom remote tasks, edit the private `configs/dpdispatcher/tasks.yaml`.

Example:

```yaml
my_python_task:
  audiences: [materials_worker]
  description: "Run my prepared Python stage."
  boot_script: "catmaster/remote/cpu/vasp_boot.py"
  resources: general_cpu
  requires: [python]
  command: "python task_script/vasp_boot.py"
  forward_files:
    - "*"
    - "task_script/vasp_boot.py"
  backward_files:
    - "*"
  task_work_path: "."
```

YAML files with `template` in the filename are examples only and are not loaded as active runtime config. Copy them to `machines.yaml`, `resources.yaml`, and `tasks.yaml` to enable them.

## 6. Use Remote Tasks In The WebUI

After remote config is ready, ask for explicit remote submission:

```text
Use vasp_inputs/CO_on_Ni_top as a prepared VASP stage. Submit it with remote_submission using task_name=vasp_execute, then return remote_context_id, receipt_rel, and status.json.
```

For MACE relax:

```text
Convert structures/ into a mace_relax_dir stage with input/, then submit it with remote_submission. Use model=mh-1 and head=omat_pbe.
```

## 7. Single Versus Batch Submission

The difference between `remote_submission` and `remote_submission_batch` is where the boot script starts:

- `remote_submission`: `work_dir` is one stage, and the boot script starts once directly inside `work_dir`.
- `remote_submission_batch`: `work_dir` is a parent root, and the boot script starts once inside each first-level child directory. The parent root itself is not used as the task cwd.

```text
remote_submission:
stage/
  INCAR
  POSCAR
  KPOINTS
  POTCAR

remote_submission_batch:
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
```

Do not switch to `remote_submission_batch` merely because one stage contains many scientific inputs. For example, one `mace_sp_dir` or `mace_relax_dir` stage may contain many structures under `input/`; that is still one `remote_submission`. Use `remote_submission_batch` only when you have multiple independent MACE stage child directories.

## 8. Real Remote Smoke Tests

After remote deployment, run the script-style smoke test first. It prepares tiny O2/H2O stages, uses CatMaster's current agent-visible `remote_submission` path, and submits real DPDispatcher jobs. It is not a dry-run.

List available suites and cases:

```bash
python scripts/remote_execution_smoke.py --list
```

Common minimal coverage:

```bash
python scripts/remote_execution_smoke.py --suite core --check-interval 30
```

`core` runs real `mace_sp`, `xtb_sp`, and `orca_sp` jobs. For material-side coverage:

```bash
python scripts/remote_execution_smoke.py --suite materials --check-interval 60
```

For all regular remote execution coverage:

```bash
python scripts/remote_execution_smoke.py --suite all --check-interval 60
```

By default the script writes staged files and the JSON report under `/tmp/catmaster_remote_execution_smoke`. To choose a stable location:

```bash
python scripts/remote_execution_smoke.py --suite core --project-space /path/to/catmaster_remote_smoke
```

The pytest entry remains available for development:

```bash
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 pytest tests/remote_execution -s -vv
```

See:

```text
tests/remote_execution/README.md
```

## 9. Troubleshooting

`Machine 'xxx' not found`

The `machine` field in `configs/dpdispatcher/resources.yaml` does not match a key in `configs/dpdispatcher/machines.yaml`.

`Resources 'xxx' not found`

The task references a resource key that is not defined in `configs/dpdispatcher/resources.yaml`.

SSH fails

Run `ssh` directly first. Confirm host, port, username, key path, and firewall access.

Remote command cannot find VASP/MACE/CP2K

Check the resource `source_list` or `prepend_script`; the remote job environment must load the required executable.

Results do not download

Check the task `backward_files`, remote permissions, and DPDispatcher logs.

## 9. Next Step

After remote submission works, use it as part of the Experiment lane described in [Features and daily workflows](03-features.en.md).
