# 8. Remote machines and execution

[Previous](07-literature-writing-review.en.md) | [Contents](README.en.md) | [Next](09-tools-skills-evolution.en.md)

CatMaster uses DPDispatcher to submit prepared calculation stages to SSH-reachable
Slurm or Shell machines. Remote configuration has four layers: a machine defines
connection, a resource defines where and with what allocation to run, a task
defines the execution contract, and an MLFF backend maps a model and operation to
a resource.

## 8.1 Responsibilities

| Role | Responsibility |
|---|---|
| Administrator | SSH, remote root, queues, resource cards, environment scripts, executables, and licenses |
| CatMaster worker | Prepare a stage from the task schema, run QC, submit, collect, and record a receipt |
| User | Choose scientific settings, approve submission, accept cost, and verify results and recovery decisions |

An agent cannot arbitrarily move a registered task to another machine or
resource. CPU/GPU overrides should be used only when the user explicitly asks
and the site permits them.

## 8.2 Create the four active configurations

Files containing `template` are not loaded by the registries. Copy all four:

```bash
cp configs/dpdispatcher/machines_template.yaml \
   configs/dpdispatcher/machines.yaml
cp configs/dpdispatcher/resources_template.yaml \
   configs/dpdispatcher/resources.yaml
cp configs/dpdispatcher/tasks_template.yaml \
   configs/dpdispatcher/tasks.yaml
cp configs/dpdispatcher/mlff_backends_template.yaml \
   configs/dpdispatcher/mlff_backends.yaml
```

Active files contain hostnames, usernames, SSH keys, paths, and site setup. They
are excluded from Git and deployment packages. Never copy real values into
documentation, issues, prompts, or shared workspaces.

Other non-template YAML, YML, and JSON files in the directory may also load.
Duplicate keys in several active files can be overwritten by later input, so
maintain one clear source of truth.

## 8.3 Machine card

A basic Slurm CPU machine has this shape:

```yaml
cpu_server:
  batch_type: Slurm
  context_type: SSHContext
  local_root: <LOCAL_WORK_ROOT>
  remote_root: <REMOTE_WORK_ROOT>
  retry_count: 0
  remote_profile:
    hostname: <CPU_LOGIN_HOST>
    port: 22
    username: <USERNAME>
    key_filename: <PATH_TO_SSH_KEY>
  env_setup: |
    ulimit -s unlimited
    module load <SITE_MODULES>
```

The template also defines:

| Machine | Batch | Typical use |
|---|---|---|
| `cpu_server` | Slurm over SSH | VASP, CP2K, LAMMPS, xTB, CREST, ORCA, and general CPU |
| `k8s_ssh_server` | Shell over SSH | Blocking SSH-to-Kubernetes bridge; its VASP task is disabled by default |
| `gpu_server` | Shell over SSH | MACE, UMA, MatterSim, ORB, and general GPU |

CatMaster replaces `local_root` with its metadata staging root for an actual
submission. Keep a valid placeholder, but focus review on `remote_root`, the SSH
profile, and the resource environment.

## 8.4 SSH and directory acceptance

Use a noninteractive key with restricted permissions:

```bash
chmod 600 <PATH_TO_SSH_KEY>
ssh -i <PATH_TO_SSH_KEY> -p 22 <USERNAME>@<CPU_LOGIN_HOST>
```

Test noninteractive access from the control plane:

```bash
ssh -o BatchMode=yes -i <PATH_TO_SSH_KEY> \
  <USERNAME>@<CPU_LOGIN_HOST> 'hostname; python3 --version'
```

Check that the remote root exists and is writable:

```bash
ssh -o BatchMode=yes -i <PATH_TO_SSH_KEY> \
  <USERNAME>@<CPU_LOGIN_HOST> \
  'mkdir -p <REMOTE_WORK_ROOT> && test -w <REMOTE_WORK_ROOT>'
```

For Slurm also check:

```bash
ssh -o BatchMode=yes -i <PATH_TO_SSH_KEY> \
  <USERNAME>@<CPU_LOGIN_HOST> \
  'command -v sbatch; command -v squeue; command -v scancel'
```

Confirm the host key interactively before automation. Do not hide a changed host
identity with `StrictHostKeyChecking=no`.

## 8.5 Environment order

The remote command environment is assembled in this order:

```text
machine.env_setup
  -> resource.source_list
      -> submission prepend_script
          -> task command
```

Every `source_list` script must exist remotely. A bad path fails before task
startup with code 127. Put modules, conda activation, license variables, and
library paths in site-controlled scripts, not in a task stage.

## 8.6 Resource card

A resource binds capability to machine, audience, queue, and allocation. Template
defaults are only examples:

| Resource | Machine | CPU | GPU | Queue | Audience/use |
|---|---|---:|---:|---|---|
| `vasp_cpu` | `cpu_server` | 52 | 0 | `batch` | Materials, VASP stage |
| `vasp_k8s_cpu` | `k8s_ssh_server` | 4 | 0 | `k8s` | Materials, K8s VASP |
| `vasp_cpu_neb` | `cpu_server` | 104 | 0 | `batch` | Materials, VASP path |
| `cp2k_cpu` | `cpu_server` | 32 | 0 | `batch` | Materials/Dynamics |
| `lammps_cpu` | `cpu_server` | 16 | 0 | `batch` | Dynamics |
| `general_cpu` | `cpu_server` | 4 | 0 | `batch` | Permitted custom CPU boot |
| `general_gpu` | `gpu_server` | 16 | 1 | `main` | Permitted custom GPU boot |
| `mace_gpu` | `gpu_server` | 16 | 1 | `main` | MACE |
| `uma_gpu` | `gpu_server` | 16 | 1 | `main` | FairChem UMA |
| `mattersim_gpu` | `gpu_server` | 16 | 1 | `main` | MatterSim |
| `orb_gpu` | `gpu_server` | 16 | 1 | `main` | ORB-v3 |
| `xtb_cpu` | `cpu_server` | 32 | 0 | `batch` | xTB |
| `crest_cpu` | `cpu_server` | 32 | 0 | `batch` | CREST |
| `orca_cpu` | `cpu_server` | 32 | 0 | `batch` | ORCA |

Adapt queue, core count, GPU, walltime, and `source_list` to the site. Resource
`audiences` determine which workers see it. Do not remove all audience controls
for convenience.

## 8.7 Task card

The current template registers:

| Task | Default resource | Main input |
|---|---|---|
| `vasp_execute` | `vasp_cpu` | One VASP stage |
| `vasp_execute_k8s` | `vasp_k8s_cpu` | Same VASP stage, disabled by default |
| `vasp_execute_neb` | `vasp_cpu_neb` | NEB/dimer directory |
| `cp2k_execute` | `cp2k_cpu` | CP2K stage |
| `lammps_execute` | `lammps_cpu` | LAMMPS stage |
| `mlff_sp` | Chosen by backend | Multi-structure single point |
| `mlff_relax` | Chosen by backend | Multi-structure relaxation |
| `mlff_md` | Chosen by backend | Single-structure trajectory |
| `mlff_neb` | Chosen by backend | Fixed-image path |
| `mace_train` | `mace_gpu` | Dataset and training parameters |
| `mace_eval` | `mace_gpu` | Dataset and evaluation parameters |
| `xtb_run` | `xtb_cpu` | Molecular input and mode settings |
| `crest_run` | `crest_cpu` | Molecule and conformer-search settings |
| `orca_execute` | `orca_cpu` | `job.inp` stage |

A task without `enabled` is enabled. Only `vasp_execute_k8s` is explicitly false
in the template. Validate the bridge, shared directory, and blocking behavior
before enabling it.

## 8.8 MLFF backend environments

Template defaults are:

| Backend | Enabled | Default model | Operations |
|---|---|---|---|
| `mace` | true and default | `mh-1` | SP, relax, MD, NEB |
| `fairchem_uma` | false | `uma-s-1p2` | SP, relax, MD, NEB |
| `mattersim` | false | `mattersim-v1-1m` | SP, relax, MD, NEB |
| `orb_v3` | false | `orb-v3-conservative-inf-omat` | SP, relax, MD, NEB |

Each provider uses an isolated environment. Do not install these requirement
sets into the control plane:

```text
requirements/mace.txt
requirements/uma.txt
requirements/mattersim.txt
requirements/orb.txt
```

Create separate remote environments and point each resource `source_list` to its
activation script. Reference scripts are under:

```text
configs/dpdispatcher/env_templates/
```

Keep model tokens, caches, and license variables in the private remote
environment, never in YAML, stages, or prompts. Set a backend to `enabled: true`
only after dependencies, weights, device, and a minimal smoke case pass.

## 8.9 Canonical stage layouts

| Task | Stage-root requirement |
|---|---|
| `vasp_execute` | `INCAR`, `POTCAR`, `POSCAR`, `KPOINTS` |
| `vasp_execute_neb` | Root `INCAR`, `POTCAR`, `KPOINTS`; `00/POSCAR ... NN/POSCAR` |
| `cp2k_execute` | `job.inp`, `manifest.json`, and referenced files |
| `lammps_execute` | `in.lammps`, `manifest.json`, `system.data` or restart, and potential files |
| `orca_execute` | `job.inp` and local referenced files |
| `xtb_run`, `crest_run` | Default `input.xyz`, or an overridden input name |
| `mlff_sp`, `mlff_relax` | Structures directly under `input/`, optional `models/` |
| `mlff_md` | Exactly one start or restart structure under `input/` |
| `mlff_neb` | `input/path/00.vasp ... NN.vasp` |
| `mace_train` | `dataset/`, `params/train_params.json` |
| `mace_eval` | `dataset/`, `params/eval_params.json` |

Every referenced input must remain inside the stage. Do not use symlinks to
outside project space. `mlff_sp` and `mlff_relax` already accept several
structures under `input/`; multiple structures alone are not a reason to use
`remote_submission_batch`.

## 8.10 Query the task schema first

The runtime task schema is the source of truth. A request can say:

```text
Call get_avail_remote_task and confirm mlff_relax is available. Then query the
full get_remote_task_spec schema for backend=mace. Prepare only
calculations/si_relax/, list defaults and overrides requiring my decision, and do
not submit yet.
```

`template_overrides` controls task scientific or method parameters.
`submission_config` controls the submission layer, including check interval,
permitted CPU/GPU overrides, and cleanup. Do not mix them. Machine and resource
come from task/backend registration, not free-form agent choice.

## 8.11 Single stage and batch

`remote_submission`:

- `work_dir` is one complete stage.
- The call blocks until a terminal state.
- Defaults are `check_interval=30` seconds and `clean_remote=false`.

`remote_submission_batch`:

- The parent contains at least two first-level child directories.
- Every child is an independent complete stage.
- Discovery is not recursive.
- All children share the same task and configuration.
- The call waits for all children to reach terminal state.

Do not poll or resubmit while the tool call is still pending. Duplicate
submission of one stage can create two billable jobs.

## 8.12 Staging, return, and receipts

CatMaster copies a stage into workspace metadata staging before DPDispatcher
uploads it. At terminal state, results merge back into the original `files/`
stage. Every stage forcibly returns:

```text
status.json
stdout.log
stderr.log
```

Receipt physical path:

```text
files/.deepagents/dpdispatcher/receipts/
  dp_<timestamp>_<hash8>.json
```

The agent sees a relative path beginning with `.deepagents/...`. Important
fields are:

| Field | Purpose |
|---|---|
| `remote_context_id` | CatMaster remote-context identity |
| `submission_hash` | DPDispatcher recovery and download identity |
| `receipt_rel` | Workspace-relative receipt path |
| `task_name`, `work_dir_rel` | Task and original stage |
| `submitted_at`, `updated_at`, `duration_s` | Timeline |
| `jobs`, `job_status_counts` | Scheduler jobs and state counts |
| `resources` | Effective resource summary |

A successful response should also include `task_count`, `task_state_counts`, and
`submission_dir`. Preserve these values instead of copying only "calculation
finished."

## 8.13 Failure and recovery

A network interruption or local exception does not prove the remote job was
cancelled. Use this order:

1. Preserve `remote_context_id`, `submission_hash`, and `receipt_rel`.
2. Determine whether the original tool call reached terminal state. Do not
   inspect or resubmit while it is pending.
3. Use the scheduler and receipt to classify the job as not created, queued,
   running, terminated, or finished but not downloaded.
4. Collect finished results and terminated logs first.
5. Resubmit only after confirming the old job will not continue consuming
   resources or overwrite results.
6. Use `clean_remote=true` or cleanup commands only after results and logs are
   downloaded.

An empty `submission_hash` normally means there is no recoverable DPDispatcher
record. For a nonempty hash, run from the corresponding project's `files/`
directory as appropriate:

```bash
dpdisp submission <submission_hash> --download-finished-task
dpdisp submission <submission_hash> --download-terminated-log
dpdisp submission <submission_hash> --reset-fail-count
dpdisp submission <submission_hash> --clean
```

Do not execute all four blindly. Download and classify first. Use the last two
only when their effect is understood.

## 8.14 Remote smoke tests

List suites and cases without submitting:

```bash
python scripts/remote_execution_smoke.py --list
```

One real example:

```bash
python scripts/remote_execution_smoke.py \
  --case mace_sp \
  --project-space /tmp/catmaster_remote_smoke \
  --stop-on-failure
```

Every mode except `--list` submits real calculations that may queue, consume
credits, and use licenses. Do not begin with `--suite all`. Run one minimal case
for a configured backend or CPU engine, inspect its stage, receipt, logs, and
returned files, then expand coverage.

## 8.15 Administrator acceptance

Before opening the service to users, confirm:

1. Every machine in use supports noninteractive login.
2. `remote_root` is writable and Slurm or Shell behavior matches the card.
3. Every `source_list` exists and loads the correct program.
4. Queue, CPU, GPU, walltime, and audience are correct.
5. All four active configurations exist and templates are not mistaken for
   active files.
6. The task catalog exposes only installed and authorized capabilities.
7. Every enabled MLFF backend passes an independent minimal case.
8. Every engine produces `status.json`, stdout, stderr, and a receipt.
9. A simulated transfer failure can be recovered from the receipt without
   recomputation.
10. Active configs, SSH keys, tokens, and license data stay out of Git and
    project files.
