# 8. Remote tasks: running prepared calculations

[Previous](07-literature-writing-review.en.md) | [Contents](README.en.md) | [Next](09-tools-skills-evolution.en.md)

CatMaster can generate structures, inspect files, and prepare calculation inputs locally, while VASP, CP2K, LAMMPS, ORCA, xTB, CREST, MACE, and other managed MLFF programs normally run on a cluster or GPU server. A remote task connects a local stage to a configured scientific program, compute resource, result transfer, and recovery record.

Users do not need to memorize submission commands. Experiment assigns the scientific work to a domain worker. The worker queries the current task catalog, validates the stage and parameter schema, submits after approval, and brings status, logs, and results back into the original workspace. A receipt preserves the identity of the remote run.

## Preparation support is not the same as configured execution

Many modeling tools work on every CatMaster installation. Materials can create slabs and VASP inputs, and Dynamics can prepare LAMMPS stages. Actual execution also requires administrator-provided SSH, remote directories, queues, environment scripts, licenses, and an enabled task.

"The VASP stage is valid, but this deployment does not expose `vasp_execute`" is therefore a precise capability boundary. CatMaster does not silently run a scientific engine on the WebUI host when managed execution is unavailable.

Workers query the catalog before every real submission. Enabled state, resource binding, MLFF backend, model, and accepted override fields come from the current deployment, not an old prompt or manual snapshot.

## Remote tasks in the repository template

The following tasks are defined by the public template. What users actually see depends on the private active configuration and worker audience.

### VASP: single stages, paths, and dimer work

`vasp_execute` runs one prepared VASP directory for relaxation, static, frequency, DOS, MD, or another stage created through the VASP preparation surface. `vasp_execute_neb` uses the larger VASP resource for NEB and dimer-style directories. The template also contains disabled `vasp_execute_k8s` support for a separately validated SSH-to-Kubernetes bridge.

Before submission, Materials verifies INCAR, POSCAR, POTCAR, and KPOINTS, checks element and pseudopotential order, and confirms the directory contract. NEB also needs root inputs and consecutively numbered image directories. A remote task executes an accepted stage. It does not replace the scientific input audit.

```text
Use Experiment to recheck the VASP relaxation stage at calculations/co_adsorption/site_03/.
Ask Materials to verify the structure, Selective Dynamics, POTCAR order, INCAR, KPOINTS, spin,
and convergence settings, then query the current vasp_execute task spec. If its execution binding is configured,
accept the registered deployment binding without asking for scheduler, module, license, revision, or prior-receipt details.

Stop only for an input problem or a concrete catalog/spec error. If all checks pass, show task, work_dir, resource,
and important settings in the Review approval card. After execution, inspect program convergence and the
final structure rather than reporting success from scheduler state alone.
```

### CP2K and LAMMPS

`cp2k_execute` is available to Materials and Dynamics. It runs a CP2K stage containing `job.inp` plus every file referenced by its manifest. Materials uses it for conventional DFT and property-oriented work. Dynamics uses it for AIMD and restart-aware workflows.

`lammps_execute` belongs to Dynamics. It runs a validated LAMMPS stage with input script, data or restart, and potential files. Successful startup does not prove that the force field is suitable. Element mapping, units, boundaries, neighbor behavior, and potential domain still need preflight review.

```text
Continue calculations/cp2k_aimd_600K_part1/.
Ask Dynamics to verify the last valid restart, coordinates, velocities, random state, and time line.
Build part2 in a separate directory and never overwrite part1. Query the current cp2k_execute spec,
show whether continuation settings match, and wait for approval before submission.

After transfer, inspect restart continuity, temperature, energy, and trajectory completeness before joining segments.
```

### Unified MLFF tasks: SP, relax, MD, and NEB

`mlff_sp`, `mlff_relax`, `mlff_md`, and `mlff_neb` use common task names. The backend configuration selects MACE, FairChem UMA, MatterSim, or ORB-v3. This keeps one scientific workflow across enabled providers rather than duplicating tools for every model family.

The public template enables only MACE `mh-1` by default. UMA, MatterSim, and ORB-v3 appear only after an administrator installs an isolated environment, weights, resource, and a passing smoke case. `get_remote_task_spec` returns valid model, device, dtype, optimizer, ensemble, and other operation fields for the current backend. Do not copy overrides from an older deployment.

`mlff_sp` and `mlff_relax` can process several structures directly under one stage's `input/`, so a multi-structure screen is not automatically a remote batch. `mlff_md` accepts one start or restart structure. `mlff_neb` accepts a locally constructed, validated fixed-image path.

```text
Pre-screen structures/adsorption_candidates/ with an enabled MLFF.
Ask Materials to query current backends and the mlff_sp and mlff_relax schemas, then recommend a model
based on element coverage and purpose. Run batch single points first, inspect abnormal energies and failures,
and relax only the candidates worth retaining.

Before approval, show model, device, dtype, input count, output location, and ranking method.
Report this as MLFF screening and identify candidates recommended for DFT with their risks.
```

### MACE training and evaluation

`mace_train` and `mace_eval` belong to ML. Training reads `dataset/` and `params/train_params.json` and returns checkpoints, logs, and other output under `output/`. Evaluation uses `params/eval_params.json` against a defined held-out or benchmark set.

The worker audits data and prepares the stage before training. Units, labels, split, E0, head, replay, and fine-tuning policy are scientific inputs that a remote task cannot repair automatically. After training, analyze held-out error and failure cases instead of reporting the final epoch alone.

```text
Check whether ml/mace_finetune_v1/ is ready for mace_train.
Ask ML to reread the dataset manifest, train/validation/test split, and train_params.json.
Verify labels, units, E0, seed, foundation checkpoint, and replay settings.

Query the current mace_train resource and estimate input scale. Show critical training parameters in Review.
After approval and execution, retain the checkpoint, complete logs, and config, then prepare mace_eval separately.
Do not substitute training error for independent testing.
```

### xTB, CREST, and ORCA

`xtb_run` supports staged molecular optimization, energy, Hessian, or short MD modes defined by the task. `crest_run` performs conformer search. `orca_execute` runs an ORCA stage containing `job.inp` and its local dependencies. All belong to ORCA/xTB.

The worker verifies charge, unpaired-electron or multiplicity convention, solvent, method, basis, and structure before submission. CREST and xTB are often low-cost filters, while selected conformers enter ORCA optimization, frequency, thermochemistry, TDDFT, NMR, or path work. The task executes one stage. Skills organize the larger molecular workflow.

```text
Run ORCA opt+freq for the six structures under molecules/conformers_selected/.
Ask ORCA/xTB to verify conformer deduplication, charge, multiplicity, solvent, method, and basis,
then create an independent stage for each structure. Query orca_execute; accept a configured execution binding
without requesting administrator-owned ORCA, MPI, scheduler, license, or historical-receipt metadata.

Before managed batch submission, show all six stage paths and common settings. After transfer, check normal
termination, gradient, and imaginary modes for every conformer. Do not hide a failed conformer behind batch totals.
```

## How an agent chooses task, resource, and parameters

A worker calls `get_avail_remote_task`, then `get_remote_task_spec` for the full schema. A listed task whose spec reports `execution_binding.status=configured` has a deployment-owned task/backend, resource, and machine binding; that is sufficient infrastructure preflight for normal submission. `get_avail_resources` lists only general custom-boot cards and is not a second audit of registered domain tasks.

The task defines the execution contract and default resource. The resource defines machine, CPU/GPU, queue, walltime, environment scripts, and audience. The machine defines SSH behavior and remote work root. Ordinary users mainly need to assess availability, scientific settings, resource suitability, and cost. Administrator configuration is in Chapter 10.

Queue/account details, resource-card revisions, module or licensed-executable identifiers, and previous smoke receipts are administrator-owned and intentionally omitted from the worker-facing surface. Their absence is not a reason to stop. The worker should block only when the catalog/spec reports an actual binding error or the managed submission returns a concrete runtime failure.

Scientific and method controls use `template_overrides`. Submission-layer controls use `submission_config`. Accepted keys come from the current task spec. Do not place model, optimizer, temperature, or scientific method in submission controls, and do not mix polling or cleanup controls into scientific overrides.

## One stage versus a remote batch

`remote_submission` accepts one complete stage. `remote_submission_batch` accepts a parent directory whose first-level children are independent complete stages sharing one task and submission configuration. It does not recursively discover deeper directories and should not split a multi-structure MLFF stage.

Batch submission preserves per-stage status within one managed call. The agent should still list every discovered child and verify the count. If part of a batch fails, preserve successful results and retry only confirmed failures.

```text
Prepare a managed VASP batch from calculations/vacancy_screen/.
List every first-level child and validate canonical VASP inputs in each. If any stage is incomplete or inconsistent,
report it before submitting anything. After confirming count, task, resource, and common settings, wait in Review.
Do not recurse into deeper folders and do not recompute existing successes.
```

## What a remote run leaves in the workspace

CatMaster copies the stage into metadata staging, DPDispatcher transfers it, and terminal results merge back into the original `files/` stage. Every task should return at least `status.json`, `stdout.log`, and `stderr.log`, plus program-specific output.

A receipt is saved under `files/.deepagents/dpdispatcher/receipts/`. It records task, original work directory, submission time, remote context, submission hash, job state, resources, and updates. This is what allows a job to remain identifiable after WebUI disconnect, network error, or local process exit.

Chat shows receipt cards, Files opens results, and Monitor records tool state. Scientific acceptance should correlate receipt, scheduler state, stdout/stderr, program convergence, and domain results.

## Stop, disconnects, and recovery

WebUI Stop ends the local agent turn. It does not cancel a Slurm or remote-shell job. A network disconnect also does not prove the job stopped. Blindly resubmitting the same stage can create duplicate billed jobs and conflicting outputs.

Recovery begins from `remote_context_id`, `submission_hash`, and `receipt_rel`. Check the scheduler and DPDispatcher record to decide whether the job was never created, queued, running, terminated, or completed without download. Retrieve completed outputs and termination logs first. Consider resubmission only after proving the old job will no longer consume resources or write results.

```text
The previous remote_submission returned an SSH error. Do not resubmit.
Read the message receipt and matching file under .deepagents/dpdispatcher/receipts/.
Confirm remote_context_id, submission_hash, task, original stage, and known job state.

Use scheduler and DPDispatcher evidence to determine whether it is running, complete but not downloaded,
or truly terminated. Retrieve finished results and failure logs first, then propose recovery. Do not resubmit
or clean the remote directory until I confirm the old job state.
```

Administrator recovery commands are listed in Chapter 11. Remote cleanup is appropriate only after results and logs are safely local.

## Protecting submission with Review mode

In Review mode, `remote_submission` and `remote_submission_batch` pause before execution. Check that the stage path is the version just reviewed, task and backend match the objective, resource and task count fit the budget, overrides come from the current schema, and cleanup will not remove needed evidence or collide with an old job.

You do not need to repeat this checklist in every prompt. "Remote submission must wait for Review approval" is enough for the agent to prepare a concrete action card.

## Sources of remote-task capability

<details>
<summary>Remote tools and execution skills visible to workers</summary>

Materials, Dynamics, ML, and ORCA/xTB receive `get_avail_remote_task`, `get_remote_task_spec`, `get_avail_resources`, `remote_submission`, and `remote_submission_batch` according to their audiences.

`remote-stage-layouts` defines canonical input layouts and preflight checks. `dpdispatcher-remote-receipts` applies only after a failed or ambiguous call, or evidence of an orphan job. It is not a polling mechanism for a normal synchronous call that is still pending.

</details>

## Connecting a machine for the first time

An administrator configures machine, resource, task, and optional MLFF backend before a worker queries the catalog. Do not begin with the full smoke suite. Submit one inexpensive real case for one installed engine, verify environment, result transfer, receipt, and failure recovery, then enable other capabilities gradually.

Ordinary users do not edit these YAML files. Ask the agent to query current availability and wait for submission approval. If the task is missing, send the precise blocker to the administrator rather than asking the agent to guess cluster configuration.
