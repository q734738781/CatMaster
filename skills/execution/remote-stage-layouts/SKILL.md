---
name: remote-stage-layouts
description: Prepare stage directories for low-level remote_submission or remote_submission_batch. Read the selected task's layout; reuse a validated handoff for unchanged stages.
license: project-local
---

# Remote stage layouts

One stage is one remote calculation working directory. Select the registered task
from the catalog and inspect its spec when the task, backend or settings are not
already established. For a non-default MLFF backend, pass that backend in
`get_remote_task_spec(template_overrides={"backend": "..."})` so the returned
schema and defaults describe the chosen implementation. Use the default compact
field table for ordinary overrides; request `detail="full"` when nested field
requirements are unclear. The selected task/backend spec is authoritative for
accepted fields and defaults. The references below explain required file layouts
and cross-field rules that parameter types alone cannot convey.

## Select the needed layout

Before preparing a task whose layout is not already in the active context, read
its reference below. Do not infer required manifests, input layouts or parameter
placement from an engine's public CLI. Reuse an already validated layout and
handoff for unchanged work.

| Registered task | Layout and method-critical input rules |
|---|---|
| `vasp_execute` | [vasp_execute](references/vasp_execute.md) |
| `vasp_execute_neb` | [vasp_execute_neb](references/vasp_execute_neb.md) |
| `cp2k_execute` | [cp2k_execute](references/cp2k_execute.md) |
| `lammps_execute` | [lammps_execute](references/lammps_execute.md) |
| `lammps_execute_kokkos` | [lammps_execute_kokkos](references/lammps_execute_kokkos.md) |
| `orca_execute` | [orca_execute](references/orca_execute.md) |
| `xtb_execute` | [xtb_execute](references/xtb_execute.md) |
| `crest_execute` | [crest_execute](references/crest_execute.md) |
| `mlff_sp` | [mlff_sp](references/mlff_sp.md) |
| `mlff_relax` | [mlff_relax](references/mlff_relax.md) |
| `mlff_ts` | [mlff_ts](references/mlff_ts.md) |
| `mlff_vib` | [mlff_vib](references/mlff_vib.md) |
| `mlff_md` | [mlff_md](references/mlff_md.md) |
| `mlff_neb` | [mlff_neb](references/mlff_neb.md) |
| `mace_train` | [mace_train](references/mace_train.md) |
| `mace_eval` | [mace_eval](references/mace_eval.md) |

## Prepare and submit

- Copy required inputs into a fresh workspace-relative stage; preserve sources
  and restart lineage. If inputs are renamed, retain their source-to-stage mapping
  in the handoff. Keep unrelated outputs and nested batches out.
- A batch discovers one complete stage per first-level child by default, with shared
  `task_name`, `template_overrides` and `submission_config`. Submit dependent
  stages only after prerequisite outputs exist. Different configurations need
  separate calls; multiple inputs inside one task do not imply multiple stages.
- Use catalog-declared overrides. Do not edit staged `task_script/` files or add
  `sitecustomize.py`. Dedicated tasks use their configured resource binding;
  `general_execute` exposes environments for uncovered script work. Resource
  catalog queries are unnecessary for an already configured dedicated task.
- Check the selected task's required inputs and batch depth, then submit.
  Both submission tools block until terminal status; wait for the call to return.
- On success use `work_dir_rel`, task status and declared scientific outputs.
  After a returned failure with ambiguous remote state, apply
  `dpdispatcher-remote-receipts` before retrying to avoid duplicating live work.

For continuation, hand off authoritative stage/restart paths, selected task and
validated overrides, completed checks and remaining work. Do not create another
manifest or repeat unchanged catalog/layout checks. Detailed scientific method
choices belong in the corresponding domain skill and native inputs.

## Tool scope and examples

To submit a selected subset without copying stages, pass `stage_paths=["branch_a/job1","branch_b/job3"]` relative to work_dir. They must be independent, nonoverlapping directories using the same task/config. Preparation failure returns all failing paths and submits nothing; fix them or deliberately select the valid subset. A returned execution failure still follows the existing receipt recovery rules.
