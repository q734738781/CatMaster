# 2. Concepts and project spaces

[Previous](01-quickstart.en.md) | [Contents](README.en.md) | [Next](03-llm-configuration.en.md)

CatMaster is not a chat window with every program installed on one computer. It
combines a control plane, project spaces, specialist agents, tools and skills,
and managed remote execution. Understanding these boundaries prevents most path,
continuation, and compute-resource mistakes.

## 2.1 System boundary

```text
Browser
  -> CatMaster WebUI and specialist runtime
      -> project-space files and metadata
      -> LLM providers, web access, and local helper tools
      -> DPDispatcher
          -> SSH/Slurm/Shell machines
              -> VASP, CP2K, LAMMPS, ORCA, xTB, CREST, and MLFF
```

The control plane owns conversation, planning, tool calls, file orchestration,
run records, and remote submission. Scientific engines run in environments
selected by resource cards. The two layers may share a host or be completely
separate.

## 2.2 Accounts, project root, and workspace

`CATMASTER_PROJECT_SPACE_ROOT` is the top-level directory managed by the WebUI,
not a single project.

Default account mode:

```text
<PROJECT_SPACE_ROOT>/
  .webui_auth/
    auth.sqlite
  users/
    <username>/
      default/
        files/
        metadata/
      <another-workspace>/
        files/
        metadata/
```

No-login mode:

```text
<PROJECT_SPACE_ROOT>/
  admin/
    files/
    metadata/
```

A workspace is a durable project boundary. Separate catalytic systems, papers,
or datasets usually deserve separate workspaces. The left rail can switch,
create, and delete workspaces. Deletion is material: switch away first, then
type the requested name to confirm.

## 2.3 `files/` and `metadata/`

Every workspace must contain both directories:

```text
workspace/
  files/
  metadata/
```

`files/` contains user inputs, agent outputs, structures, scripts, calculation
stages, reports, and remote receipts. It is the shared working area.

`metadata/` contains thread records, checkpoints, observability data, artifact
indexes, temporary remote staging, and skill-evolution state. Users normally
back it up or inspect it during diagnosis, but do not edit, rename, or delete its
contents.

The current runtime rejects the old single-root `.catmaster` layout and does not
migrate it automatically. For an old project, create `files/` and `metadata/`,
then copy user data selectively instead of dropping old internal state into the
new layout.

## 2.4 Paths visible to the agent

The agent's virtual root maps to the workspace's `files/` directory. Prefer
prompts such as:

```text
Read structures/slab.vasp
Write the report to writing/surface_report.md
Analyze calculations/co_adsorption/opt/OUTCAR
```

Do not ask the agent to access arbitrary host paths such as
`/home/user/private/...`. After uploading a file, identify its path relative to
`files/` and use that path in the request.

A useful general layout is:

```text
files/
  literature/
  structures/
  calculations/
  scripts/
  notes/
  writing/
  attachments/
  .deepagents/
```

You do not need to create every directory up front. Preserve a clear existing
layout. Put reusable scripts in `scripts/` and record their date, purpose,
inputs, outputs, and important assumptions.

## 2.5 Thread, turn, and run

These terms are not interchangeable:

| Term | Meaning | Durable |
|---|---|---|
| Workspace | Project data and history boundary | Yes |
| Thread | Continuous conversation and checkpoint | Yes |
| Turn | One user submission and agent response | Yes |
| Run | Execution and observation record for a turn, steering request, or approval resume | Yes |
| Artifact | A file or result object registered in the UI | Yes |
| Receipt | Recoverable identity and state for a remote submission | Yes |

The left rail selects a thread, not a run. Continue in the same thread to retain
its checkpoint. WebUI v2 currently has no historical run selector, thread
branching, retry control, or `resume_selected_run` option.

The thread state determines the next action:

- `idle`, `stopped`, or `error`: send an explicit continuation request.
- `running`: text is queued as `Steer` for the next safe boundary.
- `interrupted`: resume from the approval card inside the message, not with an
  ordinary composer reply.

## 2.6 Artifacts, logs, and evidence

Files, attachments, and remote results can be registered as artifacts and
opened in the right inspector. Tool cards show arguments and result summaries.
Monitor records events, model text, tool results, tokens, cost, and machine time.

Long tool output may appear only as a preview in Chat, with the complete content
written under `files/_tool_outputs/` according to `configs/tool_output.yaml`.
Final conclusions should therefore point to files, logs, receipts, or structures,
not only to a chat summary.

## 2.7 Backup and restore

Back up `files/` and `metadata/` together to restore a workspace completely.
Backing up only `files/` loses threads, checkpoints, approval state, and run
observability.

An account-enabled deployment should also back up:

```text
<PROJECT_SPACE_ROOT>/.webui_auth/auth.sqlite
```

For a consistent snapshot, stop the WebUI or make sure no run is writing.
`files/.deepagents/` may contain staged skills and DPDispatcher receipts. Do not
treat the entire hidden directory as disposable cache.

## 2.8 Execution authority and scientific responsibility

CatMaster routes each entrypoint to a suitable specialist and worker. A
coordinator does not own every scientific tool, and workers can call only tools
in their allowlists. Managed remote jobs are also constrained by task, resource,
machine, and audience declarations.

Those controls reduce misuse but do not prove that a calculation is correct.
The user must still verify:

- System, charge, spin, periodicity, and constraints.
- Potentials, functional, basis, dispersion, and convergence criteria.
- Temperature, ensemble, timestep, sampling length, and random seed.
- Units, energy references, atom mapping, and comparability.
- Software licenses, cluster policy, and compute cost.
