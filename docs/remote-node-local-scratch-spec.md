# Remote node-local scratch execution

## Scope

Selected I/O-intensive scientific boot scripts may execute a prepared remote
stage in compute-node-local scratch while keeping the DPDispatcher task
directory on shared storage as the stable transfer and recovery location.

This behavior is currently limited to registered ORCA and CP2K execution. It
does not change DPDispatcher's `remote_root`, task upload/download paths, task
status wrapper, or the scientific input prepared by the agent.

## Runtime decision

The boot script makes the decision after Slurm starts the task, when allocation
environment variables and compute-node filesystems are available.

Node-local execution is used only when all of the following are true:

- `SLURM_NNODES` or `SLURM_JOB_NUM_NODES` is present and resolves to exactly
  one allocated node;
- a candidate scratch root exists and is writable; and
- the candidate scratch root is on a different filesystem from the shared task
  stage.

An allocation with more than one node, missing or invalid node-count metadata,
or no usable local filesystem runs in the shared task stage without copying.
The scratch root is selected from the first usable value in this order:

1. `CATMASTER_SCRATCH_ROOT`;
2. `SLURM_TMPDIR`;
3. `TMPDIR`;
4. `/tmp`.

The runtime does not benchmark storage or infer node locality from a hostname.
Deployment may set `CATMASTER_SCRATCH_ROOT` when the site's local scratch path
is not one of the standard locations.

## Staging contract

For an eligible single-node task, the boot layer:

1. creates a unique directory below the selected scratch root;
2. copies the complete prepared stage into it;
3. runs the existing scientific boot logic from that directory;
4. copies all resulting scientific files back to the shared stage on normal
   completion, scientific-program failure, or a catchable Python exception;
5. removes only the scratch directory it created, and only after successful
   copy-back.

Copying never uses delete semantics and does not filter unknown scientific
outputs. CatMaster and DPDispatcher wrapper bookkeeping at the task root,
including `status.json`, `stdout.log`, `stderr.log`, `log`, and `err`, remains
in the shared stage and is not copied back over active logs.

ORCA's redirected standard output, `job.out`, is also written directly to the
stable shared stage while the ORCA subprocess runs from node-local scratch.
This keeps the main calculation log visible during a running job without moving
ORCA's high-volume temporary I/O back to shared storage. Other ORCA files remain
in scratch until copy-back. A stale scratch copy of `job.out` is removed before
launch so final stage-out cannot overwrite the live shared log. Visibility from
another node may lag behind a write because of ORCA buffering or the site's
shared-filesystem attribute and data caches.

The helper prefers `rsync -a` when available and otherwise performs an
equivalent recursive Python copy. A failure to create or populate scratch falls
back to shared-stage execution. A copy-back failure is an execution failure and
the scratch directory is retained for best-effort recovery.

## Result and failure semantics

Scientific program return codes retain their existing meaning. Boot summaries
and all paths intended for later analysis refer to the stable shared stage, not
to a temporary path that disappears after the job.

Normal program failure still triggers copy-back so partial outputs remain
reachable. A node loss, power loss, `SIGKILL`, scheduler cleanup without a
catchable termination window, or destruction of node-local storage cannot
recover files that exist only in scratch. ORCA's shared `job.out` remains
available up to the last output committed to shared storage. Periodic
checkpoint synchronization is not part of this contract.

## Engine boundary

- ORCA uses node-local execution for eligible single-node Slurm allocations.
- CP2K uses node-local execution only for eligible single-node allocations;
  multi-node MPI execution remains in the shared stage.
- xTB, CREST, VASP, LAMMPS, and MLFF tasks retain their existing work-directory
  behavior until separately enabled from representative runtime evidence.

The shared scratch implementation is a staged dependency of the selected boot
scripts. It is not an agent-facing task parameter and does not add another
scientific prepare or submission decision.

## Acceptance behavior

- A one-node Slurm environment with a distinct writable scratch filesystem runs
  the scientific subprocess under the created scratch directory and returns
  outputs to the shared stage.
- A multi-node or unknown allocation runs directly in the shared stage.
- Concurrent tasks create distinct scratch directories.
- Active task-wrapper logs are not overwritten during copy-back.
- Scientific failures retain partial results and their original return code.
- A running ORCA task exposes its flushed `job.out` in the shared stage.
- Copy-back failure cannot be reported as successful execution.
- Successful copy-back leaves no CatMaster-created scratch directory.
