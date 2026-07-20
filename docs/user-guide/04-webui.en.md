# 4. WebUI guide

[Previous](03-llm-configuration.en.md) | [Contents](README.en.md) | [Next](05-agents-and-modules.en.md)

The WebUI v2 path is: sign in, select a workspace, select a thread, set the
entrypoint and permission mode, submit a turn, observe Chat and Monitor, then
continue in the same thread. The left rail is not a historical run selector.

## 4.1 Sign-in and registration

Sign-in is enabled by default. A new user can register after completing the
arithmetic challenge, then enters a private user root. A session normally lasts
14 days. Each account is confined to its own `users/<username>/` tree.

Built-in authentication is suitable for local or protected networks, not as a
complete public identity platform. Registration is open by default, the cookie
does not use the Secure flag, and the service does not terminate TLS. See
[Deployment, operations, and security](10-deployment-operations.en.md) for a
shared deployment.

`--no-login` opens the shared `admin` space and disables Skill Evolution. Use it
only on a trusted machine bound to `127.0.0.1`.

## 4.2 Page layout

The current UI has a left rail, central work area, and right inspector, with four
main views:

| View | Purpose |
|---|---|
| Chat | Conversation, Progress, tool cards, subagent activity, approval cards, artifacts, and remote receipts |
| Monitor | Overview, Live, Events, Raw, and Details observability |
| Skill Evolution | Candidate improvements, Promote, Reject, and Rollback; account mode only |
| Files | Browse, Preview, and Uploads |

The right inspector can keep several file or artifact tabs open and shows a
read-only Todo derived from the current turn's `write_todos` call.

## 4.3 Workspace operations

Select a workspace at the top of the left rail. A new account receives a
`default` workspace.

- Create: use a meaningful short name such as `pt_co_oxidation` or
  `paper_revision_r2`.
- Switch: changing workspace changes the file, thread, and run-history boundary.
- Delete: switch to another workspace first, then type the requested name to
  confirm. This removes project data and internal state.

Do not mix unrelated research projects in one workspace. Do not casually manage
`metadata/` through Files.

## 4.4 Thread operations

Items in the left rail are threads. The current UI supports search and creation,
but not thread rename, deletion, branching, or retry.

Recommended practice:

- Keep one sustained research question in one thread.
- Create another thread for different assumptions, systems, or an independent
  audit trail.
- To continue, select the original workspace and thread, inspect recent messages
  and artifacts, then send an explicit request.
- Do not use a new thread as a continuation unless you restate all required
  context.

## 4.5 Select an entrypoint

The composer offers five entrypoints:

```text
Research
Experiment
Writing
Peer Review
Literature Review
```

The entrypoint cannot change while a run is active. See [Agents and
modules](05-agents-and-modules.en.md) for selection and boundaries. Use
Experiment for a bounded structure or calculation task. Reserve Research for an
open goal spanning literature, computation, and writing.

## 4.6 Auto and Review

The default permission mode is `Auto`. `Review` interrupts before these protected
tools:

```text
write_file
edit_file
remote_submission
remote_submission_batch
```

Review is not a global approval switch for every tool. Reads, searches, some
analysis, and tools absent from the interrupt table can still run automatically.
Review is a sensible choice when first opening a project, before overwriting
files, or before a billable submission.

Permission mode cannot change during a run. On interruption, use the approval
card in the message:

- `Approve`: execute the current action.
- `Reject`: reject it.
- `Respond`: provide more information for the agent.
- `Edit action`: edit the JSON and continue. Keep the same action count and
  retain `name` and `args` for every item.

Resume approvals from the card. The composer may also display `Respond` while
interrupted, but its current implementation performs an ordinary submission and
must not be used as the approval-resume control.

## 4.7 Write an executable request

A useful request states the objective, input, constraints, output, and stop
condition:

```text
Entry: Experiment

Read structures/POSCAR. First check elements, cell, periodicity, and Selective
Dynamics. Generate candidate terminations for a (111) slab while preserving the
constraints, and inspect dangling atoms by coordination number. Write candidates
under structures/slabs/ and the audit to notes/slab_audit.md. Do not submit a
remote calculation in this turn. Stop when I must choose a termination.
```

Give units for numerical values, a seed for stochastic work, workspace-relative
paths for existing files, and explicit permission and resource expectations for
remote submission.

## 4.8 Attachments

Attachments can be added while idle. The attachment button is disabled during a
run.

Files are first stored at the physical workspace path:

```text
files/attachments/<thread_id>/
```

The agent uses the relative path `attachments/<thread_id>/...`. Attachments are
registered as artifacts, and raw base64 is not persisted in message history.

Main limits:

| Layer | Limit |
|---|---|
| Composer browser client | 64 MiB per file |
| Backend storage | 512 MiB per file |
| Current-turn media inline | 32 MiB by default; larger files are stored only |
| Text attachment | First 20,000 characters in the current turn |
| PDF, DOCX, XLSX, PPTX parser | 50 MiB file and 60,000 characters per read |
| PDF and PPTX | 20 pages or slides |
| XLSX | 20,000 rows and 256 columns |

Images are sent according to the model's vision capability. Audio and video are
disabled by default unless the profile enables them. Legacy `.doc`, `.xls`,
`.ppt`, and unknown formats are normally stored without parsing. In Monitor
Events, inspect `multimodal.prepared` for `sent_to_model`, `sent_as`, and
warnings.

## 4.9 Send, Steer, and Stop

While idle, the button reads `Send`; `Ctrl+Enter` submits.

During a run, plain text becomes `Steer`. Steering does not immediately interrupt
the active tool or scientific task. It is queued as the next turn after a safe
boundary. Attachments cannot be added while steering.

The first two Stop requests ask for graceful termination at the next stream
boundary. A third request escalates to emergency cancellation. Stop affects the
local agent turn only. It does not cancel a Slurm or remote Shell job already
submitted. Check the receipt and scheduler separately.

## 4.10 Activity and results in Chat

Chat shows:

- Incremental text and Progress.
- Tool names, arguments, summaries, and errors.
- Specialist or worker delegation activity.
- Written artifacts.
- Remote receipts and status summaries.
- Review approval cards.

Several consecutive activity items may collapse into `Activity`. Do not read
only the last sentence. Expand failed tools, inspect artifact paths, and verify
`status.json`, `stdout.log`, and `stderr.log` for calculations.

## 4.11 Monitor

Monitor refreshes about every five seconds during a run and merges the current
thread's SSE events.

- `Overview`: status, duration, LLM calls, tokens, cost, machine time, and tool
  success or failure counts.
- `Live`: phase, active tool, Todo, subagent, recent model text, and task logs.
- `Events`: filters for thread, run, agent, tool, category, and channel.
- `Raw`: raw chat and log data.
- `Details`: task state and memory.

Monitor currently has no run selector. Its overview query is primarily scoped
by workspace and lane, so it may represent the current or latest run in that
scope rather than exactly the selected thread. For precise tracing, correlate
thread ID, run ID, artifact, and receipt in the events.

## 4.12 Files

Files provides Browse, Preview, and Uploads. It previews text, Markdown, JSON,
images, PDF, CSV/TSV, JSmol structures, trajectories, and OUTCAR vibrations.
Common structure formats include CIF, PDB, XYZ, VASP, POSCAR, CONTCAR, OUTCAR,
XDATCAR, and TRAJ.

Limits and risks:

- Text preview is about 160 KiB, directory preview 40 entries, and one file-tree
  response 500 entries.
- Trajectory preview is limited to 240 frames.
- Files backend upload limit is 512 MiB per file.
- Upload always uses overwrite mode. A same-named file is replaced without a
  second confirmation.
- Delete is permanent and recursive, with only a browser confirmation.
- The tree exposes both `files/` and `metadata/`. Never delete or rewrite
  `metadata/`.
- A directory ZIP can contain at most 20,000 files and 2 GiB total.
- The backend supports safe ZIP extraction, but the current Files UI has no
  extraction switch.

Rename important data uniquely before upload. Keep an external backup before a
bulk move, overwrite, or deletion.

## 4.13 Continue, correct, and audit

A useful continuation in the same thread is:

```text
Continue the previous task. Re-read notes/slab_audit.md and the existing
candidate structures first. List completed work, remaining work, and decisions
I must make. Do not regenerate candidates that already exist.
```

After an error, do not say only `retry`. State which files to preserve, the
failure evidence, what may change, and whether recomputation is forbidden. For a
remote failure, use receipt-driven state auditing from [Remote machines and
execution](08-remote-execution.en.md).

## 4.14 Skill Evolution

In account mode, a terminal run may trigger background candidate generation and
review. The default `observe` mode displays candidates but does not activate them
automatically. Candidates are shared by all threads in the workspace. A promoted
candidate loads from the next run; it can be rejected or rolled back while the
target remains unchanged.

See [Tools, skills, and evolution](09-tools-skills-evolution.en.md) for the safety
model.
