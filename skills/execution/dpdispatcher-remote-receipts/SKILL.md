---
name: dpdispatcher-remote-receipts
description: Use this skill when a DPDispatcher-backed managed execution tool reports remote_context_id, submission_hash, receipt_rel, DPDispatcherDispatchError, network transport failures, or possible orphan remote jobs; it guides minimal receipt inspection and controlled recovery through execute without registering a new control tool.
license: project-local
compatibility: local
allowed-tools: "execute"
---

# dpdispatcher-remote-receipts

## Overview
Use this skill to keep DPDispatcher remote job context visible after a managed execution failure and support bounded automatic recovery without creating untracked duplicate remote work.

## Quick Start
1. If a tool failure includes `remote_context_id`, `submission_hash`, or `receipt_rel`, preserve that context before deciding whether to retry.
2. Read the receipt from the reported `receipt_rel`, normally `.deepagents/dpdispatcher/receipts/<remote_context_id>.json` inside the DeepAgent workspace.
3. Bounded automatic recovery is allowed when it helps produce the requested result and the previous attempt is accounted for.
4. Use `execute` only for focused DPDispatcher or scheduler inspection, download, reset, or cleanup.
5. Report `remote_context_id`, `submission_hash`, the receipt path, and the action taken.

## Workflow

### 1. Preserve context before acting
- Treat `remote_context_id` as the stable CatMaster-side handle for this submission episode.
- Treat `submission_hash` as the DPDispatcher handle to use with `dpdisp submission ...` commands.
- If `submission_hash` is empty, there is no DPDispatcher handle to reset, download, or clean. For likely transient pre-submission transport failures, one bounded fresh submission is acceptable when the user asked for a result and the stage is still valid; record both contexts. If the same pre-submission failure repeats, stop and report evidence.
- Treat `jobs` in the receipt or failure artifact as DPDispatcher job-level evidence, not as tool-level task parsing.
- Normal successful batch returns may expose only the compact context; failures may expose `jobs` and `job_status_counts`.
- On successful submissions, do not expand per-job IDs in the final report unless the user asked for scheduler handles or you are doing failure recovery, cancellation, or cleanup.

### 2. Read the receipt before acting
Use `execute` to print the receipt JSON from the workspace-visible path. `execute` runs from the project files root, so use `receipt_rel` directly:

```bash
python -c 'import json, pathlib; p=pathlib.Path(".deepagents/dpdispatcher/receipts/<remote_context_id>.json"); print(json.dumps(json.loads(p.read_text()), indent=2, ensure_ascii=False))'
```

The receipt is a snapshot written by CatMaster. If live status matters, verify it through DPDispatcher or the scheduler before deciding that a job is gone.

### 3. Prefer recovery before fresh resubmission
Use the reported `submission_hash` with DPDispatcher CLI actions when they fit the failure:

```bash
dpdisp submission <submission_hash> --download-finished-task
dpdisp submission <submission_hash> --download-terminated-log
dpdisp submission <submission_hash> --reset-fail-count
dpdisp submission <submission_hash> --clean
```

- Use download actions before reruns when the failure is a transport/download error and remote jobs may already have finished.
- Use `--reset-fail-count` only when the DPDispatcher record is valid and retrying the same submission state is intentional.
- Use `--clean` only after outputs/logs have been collected or the user intends to cancel/cleanup that submission.
- If the receipt includes remote `job_id` values and the scheduler is known, query it through the remote execution context, for example `squeue -j <job_id>` on Slurm-capable shells. Do not assume the local workstation has the remote scheduler.

### 4. Keep recovery bounded
- Before rerunning the managed tool, account for the old submission as collected, still running, cleaned, not-yet-created, or intentionally abandoned before a recoverable handle existed.
- Prefer at most one automatic fresh retry per distinct stage after a transient connection, upload, or download failure.
- If the retry reaches a different failure mode, inspect/report that new evidence instead of escalating into repeated submissions.
- If the user's goal is explicitly a smoke test or proof of environment health, report environment instability rather than hiding it behind many retries.

## Method-critical defaults
- Do not invent new status names. Use DPDispatcher status names as exposed in the receipt.
- Do not parse tool-level batch tasks from the receipt. The receipt is submission/job context only.
- Do not assume network exceptions cancel remote jobs. A connection reset can leave remote work alive.
- Output-oriented tasks may tolerate transient recovery and performance downgrade, but the final report should still distinguish the original failure from the successful retry.

## Output Contract
Return:
- `remote_context_id`
- `submission_hash`
- `receipt_rel` or receipt file path
- whether jobs looked active, finished, terminated, unknown, or not verifiable from available evidence
- download, reset, cleanup, or retry action taken, if any
- if a retry was launched, both the previous and retry `remote_context_id`/`submission_hash` when available
- existing output/log files separately from expected-but-missing outputs; do not list missing outputs as artifacts after an early submission or transport failure

## References
- Use the original managed execution tool output/error as the primary source for the context fields.
- Use the receipt only to preserve and inspect submission/job context; scientific results still come from the normal output and analysis artifacts.
