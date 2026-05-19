---
name: dpdispatcher-remote-receipts
description: Use this skill when a DPDispatcher-backed managed execution tool reports remote_context_id, submission_hash, receipt_rel, DPDispatcherDispatchError, network transport failures, or possible orphan remote jobs; it guides minimal receipt inspection and controlled recovery through execute without registering a new control tool.
license: project-local
compatibility: local
allowed-tools: "execute"
---

# dpdispatcher-remote-receipts

## Overview
Use this skill to keep DPDispatcher remote job context visible after a managed execution failure and avoid blind resubmission.

## Quick Start
1. If a tool failure includes `remote_context_id`, `submission_hash`, or `receipt_rel`, do not immediately call the same submission tool again.
2. Read the receipt from the reported `receipt_rel`, normally `.deepagents/dpdispatcher/receipts/<remote_context_id>.json` inside the DeepAgent workspace.
3. Use `execute` only for focused DPDispatcher or scheduler inspection, download, reset, or cleanup.
4. Report `remote_context_id`, `submission_hash`, the receipt path, and the action taken.

## Suggested tools
- `execute`

## Workflow

### 1. Preserve the context
- Treat `remote_context_id` as the stable CatMaster-side handle for this submission episode.
- Treat `submission_hash` as the DPDispatcher handle to use with `dpdisp submission ...` commands.
- Treat `jobs` in the receipt or failure artifact as DPDispatcher job-level evidence, not as tool-level task parsing.
- Normal successful batch returns may expose only the compact context; failures may expose `jobs` and `job_status_counts`.

### 2. Read the receipt before acting
Use `execute` to print the receipt JSON from the workspace-visible path. `execute` runs from the project files root, so use `receipt_rel` directly:

```bash
python -c 'import json, pathlib; p=pathlib.Path(".deepagents/dpdispatcher/receipts/<remote_context_id>.json"); print(json.dumps(json.loads(p.read_text()), indent=2, ensure_ascii=False))'
```

The receipt is a snapshot written by CatMaster. If live status matters, verify it through DPDispatcher or the scheduler before deciding that a job is gone.

### 3. Prefer control over resubmission
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

### 4. Inspect scheduler state only when needed
If the receipt includes remote `job_id` values and the machine scheduler is known, use a targeted scheduler query through `execute`, such as `squeue -j <job_id>` on Slurm-capable shells. Do not infer that a local workstation has the remote scheduler available; use this only when the execution environment makes it meaningful.

### 5. Retry only after accounting for the old submission
Before rerunning the original managed tool, decide what happened to the previous `submission_hash`: collected, still running, cleaned, or intentionally abandoned. Mention that decision in the handoff so the next agent loop does not create duplicate remote work.

## Method-critical defaults
- Do not invent new status names. Use DPDispatcher status names as exposed in the receipt.
- Do not parse tool-level batch tasks from the receipt. The receipt is submission/job context only.
- Do not assume network exceptions cancel remote jobs. A connection reset can leave remote work alive.

## Output Contract
Return:
- `remote_context_id`
- `submission_hash`
- `receipt_rel` or receipt file path
- whether jobs looked active, finished, terminated, unknown, or not verifiable from available evidence
- download, reset, cleanup, or retry action taken, if any

## References
- Use the original managed execution tool output/error as the primary source for the context fields.
- Use the receipt only to preserve and inspect submission/job context; scientific results still come from the normal output and analysis artifacts.
