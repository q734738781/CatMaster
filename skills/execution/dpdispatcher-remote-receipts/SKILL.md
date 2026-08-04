---
name: dpdispatcher-remote-receipts
description: Use only after a DPDispatcher-backed managed execution tool returns a failure with receipt/context fields, an ambiguous transport error, or evidence of a possible orphan job. Do not use for a pending synchronous call or ordinary success.
license: project-local
allowed-tools: "execute"
---

# dpdispatcher-remote-receipts

## Overview

Account for possibly live remote work after a returned managed-execution failure, then recover without duplicate submissions.
This is a narrow operational-recovery exception: receipt IDs and submission hashes may be inspected here to preserve job identity, but they are not scientific QC and must not be copied into scientific Results or ordinary successful-run reports.

## Quick Start

1. While a managed submission call is pending, wait; it blocks until terminal status.
2. On ordinary success, use the returned outputs and do not inspect its receipt.
3. After a returned failure with ambiguous remote state, preserve the context, inspect once, and perform at most one justified recovery.

## Workflow

### 1. Enforce the trigger boundary

- Never infer call health, poll a receipt, or start another submission while `remote_submission` or `remote_submission_batch` is pending.
- Trigger recovery only after the tool returns a failure or an ambiguous transport/download result. Preserve `remote_context_id`, `submission_hash`, and `receipt_rel`.
- If `submission_hash` is empty, no DPDispatcher record exists to inspect or recover. One fresh retry is allowed only for a likely pre-submission transient failure.

### 2. Inspect and recover once

Read the reported `receipt_rel`; when a DPDispatcher handle exists, use only the action required by the failure:

```bash
python -c 'import json, pathlib; p=pathlib.Path(".deepagents/dpdispatcher/receipts/<remote_context_id>.json"); print(json.dumps(json.loads(p.read_text()), indent=2, ensure_ascii=False))'
dpdisp submission <submission_hash> --download-finished-task
dpdisp submission <submission_hash> --download-terminated-log
dpdisp submission <submission_hash> --reset-fail-count
dpdisp submission <submission_hash> --clean
```

- Use download actions before reruns when the failure is a transport/download error and remote jobs may already have finished.
- Use `--reset-fail-count` only when the DPDispatcher record is valid and retrying the same submission state is intentional.
- Use `--clean` only after outputs/logs are collected or cleanup was requested.
- Treat the receipt as a snapshot; if live status matters, verify through the known remote scheduler context.

### 3. Keep recovery bounded

- Before rerunning the managed tool, account for the old submission as collected, still running, cleaned, not-yet-created, or intentionally abandoned before a recoverable handle existed.
- Allow at most one automatic fresh retry per stage after a transient connection, upload, or download failure.
- On repeated or changed failure, stop and report the evidence.

## Method-critical defaults
- Do not invent new status names. Use DPDispatcher status names as exposed in the receipt.
- Do not parse tool-level batch tasks from the receipt. The receipt is submission/job context only.
- Do not assume network exceptions cancel remote jobs. A connection reset can leave remote work alive.

## Output Contract
Return:
- `remote_context_id`
- `submission_hash`
- `receipt_rel` or receipt file path
- observed remote state and recovery action, if any
- both old and retry context identifiers when a retry was launched
- existing output/log files, separate from expected-but-missing outputs

## References
- Use the original managed execution tool output/error as the primary source for the context fields.
- Use the receipt only to preserve and inspect submission/job context; scientific results still come from the normal output and analysis artifacts.
