---
name: scan-to-ts
description: Use this skill for a bounded molecular reaction-coordinate workflow that starts from a relaxed scan, identifies a TS-side guess, and refines it with ORCA OptTS.
---

# scan-to-ts

## Overview
Use this skill when the task is one scan-to-TS episode on a molecular reaction coordinate.

## Quick Start
1. Build the relaxed scan with `orca_scan_prepare`.
2. Submit the scan stage with `remote_submission` or `remote_submission_batch` using `task_name="orca_execute"`.
3. Inspect the returned profile and structures with `analyze_orca_results`.
4. Promote one TS-side guess into `orca_optts_prepare`.
5. Submit the OptTS refinement and summarize it again with `analyze_orca_results`.

## Allowed tools
- `orca_scan_prepare`
- `orca_optts_prepare`
- `remote_submission`
- `remote_submission_batch`
- `analyze_orca_results`

## Output Contract
Return:
- scan run directory
- OptTS run directory when launched
- ORCA summary path(s)
