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

## Method-critical defaults
- Parameter priority: honor explicit user requirements first; otherwise choose ORCA scan and OptTS settings from the molecule class and reaction-coordinate objective; if that judgment remains uncertain, run a narrow literature or official documentation check before finalizing the override.
- Do not add ORCA overrides just to restate the tool baseline; only override when the user, molecule class, task objective, or a checked source justifies it.
- The `orca_scan_prepare` and `orca_optts_prepare` auto level resolves to `r2SCAN-3c`, which is intended as a practical scan/geometry/TS-search layer.
- For final reaction barriers, refine accepted TS candidates and endpoints with a higher-level hybrid/TZ-or-larger single-point stage unless the user or checked source says the scan/search level is sufficient.
- Keep the relaxed scan, OptTS refinement, frequency validation, and final single-point stages traceable to the chosen method/basis/solvation/spin level; do not mix levels silently.

## Output Contract
Return:
- scan run directory
- OptTS run directory when launched
- ORCA summary path(s)
