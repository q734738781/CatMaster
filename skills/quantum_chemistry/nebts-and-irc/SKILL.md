---
name: nebts-and-irc
description: Use this skill for bounded molecular NEB-TS and IRC workflows in ORCA when the task already has explicit reactant/product or TS-side starting structures.
---

# nebts-and-irc

## Overview
Use this skill when the task is one ORCA pathway-validation episode rooted in explicit endpoints or a TS guess.

## Quick Start
1. Use `orca_nebts_prepare` when reactant and product endpoints are available.
2. Submit with `remote_submission` or `remote_submission_batch` using `task_name="orca_execute"`.
3. Use `analyze_orca_results` to check whether the NEB-TS stage produced an acceptable TS candidate.
4. Use `orca_irc_prepare` on the accepted TS structure when the task needs forward/backward path validation.
5. Submit the IRC stage and summarize it again with `analyze_orca_results`.

## Allowed tools
- `orca_nebts_prepare`
- `orca_irc_prepare`
- `remote_submission`
- `remote_submission_batch`
- `analyze_orca_results`

## Method-critical defaults
- Parameter priority: honor explicit user requirements first; otherwise choose ORCA pathway settings from the molecule class and reaction-path objective; if that judgment remains uncertain, run a narrow literature or official documentation check before finalizing the override.
- Do not add ORCA overrides just to restate the tool baseline; only override when the user, molecule class, task objective, or a checked source justifies it.
- The `orca_nebts_prepare` default `XTB2` level is a cheap pathway-search default, not a final quantum-chemistry validation level.
- For accepted TS candidates and IRC validation, prefer an ORCA DFT refinement level comparable to the surrounding opt/freq workflow; `r2SCAN-3c` is a reasonable default structure/frequency layer unless the system needs benchmarking.
- For final reaction barriers, add a higher-level hybrid/TZ-or-larger single-point stage on accepted endpoints and TS structures unless the user or checked source says the structure/frequency level is sufficient.
- Keep NEB-TS, TS refinement, frequency validation, IRC, and final single-point stages traceable to the chosen method/basis/solvation/spin level; do not mix levels silently.

## Output Contract
Return:
- NEB-TS run directory
- IRC run directory when launched
- ORCA summary path(s)
