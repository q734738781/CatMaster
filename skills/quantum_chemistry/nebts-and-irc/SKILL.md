---
name: nebts-and-irc
description: Use this skill for bounded molecular NEB-TS and IRC workflows in ORCA when the task already has explicit reactant/product or TS-side starting structures.
---

# nebts-and-irc

## Overview
Use this skill when the task is one ORCA pathway-validation episode rooted in explicit endpoints or a TS guess.

## Quick Start
1. Use `orca_nebts_prepare` when reactant and product endpoints are available.
2. Submit with `orca_execute_batch`.
3. Use `analyze_orca_results` to check whether the NEB-TS stage produced an acceptable TS candidate.
4. Use `orca_irc_prepare` on the accepted TS structure when the task needs forward/backward path validation.
5. Submit the IRC stage and summarize it again with `analyze_orca_results`.

## Allowed tools
- `orca_nebts_prepare`
- `orca_irc_prepare`
- `orca_execute_batch`
- `analyze_orca_results`

## Output Contract
Return:
- NEB-TS run directory
- IRC run directory when launched
- ORCA summary path(s)

