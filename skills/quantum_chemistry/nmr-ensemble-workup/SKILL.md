---
name: nmr-ensemble-workup
description: Use this skill for a bounded flexible-molecule NMR workflow that requires conformer generation, xTB cleanup, ORCA NMR execution, and evidence handoff for later Boltzmann aggregation.
---

# nmr-ensemble-workup

## Overview
Use this skill when the task is one ensemble-aware molecular NMR episode.

## Quick Start
1. Generate or collect the conformer ensemble.
2. Use xTB to optimize and prune the ensemble before ORCA.
3. Extract the accepted structures into a clean directory.
4. Prepare ORCA with `task="nmr"`.
5. Submit with `remote_submission` or `remote_submission_batch` using `task_name="orca_execute"` and summarize with `analyze_orca_results`.

## Allowed tools
- `enumerate_molecular_conformers`
- `filter_conformer_ensemble`
- `remote_submission`
- `remote_submission_batch`
- `analyze_xtb_results`
- `extract_optimized_molecules`
- `orca_prepare`
- `analyze_orca_results`

## Output Contract
Return:
- retained conformer directory
- ORCA NMR batch root
- ORCA summary path
