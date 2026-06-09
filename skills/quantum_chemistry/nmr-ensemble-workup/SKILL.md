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

## Method-critical defaults
- Parameter priority: honor explicit user requirements first; otherwise choose ORCA NMR settings from the molecule class and spectral objective; if that judgment remains uncertain, run a narrow literature or official documentation check before finalizing the override.
- Do not add ORCA overrides just to restate the tool baseline; only override when the user, molecule class, task objective, or a checked source justifies it.
- The `orca_prepare` auto level for `task="nmr"` resolves to `WB97X-D4/def2-TZVP`; override to an NMR-oriented basis, solvent model, or functional when the spectrum target or checked source justifies it.
- Prefer xTB/CREST followed by `r2SCAN-3c` for conformer cleanup or structure refinement before NMR unless the user provides a different ensemble protocol.
- Keep conformer pruning, ORCA NMR preparation, and later Boltzmann aggregation traceable to the selected charge, multiplicity, method, basis, solvation model, and retained conformer set.

## Output Contract
Return:
- retained conformer directory
- ORCA NMR batch root
- ORCA summary path
