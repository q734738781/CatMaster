---
name: xtb-screen-and-prune
description: Use this skill for bounded xTB screening, optimization, Hessian checking, and ensemble pruning when the main objective is to cheaply rank or clean up molecular candidates before higher-level calculations.
allowed-tools: "xtb_prepare remote_submission remote_submission_batch analyze_xtb_results filter_conformer_ensemble extract_optimized_molecules"
---

# xtb-screen-and-prune

## Overview
Use this skill when xTB is the main execution layer for one bounded molecular screening episode.

## Quick Start
1. Call `xtb_prepare` with the cheapest mode that can answer the current decision.
2. Submit its prepared stage with `task_name="xtb_execute"` and no scientific `template_overrides`.
3. Use `analyze_xtb_results` immediately after collection.
4. Prune or extract the accepted structures before handing them downstream.

## Allowed tools
- `xtb_prepare`
- `remote_submission`
- `remote_submission_batch`
- `analyze_xtb_results`
- `filter_conformer_ensemble`
- `extract_optimized_molecules`

## Workflow

### 1. Materialize the calculation contract
- Put mode, GFN family, charge, unpaired-electron count, solvation, and optimization level in `xtb_prepare`.
- Use typed constraint fields for common fixes or internal-coordinate constraints. Use `xcontrol_path` when a complete custom xTB detailed input already exists.
- Batch preparation applies one constraint set to every structure, so atom indices and topology must be compatible across the batch.

### 2. Submit only complete stages
- One prepared stage goes to `remote_submission`; a parent containing two or more first-level prepared stages goes to `remote_submission_batch`.
- Use `task_name="xtb_execute"` without mode, method, solvent, or constraint overrides.

### 3. Analyze and hand off
- Run `analyze_xtb_results` on the collected stage or batch root.
- Preserve `manifest.json` and `xtb.inp` with the result so the executed setup remains auditable.

## Method-critical defaults
- Use `mode="sp"` only for ranking when geometry quality is already acceptable.
- Use `mode="opt"` before comparing structures whose geometries may differ materially.
- Use `mode="hess"` when you need a quick imaginary-frequency sanity check on a candidate TS or minimum.
- Keep one explicit charge and spin assignment across the whole screen.
- Keep identical GFN, solvation, charge, and unpaired-electron settings across structures being ranked.
- `$fix` is not valid as an MD restraint; use `$constrain` or a reviewed custom `xtb.inp` for MD.

## Output Contract
Return:
- prepared stage root and prepare manifest
- xTB result root
- xTB summary path
- retained structure directory if pruning was applied

## References
- `/.deepagents/skills/execution/remote-stage-layouts/SKILL.md#xtb_execute`
