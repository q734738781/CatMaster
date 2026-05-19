---
name: xtb-screen-and-prune
description: Use this skill for bounded xTB screening, optimization, Hessian checking, and ensemble pruning when the main objective is to cheaply rank or clean up molecular candidates before higher-level calculations.
---

# xtb-screen-and-prune

## Overview
Use this skill when xTB is the main execution layer for one bounded molecular screening episode.

## Quick Start
1. Run `xtb_run_batch` in the cheapest mode that can answer the current decision.
2. Use `analyze_xtb_results` immediately after collection.
3. If the task is ensemble ranking, prune with `filter_conformer_ensemble`.
4. Extract cleaned structures with `extract_optimized_molecules` before handing off downstream.

## Allowed tools
- `xtb_run_batch`
- `analyze_xtb_results`
- `filter_conformer_ensemble`
- `extract_optimized_molecules`

## Method-critical defaults
- Use `mode="sp"` only for ranking when geometry quality is already acceptable.
- Use `mode="opt"` before comparing structures whose geometries may differ materially.
- Use `mode="hess"` when you need a quick imaginary-frequency sanity check on a candidate TS or minimum.
- Keep one explicit charge and spin assignment across the whole screen.

## Output Contract
Return:
- xTB result root
- xTB summary path
- retained structure directory if pruning was applied

