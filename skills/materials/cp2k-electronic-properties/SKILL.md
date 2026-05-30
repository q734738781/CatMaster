---
name: cp2k-electronic-properties
description: Use this skill for CP2K DOS/PDOS, band-style, and population-analysis follow-up planning where parsing is task-specific and should usually be scripted.
allowed-tools: "cp2k_prepare remote_submission remote_submission_batch get_avail_remote_task execute"
---

# cp2k-electronic-properties

## Overview
Use this skill for CP2K electronic-property stages after the structural footing is clear. The preparation can be managed, but the analysis is often task-specific.

## Quick Start
1. Start from an accepted structure or prior CP2K stage.
2. Prepare a property stage with `cp2k_prepare`, usually `recipe="dos"` or `recipe="sp"` with explicit `settings.properties`.
3. Submit with `task_name="cp2k_execute"`.
4. Parse property files with a focused workspace script or a trusted parser.

## Allowed tools
- `cp2k_prepare`
- `remote_submission`
- `remote_submission_batch`
- `get_avail_remote_task`
- `execute`

## Workflow

### 1. Prepare properties explicitly
- Use `settings.properties` to request the property family needed by the task.
- Do not assume DOS, PDOS, band, or population outputs exist unless they were requested in `job.inp`.

### 2. Keep analysis task-specific
- Prefer a short workspace parser for the requested property file rather than a broad bound CP2K analyzer.
- Report the exact files parsed and the assumptions used to align energy windows, spin channels, atom groups, or orbital projections.

## Method-critical defaults
- DOS/PDOS settings depend on the scientific question; do not present a generic grid as converged evidence.
- For charge or population analysis, state which CP2K property output was requested and parsed.

## Output Contract
Return:
- property-stage path
- submitted receipt/context if executed
- parser script path when created
- output JSON/CSV/plot paths
- limitations of the property extraction

## References
- Local source note: `references/cp2k_properties_reference.md`
- CP2K FORCE_EVAL/PROPERTIES: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/PROPERTIES.html
- CP2K BANDSTRUCTURE: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/PROPERTIES/BANDSTRUCTURE.html
- CP2K DOS: https://manual.cp2k.org/cp2k-2024_1-branch/CP2K_INPUT/FORCE_EVAL/PROPERTIES/BANDSTRUCTURE/DOS.html
