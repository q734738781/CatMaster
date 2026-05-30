---
name: cp2k-run-analysis
description: Use this skill for generic CP2K run-health analysis after cp2k_execute, without replacing property-specific parsers.
allowed-tools: "cp2k_output_summary md_trajectory_summary analyze_trajectory execute"
---

# cp2k-run-analysis

## Overview
Use this skill for reusable CP2K evidence: completion, wrapper status, output files, energies, SCF markers, optimization/frequency markers, and AIMD energy/trajectory health.

## Quick Start
1. Run `cp2k_output_summary` on the CP2K result directory or batch root.
2. For AIMD, run `md_trajectory_summary` on the result directory.
3. For DOS, charges, band structures, or custom properties, write a focused parser for the exact file requested.

## Allowed tools
- `cp2k_output_summary`
- `md_trajectory_summary`
- `analyze_trajectory`
- `execute`

## Workflow

### 1. Separate run health from scientific interpretation
- `cp2k_output_summary` can report reusable run evidence.
- It must not be treated as a DOS, charge, band, PLUMED, or reaction-barrier analyzer.

### 2. Inspect common artifacts
- Check `cp2k_summary.json`, `status.json`, `job.out`, CP2K `.ener` files, trajectories, restart files, and generated property files.
- Report exact files parsed.

### 3. Escalate to focused scripts when needed
- Use focused scripts for property files whose layout depends on the requested CP2K section.
- Keep parser outputs under `scripts/` or an analysis directory and report their paths.

## Method-critical defaults
- Completion and energy extraction are not proof of convergence or scientific validity.
- Frequency, DOS, charge, and free-energy outputs require task-specific checks.

## Output Contract
Return:
- CP2K output summary JSON path
- trajectory summary path when relevant
- exact property parser path if created
- key warnings/errors and unsupported parser scope

## References
- Local source note: `references/cp2k_run_analysis_reference.md`
- CP2K manual index: https://manual.cp2k.org/
- CP2K GLOBAL/RUN_TYPE: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/GLOBAL.html
- CP2K FORCE_EVAL/PROPERTIES: https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/PROPERTIES.html
- CP2K VIBRATIONAL_ANALYSIS: https://manual.cp2k.org/cp2k-2024_1-branch/CP2K_INPUT/VIBRATIONAL_ANALYSIS.html
- CP2K MOTION/MD: https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/MD.html
