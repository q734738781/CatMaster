---
name: cp2k-vibrational-analysis
description: Use this skill for CP2K vibrational-analysis preparation and task-specific thermochemistry parsing after an accepted stationary point.
allowed-tools: "cp2k_prepare remote_submission remote_submission_batch get_avail_remote_task execute"
---

# cp2k-vibrational-analysis

## Overview
Use this skill when a CP2K frequency or vibrational-analysis stage is requested after a stationary structure has been accepted.

## Quick Start
1. Confirm the input structure is the intended stationary point.
2. Prepare the stage with `cp2k_prepare(recipe="freq")`.
3. Submit with `remote_submission(task_name="cp2k_execute")`.
4. Parse vibrational outputs with a focused workspace script; do not use a broad generic CP2K analyzer.

## Allowed tools
- `cp2k_prepare`
- `remote_submission`
- `remote_submission_batch`
- `get_avail_remote_task`
- `execute`

## Workflow

### 1. Start from accepted geometry
- Do not launch vibrational analysis from an unconverged or unreviewed geometry.
- Keep the electronic settings comparable to the energy/optimization stage unless the task explicitly changes them.

### 2. Prepare and submit
- Use the generic `cp2k_execute` task.
- Keep the prepared stage separate from the geometry optimization stage so files and parser outputs are traceable.

### 3. Parse narrowly
- Write a focused parser for frequencies and thermochemistry fields needed by the current task.
- Report imaginary-frequency count and the exact output file parsed.

## Method-critical defaults
- Parameter priority: honor explicit user requirements first; otherwise choose CP2K vibrational-stage overrides from the system class and frequency/thermochemistry objective; if that judgment remains uncertain, run a narrow literature or official documentation check before finalizing the override.
- Do not add CP2K `settings` overrides just to restate the tool baseline; only override when the user, system class, task objective, or a checked source justifies it.
- Vibrational analysis is a validation/post-processing stage, not a substitute for geometry acceptance.
- For surfaces or constrained systems, state the atom set and constraint assumptions used in the model.

## Output Contract
Return:
- vibrational stage path
- parser script path if created
- frequency/thermochemistry artifact paths
- limitations or imaginary-mode warnings

Keep receipt and platform fields in runtime records unless failure recovery needs them; provide them whenever the user explicitly asks to inspect, compare, record, or report them.

## References
- Local source note: `references/cp2k_vibrational_reference.md`
- CP2K VIBRATIONAL_ANALYSIS: https://manual.cp2k.org/cp2k-2024_1-branch/CP2K_INPUT/VIBRATIONAL_ANALYSIS.html
- CP2K manual index: https://manual.cp2k.org/
