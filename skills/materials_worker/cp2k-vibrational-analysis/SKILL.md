---
name: cp2k-vibrational-analysis
description: Use this skill to author and analyze a CP2K vibrational calculation for an accepted stationary structure.
allowed-tools: "cp2k_prepare cp2k_output_summary remote_submission remote_submission_batch get_avail_remote_task execute"
---

# CP2K vibrational analysis

## Workflow

1. Confirm the intended stationary structure, constraints, electronic state, and comparable electronic settings.
2. Author a complete CP2K input with `RUN_TYPE VIBRATIONAL_ANALYSIS` and the required `VIBRATIONAL_ANALYSIS` and print sections.
3. Stage it unchanged with `cp2k_prepare`, including every referenced asset, and submit with `cp2k_execute`.
4. Use `cp2k_output_summary` for formal completion and parsed `cm^-1` frequencies; use a focused parser for thermochemistry or mode files not covered by the summary.

## Scientific choices

- A completed optimization does not imply that frequencies were calculated.
- Missing or unparsed frequencies are unknown, not zero imaginary modes.
- Interpret imaginary modes against the intended minimum or transition state and the actual constrained subspace.

## Handoff

Return the vibrational stage, frequency state and values, imaginary-mode count when calculated, and any thermochemistry or mode artifacts.

## References

- [CP2K VIBRATIONAL_ANALYSIS](https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/VIBRATIONAL_ANALYSIS.html)
