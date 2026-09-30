---
name: cp2k-aimd-restart
description: Use this skill to continue CP2K AIMD from an explicitly selected restart while preserving the scientific trajectory lineage.
allowed-tools: "cp2k_prepare cp2k_output_summary md_trajectory_summary remote_submission remote_submission_batch get_avail_remote_task execute"
---

# CP2K AIMD restart

## Workflow

1. Inspect the prior scientific result and select the intended CP2K restart, structure, wavefunction, and any PLUMED state explicitly.
2. Author a new complete `job.inp` with the intended `EXT_RESTART` references and any deliberate changes to ensemble, timestep, thermostat/barostat, or output strides.
3. Stage the input and selected restart assets into a fresh directory with `cp2k_prepare`.
4. Submit the new stage with `cp2k_execute`; summarize the continuation with `cp2k_output_summary` and `md_trajectory_summary`.

## Scientific choices

- Do not select a file merely because it is newest when the user or trajectory lineage identifies another restart.
- Preserve velocities and other restart state when continuity is required; state any intentionally reset state.
- Keep old outputs separate from the continuation stage.

## Handoff

Return the source scientific result, selected restart assets, new stage, intentionally changed simulation settings, and continuation summaries.

## References

- [CP2K EXT_RESTART](https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/EXT_RESTART.html)
- [CP2K restart printing](https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/PRINT/RESTART.html)
