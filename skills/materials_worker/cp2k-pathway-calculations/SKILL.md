---
name: cp2k-pathway-calculations
description: Use this skill to author complete native CP2K BAND/NEB or transition-state refinement inputs from an existing mapped path or TS guess.
allowed-tools: "cp2k_prepare cp2k_output_summary remote_submission remote_submission_batch get_avail_remote_task execute"
---

# CP2K pathway calculations

## Workflow

1. Verify that all path images have identical atom count and per-index element ordering, or identify the intended TS guess and dimer vector.
2. Author the complete native `job.inp`, including the intended `MOTION/BAND` or `MOTION/GEO_OPT/TRANSITION_STATE` sections and explicit replica file references.
3. Stage `job.inp` and every replica/vector/restart asset with `cp2k_prepare`; it does not interpolate endpoints or choose pathway settings.
4. Submit with `cp2k_execute`. Parse image energies, convergence, endpoint behavior, and barrier evidence with a focused script.

## Scientific choices

- Make image count, band type, spring/optimizer settings, endpoint treatment, and convergence thresholds explicit.
- Preparation alone is not barrier evidence. Report the actual converged image energies and endpoint reference used.
- Reject an atom-order mismatch before staging because it changes the physical path.

## Handoff

Return the mapped source path or TS guess, prepared stage, and pathway analysis artifacts.

## References

- [CP2K NEB exercise](https://www.cp2k.org/exercises%3Acommon%3Aneb)
- [MOTION/BAND](https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/BAND.html)
- [Transition-state input](https://manual.cp2k.org/cp2k-2025_1-branch/CP2K_INPUT/MOTION/GEO_OPT/TRANSITION_STATE.html)
