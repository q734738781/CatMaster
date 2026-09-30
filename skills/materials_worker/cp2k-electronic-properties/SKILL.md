---
name: cp2k-electronic-properties
description: Use this skill to author CP2K DOS, PDOS, band-structure, and population-analysis follow-up inputs and parse their task-specific outputs.
allowed-tools: "cp2k_prepare cp2k_output_summary remote_submission remote_submission_batch get_avail_remote_task execute"
---

# CP2K electronic properties

## Workflow

1. Start from the accepted structure and electronic state.
2. Author a complete `job.inp` containing the exact `FORCE_EVAL/PROPERTIES` and print sections needed for the requested property. Do not assume an energy calculation produces DOS, PDOS, bands, or populations automatically.
3. Stage the unchanged input and all referenced files with `cp2k_prepare`.
4. Submit with `cp2k_execute`, then use `cp2k_output_summary` for run state and a focused parser for the requested property files.

## Scientific choices

- Match k-point sampling, spin treatment, smearing, energy windows, orbital projections, and atom groups to the stated property question.
- Keep reference-energy alignment explicit when comparing DOS or band outputs.
- Report the exact property files used and do not substitute unrelated files found in the result directory.

## Handoff

Return the property stage, the relevant electronic settings, and JSON/CSV/figure artifacts from the focused analysis.

## References

- [CP2K FORCE_EVAL/PROPERTIES](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/PROPERTIES.html)
- [CP2K BANDSTRUCTURE](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/PROPERTIES/BANDSTRUCTURE.html)
