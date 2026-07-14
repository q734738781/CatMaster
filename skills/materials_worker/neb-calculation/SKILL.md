---
name: neb-calculation
description: "Use this skill for the execution stage of NEB and dimer workflows, especially the detailed run protocol for plain-NEB to CI-NEB refinement or NEB/frequency/dimer refinement."
---

# neb-calculation

## Overview
Use this skill once the pathway inputs are already prepared and the question is how to run them robustly. It covers the two common execution branches: `plain-NEB -> CI-NEB` and `NEB or TS guess -> frequency/mode guess -> dimer`. For image generation and dimer input assembly, use `neb-prepare`. For post-run barrier interpretation and QC, use `neb-analysis`.

## Quick Start
1. Decide whether the task is a NEB refinement workflow or a dimer-refinement workflow.
2. Confirm the prepared root, execution branch, climb policy, dtype, output root, and whether the run is coarse convergence or refinement.
3. Dispatch the prepared calculation root through `remote_submission` or `remote_submission_batch`, normally with `task_name="vasp_execute_neb"` for VASP pathway work.
4. For MACE pathway optimization, prepare the MACE NEB stage layout and submit with `task_name="mace_neb_dir"`.
5. Keep coarse convergence and refinement as separate episodes instead of mixing them into one opaque run root.

## Allowed tools
- `get_avail_remote_task`
- `remote_submission`
- `remote_submission_batch`

## Workflow

### 1. Default NEB route: coarse `plain-NEB`, then `CI-NEB`
- Do not use managed NEB execution as a diagnostic for missing preparation. If image trees, endpoint provenance, or dimer modes are uncertain, return to `neb-prepare` first.
- A NEB tree that carries an overlap/short-distance preparation warning is very likely to contain abnormal interpolation. In CatMaster tools, `short_distance_count > 0` means at least one image has a minimum interatomic distance below the configured threshold, default `0.8 Å`; verify or remediate the image tree before deciding to submit.
- Start from a prepared NEB root with climbing image disabled.
- Run a coarse plain `NEB` first to localize the band and reduce gross path noise.
- Once the band is reasonably converged, restart from those coarse-converged images with climbing image enabled for `CI-NEB` refinement.
- Do not present a direct first-shot climbing-image run as the default; use it only when the saddle is already well localized and the task has a reason to skip the coarse stage.

### 2. Keep refinement episodes explicit
- Keep the coarse plain-`NEB` run and the `CI-NEB` refinement run in separate output roots.
- Do not change image count, endpoint definition, and convergence strategy all at once if you expect to compare outcomes scientifically.
- Record which stage produced which artifact; “NEB completed” is not enough.

### 3. Dimer route: NEB or TS guess, then frequency-guided mode, then dimer
- Use this route when the goal is TS refinement rather than only a band profile.
- A common pattern is:
  `coarse NEB or TS guess -> frequency/mode estimate -> dimer refinement`.
- The practical role of the frequency/mode step is to produce a chemically meaningful reaction-direction guess before launching the dimer.
- If the initial mode guess is poor, fix the mode selection rather than blindly rerunning the same dimer job.

### 4. Dispatch details for VASP pathway jobs
- Prefer `task_name="vasp_execute_neb"` for NEB or dimer-style VASP runs so they use the dedicated submission preset rather than the generic VASP one.
- Report the prepared root, task name, and output root together.
- Treat launch success as only one checkpoint; it does not prove the pathway is physically meaningful.

### 5. Managed MACE NEB
- Use `task_name="mace_neb_dir"` for managed MACE NEB rather than ad hoc scripts.
- Prepare one complete stage per independent MACE NEB path and use `remote_submission_batch` for multiple paths. The current runner processes multiple path directories inside one stage sequentially and constructs the path calculator separately, so grouping independent long paths under one `input/` does not provide useful model-reuse parallelism.
- Keep `default_dtype=float64` by default for MACE pathway optimization.
- Use `plain` mode for a fixed image set and `autoneb` only when the workflow explicitly benefits from adaptive image insertion.
- Keep `climb` as an explicit decision rather than an implicit default.

## Method-critical defaults
- `plain-NEB -> CI-NEB` is the default robust VASP barrier workflow.
- Do not enable climbing image in the first rough-convergence stage unless there is a clear reason.
- If the workflow takes the dimer branch, keep the mode-generation evidence explicit; dimer runs without a credible reaction direction are weakly interpretable.
- Keep `default_dtype=float64` as the default for MACE geometry/path optimization; only use `float32` when the run is explicitly exploratory.
- For pathway work, separate preparation, execution, and analysis artifacts cleanly.
- Before dispatch, verify endpoint ordering and image-distance QC; launch success does not repair an overlapped band.

## Output Contract
Return:
- which branch was run (`plain_neb`, `cineb_refinement`, `dimer_refinement`, `mace_neb`, or similar)
- execution root and submission evidence
- whether the run was coarse convergence or refinement
- any follow-up action the next stage should take

## References
- Use `neb-prepare` before this skill if the image tree or dimer inputs are not ready yet.
- Use `neb-analysis` after collection to interpret the barrier, profile shape, and common pitfalls.
