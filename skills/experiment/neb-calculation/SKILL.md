---
name: neb-calculation
description: "Use this skill for the execution stage of NEB and dimer workflows, especially the detailed run protocol for plain-NEB to CI-NEB refinement or NEB/frequency/dimer refinement."
---

# neb-calculation

## Overview
Use this skill once the pathway inputs are already prepared and the question is how to run them robustly. It covers the two common execution branches: `plain-NEB -> CI-NEB` and `NEB or TS guess -> frequency/mode guess -> dimer`. For image generation and dimer input assembly, use `neb-prepare`. For post-run barrier interpretation and QC, use `neb-analysis`.

## Quick Start
1. Decide whether the task is a NEB refinement workflow or a dimer-refinement workflow.
2. Dispatch the prepared calculation root through `vasp_execute_batch`, normally with `task_name="vasp_execute_neb"` for VASP pathway work.
3. For MACE pathway optimization, use `mace_neb_batch`.
4. Keep coarse convergence and refinement as separate episodes instead of mixing them into one opaque run root.

## Allowed tools
- `vasp_execute_batch`
- `mace_neb_batch`

## Workflow

### 1. Default NEB route: coarse `plain-NEB`, then `CI-NEB`
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
- Prefer `vasp_execute_batch(task_name="vasp_execute_neb")` for NEB or dimer-style VASP runs so they use the dedicated submission preset rather than the generic VASP one.
- Report the prepared root, task name, and output root together.
- Treat launch success as only one checkpoint; it does not prove the pathway is physically meaningful.

### 5. Managed MACE NEB
- Use `mace_neb_batch` for managed MACE NEB rather than ad hoc scripts.
- Keep `default_dtype=float64` by default for MACE pathway optimization.
- Use `plain` mode for a fixed image set and `autoneb` only when the workflow explicitly benefits from adaptive image insertion.
- Keep `climb` as an explicit decision rather than an implicit default.

## Method-critical defaults
- `plain-NEB -> CI-NEB` is the default robust VASP barrier workflow.
- Do not enable climbing image in the first rough-convergence stage unless there is a clear reason.
- If the workflow takes the dimer branch, keep the mode-generation evidence explicit; dimer runs without a credible reaction direction are weakly interpretable.
- Keep `default_dtype=float64` as the default for MACE geometry/path optimization; only use `float32` when the run is explicitly exploratory.
- For pathway work, separate preparation, execution, and analysis artifacts cleanly.

## Output Contract
Return:
- which branch was run (`plain_neb`, `cineb_refinement`, `dimer_refinement`, `mace_neb`, or similar)
- execution root and submission evidence
- whether the run was coarse convergence or refinement
- any follow-up action the next stage should take

## References
- Use `neb-prepare` before this skill if the image tree or dimer inputs are not ready yet.
- Use `neb-analysis` after collection to interpret the barrier, profile shape, and common pitfalls.
