---
name: transition-state-neb
description: Use this skill for transition-state and NEB workflows, including image generation, NEB VASP input setup, official VASP improved-dimer preparation, reaction-mode guessing, and execution/evidence checks for pathway calculations.
license: project-local
compatibility: local
allowed-tools: "make_neb_geometry vasp_neb_prepare vasp_dimer_prepare make_dimer_mode_from_neb make_dimer_mode_from_mace vasp_execute_batch mace_neb_batch analyze_vasp_neb_results"
---

# transition-state-neb

## Overview
Use this skill to generate NEB image directories, prepare NEB-ready and dimer-ready VASP roots, derive reaction-direction guesses, and hand off a valid pathway batch for execution. Use it for setup-focused NEB work, dimer refinement, or custom mode-generation branches. Do not use it when a standard endpoint-pair-to-barrier workflow is enough; prefer `reaction-neb-analysis` for that narrower path.

## Quick Start
1. First choose the narrow branch: standard NEB barrier workflow, setup-only NEB, dimer-from-NEB, or dimer-from-mode.
2. Validate the initial and final structures before generating images.
3. Use `make_neb_geometry` to create the flat numbered image-file tree.
4. Use `vasp_neb_prepare` to assemble the NEB root with canonical support files and NEB-critical INCAR settings.
5. If switching from NEB to an official VASP improved dimer refinement, derive a raw reaction-direction text block first, then feed it into `vasp_dimer_prepare`.
6. Run the resulting NEB or dimer folders through the standard VASP batch execution path.

## Allowed tools
- `make_neb_geometry`
- `vasp_neb_prepare`
- `vasp_dimer_prepare`
- `make_dimer_mode_from_neb`
- `make_dimer_mode_from_mace`
- `vasp_execute_batch`
- `mace_neb_batch`
- `analyze_vasp_neb_results`

## Workflow

### 0. Choose the narrowest branch before doing setup
- Use `reaction-neb-analysis` when the task is simply “endpoint pair -> NEB dispatch -> barrier summary”.
- Use this skill when you need one of the wider branches: primitive NEB setup without full barrier reporting, official VASP improved-dimer preparation, dimer-mode extraction from NEB, or dimer-mode extraction from MACE finite differences.
- Do not mix NEB and dimer outputs inside one ambiguous run root; keep the branch identity explicit.

### 1. Build the image set from a valid endpoint pair
- `make_neb_geometry` validates the endpoint pair before interpolation.
- It writes a flat numbered image-file tree (`00.vasp`, `01.vasp`, ...) under `output_dir`. This is the preferred shared geometry format because it can be consumed directly by `mace_neb_batch`.
- For high-throughput work, `make_neb_geometry` also supports `input_root/output_root` batch mode: `input_root/task0/IS.vasp + FS.vasp -> output_root/task0/00.vasp...`.
- If `output_dir` already exists, `overwrite=true` is required to replace it.

### 2. Prepare the NEB VASP root
- `vasp_neb_prepare` keeps geometry as a separate primitive: it can either consume an endpoint pair or reuse an existing image tree.
- In image-tree mode, prefer a flat numbered file tree (`00.vasp`, `01.vasp`, ...). Legacy numbered directories (`00/POSCAR`, `01/POSCAR`, ...) are still accepted.
- In image-tree mode, the same task directory must contain `IS_OUTCAR` and `FS_OUTCAR`. Do the preprocessing copy explicitly before calling the tool.
- For high-throughput work, `vasp_neb_prepare` also supports `input_root/output_root` batch mode where each child task directory is one complete NEB task containing its image tree plus `IS_OUTCAR` and `FS_OUTCAR`.
- It enforces the core NEB settings; `iopt` must be one of `7`, `2`, or `1`.
- It writes the resulting support files plus `neb_incar_patch.json`, which is the authoritative diff from the canonical support-file baseline.
- In `patch_policy="safe"`, NEB-critical keys remain protected; use `force` only for intentional overrides.

### 3. Hand off to execution as a VASP batch
- Treat the NEB image tree as a prepared VASP input set.
- Prefer a two-stage pathway run when the saddle is not already well localized: first run plain `NEB` with climbing image disabled to coarse-converge the band, then restart from those images with `CI-NEB` enabled for saddle refinement.
- Do not enable climbing image in the first rough-convergence stage unless the task has a strong reason to skip directly to refinement.
- When NEB/TS should use a separate submission preset, call `vasp_execute_batch` with `task_name="vasp_execute_neb"` so it routes through the dedicated DPDispatcher task/resources config instead of the generic VASP preset.
- Prefer `task_name="vasp_execute_neb"` by default for NEB/TS runs: the generic `vasp_execute` path can still run, but it will use the generic VASP resource preset rather than the NEB-specific submission configuration.
- Report image count, INCAR patch path, and execution status together; launch status alone is not enough.
- If the task is only setup or mode preparation, say that explicitly instead of implying the barrier-analysis branch was also completed.

### 4. Close the loop with barrier extraction
- Use `analyze_vasp_neb_results` after collection to produce the barrier summary, profile CSV, and profile plot.
- If image energies are incomplete, report partial collection rather than inferring a barrier.

### 5. Prepare official VASP improved-dimer jobs with a raw reaction-direction text block
- Use `vasp_dimer_prepare` for the official VASP improved dimer method (`IBRION=44`).
- Treat it as a relax-style VASP setup with one special extra requirement: the raw reaction-direction text block.
- `vasp_dimer_prepare` internally performs the required per-atom `1/sqrt(mass)` transformation and then appends the resulting normalized vectors to the end of `POSCAR` after a separating blank line, which is what official `IBRION=44` reads.
- Do not pre-mass-normalize the mode before passing it into `vasp_dimer_prepare`, or you will double-apply the mass weighting.
- Use this path for the official VASP dimer line. Do not assume `MODECAR` is part of this workflow.

### 6. Derive a dimer direction from NEB when a TS-adjacent path is already available
- `make_dimer_mode_from_neb` takes one NEB image tree and uses the displacement between the two images adjacent to the chosen TS image as the raw reaction-direction guess.
- Prefer this when a converged or nearly converged NEB already exists and the barrier region is well localized.
- If no `ts_image_index` is provided, the tool defaults to the central non-endpoint image.
- This tool writes both a raw text block and a mass-normalized text block, but the raw text block is the one you should feed into `vasp_dimer_prepare`.

### 7. Derive a dimer direction from ASE/MACE finite differences when NEB guidance is not enough
- `make_dimer_mode_from_mace` uses ASE finite-difference vibrations with a MACE calculator to estimate candidate modes on a TS guess.
- Restrict the finite-difference region to the chemically active atoms whenever possible; do not vibrate an entire large slab unless the task explicitly requires it.
- The tool reports all frequencies it found and exports one selected raw mode.
- If there are multiple imaginary modes, do not blindly trust the most negative one. Choose the mode that matches the chemically meaningful reaction coordinate.
- If no imaginary mode appears, the exported lowest-frequency mode is only a heuristic initial dimer direction.

### 8. Use VASP finite-difference frequencies only as a targeted local check, not a brute-force whole-slab default
- VASP finite-difference frequencies can also provide a dimer direction, but this is usually expensive.
- If you go this route, restrict the frequency job to the active reaction center and nearby atoms instead of the full slab whenever the scientific objective allows it.
- If you copy a mode from a VASP `OUTCAR`, extract the ordinary raw `dx dy dz` block into a text file and pass that text file into `vasp_dimer_prepare`. Do not feed the already mass-divided `Eigenvectors after division by SQRT(mass)` block into the tool.

### 9. For MACE NEB, use the dedicated managed tool instead of ad hoc scripts
- Prefer `mace_neb_batch` for managed MACE NEB work via DPDispatcher.
- The expected input contract is explicit task directories: either one task directory directly, or a batch root containing `task0/`, `task1/`, ... where each task directory contains flat numbered image files such as `00.vasp`, `01.vasp`, ...
- Do not create deeper nested task trees under a task directory.
- `mace_neb_batch` writes one task-level output directory per task, with final converged images as top-level `00.vasp`, `01.vasp`, ... plus a `summary.json` and shared artifacts like `image_energies.csv`, `profile.png`, and `neb.traj`.

## Method-critical defaults
- Keep endpoint preparation, image generation, and execution settings scientifically consistent across the whole pathway calculation.
- Do not treat launch success as pathway validity; evidence must include image count, INCAR patch, and outcome diagnostics.
- If the workflow does not require dimer refinement or custom mode logic, do not use this broader skill by default; the narrower `reaction-neb-analysis` route is easier to audit.
- Treat `plain-NEB -> CI-NEB` as the default convergence pattern for pathway searches: coarse-converge the band without climbing image first, then refine the saddle with climbing image enabled.
- When choosing NEB interpolation counts for routine runs, prefer small image counts such as `3`, `4`, `5`, or `6` unless the pathway is clearly too sharp for that range.
- For `CI-NEB` / climbing-image runs, prefer an odd number of intermediate images so there is a natural central image to climb.
- For plain `NEB` without climbing image, prefer an even number of intermediate images when the path is otherwise symmetric and no single central climbing image is needed.
- For official VASP dimer jobs, keep the electronic-structure footing aligned with the relax campaign and treat the appended dimer direction as the only special input.
- Mass weighting changes the direction, not just the scale. Never skip the per-atom `1/sqrt(mass)` correction when converting a raw vibrational mode into a dimer direction.
- When several imaginary modes exist, the correct reaction-direction guess is the chemically meaningful one, not automatically the largest-magnitude imaginary mode.

## Output Contract
Return:
- branch label (`neb_setup`, `neb_run`, `dimer_from_neb`, `dimer_from_mace`, or similar)
- NEB image root
- image count
- INCAR path and patch JSON path
- execution evidence path(s) if the run was dispatched
- For dimer work, also return the raw mode text path, the dimer-ready `POSCAR` path, and enough evidence to show which mode-selection logic was used.

## References
- Pair this skill with `vasp-batch-execution` for dispatch and rerun handling instead of inventing a separate NEB execution path.
- Use `reaction-neb-analysis` when the images are finished and you need barrier extraction and profile artifacts.
