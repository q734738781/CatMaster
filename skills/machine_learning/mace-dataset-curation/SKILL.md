---
name: mace-dataset-curation
description: Use this skill for turning VASP result trees into extxyz training datasets that follow the validated reference-script conventions for REF labels, optional head/config_type tags, and fixed split artifacts before MACE training.
---

# mace-dataset-curation

## Overview
Use this skill to convert collected VASP runs into a reusable extxyz dataset directory while staying close to the validated `reference_scripts/mace_training_example` export contract.

## Quick Start
1. Point `build_dataset_from_runs` at one result root.
2. Choose `frame_mode` deliberately: `final` or `all_ionic_steps`.
3. If the downstream training uses a multi-head foundation model, set `head_label` explicitly, typically `omat_pbe`.
4. Keep the split fractions explicit and reproducible, and leave `split_unit="source_run"` unless you intentionally want frame-level leakage.
5. Leave `require_converged=false` unless you are intentionally building a guessed-converged subset.
6. Carry forward the emitted summary JSON and split file paths.

## Allowed tools
- `build_dataset_from_runs`

## Workflow

### 1. Choose the frame policy first
- The validated reference workflow starts from ionic-step data rather than only final frames, so `all_ionic_steps` is the default starting point when you want to match that path.
- `final` is only for deliberately reduced datasets where relaxed endpoints alone are the training target.

### 2. Keep split artifacts stable
- Treat `dataset.extxyz`, `train.extxyz`, `valid.extxyz`, and `test.extxyz` as the canonical handoff set.
- Use the summary JSON as the ledger for skipped runs, frame counts, stress encoding assumptions, and any `head_label` / `config_type` tags.

### 3. Keep the reference labeling contract explicit
- The validated reference export path writes `REF_energy`, `REF_forces`, `REF_stress`, `config_type`, and optionally `head`.
- If the downstream finetune uses `mace-mh-1` with `omat_pbe`, set `head_label="omat_pbe"` instead of assuming the head will be inferred later.

### 4. Stop before training
- This skill ends at a curated dataset directory.
- Hand the resulting dataset to `mace-finetuning-and-benchmark` or `active-learning-relabel-loop` instead of mixing curation and training into one opaque step.
- Use this skill once the working artifact is a dataset directory rather than a structure-screening batch.

## Method-critical defaults
- The reference-aligned default is to keep all ionic steps and store `step_electronic_converged_guess` for later filtering; use `require_converged=true` only when that hard subset is the actual dataset target.
- Keep `alignment_check=true` unless you are deliberately debugging malformed XML; XML/ASE step-order mismatches should cause the run to be skipped, not silently truncated into the dataset.
- Report whether the dataset contains final frames only or full ionic trajectories.
- Surface the `head_label` when the dataset is intended for multi-head foundation-model finetuning.
- Keep `split_unit="source_run"` for trajectory-style data unless you intentionally want frame-level mixing across train/valid/test.
- Do not silently reshuffle split fractions between model comparisons.
- Keep the handoff artifact explicit: this skill should start from collected result trees and end with a reproducible dataset directory plus split files.

## Output Contract
Return:
- dataset directory
- split file paths
- dataset summary JSON
- any skipped-run ledger

## References
- Reference flow: [vasp_to_mace_finetune.md](/home/chenhh/python_projects/CatMaster/reference_scripts/mace_training_example/vasp_to_mace_finetune.md)
- Export conventions: [export_ase_db_to_mace_xyz.py](/home/chenhh/python_projects/CatMaster/reference_scripts/mace_training_example/export_ase_db_to_mace_xyz.py)
