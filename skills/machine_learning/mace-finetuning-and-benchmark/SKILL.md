---
name: mace-finetuning-and-benchmark
description: Use this skill for remote MACE fine-tuning/training plus held-out evaluation using the validated reference-script conventions, especially the `mace-mh-1 + omat_pbe` replay-style path with explicit E0 and replay controls.
license: project-local
compatibility: local
allowed-tools: "mace_train mace_evaluate"
---

# mace-finetuning-and-benchmark

## Overview
Use this skill to train or fine-tune a MACE model on a prepared dataset while matching the validated `reference_scripts/mace_training_example` behavior instead of inventing ad hoc MACE CLI settings.

## Quick Start
1. Start from a dataset directory that already contains explicit split files.
2. For the validated baseline, use `foundation_model=models/mace-mh-1.model` and `foundation_head=omat_pbe`.
3. Launch training with `mace_train`, choosing `e0s="estimated"` or a fixed E0 JSON path explicitly.
4. Use `mace_evaluate` only when you need an extra post-training benchmark pass; the training run itself may already include `test.extxyz`.

## Allowed tools
- `mace_train`
- `mace_evaluate`

## Workflow

### 1. Keep dataset provenance fixed
- Do not retrain on a moving dataset root while comparing hyperparameters or base models.
- Keep the split file names and the exact foundation-model/head choice explicit.

### 2. Train as one remote job
- `mace_train` stages the dataset, optional foundation model, optional E0 JSON, params JSON, and logs under one output root.
- Report the collected batch state and summary artifacts, not just that the submission launched.
- The reference-validated route is replay-style finetuning with explicit `foundation_head`, `multiheads_finetuning`, `pt_train_file`, replay sampling knobs, and explicit loss weights.

### 3. Benchmark separately
- The training run can already carry `test.extxyz`; use `mace_evaluate` when you need an additional benchmark pass on a retained checkpoint or an alternate split.
- Keep the evaluation output root separate from the training root.
- Choose the evaluation device explicitly when the remote resource is not guaranteed to expose CUDA.

### 4. Use this skill once the workflow artifact is a dataset or model
- Start from a prepared dataset directory, a checkpoint to benchmark, or an explicit model-comparison plan.
- If the training run identifies new structures that need relabeling or new reference calculations, hand those artifacts back into the materials-side workflow before the next dataset rebuild.

## Method-critical defaults
- The validated baseline in this repo is `mace-mh-1` with `foundation_head=omat_pbe`, explicit replay controls, `compute_stress=True`, `energy_weight=1.0`, `forces_weight=100.0`, `stress_weight=1.0`, `default_dtype=float32`, `batch_size=4` as the conservative starting point, and `seed=42`.
- Surface the foundation-model choice, head, E0 strategy, replay controls, batch size, learning rate, and epoch cap when they differ across runs.
- Treat benchmark coverage honestly: the evaluator reports energy/force metrics, and reports stress metrics only when reference stress is present in the dataset and the model/calculator exposes stress.
- Do not compare metrics across different train/valid/test splits as if they came from the same benchmark.
- Keep the training artifact chain explicit: dataset inputs, checkpoint outputs, and benchmark reports should remain separately identifiable.

## Output Contract
Return:
- training output root
- evaluation output root
- collected batch-state paths
- model artifact path(s)
- metrics JSON / per-config CSV path(s)

## References
- Use `mace-dataset-curation` first when the dataset root has not yet been built from VASP outputs.
- Reference flow: [vasp_to_mace_finetune.md](/home/chenhh/python_projects/CatMaster/reference_scripts/mace_training_example/vasp_to_mace_finetune.md)
- Validated training command: [run_train.sh](/home/chenhh/python_projects/CatMaster/reference_scripts/mace_training_example/run_train.sh)
