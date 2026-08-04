---
name: mace-finetuning-and-benchmark
description: Use this skill for remote MACE fine-tuning/training plus held-out evaluation using the validated reference-script conventions, especially the `mace-mh-1 + omat_pbe` replay-style path with explicit E0 and replay controls.
---

# mace-finetuning-and-benchmark

## Overview
Use this skill to train or fine-tune a MACE model on a prepared dataset while matching the validated `reference_scripts/mace_training_example` behavior instead of inventing ad hoc MACE CLI settings.

## Quick Start
1. Start from a dataset directory that already contains explicit split files.
2. For the validated baseline, use `foundation_model=mh-1` or an explicit staged/local foundation-model path, and `foundation_head=omat_pbe`.
3. Launch training by preparing the `mace_train` stage layout and calling `remote_submission`, choosing `e0s="estimated"` or a fixed E0 JSON path explicitly in the staged params.
4. Use `mace_eval` through `remote_submission` only when you need an extra post-training benchmark pass; the training run itself may already include `test.extxyz`.
5. Do not replace the managed remote training task with a local training wrapper when the catalog task already fits the job.

## Allowed tools
- `get_avail_remote_task`
- `remote_submission`

## Workflow

### 1. Keep dataset provenance fixed
- Do not retrain on a moving dataset root while comparing hyperparameters or base models.
- Keep the split file names and the exact foundation-model/head choice explicit.

### 2. Train as one remote job
- The training stage must contain the dataset, optional foundation model, optional E0 JSON, optional replay/statistics/local CLI assets, and `params/train_params.json`.
- Do not split one train/valid/test dataset across inference-style structure chunks. Training batch size and distributed-training behavior belong to the MACE training configuration; an independent model, seed, or hyperparameter trial is a separate complete training stage.
- Report stage-local model and metric outputs, not just that the submission launched. Keep receipt/context fields in runtime recovery records.
- The reference-validated route is replay-style finetuning with explicit `foundation_head`, `multiheads_finetuning`, `pt_train_file`, replay sampling knobs, and explicit loss weights.
- If the user did not ask for a custom ablation, keep the validated baseline explicit: `mh-1`, `omat_pbe`, and `estimated`/fixed estimated E0s. Do not silently swap to another foundation model or another head.
- When the user needs additional official MACE CLI knobs beyond the common first-class fields, pass them through `cli_args` rather than writing a local wrapper script.

### 3. Benchmark separately
- The training run can already carry `test.extxyz`; use `mace_eval` when you need an additional benchmark pass on a retained checkpoint or an alternate split.
- Keep one checkpoint-and-dataset evaluation as one complete evaluation stage. Do not manually shard the held-out dataset into remote stages unless the evaluation workflow explicitly supports deterministic shard merging.
- Keep the evaluation output root separate from the training root.
- Use the task's supported device control when a concrete compatibility constraint requires it; device identity is runtime metadata rather than benchmark evidence.

### 4. Use this skill once the workflow artifact is a dataset or model
- Start from a prepared dataset directory, a checkpoint to benchmark, or an explicit model-comparison plan.
- If the training run identifies new structures that need relabeling or new reference calculations, hand those artifacts back into the materials-side workflow before the next dataset rebuild.

## Method-critical defaults
- The validated baseline in this repo is `mace-mh-1` with `foundation_head=omat_pbe`, explicit replay controls, `compute_stress=True`, `energy_weight=1.0`, `forces_weight=10.0`, `stress_weight=1.0`, `default_dtype=float32`, `batch_size=4` as the conservative starting point, and `seed=42`.
- For typical fine-tuning runs in this workflow, use an epoch cap in the `15-25` range as the default starting band. Keep `25` as the normal upper cap unless the user explicitly asks for a longer ablation, and prefer the best validation checkpoint over blindly extending epochs.
- Prefer the managed remote task path for that baseline. Only fall back to custom local scripts when the requested workflow is genuinely outside what `mace_train` / `mace_eval` can express.
- Surface the foundation-model choice, head, E0 strategy, replay controls, batch size, learning rate, and epoch cap when they differ across runs.
- Treat benchmark coverage honestly: the evaluator reports energy/force metrics, and reports stress metrics only when reference stress is present in the dataset and the model/calculator exposes stress.
- Do not compare metrics across different train/valid/test splits as if they came from the same benchmark.
- Keep the training artifact chain explicit: dataset inputs, checkpoint outputs, and benchmark reports should remain separately identifiable.

## Output Contract
Return:
- training output root
- evaluation output root
- model artifact path(s)
- metrics JSON / per-config CSV path(s)

Keep receipt/context identifiers, hardware/device identity, scheduler layout, software build, and performance telemetry in runtime records unless a concrete failure or compatibility issue makes them relevant. If the user explicitly asks to inspect, compare, record, or report any of these fields, follow that request directly.

## References
- Use `mace-dataset-curation` first when the dataset root has not yet been built from VASP outputs.
- Reference flow: [vasp_to_mace_finetune.md](../../../demos/reference_scripts/mace_training_example/vasp_to_mace_finetune.md)
- Validated training command: [run_train.sh](../../../demos/reference_scripts/mace_training_example/run_train.sh)
