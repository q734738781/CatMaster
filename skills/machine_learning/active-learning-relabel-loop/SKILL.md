---
name: active-learning-relabel-loop
description: Use this skill for candidate ranking and relabel-loop bookkeeping when the task is to select the next structures for expensive reference calculations from a structure pool or curated dataset.
---

# active-learning-relabel-loop

## Overview
Use this skill to rank candidate structures for the next relabel round and maintain a clean handoff between current dataset, current model, and selected next-step structures.

## Quick Start
1. Start from either a candidate structure pool or a curated dataset.
2. Use `calculate_al_candidates` to rank and select the next structures.
3. After new reference calculations are collected, rebuild the dataset with `build_dataset_from_runs`.
4. Retrain and benchmark with `mace_train` and `mace_evaluate`.

## Allowed tools
- `calculate_al_candidates`
- `build_dataset_from_runs`
- `mace_train`
- `mace_evaluate`

## Workflow

### 1. Select under one explicit ranking rule
- If a committee is available, say whether disagreement is part of the ranking.
- If not, state that the round is diversity-first.
- Treat the current selector as a baseline heuristic: diversity in a simple structure-feature space plus optional committee disagreement from per-atom energy variance.

### 2. Keep relabel bookkeeping clean
- Preserve the emitted ranking JSON/CSV and any selected-structure export as the loop ledger.
- Do not overwrite the previous round’s selection artifacts.

### 3. Rebuild, then retrain
- After the new reference calculations finish, rebuild the dataset rather than manually appending hidden frames.
- Retrain and re-benchmark from the new dataset state so round-to-round changes are auditable.
- Keep the MACE retrain leg aligned with the same validated finetune path used in `mace-finetuning-and-benchmark`, especially the foundation-model head and E0 strategy.

### 4. Use this skill for the model-update portion of the loop
- Candidate structures may originate from materials-side screening, but this skill starts when the loop object is selection, dataset refresh, retraining, and benchmark comparison.
- Keep the selection ledger, rebuilt dataset, and updated model artifacts explicit so the next relabel round is auditable.

## Method-critical defaults
- Surface the candidate source, committee size, and selection size in every loop round.
- Do not describe the current scorer as a richer uncertainty engine than it is; it is not force-uncertainty AL, Bayesian optimization, or a calibrated acquisition function.
- Do not compare active-learning rounds if the held-out benchmark split changed without being reported.
- Keep the round boundary explicit: each cycle should expose the incoming candidate pool, the selected subset, the rebuilt dataset, and the updated benchmark outputs.

## Output Contract
Return:
- AL ranking JSON/CSV path
- selected candidate artifact path
- rebuilt dataset summary path after relabel
- updated training/evaluation artifact paths

## References
- This skill coordinates the loop contract; the expensive reference calculations still happen outside the ML lane.
- Use the same validated training recipe documented in [vasp_to_mace_finetune.md](/home/chenhh/python_projects/CatMaster/reference_scripts/mace_training_example/vasp_to_mace_finetune.md) when comparing AL rounds.
