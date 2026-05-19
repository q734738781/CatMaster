#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=3
OMP_NUM_THREADS=8
# This script is bundle-friendly: after unpacking the training archive, run it
# from the bundle root and it will use only local files.
mace_run_train \
  --name battery_mh1_replay \
  --foundation_model models/mace-mh-1.model \
  --foundation_head omat_pbe \
  --train_file DFT_files/mace_ft/train.extxyz \
  --valid_file DFT_files/mace_ft/valid.extxyz \
  --test_file DFT_files/mace_ft/test.extxyz \
  --pt_train_file omat \
  --num_samples_pt 50000 \
  --filter_type_pt combinations \
  --subselect_pt fps \
  --weight_pt 1.0 \
  --atomic_numbers "[8, 11, 12, 13, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 38, 39, 40, 41, 42, 44, 45, 46, 49, 50]" \
  --multiheads_finetuning True \
  --E0s DFT_files/mace_ft/e0s_estimated.json \
  --compute_stress True \
  --energy_weight 1.0 \
  --forces_weight 10.0 \
  --stress_weight 1.0 \
  --default_dtype float32 \
  --max_num_epochs 25 \
  --batch_size 4 \
  --device cuda \
  --seed 42 \
  --checkpoints_dir checkpoints \
  --restart_latest
