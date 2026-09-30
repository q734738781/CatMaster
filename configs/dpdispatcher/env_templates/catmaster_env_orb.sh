#!/usr/bin/env bash
# Copy to the remote path referenced by orb_gpu.source_list.
set -euo pipefail
export PATH="<REMOTE_CONDA_ROOT>/condabin:${PATH}"
eval "$(conda shell.bash hook)"
conda activate "<ORB_ENV_NAME>"
export PYTHONUNBUFFERED=1
# GPU inference / Sella uses small CPU matrices. Avoid per-process BLAS
# oversubscription; resource envs can explicitly select a different count.
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
