#!/usr/bin/env bash
# Copy to the remote path referenced by uma_gpu.source_list. Keep tokens remote.
set -euo pipefail
export PATH="<REMOTE_CONDA_ROOT>/condabin:${PATH}"
eval "$(conda shell.bash hook)"
conda activate "<UMA_ENV_NAME>"
export PYTHONUNBUFFERED=1
# export HF_HOME=<REMOTE_HF_CACHE>
# export HF_TOKEN from a protected remote secret source when the model is gated.
