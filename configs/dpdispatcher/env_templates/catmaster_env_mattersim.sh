#!/usr/bin/env bash
# Copy to the remote path referenced by mattersim_gpu.source_list.
set -euo pipefail
export PATH="<REMOTE_CONDA_ROOT>/condabin:${PATH}"
eval "$(conda shell.bash hook)"
conda activate "<MATTERSIM_ENV_NAME>"
export PYTHONUNBUFFERED=1
