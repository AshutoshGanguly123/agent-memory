#!/usr/bin/env bash
set -euo pipefail
AMEM_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$AMEM_ROOT"
export PYTHONPATH="$AMEM_ROOT/src"
export PATH="$AMEM_ROOT/.venv/bin:$AMEM_ROOT/tools:$PATH"
export PYTHONHASHSEED=42
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MPLCONFIGDIR="$AMEM_ROOT/results/setup/matplotlib"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export UV_LINK_MODE=copy
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
