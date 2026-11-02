#!/bin/bash
# Start Qwen3.8-27B + run exp1 one-per-template smoke. Run ON a GPU node.
set -euo pipefail

ROOT=${REPO_ROOT}/sql-agent
EXP=$ROOT/experiment1
VENV=<local-home>/pablo-cu129
export HF_HOME=<local-home>/.cache/huggingface
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
export CUDA_HOME=/usr/local/cuda-12.2
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
mkdir -p "$EXP/runs" "$TMPDIR"

source "$VENV/bin/activate"

IFS=',' read -r -a DEV <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
if [[ ${#DEV[@]} -lt 4 ]]; then
  echo "need 4 visible GPUs, got CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
  exit 1
fi
echo "GPUs: ${DEV[*]}"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

CHEM_PORT="${CHEM_PORT:-8000}"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=$EXP/runs/exp1-smoke-$STAMP.jsonl
LOG=$EXP/runs/vllm-$STAMP.log

cleanup() {
  if [[ -n "${CHEM_PID:-}" ]]; then kill "$CHEM_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

wait_health() {
  local port=$1 name=$2
  for _ in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
      echo "$name is up on :$port"
      return 0
    fi
    sleep 10
  done
  echo "$name failed to become healthy" >&2
  tail -50 "$LOG" >&2 || true
  return 1
}

echo "starting vLLM Qwen3.8-27B on ${DEV[0]},${DEV[1]},${DEV[2]},${DEV[3]} → :$CHEM_PORT"
CUDA_VISIBLE_DEVICES="${DEV[0]},${DEV[1]},${DEV[2]},${DEV[3]}" \
  vllm serve Qwen/Qwen3.8-27B \
    --served-model-name controller \
    --tensor-parallel-size 4 \
    --disable-custom-all-reduce \
    --language-model-only \
    --max-model-len 32768 \
    --max-num-seqs 2 \
    --gpu-memory-utilization 0.90 \
    --port "$CHEM_PORT" \
    > "$LOG" 2>&1 &
CHEM_PID=$!

wait_health "$CHEM_PORT" vllm

echo "running one-per-template smoke → $OUT"
python "$EXP/run.py" \
  --model-url "http://127.0.0.1:$CHEM_PORT" \
  --model-name controller \
  --db "$ROOT/data/biomedsql.sqlite" \
  --benchmark "$ROOT/data/benchmark_dev_sample.csv" \
  --one-per-template \
  --out "$OUT"

echo "done $OUT"
