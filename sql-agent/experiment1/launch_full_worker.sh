#!/bin/bash
# One parallel worker: vLLM (4 GPUs) + full shard run.
# Env: CUDA_VISIBLE_DEVICES, CHEM_PORT, SHARD_CSV, WORKER_ID
set -euo pipefail

ROOT=${REPO_ROOT}/sql-agent
EXP=$ROOT/experiment1
VENV=<local-home>/pablo-cu129
WORKER_ID="${WORKER_ID:-0}"
CHEM_PORT="${CHEM_PORT:-8010}"
SHARD_CSV="${SHARD_CSV:?set SHARD_CSV}"
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

IFS=',' read -r -a DEV <<< "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES}"
if [[ ${#DEV[@]} -lt 4 ]]; then
  echo "need 4 GPUs, got ${CUDA_VISIBLE_DEVICES}" >&2
  exit 1
fi

STAMP=$(date +%Y%m%d-%H%M%S)
OUT=$EXP/runs/exp1-full-w${WORKER_ID}-$STAMP.jsonl
LOG=$EXP/runs/vllm-full-w${WORKER_ID}-$STAMP.log

cleanup() {
  if [[ -n "${CHEM_PID:-}" ]]; then kill "$CHEM_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

wait_health() {
  local port=$1
  for _ in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
      echo "vllm w${WORKER_ID} up on :$port"
      return 0
    fi
    sleep 10
  done
  echo "vllm w${WORKER_ID} failed" >&2
  tail -50 "$LOG" >&2 || true
  return 1
}

echo "worker $WORKER_ID GPUs=${DEV[*]} port=$CHEM_PORT shard=$SHARD_CSV"
nvidia-smi --query-gpu=index,memory.used --format=csv | head -20

if curl -sf "http://127.0.0.1:${CHEM_PORT}/health" >/dev/null; then
  echo "reusing existing vllm on :$CHEM_PORT"
  CHEM_PID=""
else
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
  wait_health "$CHEM_PORT"
fi

echo "running shard → $OUT"
python "$EXP/run.py" \
  --model-url "http://127.0.0.1:$CHEM_PORT" \
  --model-name controller \
  --db "$ROOT/data/biomedsql.sqlite" \
  --benchmark "$SHARD_CSV" \
  --out "$OUT"

echo "DONE worker=$WORKER_ID out=$OUT"
echo "$OUT" > "$EXP/runs/exp1-full-w${WORKER_ID}-latest.path"
