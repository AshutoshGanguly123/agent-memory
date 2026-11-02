#!/bin/bash
# Rerun the 16 EX=0 smoke failures with updated prompts.
set -euo pipefail

ROOT=${REPO_ROOT}/sql-agent
EXP=$ROOT/experiment1
VENV=<local-home>/pablo-cu129
UUIDS_FILE=$EXP/runs/failed16.uuids
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

IFS=',' read -r -a DEV <<< "${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
if [[ ${#DEV[@]} -lt 4 ]]; then
  echo "need 4 visible GPUs" >&2
  exit 1
fi

CHEM_PORT="${CHEM_PORT:-8010}"
STAMP=$(date +%Y%m%d-%H%M%S)
OUT=$EXP/runs/exp1-rerun16-$STAMP.jsonl
LOG=$EXP/runs/vllm-rerun16-$STAMP.log
mapfile -t UUIDS < "$UUIDS_FILE"

cleanup() {
  if [[ -n "${CHEM_PID:-}" ]]; then kill "$CHEM_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT

wait_health() {
  local port=$1
  for _ in $(seq 1 360); do
    if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
      echo "vllm up on :$port"
      return 0
    fi
    sleep 10
  done
  echo "vllm failed" >&2
  tail -40 "$LOG" >&2 || true
  return 1
}

# reuse existing server if healthy
if curl -sf "http://127.0.0.1:${CHEM_PORT}/health" >/dev/null; then
  echo "reusing vllm on :$CHEM_PORT"
  CHEM_PID=""
else
  echo "starting vLLM on ${DEV[*]} → :$CHEM_PORT"
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

echo "rerunning ${#UUIDS[@]} failed uuids → $OUT"
python "$EXP/run.py" \
  --model-url "http://127.0.0.1:$CHEM_PORT" \
  --model-name controller \
  --db "$ROOT/data/biomedsql.sqlite" \
  --benchmark "$ROOT/data/benchmark_dev_sample.csv" \
  --uuid "${UUIDS[@]}" \
  --out "$OUT"

echo "done $OUT"
