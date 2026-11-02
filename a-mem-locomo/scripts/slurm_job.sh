#!/usr/bin/env bash
#SBATCH --partition=gpu
#SBATCH --nodelist=gpu-04
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --requeue
set -euo pipefail
source "${AMEM_PROJECT_ROOT:?}/scripts/runtime.sh"
test -n "${SLURM_JOB_ID:-}"
mkdir -p results/operations
export AMEM_STOP_FILE="$AMEM_ROOT/results/operations/job_${SLURM_JOB_ID}.stop"
export AMEM_STOP_EPOCH="$(python -c 'import time; print(time.time()+3*3600+45*60)')"
trap 'touch "$AMEM_STOP_FILE"' USR1 TERM
AMEM_EXIT="$AMEM_ROOT/results/operations/job_${SLURM_JOB_ID}.exit.json"
if [ -f "$AMEM_EXIT" ]; then mv "$AMEM_EXIT" "${AMEM_EXIT}.previous.$(date +%s)"; fi
if [ -f "$AMEM_STOP_FILE" ]; then mv "$AMEM_STOP_FILE" "${AMEM_STOP_FILE}.previous.$(date +%s)"; fi
args=(--root "$AMEM_ROOT" --worker "${AMEM_WORKER:?}")
if [ "${AMEM_MODE:?}" = preflight ]; then args+=(--preflight); fi
python scripts/telemetry.py --job "$SLURM_JOB_ID" --worker "$AMEM_WORKER" --mode "$AMEM_MODE" &
AMEM_TELEMETRY_PID=$!
set +e
srun --ntasks=1 --cpus-per-task=4 --gres=gpu:1 .venv/bin/python -u -m amem_reproduction.runner "${args[@]}" >> "results/operations/job_${SLURM_JOB_ID}.worker.log" 2>&1 &
AMEM_STEP_PID=$!
while true; do
  wait "$AMEM_STEP_PID"
  AMEM_STEP_RC=$?
  if ! kill -0 "$AMEM_STEP_PID" 2>/dev/null; then break; fi
done
.venv/bin/python scripts/record_exit.py "$SLURM_JOB_ID" "$AMEM_STEP_RC"
AMEM_RECORD_RC=$?
touch "results/operations/job_${SLURM_JOB_ID}.telemetry.stop"
wait "$AMEM_TELEMETRY_PID" || true
if [ "$AMEM_RECORD_RC" -ne 0 ]; then exit 1; fi
exit "$AMEM_STEP_RC"
