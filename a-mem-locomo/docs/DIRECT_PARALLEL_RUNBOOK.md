# Direct conversation parallelism

The user explicitly requested this execution mode on 10 September 2026: stop Slurm, retain progress, SSH directly into `<gpu-node>`, and use available cards concurrently. The historical Slurm instructions in HPC_RUNBOOK.md describe the first phase, not the active launcher. No new Slurm submissions are made by this adapter. Direct access does not reserve cards in Slurm; physical occupancy and read-only scheduler checks are used to yield our own workers when competing use appears.

## Active supervisor update — GPU reuse

The user subsequently requested using GPUs freed by failed conversations for other queued work. The validated replacement supervisor is `operations/direct_reuse_v2`, with tmux `amem_direct_full_v2`. It preserves failed conversations and reuses a GPU only after proving that the worker exhausted schema retries; unknown/device errors remain blocked. See [GPU_REUSE_RUNBOOK.md](GPU_REUSE_RUNBOOK.md) for the audited handover, exact failure classification and current recovery instructions. The v1 worker/backend and all original manifests remain unchanged; the same authoritative coordinator and worker-status paths are used.

## Preserved experiment

The original Slurm worker exited 75 at 15:02:50 UTC with 144 committed notes, zero QA, and durable call results beyond the last committed note. The original run ID, all 49 frozen files, model weights, configuration, call IDs, seeds, maximum three schema attempts, stale embedding checkpoints and scoring remain unchanged. A complete pre-migration raw archive and file hashes are saved under `results/operations/direct_parallel_v1/` before resumption. No TT data is involved.

Only execution and whole-conversation dispatch change. New code is under `operations/direct_parallel_v1/`, outside the original frozen source tree, with its own immutable manifest and archive. DirectCudaLlama inherits all inference methods from the original backend. Its copied initializer changes the explicit runtime context check and execution metadata; tests compare its computational initialization AST to the original. It does not fabricate SLURM_JOB_ID. Before full launch, every available GPU must exactly reproduce the token IDs and parsed JSON of a saved original-run note call, using its existing seed, prompt and schema. This is a targeted execution check, not proof that every future stochastic output is bit-identical.

## Architecture

One persistent tmux supervisor on `<gpu-node>` starts up to eight independent Python processes. Each is pinned to one physical GPU UUID and loads its own BF16 Llama plus CPU MiniLM. Shared model files are read-only. Every worker exclusively locks one conversation, executes the unchanged run_conversation function, freezes memory, answers its QA, then claims another conversation. Conversations never share a mutable graph, journal, checkpoint or QA file. conv-48 resumes from its checkpoint; other conversations begin fresh. Remaining conversations are ordered by note count to reduce the final imbalance. Ten conversations form the queue, so the first eight can run together and the final two start as workers finish. Updates within each conversation remain chronological.

The supervisor holds the legacy coordinator lock and all four old worker locks to prevent accidental concurrent execution of the old launcher. Per-GPU locks and per-conversation locks prevent duplicates in the new launcher. Locks release automatically on process exit. No manually deleting lock files is needed.

## Resource coexistence

Before model loading, validate <gpu-node>, actual tmux provenance, no Slurm job identity, one visible GPU, BF16 support, CUDA/NVML UUID agreement, no other compute PID and at least 28 GiB free VRAM. The supervisor also requires 32 GiB available host RAM before each launch. CPU thread settings stay at two per worker. No GPU resets, other-user signals, allocations or job submissions are performed.

Every ten seconds the supervisor samples all physical cards and its own process RAM/CPU counters. If a foreign compute PID appears on an occupied card, only our worker is asked to stop at a durable call boundary. If scheduled work starts on <gpu-node>, all our direct workers yield at a durable boundary. Stopped/busy workers wait at least 15 minutes before another availability check can relaunch them. A race remains between inspection and another process starting; this is not an exclusive scheduler reservation.

## Durable operations

Use the active home-backed project via either `${REPO_ROOT}/a-mem-locomo` or the canonical `${REPO_ROOT}/a-mem-locomo` symlink. Long operations always use tmux and this project's `.venv`.

Historical v1 commands: `python operations/direct_parallel_v1/start.py prepare` on the login node snapshots the clean stop, runs checks and freezes that adapter. On <gpu-node>, `python operations/direct_parallel_v1/start.py probe` or `python operations/direct_parallel_v1/start.py full` started `amem_direct_probe` and `amem_direct_full`. After the audited v2 handover, follow GPU_REUSE_RUNBOOK.md instead of restarting the historical full supervisor. Never launch a second full supervisor.

Authoritative live state is `results/operations/direct_parallel_v1/full_coordinator.json`, plus `workers/*.json`, `exits/*.json`, `logs/*.log`, `events.jsonl`, `full_telemetry.jsonl` and `failures/*.json`. Old Slurm files are historical. Actual progress is always the raw conversation memory.json, answers.jsonl and complete.json.

Create `results/operations/direct_parallel_v1/STOP` to request a full durable stop. Existing calls finish and are fsynced; next-call boundaries stop. SIGTERM/USR1 to an owned direct supervisor also requests this behavior. Unexpected crashes can lose unfinished computation, but durable completed calls and retry counts remain. Unknown failures are visible and are not automatically retried. A conversation failure does not discard or halt other independent conversations. Its failure marker blocks blind resumption. Inspect and record any explicit compatible recovery; never increase or reset the scientific retry budget.

## Additional ingestion cost reporting — user request, 10 September 2026

Record arithmetic mean input/output tokens and latency by note/evolution stage, per-committed-turn cost, median/p95, failure/retry costs, elapsed throughput and hardware/concurrency context. Existing raw logs provide these measurements. Use read-only `operations/ingestion_profile_v1/profile.py` with the project `.venv` in a separate tmux session, retaining logs and an exit record. It writes timestamped JSON and per-turn CSV under `results/operations/ingestion_profile_v1/`, verifies frozen hashes and archives its source. It does not alter inference or the original analyzer.

Recompute after ingestion ends and incorporate these distributions and accounting limitations in final artifact review. This step is not automatically invoked by the frozen analyzer. See `results/further_work/memory_ingestion_optimization.md` for required measurements, the partial baseline, hypotheses and comparison criteria. Keep sequential/direct phases separate, retain unknown interrupted costs, and distinguish cumulative worker-seconds from elapsed wall time.

## Timing and final report

Record sequential and parallel phases separately. Parallelism changes wall time and contention; aggregate build seconds are not elapsed wall time. Do not simply divide the old ETA by eight: conversations have different lengths, only ten exist, QA throughput has not yet been measured, and two conversations start later. Re-estimate from observed aggregate throughput and the slowest remaining conversation. Preserve these execution limitations in the final report. All 5,882 notes, 1,986 unique QA records, ten completions, six metrics and four figures are required. The original analyzer runs after successful inference; the final report still requires manual review, inclusion of the execution phase history, and PDF visual QA.
