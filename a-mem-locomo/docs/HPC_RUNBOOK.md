# HPC operational design

## Deployment and separation

Project root: `${REPO_ROOT}/a-mem-locomo`. SSH alias `<hpc-host>`. Use `.venv`; all setup/tests/launches/analysis use named tmux sessions. GPU workers run as Slurm-managed srun steps inside persistent batch allocations launched by the tmux coordinator. The Mac source tree is `agent_memory/A_Mem_reproduction`; it is separate from the old `agentic_memory` TT mirror. Source code and specifications may be deployed from Mac. No TT dataset, weights, outputs or checkpoints are imported.

**Storage workaround:** the canonical `/l` project path links to `${REPO_ROOT}/a-mem-locomo` on the HPC home filesystem. `/l` produced repeated write errors and then blocked even on the dataset file (`cl_sync_io_wait`). A 32 MiB home-filesystem write/fsync/read probe passed. The failed setup is preserved as `A_Mem_reproduction_lustre_setup_20260910` alongside the canonical path. Fresh active setup, models, checkpoints and logs use HPC home storage. This preserves the requested access path and separation from TT; it does not claim the shared filesystem's underlying problem has been diagnosed.

Original assets are downloaded directly on HPC: author A-Mem commit, original LoCoMo commit, pinned Meta weights, pinned MiniLM and official NeurIPS author kit. Setup verifies the known model shard and dataset hashes and records all asset hashes. Existing HPC Hugging Face authentication is used without printing or copying its token.

The first large model downloads to `/l` failed with writer errors (including `OSError 14: Bad address`). Their cause is unconfirmed. Weights are therefore independently downloaded to an HPC home cache and exposed through `models/weights/Llama-3.2-3B-Instruct` as a symlink. This is HPC-internal storage, with no TT transfer. The asset manifest records the resolved path; failed partial downloads stay excluded from inference and Git. Standard HTTP transport is used with Xet disabled.

## Resource policy verified on 10 September 2026

The `gpu` partition has four servers with eight RTX 5000 Ada Generation cards each. An allocated `nvidia-smi` probe reported 32,760 MiB per GPU and driver 570.195.03. Free capacity varies. Slurm accepted test requests for one GPU and 16 GiB host RAM, including an eight-hour time limit; two- and four-GPU requests failed `QOSMaxGRESPerUser`. A twelve-hour request failed the wall-time policy. Thus this deployment requests one GPU, four CPUs, 32 GiB RAM and four hours per allocation. It does not try to evade the quota with multiple jobs or accounts.

Four logical shards preserve independent memory states and the prior assignments. One is scheduled at a time; after a 3-hour-45-minute compute slice, the next shard is selected. Completion of a shard removes it from the rotation. Slurm queues the request if the GPU is unavailable. There is no shared mutable graph, and no other users' processes are touched. A physical-occupancy recheck supplements Slurm, as described below.

**Physical occupancy guard:** a later audit found that <gpu-node>'s Slurm-assigned GPU was already nearly full with another user's process. Slurm bookkeeping does not establish physical availability on this cluster. The active job prefers `<gpu-node>`, whose eight GPUs were independently observed idle, and checks the allocated GPU with NVML before model loading. Any other compute PID or less than 28 GiB free causes an exit without model loading, release of the allocation and a 15-minute retry delay. CUDA/NVML UUIDs must agree. No unallocated GPU is selected. Occupancy can still change after the check; a subsequent unexpected error stops for inspection. No user process or GPU is reset.

## Backend

Pinned PyTorch 2.7.1 CUDA 12.8 and Transformers 4.53.0 load the model entirely into BF16 on `cuda:0`, where the visible device is defined by Slurm. The backend refuses login-node execution, absent tmux-launch provenance, absent allocation or more than one visible GPU. MiniLM, tokenization, JSON token filtering, stochastic sampling and scoring stay on CPU. No CPU language-model fallback is allowed.

SDPA and a request-local dynamic KV cache support full-context prefill in 4,096-token chunks without deleting or summarizing context. Only final-position vocabulary logits are materialized. CUDA execution differs numerically from TT BFP8/BF16, and the new config/run ID changes deterministic request seeds. The sampling policy, prompts, memory evolution and scoring remain fixed. Python hash seed 42 improves repeatability relative to TT's unpinned hash ordering. Do not claim bit-identical cross-backend outputs.

Preflight must verify synthetic note/evolution/rewrite/answer calls, resume without new calls, no leaked distractor, and real 131,071-token prefill followed by one decode token (131,072 cached positions). The latter is a capacity test, not evidence that the model reasons accurately across the entire context.

## Checkpointing and shutdown

- Every call start and complete result is fsynced. Successful outputs can be replayed to finish an interrupted turn without generating them again.
- Completed schema failures consume a maximum of three attempts across restarts. Backend errors stop for investigation. Incomplete operational attempts retain their seed and available partial output.
- Each committed turn atomically replaces the memory checkpoint, preserving updated notes AND stale vectors/index documents.
- QA is a durable append-only journal; completed questions are skipped on restart. Frozen-memory fingerprints are verified before resuming.
- The allocation deadline requests a stop at a durable call boundary with exit 75. The coordinator resubmits unfinished work. Slurm USR1/TERM also requests a boundary stop. A hard preemption can lose an unfinished call, while prior durable artifacts remain.
- Unknown exits, OOM, missing exit records or vanished jobs stop automatic scheduling and require inspection. Do not call this automatic recovery from every failure.
- Source/config/model/dataset/environment hashes are frozen before full run; the launcher checks frozen source/config hashes on resumption. Models and datasets are independently verified at setup; dataset hash is also verified by each worker. Keep model files immutable.

## Files to inspect

- `results/setup/setup.log`, `setup.exit`, `checks.log`, `checks.exit`.
- `results/preflight/<run_id>/passed.json`, `context_capacity.json`, synthetic raw traces.
- `results/operations/full_coordinator.json`: authoritative scheduler state and current Slurm ID.
- `results/operations/job_<id>.worker.log`, `.exit.json`, `.telemetry.jsonl`, `slurm_<id>.log`.
- `results/raw_results/<run_id>/manifest.json`, `source_snapshot.tar.gz`, `worker_<n>.json`.
- `results/raw_results/<run_id>/conversations/<id>/{calls.jsonl,memory.json,memory_events.jsonl,memory_frozen.json,answers.jsonl,complete.json}`.

`squeue -u ashutosh.ganguly` and `python scripts/status.py` are read-only status commands. Resource telemetry samples only the UUID reported by our allocated worker, every 30 seconds. GPU memory/utilization/power, host RSS and cumulative CPU times are operational diagnostics. Per-call CUDA peaks are retained alongside timing.

## Interruption and intentional resume

If only SSH/Mac disconnects, the tmux coordinator and Slurm-managed worker continue. A new login can inspect them. The server-side coordinator schedules slices without the Mac; conversational updates still require an available Codex session.

If the coordinator itself ends, inspect `full_coordinator.json`, Slurm and the owned exit log before launching it again. A recorded failed coordinator is not automatically cleared. After establishing a safe operational cause, archive that coordinator JSON under a dated name and explicitly resume. Do not delete checkpoints or change the run ID to conceal a failure. Scientific changes require a distinct fresh experiment.

## Completion and reporting

The coordinator invokes offline analysis only after all ten conversation completion markers exist. Completion still requires matching note/question counts, frozen memory, six metrics, four figures, failure traces and manual PDF visual review. The report generator produces a NeurIPS-style `.tex` and PDF when the local compiler is installed. No submission or publication is automated.

Maintain `results/further_work/observations.md` throughout; label measured findings versus untested explanations. Preserve the user's edits. Never merge TT and HPC scores.

## Preflight launcher correction

The first detached compute-node tmux attempt (183439) ended with an empty shell-written exit record. Its exact early-exit cause is unconfirmed. An independent Slurm srun diagnostic (183441) passed CUDA, MiniLM and compute-node storage checks. The active launcher therefore uses a directly tracked srun worker step under the persistent login-node tmux coordinator, 32 GiB host RAM, atomic JSON exit records and Python fault diagnostics. This is an operational change before the scientific run; the failed attempt remains archived.

## Successful validation — 2026-09-10

Preflight job 183454 completed on <gpu-node>. All four LLM stages ran over six synthetic notes and four questions, and resume reused all completed calls. The real capacity probe cached 131,072 positions (131,071 input tokens plus one decode), taking 55.14 seconds with 25,914,792,448 bytes allocated peak and 32,858,177,536 bytes reserved peak. These are stress-test measurements, not normal query memory or latency. The independent FP32 chunked/unsplit check differed by at most 0.00006652; BF16 differences were recorded separately. The full experiment remains BF16.
