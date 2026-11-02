# HPC build status — 10 September 2026

**Infrastructure is built and the fresh full LoCoMo run is running. Benchmark results are not complete.**

- Canonical access: `${REPO_ROOT}/a-mem-locomo`.
- Active storage: `${REPO_ROOT}/a-mem-locomo`, reached through the canonical symlink after `/l` write failures.
- Run ID: `locomo_llama32_3b_amem_cuda_v1`.
- Started: 13:23:42 UTC, 10 September 2026.
- Initial Slurm job: **183458**, node **<gpu-node>**, one RTX 5000 Ada GPU.
- Persistent coordinator: tmux **amem_hpc_full**. Its current job and state are in `results/operations/full_coordinator.json`.
- First real progress verified: **5 saved LoCoMo notes, 0 QA**, at approximately 13:25 UTC. Later status must be read from the live artifacts.

## Verified before full inference

Twenty software tests passed. Independently downloaded Llama and LoCoMo files passed their pinned checksums; the dataset contains ten conversations, 5,882 turns and 1,986 questions. The successful GPU preflight (183454) exercised all four LLM roles, six synthetic notes and four synthetic questions, and resumed without regenerating completed calls.

The real context test processed 131,071 input tokens plus one decode token, reaching 131,072 cached positions. It took 55.14 seconds, with 24.14 GiB peak allocated GPU memory and 30.60 GiB peak reserved memory. These are context stress-test figures, not typical inference usage. FP32 chunked/unsplit correctness was independently checked; the actual experiment remains BF16.

Forty-nine source, configuration, upstream-reference, environment and asset-metadata files were hashed before the full run. The immutable manifest and source snapshot are in the run's raw-results directory.

## What stays fixed

The same pinned Llama-3.2-3B-Instruct, CPU MiniLM, A-Mem prompts/evolution/retrieval semantics, full LoCoMo evaluation, six primary metrics, four planned diagnostic views and observations workflow are retained. CUDA BF16 replaces native TT execution; its numerical differences are documented. Four logical conversation shards remain isolated and run within the one-GPU account quota.

TT's 732-note run is preserved separately and stopped. No TT weights, dataset, checkpoints or results were transferred to HPC. Scientific results must never be merged across the two runs.

## Persistence and follow-up

Each completed call is journaled durably; every turn is checkpointed atomically; QA uses frozen memory and per-question records. Slurm compute slices stop at durable boundaries and the tmux coordinator schedules unfinished shards. Physically occupied allocations are released and retried after 15 minutes. Unexpected failures stop for inspection. The existing Codex monitor checks this HPC experiment every 15 minutes and updates `results/further_work/observations.md`.

The final metrics, figures, failure-case analysis and NeurIPS-format report follow completion of all 5,882 notes and 1,986 questions. The report is not yet complete or submitted.

See [the full TT implementation PRD](PRD_TT_AS_BUILT.md) and [HPC runbook](HPC_RUNBOOK.md).
