# autonomous-agent-memory

Protocols and code for **long-horizon LLM agents with explicit memory**: A-Mem on LoCoMo (NVIDIA HPC reproduction stack) and an inference-only Mem-α → LoCoMo generalization comparison.

**Author:** Ashutosh Ganguly — AI researcher @ MBZUAI.

## Tracks

| Track | Path | Question |
|-------|------|----------|
| A-Mem @ LoCoMo | [`a-mem-locomo/`](a-mem-locomo/) | Reproduce / adapt A-Mem with Llama-3.2-3B-Instruct; six primary metrics; resumable CUDA/Slurm runners |
| Mem-α generalization | [`memalpha-locomo-generalization/`](memalpha-locomo-generalization/) | Does RL-trained Mem-α-4B writer beat frozen Qwen3-4B on LoCoMo under a frozen protocol? |

## Headline results (Mem-α track)

Full coverage **3972/3972** QA, **20/20** memories. From committed `runs/official_memalpha_locomo_v1/report/`:

| Metric | Mem-α-4B writer | Base Qwen3-4B writer |
|--------|----------------:|---------------------:|
| Overall micro | 0.3255 | **0.3376** |
| Category macro | 0.2936 | **0.3112** |
| Non-adversarial micro | 0.1841 | **0.2030** |
| Adversarial | **0.8139** | 0.8027 |

Honest takeaway: under this frozen inference-only protocol, the RL-trained writer does **not** dominate the base writer on overall micro-F1; adversarial is slightly better for Mem-α. See report for category breakdowns and limitations.

## A-Mem metrics protocol

Six primary metrics (see track README / docs): (1) category F1 / adversarial abstention, (2) gold-evidence coverage, (3) memory-build & QA latency, (4) tokens by phase, (5) semantic footprint + link counts, (6) context overflow / utilization. Partial runs are labeled partial. This is an experimental adaptation, **not** a claim of exact paper numerical reproduction.

## Related work (not in this repo)

Agentic BiomedSQL controller+DB baseline (local SQLite, 546-question sample) used as a separate agentic eval line — ask for `crack_biomedsql` privately if needed.

## Limitations / non-claims

- Weights, raw completion dumps, and large result shards are excluded; regenerate via scripts + HF auth.
- Mem-α study is inference-only (no training/reward updates) and uses official LoCoMo prompts/scoring with adapter bridges — not an end-to-end replica of every paper reader setting.
- A-Mem CUDA stack assumes Slurm/HPC; laptop smoke tests are limited.
- Absolute cluster paths have been parameterized; you must set local roots.

## License

MIT for original packaging/scripts here. Upstream A-Mem / Mem-α / LoCoMo / model licenses still apply.
