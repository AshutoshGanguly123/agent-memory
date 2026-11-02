# autonomous-agent-memory

Long-horizon **LLM agents** with explicit memory and tool use — plus a biomedical **SQL agent** that beats published BiomedSQL baselines under a transparent local protocol.

**Author:** Ashutosh Ganguly — AI researcher @ MBZUAI.

## Tracks

| Track | Path | Headline |
|-------|------|----------|
| **SQL agent (BiomedSQL)** | [`sql-agent/`](sql-agent/) | **69.2% EX** on 546 Q vs paper Gemini **58.1%** / BMSQL **62.6%** |
| A-Mem @ LoCoMo | [`a-mem-locomo/`](a-mem-locomo/) | Resumable A-Mem reproduction stack; six primary metrics; not exact paper-score claim |
| Mem-α generalization | [`memalpha-locomo-generalization/`](memalpha-locomo-generalization/) | Inference-only Mem-α-4B vs base Qwen3-4B on full LoCoMo (3972/3972); base slightly higher overall micro |

## Strongest systems result (SQL agent)

Controller + DB-agent dialogue on a SQLite BiomedSQL KB:

| System | EX |
|--------|---:|
| **Ours** | **69.2%** |
| BMSQL (paper) | 62.6% |
| Gemini-3-Pro (paper) | 58.1% |

Details, caveats, and code: [`sql-agent/README.md`](sql-agent/README.md). Metrics: [`sql-agent/metrics/summary.json`](sql-agent/metrics/summary.json).

## Memory tracks (honest)

**Mem-α LoCoMo** (committed report): overall micro **0.3255 (Mem-α)** vs **0.3376 (base)** — RL writer does **not** dominate under this frozen inference-only protocol; adversarial is slightly better for Mem-α (0.8139 vs 0.8027). That negative/nuanced result is intentional transparency.

**A-Mem:** protocol + harness for long-context memory agents with latency/token/footprint metrics; partial runs labeled partial.

## Limitations

- Memory tracks exclude weights and raw shards; SQL agent excludes raw jsonl / DB dumps.
- Cross-paper baseline comparisons always state stack differences.
- HPC launchers assume Slurm-class environments.

## Reference papers

[`reference_research_papers/`](reference_research_papers/) — BiomedSQL, agent memory, tool-use.

## License

MIT for original packaging here; upstream licenses apply to vendored/benchmark assets.
