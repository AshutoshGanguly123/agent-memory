# autonomous-agent-memory

Long-horizon LLM agents with explicit memory, plus a biomedical SQL agent.

**Author:** Ashutosh Ganguly — AI researcher @ MBZUAI.

## Tracks

| Track | Path | Notes |
|-------|------|-------|
| SQL agent | [`sql-agent/`](sql-agent/) | 69.2% EX on 546Q with **Qwen3.8-27B**; paper BMSQL 62.6% used **GPT-o3-mini** — not matched. Controller ablation pending. |
| A-Mem @ LoCoMo | [`a-mem-locomo/`](a-mem-locomo/) | Reproduction / adaptation harness; six primary metrics |
| Mem-α generalization | [`memalpha-locomo-generalization/`](memalpha-locomo-generalization/) | Inference-only Mem-α vs base writer; base slightly higher overall micro; diagnosis notes under `diagnosis/` |

## References

[`reference_research_papers/`](reference_research_papers/) — arXiv / venue links only (no PDF bulk).

## License

MIT for original packaging; upstream licenses apply.
