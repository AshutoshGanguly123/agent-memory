# sql-agent (BiomedSQL)

Controller + DB-agent dialogue for biomedical text-to-SQL on a **local SQLite** copy of the BiomedSQL knowledge base.

**Author:** Ashutosh Ganguly — AI researcher @ MBZUAI.

## What this is

A two-agent stack (controller agrees on the scientific brief; DB agent runs bounded SQL ReAct) evaluated on **546 BiomedSQL-style questions / 38 templates**.

## Numbers (read carefully)

| System | Exact-match EX | Model / stack |
|--------|---------------:|---------------|
| Ours (controller + DB agent) | **69.2%** | **Qwen3.8-27B** (vLLM), local SQLite |
| BiomedSQL paper — BMSQL | 62.6% | **GPT-o3-mini** (paper setup) |
| BiomedSQL paper — Gemini-3-Pro | 58.1% | Gemini (paper setup) |

These are **not matched-model comparisons**. A large part of the gap vs 62.6% may be the backbone LLM, not the controller. The matched ablation (`--no-controller`, same model/DB/grader) is implemented but **not yet re-run** — see `metrics/controller_ablation.json`.

Cost proxies on the full run (`metrics/cost_table.json`): mean talk_turns ≈ 1.10, mean SQL queries ≈ 2.57, mean rounds ≈ 2.30.

Gold audit: `metrics/gold_audit.json` — 546 rows, 0 malformed under strict parse (105 empty gold lists).

## Method

1. Controller drafts the scientific need.
2. Brief talk until agreement (skipped with `--no-controller`).
3. DB agent SQL ReAct against schema + SQLite.
4. Grade with EX / JAC / SER (`experiment1/eval_metrics.py`).

## Setup

```bash
python3 experiment1/build_sqlite.py --parquet-dir <data> --out biomedsql.sqlite --only-core
python3 experiment1/run.py --one-per-template
python3 experiment1/run.py --no-controller --one-per-template   # matched ablation
```

## Venue note

BiomedSQL is an **ICLR 2026 workshop** dataset/paper. Prefer the authors’ official page for the canonical citation.

## Limitations

- SQLite rewrite ≠ paper BigQuery deployment.
- Cross-paper EX numbers mix different models and stacks.
- Raw jsonl / DB dumps not shipped; summary metrics only.

## References (links)

See [`../reference_research_papers/`](../reference_research_papers/) and the BiomedSQL / ReAct / tool-use abstracts linked there.

## License

MIT for agent code. Upstream dataset / model licenses still apply.
