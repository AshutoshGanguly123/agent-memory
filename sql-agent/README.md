# sql-agent (BiomedSQL)

Controller + DB-agent dialogue for biomedical text-to-SQL on a **local SQLite** rewrite of the BiomedSQL knowledge base.

## Claim

On the **546-question / 38-template** BiomedSQL-style eval, our two-agent stack reaches **69.2% EX** — above the BiomedSQL paper’s reported **Gemini-3-Pro 58.1% EX** and **BMSQL 62.6% EX** under our protocol (see caveats).

## Headline metrics

From [`metrics/summary.json`](metrics/summary.json) (computed from the full merged run; raw jsonl not shipped):

| System | Exact-match EX |
|--------|---------------:|
| **Ours (controller + DB agent, Qwen3.8-27B)** | **69.2%** |
| BiomedSQL paper — BMSQL multi-step agent | 62.6% |
| BiomedSQL paper — Gemini-3-Pro | 58.1% |

Also: mean JAC **71.0%**, mean SER **~0.002**, syntax_ok **high**. Per-template rollup: `metrics/per_template_ex.csv`.

**Δ EX vs Gemini ≈ +11.1 pp; vs BMSQL ≈ +6.6 pp.**

## Method

1. **Controller** owns the scientific need (entities, thresholds, success criteria) and rates satisfaction.
2. **DB agent** restates the brief, clarifies, then runs a read-only SQL ReAct loop against schema + SQLite.
3. Agreement gate before search; bounded revise rounds on error/empty results.
4. Graded with EX / JAC / SER aligned to the BiomedSQL evaluation spirit (`experiment1/eval_metrics.py`).

Architecture notes: `experiment1/PLAN.md`, prompts: `experiment1/PROMPTS.md`.

## Setup

```bash
# Build SQLite KB once (parquet → sqlite); see experiment1/build_sqlite.py
python3 experiment1/build_sqlite.py --parquet-dir <data> --out biomedsql.sqlite --only-core

# Smoke / full runs need a local OpenAI-compatible LLM endpoint (we used vLLM Qwen3.8-27B)
python3 experiment1/run.py --one-per-template
```

Weights and the 73MB raw run dumps are **not** in git. Metrics summaries are.

## Papers in-tree

- [`papers/BiomedSQL_COLM2026.pdf`](papers/BiomedSQL_COLM2026.pdf) — benchmark + published baselines
- [`papers/Controller-Guided_ChEMBL_Search_for_Agentic_Molecular_Design.pdf`](papers/Controller-Guided_ChEMBL_Search_for_Agentic_Molecular_Design.pdf) — controller/DB dialogue lineage
- [`papers/our_technical_note.pdf`](papers/our_technical_note.pdf) — local technical write-up

## Limitations / non-claims

- Local **SQLite** KB + schema projection is not identical to the paper’s BigQuery deployment.
- Model stack (**Qwen3.8-27B** via vLLM) differs from Gemini / BMSQL paper configurations — treat baseline deltas as **cross-system** comparisons with a fixed public question set, not a bitwise reproduction.
- EX uses our grader on execution results; always cite `metrics/summary.json` when quoting numbers.
- Dataset tables / DB file are rebuilt locally (licensing).

## License

MIT for agent code here. Upstream BiomedSQL / model / data licenses still apply.
