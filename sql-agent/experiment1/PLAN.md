# Experiment 1 — Controller + DB Agent baseline on BiomedSQL

**Status:** planned, not run.  
**Goal:** Put a ChEMBL-style two-LLM dialogue channel between the English question and the BiomedSQL BigQuery DB, and measure whether brief-agreement before SQL helps vs single-shot / BMSQL.

## Why this experiment

BiomedSQL (COLM 2026) shows a large gap: Gemini-3-Pro **58.1% EX**, their multi-step agent BMSQL **62.6% EX**, experts **~90%**. Failures are often scientific conventions (GWAS thresholds, effect direction, trial phase) that are not in the schema.

Our ChEMBL exp9 architecture already separates **agreeing what to find** from **executing the search**:

```
controller  ↔  db agent   (bounded talk, dual scores ≥ 8/10 or 6 turns)
     ↓
db agent ReAct over DB
```

Exp1 ports only that spine to BiomedSQL. **No workers, no curators, no molecular oracle.** One question → one dialogue → one SQL search → one answer.

## Architecture (baseline)

```
NL question (BiomedSQL)
        │
        ▼
┌───────────────────┐     shared channel      ┌────────────────────┐
│ CONTROLLER        │◄───────────────────────►│ DB AGENT           │
│ Understand NL /   │  understanding 0–10     │ Restate + search   │
│ scientific intent │  satisfaction 0–10      │ plan; write SQL    │
│ Refine the brief  │  stop if both ≥ 8       │                    │
└───────────────────┘  or after 6 turns       └─────────┬──────────┘
                                                        │
                                                        ▼
                                              BigQuery (BiomedSQL)
                                                        │
                                                        ▼
                                              NL answer + EX / JAC
```

### Role split (two prompts, same or different model)

| Role | Job | Must not |
|------|-----|----------|
| **Controller** | Own the scientific need: entities, implied thresholds, what a useful answer looks like. Rate `satisfaction`. | Write SQL / name tables in talk phase (mirror ChEMBL: idea-level only). |
| **DB agent** | Restate brief, ask clarifying questions, rate `understanding`, then after agreement run SQL ReAct against schema + BigQuery. | Invent scientific thresholds the controller did not agree. |

Talk JSON mirrors exp9:

- DB turn: `{restatement, plan, questions, understanding}`
- Controller turn: `{reasoning, reply, need, satisfaction}`
- Pass: `understanding ≥ 8` and `satisfaction ≥ 8`, else stop at turn 6 with last `need` as the brief.

### After agreement: DB search loop

Borrow BiomedSQL tooling + ChEMBL ReAct budget:

1. Propose read-only BigQuery SQL (schema from `table_info*.txt`).
2. Execute via their `handlers/gcp/big_query.py` (or thin wrapper).
3. On error / empty / obviously wrong shape → revise (≤ 6 rounds, ≤ 5 statements/round, `LIMIT 100` default like BMSQL).
4. Produce final SQL + short NL answer from rows.

Optional later (not in exp1): fulfillment self-score of rows vs brief (like ChEMBL `db_score`), without changing EX.

## What we compare against

On the **same 546-question `dev_sample` / paper eval set**:

| Condition | Description |
|-----------|-------------|
| **A. Single-shot** | Their baseline SQL prompt (schema + question → SQL). |
| **B. BMSQL** | Their multi-step pipeline (column select → SQL → refine thresholds → answer). Reuse upstream if keys exist. |
| **C. Ours (exp1)** | Controller↔DB talk → ReAct SQL → answer. |

Primary metrics (paper defs): **EX**, **JAC**, **SER**. Secondary: talk turns to convergence, token cost, qualitative failure codes (wrong threshold, wrong table, entity miss, syntax).

## Eval slice for first run

- Full official eval set is already local: `data/benchmark_dev_sample.csv` (**546** questions, **38** templates).
- First smoke: **1 question per template** (~38) before full 546.
- Stratify reporting by `bio_category` (GWAS / SMR / Approval / Metadata / …) — paper’s hardest gaps are threshold-heavy categories.

## Implementation sketch (when greenlit)

```
crack_biomedsql/
  experiment1/
    PLAN.md              ← this file
    prompts.py           ← controller_talk / db_talk / db_query / answer (adapt from PABLO/exp9)
    talk.py              ← copy exp9 talk() loop (TALK_MAX=6, TALK_PASS=8)
    search.py            ← ReAct SQL against BigQuery
    run.py               ← per-question pipeline + logging
    eval.py              ← EX / JAC vs gold execution_results
    table_info*.txt      ← schema (copied from upstream)
  data/                  ← benchmark CSVs
  upstream_biomedsql/    ← NIH-CARD code (eval helpers, BQ client)
  research/              ← papers
```

Reuse, don’t reinvent: `upstream_biomedsql/handlers/gcp/big_query.py`, EX/JAC helpers in `utils/`, schema prompts.

## Infra decisions (updated)

### LLM — Qwen3.8-27B (local vLLM), no cloud needed
Same stack as PABLO exp9:
- Model cache: `~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B`
- Serve: `vllm serve Qwen/Qwen3.8-27B --served-model-name controller` (4 GPUs, see `PABLO/exp9/exp9.sbatch`)
- Client: OpenAI-compatible `http://127.0.0.1:8000`, both controller + DB agent hit the same endpoint with different prompts
- Venv: `<local-home>/pablo-cu129`

### DB — local SQLite from HF parquet, **no GCP**
1. Download `db_data/*.parquet` from HF into `data/db_data/`.
2. `build_sqlite.py` materializes `data/biomedsql.sqlite`.
3. Rewrite BigQuery refs: `` `{project_id}.{dataset_name}.Table` `` → `Table`; `"strings"` → `'strings'`.
4. **EX/JAC** vs CSV `execution_results`.

### Still needed before run
1. Finish parquet download + `build_sqlite.py` (Alzheimer table is required for eval).
2. GPU node + start vLLM Qwen3.8-27B (same as exp9).
3. Smoke: `--one-per-template` (~38) then full 546.

## Success criteria for “baseline works”

- Smoke (38): pipeline completes end-to-end with logs of talk scores + SQL + EX.
- Full (546): EX reported with CI; at least competitive with single-shot on threshold categories, or clear failure analysis if not.
- Logs show dialogue actually changing the brief (not rubber-stamping) on a manual sample of 10 traces.

## Explicit non-goals (exp1)

- No worker / edit / molecule oracle.
- No persistent task/tool memory across questions (per-question fresh).
- No beating BMSQL on day one — establish the dialogue baseline first.
