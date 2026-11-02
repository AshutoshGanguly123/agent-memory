#!/usr/bin/env python3
"""Experiment 1: controller + DB agent on BiomedSQL (SQLite).

Pipeline per question:
  1. controller_seed  → initial scientific need
  2. talk             → dual-score dialogue until ≥8/10 or 6 turns
  3. db_react         → iterative SQL against SQLite
  4. answer           → NL answer from rows
  5. EX / JAC vs gold execution_results

Does not start the model server. Point --model-url at a running vLLM
(Qwen3.8-27B), same as PABLO exp9.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(10_000_000)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from eval_metrics import execution_accuracy, jaccard  # noqa: E402
from llm import Log  # noqa: E402
from search import db_react, write_answer  # noqa: E402
from talk import seed_need, talk  # noqa: E402

DEFAULT_BENCH = ROOT.parent / "data" / "benchmark_dev_sample.csv"
DEFAULT_DB = ROOT.parent / "data" / "biomedsql.sqlite"


def load_benchmark(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pick_rows(rows: list[dict], args) -> list[dict]:
    if args.uuid:
        wanted = set(args.uuid)
        picked = [row for row in rows if row["uuid"] in wanted]
        if len(picked) != len(wanted):
            missing = wanted - {row["uuid"] for row in picked}
            raise SystemExit(f"uuids not found: {sorted(missing)}")
        return picked
    if args.one_per_template:
        seen: set[str] = set()
        picked = []
        for row in rows:
            tid = row["template_uuid"]
            if tid in seen:
                continue
            seen.add(tid)
            picked.append(row)
        return picked
    if args.limit is not None:
        return rows[: args.limit]
    return rows


def run_one(args, log: Log, row: dict) -> dict:
    qid = row["uuid"]
    question = row["question"]
    print(f"\n=== {qid}: {question} ===", flush=True)

    need = seed_need(args, log, qid, question)
    log.write({"uuid": qid, "action": "seed", "question": question, "need": need})
    print(f"[{qid}] seed need: {need[:200]}", flush=True)

    session = talk(args, log, qid, need)
    brief = session["brief"]
    print(f"[{qid}] brief ({session['stop_reason']}, {session['n_turns']} turns): {brief[:200]}", flush=True)

    search = db_react(args, log, qid, brief, args.db)
    answer = write_answer(args, log, qid, question, brief, search)

    gold = row.get("execution_results") or ""
    ex = execution_accuracy(search.get("rows") or [], gold)
    jac = jaccard(search.get("rows") or [], gold)
    syntax_ok = 1.0 if search.get("best_sql") and not (
        search.get("sql_errors") == search.get("n_queries") and search.get("n_queries")
    ) else (1.0 if search.get("best_sql") else 0.0)
    # SER: failed to produce any executable best_sql
    ser = 0.0 if search.get("best_sql") else 1.0

    result = {
        "uuid": qid,
        "template_uuid": row.get("template_uuid"),
        "question": question,
        "gold_answer": row.get("answer"),
        "need": need,
        "brief": brief,
        "talk_turns": session["n_turns"],
        "talk_stop": session["stop_reason"],
        "sql": search.get("best_sql"),
        "n_rows": len(search.get("rows") or []),
        "n_queries": search.get("n_queries"),
        "sql_errors": search.get("sql_errors"),
        "rounds": search.get("rounds"),
        "answer": answer,
        "EX": ex,
        "JAC": jac,
        "SER": ser,
        "syntax_ok": syntax_ok,
    }
    log.write({"uuid": qid, "action": "result", **result})
    print(
        f"[{qid}] EX={ex} JAC={jac:.3f} SER={ser} rows={result['n_rows']} sql={str(result['sql'])[:120]}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="BiomedSQL exp1: controller + DB agent")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCH)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--model-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model-name", default="controller")
    parser.add_argument("--out", type=Path, default=None, help="jsonl results path")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--one-per-template", action="store_true")
    parser.add_argument("--uuid", nargs="+", default=None)
    parser.add_argument("--reason-temperature", type=float, default=0.4)
    parser.add_argument("--talk-tokens", type=int, default=800)
    parser.add_argument("--summary-tokens", type=int, default=400)
    parser.add_argument("--query-tokens", type=int, default=1200)
    parser.add_argument("--observe-tokens", type=int, default=600)
    parser.add_argument("--answer-tokens", type=int, default=500)
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"SQLite DB missing: {args.db}\nBuild it with build_sqlite.py first.")
    if not args.benchmark.exists():
        raise SystemExit(f"benchmark missing: {args.benchmark}")

    rows = pick_rows(load_benchmark(args.benchmark), args)
    stamp = Path(args.out) if args.out else (
        ROOT / "runs" / f"exp1-{Path(args.db).stem}.jsonl"
    )
    # unique stamp if default
    if args.out is None:
        from datetime import datetime

        stamp = ROOT / "runs" / f"exp1-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"

    log = Log(stamp)
    log.write(
        {
            "action": "init",
            "experiment": "biomedsql_exp1",
            "n_questions": len(rows),
            "model": args.model_name,
            "model_url": args.model_url,
            "db": str(args.db),
            "benchmark": str(args.benchmark),
        }
    )
    print(f"running {len(rows)} questions → {stamp}", flush=True)

    results = []
    try:
        for row in rows:
            try:
                results.append(run_one(args, log, row))
            except Exception as exc:  # noqa: BLE001
                print(f"[{row['uuid']}] FAILED: {exc}", flush=True)
                log.write({"uuid": row["uuid"], "action": "error", "error": str(exc)})
                results.append({"uuid": row["uuid"], "EX": 0.0, "JAC": 0.0, "SER": 1.0, "error": str(exc)})
    finally:
        if results:
            n = len(results)
            mean_ex = sum(r.get("EX", 0) for r in results) / n
            mean_jac = sum(r.get("JAC", 0) for r in results) / n
            mean_ser = sum(r.get("SER", 0) for r in results) / n
            summary = {"action": "summary", "n": n, "EX": mean_ex, "JAC": mean_jac, "SER": mean_ser}
            log.write(summary)
            print(json.dumps(summary, indent=2), flush=True)
        log.close()


if __name__ == "__main__":
    main()
