#!/usr/bin/env python3
"""Smoke-test BigQuery→SQLite rewrite + gold query execution (no LLM)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

csv.field_size_limit(10_000_000)

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from db import execute_select, list_tables, rewrite_sql  # noqa: E402
from eval_metrics import execution_accuracy, jaccard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=ROOT.parent / "data" / "biomedsql.sqlite")
    parser.add_argument("--benchmark", type=Path, default=ROOT.parent / "data" / "benchmark_dev_sample.csv")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    if not args.db.exists():
        raise SystemExit(f"missing db: {args.db}")

    tables = list_tables(args.db)
    print("tables:", tables)

    with args.benchmark.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))[: args.limit]

    ok = 0
    for row in rows:
        gold_sql = row["benchmark_query"]
        rewritten = rewrite_sql(gold_sql)
        result = execute_select(args.db, gold_sql)
        ex = execution_accuracy(result["rows"], row["execution_results"])
        jac = jaccard(result["rows"], row["execution_results"])
        status = "OK" if not result["error"] and ex == 1.0 else "FAIL"
        if status == "OK":
            ok += 1
        print(
            f"{status} {row['uuid']} EX={ex} JAC={jac:.2f} "
            f"err={result['error']!r} rows={len(result['rows'])}"
        )
        if status == "FAIL":
            print("  gold:", gold_sql[:180].replace("\n", " "))
            print("  rw  :", rewritten[:180].replace("\n", " "))
    print(f"{ok}/{len(rows)} gold queries matched EX=1 on SQLite")


if __name__ == "__main__":
    main()
