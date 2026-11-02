#!/usr/bin/env python3
"""Merge parallel full-run jsonl shards and print EX/JAC/SER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonls", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for path in args.jsonls:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get("action") == "result":
                results.append(r)

    # de-dupe by uuid (keep last)
    by_uid = {r["uuid"]: r for r in results}
    results = list(by_uid.values())
    n = len(results)
    if not n:
        raise SystemExit("no results")

    summary = {
        "action": "summary_merged",
        "n": n,
        "EX": sum(r.get("EX", 0) for r in results) / n,
        "JAC": sum(r.get("JAC", 0) for r in results) / n,
        "SER": sum(r.get("SER", 0) for r in results) / n,
        "wins": sum(1 for r in results if r.get("EX") == 1),
        "sources": [str(p) for p in args.jsonls],
    }
    print(json.dumps(summary, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.write(json.dumps(summary, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
