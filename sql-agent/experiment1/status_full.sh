#!/bin/bash
# Quick status for 2x full run.
python3 - <<'PY'
import json, re
from pathlib import Path
runs = Path("${REPO_ROOT}/sql-agent/experiment1/runs")
for wid in (0, 1):
    logs = sorted(runs.glob(f"exp1-full-w{wid}-*.jsonl"))
    logs = [p for p in logs if "latest" not in p.name]
    if not logs:
        print(f"w{wid}: no jsonl yet")
        continue
    p = logs[-1]
    results = []
    for line in p.read_text().splitlines():
        r = json.loads(line)
        if r.get("action") == "result":
            results.append(r)
    n = len(results)
    if n:
        ex = sum(r["EX"] for r in results) / n
        print(f"w{wid}: {n}/273  EX={ex:.3f}  file={p.name}")
    else:
        print(f"w{wid}: 0/273  loading/talking  file={p.name}")
PY
echo '---'
ssh -o StrictHostKeyChecking=no gpu-04 'pgrep -af "run.py|vllm serve" | grep -v pgrep | head -3 || echo w0 idle'
ssh -o StrictHostKeyChecking=no gpu-02 'pgrep -af "run.py|vllm serve" | grep -v pgrep | head -3 || echo w1 idle'
