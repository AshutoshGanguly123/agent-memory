"""Finish audit and packaging on the HPC after the durable pipeline completes."""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import atomic_json, config, now, read_json

DIRECTORY = ROOT / "results/finalization"


def state(status, **fields):
    receipt = {"status": status, "utc": now(), **fields}
    atomic_json(DIRECTORY / "status.json", receipt)
    print(receipt, flush=True)


def main():
    if not os.environ.get("TMUX"):
        raise RuntimeError("Run the finalizer in a durable HPC tmux session")
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    with (DIRECTORY / "finalize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state("waiting_for_pipeline")
        while True:
            if (DIRECTORY / "STOP").exists():
                state("stopped")
                return 75
            pipeline = read_json(ROOT / "results/pipeline/status.json")
            if pipeline["state"] == "complete":
                break
            if pipeline["state"] in {"failed", "complete_with_errors", "stop_requested"}:
                state("blocked", pipeline=pipeline)
                return 2
            time.sleep(30)
        base = ROOT / "runs" / config()["run_id"]
        commands = [
            [sys.executable, str(ROOT / "verification/verify_run.py"), str(ROOT),
             "--output", str(base / "verification/integrity.json")],
            [sys.executable, str(ROOT / "verification/package_run.py"), str(ROOT)],
        ]
        for phase, command in zip(("verifying", "packaging"), commands):
            if (DIRECTORY / "STOP").exists():
                state("stopped")
                return 75
            state(phase, command=command)
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                state("failed", phase=phase, exit_code=result.returncode)
                return result.returncode
        state("complete", report=str(base / "report/report.md"),
              integrity=str(base / "verification/integrity.json"),
              manifest=str(base / "run_manifest.json"))
        return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        state("failed", error=repr(exc))
        raise
