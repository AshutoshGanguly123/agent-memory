"""Durable supervisor for the already-frozen full experiment, never training."""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time

from common import ROOT, atomic_json, now, read_json, run_root, verify_freeze

PYTHON = str(ROOT / ".venv/bin/python")
STATE = ROOT / "results/pipeline/status.json"
OPS = ROOT / "results/operations"
ACTIVE_STATES = {"starting", "running"}
TERMINAL_STATES = {"complete", "failed", "stopped"}
STARTUP_STATUS_TIMEOUT_SECONDS = 180


class StopRequested(Exception):
    """The supervisor stops while durable lease monitors drain owned children."""


def check_stop(current_operation=None):
    if not (ROOT / "results/pipeline/STOP").exists():
        return
    operations = {path.name for path in OPS.glob("full_*") if path.is_dir()}
    if current_operation:
        operations.add(current_operation)
    active = []
    for name in sorted(operations):
        status_path = OPS / name / "status.json"
        if status_path.exists() and read_json(status_path).get("state") in TERMINAL_STATES:
            continue
        atomic_json(OPS / name / "STOP", {"requested_utc": now()})
        active.append(name)
    update(state="stop_requested", stopping_operations=active)
    raise StopRequested()


def update(**fields):
    previous = read_json(STATE) if STATE.exists() else {}
    previous.update(fields, updated_utc=now())
    atomic_json(STATE, previous)
    print(json.dumps(fields), flush=True)


def dispatch(stage, count, args, attempt):
    check_stop()
    name = f"full_{stage}_{attempt:03d}"
    environment = []
    compatibility = ROOT / "results/setup/reader_runtime.json"
    if stage == "reader" and compatibility.exists():
        settings = read_json(compatibility).get("environment", {})
        allowed = {"NCCL_P2P_DISABLE", "NCCL_IB_DISABLE", "NCCL_DEBUG", "NCCL_SOCKET_IFNAME", "GLOO_SOCKET_IFNAME"}
        if not set(settings) <= allowed:
            raise RuntimeError("Unreviewed reader runtime setting")
        environment = ["env", *(f"{key}={value}" for key, value in sorted(settings.items()))]
    command = [PYTHON, str(ROOT / "ops.py"), "dispatch", "--name", name,
               "--gpu-count", str(count), "--min-host-gib", str(128 if count == 4 else 64),
               "--", *environment, PYTHON, str(ROOT / "run.py"), *args]
    result = subprocess.run(command, capture_output=True, text=True, timeout=150)
    if result.returncode:
        check_stop(name)
        latest = OPS / name / "status.json"
        if latest.exists() and read_json(latest).get("child_pid"):
            raise RuntimeError("A dispatched child needs inspection: " + result.stderr[-3000:])
        if "no_available_node" not in result.stderr:
            raise RuntimeError(result.stderr[-3000:] or result.stdout[-3000:])
        update(stage=stage, state="waiting_for_resources", last_dispatch_error=result.stderr[-3000:])
        return None
    receipt = json.loads(result.stdout)
    atomic_json(ROOT / "results/pipeline" / (name + ".json"), {"command": command, "receipt": receipt, "utc": now()})
    check_stop(name)
    update(stage=stage, state="running", operation=name, dispatch=receipt)
    return name


def wait_operation(name):
    status_deadline = time.monotonic() + STARTUP_STATUS_TIMEOUT_SECONDS
    while True:
        check_stop(name)
        status_path = OPS / name / "status.json"
        if not status_path.exists():
            if time.monotonic() >= status_deadline:
                raise RuntimeError("Dispatched operation produced no status; inspect before retry: " + name)
            time.sleep(15)
            continue
        status = read_json(status_path)
        if status["state"] in TERMINAL_STATES:
            return status
        if status["state"] not in ACTIVE_STATES:
            raise RuntimeError("Unknown operation state: " + json.dumps(status))
        if status["state"] == "starting" and time.monotonic() >= status_deadline:
            raise RuntimeError("Operation never confirmed its child; inspect before retry: " + name)
        time.sleep(15)


def _run():
    if not os.environ.get("TMUX"):
        raise RuntimeError("Run the pipeline in a durable login-node tmux session")
    lock_path = ROOT / "results/pipeline/pipeline.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_freeze()
    check_stop()
    update(state="starting", started_utc=now())
    jobs = []
    for stage, arm in (("memalpha", "memalpha"), ("base", "qwen3_base")):
        check_stop()
        status_path = run_root() / arm / "build_status.json"
        if status_path.exists() and read_json(status_path)["status"] == "complete":
            continue
        for attempt in range(1, 10000):
            check_stop()
            existing = OPS / f"full_{stage}_{attempt:03d}" / "status.json"
            if existing.exists():
                s = read_json(existing)
                if s["state"] in ACTIVE_STATES:
                    jobs.append((stage, arm, existing.parent.name, attempt))
                    break
                if s["state"] != "complete" and s.get("exit_code") != 75 and s.get("child_pid"):
                    raise RuntimeError("Previous writer operation needs inspection: " + json.dumps(s))
                continue
            operation = dispatch(stage, 1, ["build", "--arm", arm], attempt)
            if operation:
                jobs.append((stage, arm, operation, attempt))
                break
            check_stop()
            time.sleep(60)
    for stage, arm, name, attempt in jobs:
        while True:
            completed = wait_operation(name)
            if completed["state"] == "complete": break
            check_stop()
            if completed.get("exit_code") != 75:
                raise RuntimeError("Writer operation needs inspection: " + json.dumps(completed))
            attempt += 1
            while True:
                check_stop()
                name = dispatch(stage, 1, ["build", "--arm", arm], attempt)
                if name: break
                time.sleep(60)
                attempt += 1
    if len(list(run_root().glob("*/conversations/*/memory_frozen.json"))) != 20:
        raise RuntimeError("All 20 frozen memories are required before reader launch")
    answer_status = run_root() / "answer_status.json"
    if not answer_status.exists() or read_json(answer_status)["status"] != "complete":
        invalid_attempts = 0
        for attempt in range(1, 10000):
            check_stop()
            name = f"full_reader_{attempt:03d}"
            existing = OPS / name / "status.json"
            if existing.exists():
                status = read_json(existing)
                if status["state"] in ACTIVE_STATES:
                    completed = wait_operation(name)
                else:
                    if status["state"] != "complete" and status.get("exit_code") not in (2, 75) and status.get("child_pid"):
                        raise RuntimeError("Previous reader operation needs inspection: " + json.dumps(status))
                    continue
            else:
                name = dispatch("reader", 4, ["answer"], attempt)
                if not name:
                    time.sleep(60)
                    continue
                completed = wait_operation(name)
            if completed["state"] == "complete": break
            check_stop()
            if completed.get("exit_code") == 2:
                invalid_attempts += 1
                if invalid_attempts >= 3: break
            elif completed.get("exit_code") != 75:
                raise RuntimeError("Reader operation needs inspection: " + json.dumps(completed))
    check_stop()
    result = subprocess.run([PYTHON, str(ROOT / "run.py"), "score"], cwd=ROOT)
    if result.returncode:
        raise RuntimeError("Scoring failed")
    metrics = read_json(run_root() / "report/metrics.json")
    update(state="complete" if metrics["complete"] else "complete_with_errors", report=str(run_root() / "report/report.md"))
    return 0 if metrics["complete"] else 2


def run():
    try:
        return _run()
    except StopRequested:
        return 75


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except Exception as exc:
        update(state="failed", error=repr(exc))
        raise
