"""Atomic experiment artifacts and immutable source provenance."""
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARMS = ("memalpha", "qwen3_base")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def config():
    result = read_json(ROOT / "config.json")
    if result.get("training_enabled") is not False:
        raise RuntimeError("This experiment permits inference only")
    return result


def source_manifest():
    paths = [ROOT / x for x in ("config.json", "common.py", "run.py", "bridge.py", "locomo.py", "compatibility.patch")]
    paths += sorted((ROOT / "vendor").glob("*.py"))
    result = {str(p.relative_to(ROOT)): file_hash(p) for p in paths if p.exists()}
    result["upstream_commit"] = subprocess.check_output(["git", "-C", str(ROOT / "upstream"), "rev-parse", "HEAD"], text=True).strip()
    result["upstream_worktree_diff"] = hashlib.sha256(subprocess.check_output(["git", "-C", str(ROOT / "upstream"), "diff", "HEAD"])).hexdigest()
    return result


def run_root(synthetic=False):
    return ROOT / "runs" / ("synthetic_smoke" if synthetic else config()["run_id"])


def verify_freeze():
    freeze = read_json(run_root() / "freeze.json")
    if freeze["sources"] != source_manifest():
        raise RuntimeError("Frozen experiment source/config differs from current files")
    for name, expected in freeze["prepared_hashes"].items():
        if file_hash(ROOT / "data" / "prepared" / name) != expected:
            raise RuntimeError("Prepared input changed: " + name)
    return freeze


def machine():
    return {"hostname": socket.gethostname(), "pid": os.getpid(), "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "utc": now()}
