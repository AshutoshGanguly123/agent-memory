"""Index a completed, verified experiment without copying weights or changing it.

The one generated artifact is runs/<run_id>/run_manifest.json. A saved passing
integrity report, a fresh passing audit, completed pipeline, and terminal released
full operations are required. No model or scoring package is imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

MANIFEST_TYPE = "memalpha_locomo_run_package"
TERMINAL = {"complete", "failed", "stopped"}
EXCLUDED_DIRS = {".git", ".venv", ".cache", "__pycache__", ".pytest_cache", "locks", ".locks"}
WEIGHT_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fresh_integrity(root):
    path = Path(__file__).with_name("verify_run.py")
    spec = importlib.util.spec_from_file_location("integrity_for_packaging", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify(root)


def prove_operation_lock_released(path, allow_absent=False):
    if not path.is_file() and allow_absent:
        # A launch rejected before acquiring its operation lock has nothing to
        # release. Never accept absence after any child was actually launched.
        return
    require(path.is_file(), "Missing operation lease evidence: " + str(path))
    with path.open("rb") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Full operation still holds its lease: " + str(path)) from exc
        finally:
            # Unlocking an unacquired flock is harmless; this never creates or
            # changes a lock file and does not touch another owner's lock.
            fcntl.flock(handle, fcntl.LOCK_UN)


def validate_operations(root):
    operations = root / "results/operations"
    directories = sorted(path for path in operations.glob("full_*") if path.is_dir())
    require(bool(directories), "No full operation provenance was found")
    references, successful_stages = [], set()
    for directory in directories:
        latest_path = directory / "status.json"
        latest = read(latest_path)
        require(latest.get("state") in TERMINAL, "Nonterminal full operation: " + directory.name)
        require(latest.get("owned_pids") == [], "Full operation has unproven/live descendants: " + directory.name)
        require(bool(latest.get("finished_utc")), "Full operation has no terminal timestamp: " + directory.name)
        lock = operations / "locks" / f"operation_{directory.name}.lock"
        attempt_directories = sorted(path for path in (directory / "attempts").iterdir() if path.is_dir())
        require(bool(attempt_directories), "No attempt receipts for " + directory.name)
        require(any(path.name == latest.get("attempt") for path in attempt_directories),
                "Latest operation references a missing attempt: " + directory.name)
        any_launched = False
        for attempt in attempt_directories:
            status_path, exit_path = attempt / "status.json", attempt / "exit.json"
            status, exited = read(status_path), read(exit_path)
            require(status == exited, "Attempt status and exit evidence disagree: " + str(attempt))
            require(status.get("state") in TERMINAL and status.get("owned_pids") == [] and status.get("finished_utc"),
                    "Attempt has not completely released its descendants: " + str(attempt))
            require(status.get("name") == directory.name and status.get("attempt") == attempt.name,
                    "Operation/attempt identity mismatch: " + str(attempt))
            if attempt.name == latest["attempt"]:
                require(status == latest, "Latest status differs from its final attempt receipt")
            command = status.get("command")
            require(isinstance(command, list) and command, "Actual child command missing: " + str(attempt))
            launched = bool(status.get("child_pid"))
            any_launched = any_launched or launched
            if launched:
                require(bool(status.get("node")) and bool(status.get("gpus")) and
                        bool(status.get("physical_inventory")) and bool(status.get("cuda_mapping")),
                        "Launched child lacks physical hardware evidence: " + str(attempt))
                inventory = status["physical_inventory"]
                require(inventory.get("node") == status["node"], "Hardware node identity mismatch")
                cards = {card["uuid"] for card in inventory.get("cards", [])}
                require(set(status["gpus"]) <= cards and
                        {card["uuid"] for card in status["cuda_mapping"]} == set(status["gpus"]),
                        "Physical GPU evidence does not cover every leased device")
            stage = next((stage for stage in ("memalpha", "base", "reader")
                          if directory.name == "full_" + stage or directory.name.startswith("full_" + stage + "_")), None)
            if status["state"] == "complete" and status.get("exit_code") == 0 and launched and stage:
                successful_stages.add(stage)
            references.append({"operation": directory.name, "attempt": attempt.name,
                               "stage": stage, "state": status["state"], "exit_code": status.get("exit_code"),
                               "child_launched": launched,
                               "actual_child_command": command, "node": status.get("node"),
                               "physical_gpu_uuids": status.get("gpus", []), "cuda_mapping": status.get("cuda_mapping", []),
                               "operations_version": status.get("operations_version"),
                               "operations_sha256": status.get("operations_sha256"),
                               "status_evidence": str(status_path.relative_to(root)),
                               "exit_evidence": str(exit_path.relative_to(root)),
                               "hardware_evidence": str(status_path.relative_to(root)) + "#/physical_inventory" if launched else None,
                               "lease_release_evidence": {"terminal_owned_pids": [], "operation_lock_available": True,
                                                          "operation_lock_file_exists": lock.is_file()}})
        prove_operation_lock_released(lock, allow_absent=not any_launched)
    require(successful_stages == {"memalpha", "base", "reader"},
            "Successful released operations for both writers and reader are required")
    return references


def excluded(path, root, output):
    relative = path.relative_to(root)
    if path == output or path.name.startswith("run_manifest") and path.suffix == ".json":
        return True
    if any(part in EXCLUDED_DIRS or part.startswith(".") for part in relative.parts):
        return True
    if path.suffix in WEIGHT_SUFFIXES or path.name.startswith("pytorch_model") and path.suffix == ".bin":
        return True
    if path.suffix in {".lock", ".tmp", ".pyc"}:
        return True
    return False


def artifact_paths(root, base, frozen, output):
    paths = set()
    trees = [base, root / "runs/synthetic_smoke", root / "validation", root / "data/prepared",
             root / "results/setup", root / "results/operations",
             root / "results/pipeline", root / "results/provenance", root / "scripts",
             root / "verification", root / "tests", root / "vendor"]
    for tree in trees:
        if tree.is_dir():
            paths.update(path for path in tree.rglob("*") if path.is_file())
    paths.update(root.glob("*.py"))
    paths.update(root.glob("README*.md"))
    paths.update([root / "data/locomo10.json", root / "models/manifest.json",
                  root / "results/setup/requirements.resolved.txt"])
    paths.update(root / name for name in frozen["sources"]
                 if name not in {"upstream_commit", "upstream_worktree_diff"})
    tracked = subprocess.check_output(["git", "-C", str(root / "upstream"), "ls-files", "-z"])
    paths.update(root / "upstream" / item.decode() for item in tracked.split(b"\0") if item)
    selected = []
    for path in sorted(paths):
        if excluded(path, root, output):
            continue
        require(path.is_file(), "Required indexed artifact is missing: " + str(path))
        require(path.resolve().is_relative_to(root), "Refusing an artifact symlink outside the experiment: " + str(path))
        selected.append(path)
    return selected


def fingerprint(path):
    before = path.stat()
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            "Artifact changed while hashing: " + str(path))
    return {"bytes": after.st_size, "sha256": result.hexdigest()}, (after.st_size, after.st_mtime_ns)


def build_manifest(root):
    root = Path(root).resolve()
    cfg = read(root / "config.json")
    base = root / "runs" / cfg["run_id"]
    output = base / "run_manifest.json"
    frozen = read(base / "freeze.json")
    pipeline = read(root / "results/pipeline/status.json")
    require(pipeline.get("state") == "complete", "Packaging requires a completed pipeline")
    require(cfg.get("training_enabled") is False and frozen.get("training") is False and frozen.get("config") == cfg,
            "No-training configuration/freeze mismatch")
    integrity_path = base / "verification/integrity.json"
    saved = read(integrity_path)
    require(saved.get("passed") is True and saved.get("prediction_scoring_performed") is False,
            "A passing saved integrity report is required")
    require(saved.get("counts", {}).get("frozen_memories") == 20 and
            saved.get("counts", {}).get("latest_answers") == 3972,
            "The saved integrity report does not cover the full experiment")
    fresh = fresh_integrity(root)
    require(fresh.get("passed") is True, "Fresh integrity audit failed: " + str(fresh.get("issues", [])[:3]))
    for key in ("dataset_sha256", "source_hash", "config_hash", "shared_question_option_manifest_sha256", "counts", "expected", "completed_categories", "report"):
        # JSON round-trips category keys as strings; canonical JSON comparison
        # treats a fresh integer-keyed counter and its saved form identically.
        require(digest(saved.get(key)) == digest(fresh.get(key)), "Saved integrity evidence is stale: " + key)
    require(fresh["source_hash"] == digest(frozen["sources"]) and fresh["config_hash"] == digest(cfg),
            "Integrity result does not identify this freeze")
    models = read(root / "models/manifest.json")
    for alias, expected in cfg["models"].items():
        model = models.get("models", {}).get(alias, {})
        require(model.get("state") == "ready" and model.get("repo_id") == expected["repo"] and
                model.get("revision") == expected["revision"], "Pinned model manifest mismatch: " + alias)
    operations = validate_operations(root)
    files, stats = [], {}
    for path in artifact_paths(root, base, frozen, output):
        hashed, stat = fingerprint(path)
        files.append({"path": str(path.relative_to(root)), **hashed})
        stats[path] = stat
    # A terminal pipeline should be stable. Reject a report/source/receipt edit
    # during packaging instead of emitting a mixed-time artifact index.
    for path, expected_stat in stats.items():
        actual = path.stat()
        require((actual.st_size, actual.st_mtime_ns) == expected_stat,
                "Artifact changed before packaging completed: " + str(path))
    require(read(root / "results/pipeline/status.json") == pipeline, "Pipeline changed during packaging")
    # Confirm leases again after hashing; no weights or GPU runtime is touched.
    require(validate_operations(root) == operations, "Full operation evidence changed during packaging")
    result = {"manifest_type": MANIFEST_TYPE, "schema_version": 1,
              "created_utc": datetime.now(timezone.utc).isoformat(), "run_id": cfg["run_id"],
              "experiment_root": str(root), "no_training": True, "model_revisions": cfg["models"],
              "dataset_sha256": fresh["dataset_sha256"], "scientific_source_hash": fresh["source_hash"],
              "config_hash": fresh["config_hash"], "upstream_commit": frozen["sources"]["upstream_commit"],
              "freeze": str((base / "freeze.json").relative_to(root)),
              "integrity_report": str(integrity_path.relative_to(root)),
              "shared_question_option_manifest": "data/prepared/questions.json",
              "model_asset_manifest": "models/manifest.json",
              "resolved_dependencies": "results/setup/requirements.resolved.txt",
              "pipeline_evidence": "results/pipeline/status.json", "pipeline_state": "complete",
              "coverage": fresh["counts"], "completed_categories_per_arm": fresh["completed_categories"],
              "operations": operations, "artifact_count": len(files), "indexed_bytes": sum(item["bytes"] for item in files),
              "artifact_index_sha256": digest(files), "artifacts": files,
              "excluded": ["model weights", "lock files/directories", "caches", "temporary files", "run_manifest*.json"],
              "packager_sha256": fingerprint(Path(__file__))[0]["sha256"],
              "prediction_scoring_performed": False}
    return output, result


def write_manifest(path, manifest):
    fd, temporary = tempfile.mkstemp(prefix=".run_manifest.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(manifest, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    try:
        path, result = build_manifest(args.experiment)
        write_manifest(path, result)
    except Exception as exc:
        print(json.dumps({"packaged": False, "error": str(exc)}, indent=2))
        return 1
    print(json.dumps({"packaged": True, "manifest": str(path), "artifact_count": result["artifact_count"],
                      "indexed_bytes": result["indexed_bytes"], "artifact_index_sha256": result["artifact_index_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
