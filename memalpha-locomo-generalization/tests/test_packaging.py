"""Packaging preconditions and checksums, using synthetic metadata only."""
import copy
import fcntl
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "verification/package_run.py"
spec = importlib.util.spec_from_file_location("packaging_under_test", SCRIPT)
package = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class PackagingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.base = self.root / "runs/test"
        self.cfg = {"run_id": "test", "training_enabled": False,
                    "models": {alias: {"repo": alias, "revision": "pinned-" + alias}
                               for alias in ("memalpha", "qwen3_base", "reader")}}
        self.sources = {"upstream_commit": "pinned", "config.json": "frozen", "bridge.py": "frozen"}
        save(self.root / "config.json", self.cfg)
        save(self.base / "freeze.json", {"config": self.cfg, "training": False, "sources": self.sources})
        save(self.root / "results/pipeline/status.json", {"state": "complete"})
        save(self.root / "models/manifest.json", {"models": {
            alias: {"repo_id": config["repo"], "revision": config["revision"], "state": "ready"}
            for alias, config in self.cfg["models"].items()}})
        self.fresh = {"passed": True, "prediction_scoring_performed": False,
                      "dataset_sha256": "dataset", "source_hash": package.digest(self.sources),
                      "config_hash": package.digest(self.cfg), "shared_question_option_manifest_sha256": "options",
                      "counts": {"frozen_memories": 20, "latest_answers": 3972, "chunk_receipts": 276},
                      "expected": {"frozen_memories": 20, "latest_answers": 3972, "chunk_receipts": 276},
                      "completed_categories": {arm: {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
                                               for arm in ("memalpha", "qwen3_base")},
                      "report": {"present": True}}
        save(self.base / "verification/integrity.json", self.fresh)
        for relative in ("bridge.py", "report.py", "ops.py", "pipeline.py", "upstream/agent.py",
                         "data/locomo10.json", "data/prepared/questions.json", "results/setup/requirements.resolved.txt"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("synthetic fixture")
        for stage in ("memalpha", "base", "reader"):
            name = "full_" + stage + "_001"
            operation = self.root / "results/operations" / name
            status = {"name": name, "attempt": "attempt1", "state": "complete", "exit_code": 0,
                      "finished_utc": "finished", "owned_pids": [], "child_pid": 123,
                      "command": ["python", "run.py", stage], "node": "gpu-04", "gpus": ["GPU-test"],
                      "cuda_mapping": [{"uuid": "GPU-test", "physical_index": 0}],
                      "physical_inventory": {"node": "gpu-04", "cards": [{"uuid": "GPU-test"}]}}
            for name_ in ("status.json", "attempts/attempt1/status.json", "attempts/attempt1/exit.json"):
                save(operation / name_, status)
            lock = self.root / "results/operations/locks" / f"operation_{name}.lock"
            lock.parent.mkdir(parents=True, exist_ok=True)
            lock.touch()
        self.fresh_patch = patch.object(package, "fresh_integrity", lambda root: copy.deepcopy(self.fresh))
        self.fresh_patch.start()
        self.addCleanup(self.fresh_patch.stop)
        self.git_patch = patch.object(package.subprocess, "check_output", return_value=b"agent.py\0")
        self.git_patch.start()
        self.addCleanup(self.git_patch.stop)

    def test_complete_manifest_indexes_sources_without_weights_or_itself(self):
        weights = self.root / "models/reader/model.safetensors"
        weights.parent.mkdir(parents=True, exist_ok=True)
        weights.write_text("must not be indexed")
        save(self.root / "validation/cpu_tests.json", {"passed": True})
        save(self.root / "runs/synthetic_smoke/validation.json", {"passed": True})
        save(self.root / "runs/synthetic_smoke/memalpha/conversations/synthetic-1/chunks/00000.json",
             {"raw_response": "synthetic memory operations"})
        (self.base / "run_manifest_old.json").write_text("old package excluded")
        output, manifest = package.build_manifest(self.root)
        self.assertEqual(output, self.base / "run_manifest.json")
        paths = {item["path"] for item in manifest["artifacts"]}
        self.assertIn("models/manifest.json", paths)
        self.assertIn("upstream/agent.py", paths)
        self.assertIn("report.py", paths)
        self.assertIn("validation/cpu_tests.json", paths)
        self.assertIn("runs/synthetic_smoke/validation.json", paths)
        self.assertIn("runs/synthetic_smoke/memalpha/conversations/synthetic-1/chunks/00000.json", paths)
        self.assertFalse(any("model.safetensors" in path or "run_manifest" in path or "/locks/" in path for path in paths))
        self.assertEqual(manifest["artifact_index_sha256"], package.digest(manifest["artifacts"]))
        self.assertEqual(len(manifest["operations"]), 3)
        self.assertTrue(manifest["no_training"])
        self.assertFalse(manifest["prediction_scoring_performed"])

    def test_repackaging_does_not_recursively_index_previous_output(self):
        output, first = package.build_manifest(self.root)
        package.write_manifest(output, first)
        _, second = package.build_manifest(self.root)
        self.assertEqual(first["artifact_index_sha256"], second["artifact_index_sha256"])

    def test_pipeline_must_be_complete(self):
        save(self.root / "results/pipeline/status.json", {"state": "running"})
        with self.assertRaisesRegex(RuntimeError, "completed pipeline"):
            package.build_manifest(self.root)

    def test_integrity_report_must_be_fresh(self):
        saved = copy.deepcopy(self.fresh)
        saved["shared_question_option_manifest_sha256"] = "stale"
        save(self.base / "verification/integrity.json", saved)
        with self.assertRaisesRegex(RuntimeError, "stale"):
            package.build_manifest(self.root)

    def test_fresh_audit_failure_blocks_packaging(self):
        self.fresh["passed"] = False
        with self.assertRaisesRegex(RuntimeError, "Fresh integrity"):
            package.build_manifest(self.root)

    def test_held_operation_lock_blocks_packaging(self):
        lock = self.root / "results/operations/locks/operation_full_reader_001.lock"
        with lock.open("rb") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "still holds"):
                package.build_manifest(self.root)
        self.assertEqual(len(package.validate_operations(self.root)), 3)

    def test_terminal_status_with_owned_descendants_is_rejected(self):
        path = self.root / "results/operations/full_reader_001/status.json"
        status = package.read(path)
        status["owned_pids"] = [123]
        save(path, status)
        with self.assertRaisesRegex(RuntimeError, "descendants"):
            package.build_manifest(self.root)

    def test_mismatched_attempt_exit_is_rejected(self):
        path = self.root / "results/operations/full_reader_001/attempts/attempt1/exit.json"
        status = package.read(path)
        status["exit_code"] = 2
        save(path, status)
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            package.build_manifest(self.root)

    def test_pinned_model_manifest_must_match_config(self):
        path = self.root / "models/manifest.json"
        value = package.read(path)
        value["models"]["reader"]["revision"] = "different"
        save(path, value)
        with self.assertRaisesRegex(RuntimeError, "Pinned model"):
            package.build_manifest(self.root)

    def test_prelaunch_failure_can_have_no_acquired_operation_lock(self):
        status = {"name": "full_reader_000", "attempt": "prelaunch", "state": "failed", "exit_code": 1,
                  "owned_pids": [], "finished_utc": "finished", "command": ["python", "run.py", "answer"]}
        directory = self.root / "results/operations/full_reader_000"
        for name in ("status.json", "attempts/prelaunch/status.json", "attempts/prelaunch/exit.json"):
            save(directory / name, status)
        _, manifest = package.build_manifest(self.root)
        item = next(x for x in manifest["operations"] if x["operation"] == "full_reader_000")
        self.assertFalse(item["child_launched"])
        self.assertIsNone(item["hardware_evidence"])
        self.assertFalse(item["lease_release_evidence"]["operation_lock_file_exists"])


if __name__ == "__main__":
    unittest.main()
