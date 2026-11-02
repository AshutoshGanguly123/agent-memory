"""Resume and provenance regression checks without GPU inference."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge
import run as runner
from common import atomic_json, digest, read_json


class FakeWriter:
    calls = 0

    def __init__(self, *args):
        pass

    def complete_batch(self, items):
        FakeWriter.calls += len(items)
        outputs = []
        for item in items:
            memory = copy.deepcopy(item["memory_state"])
            memory["core"] += item["chunk_text"]
            outputs.append({"memory": memory, "raw_response": "Done", "errors": [], "tool_calls": []})
        return outputs


class FakeReader:
    calls = 0

    def __init__(self, *args):
        pass

    def answer_batch(self, items):
        FakeReader.calls += len(items)
        return [{"raw_response": "Pixel", "errors": []} for _ in items]


class RunnerReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {"writer": {}, "reader": {}, "writer_batch_size": 2, "reader_batch_size": 2,
                    "source_commit": "pinned", "run_id": "test", "training_enabled": False}
        self.conversations = [{"sample_id": "one", "chunks": [{"text": "first"}, {"text": "second"}]}]
        self.questions = [{"sample_id": "one", "qa_index": 0, "question": "Dog name?", "reader_prompt": "Dog name?"}]
        FakeWriter.calls = 0
        FakeReader.calls = 0
        self.patches = [
            patch.object(runner, "ROOT", self.root),
            patch.object(runner, "config", lambda: self.cfg),
            patch.object(runner, "run_root", lambda synthetic=False: self.root / ("smoke" if synthetic else "run")),
            patch.object(runner, "inputs", lambda synthetic=False: (self.conversations, self.questions)),
            patch.object(runner, "machine", lambda: {}),
            patch.object(runner, "STOP", False),
            patch.object(bridge, "WriterBridge", FakeWriter),
            patch.object(bridge, "ReaderBridge", FakeReader),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_completed_writer_resume_performs_no_generation(self):
        self.assertEqual(runner.build("memalpha", synthetic=True), 0)
        self.assertEqual(FakeWriter.calls, 2)
        first = read_json(self.root / "smoke/memalpha/conversations/one/memory_frozen.json")
        self.assertEqual(runner.build("memalpha", synthetic=True), 0)
        self.assertEqual(FakeWriter.calls, 2)
        self.assertEqual(first, read_json(self.root / "smoke/memalpha/conversations/one/memory_frozen.json"))

    def test_missing_checkpoint_in_chain_is_rejected(self):
        runner.build("memalpha", synthetic=True)
        directory = self.root / "smoke/memalpha/conversations/one"
        (directory / "chunks/00000.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "Noncontiguous"):
            runner.checkpoint_state(self.conversations[0], directory, digest(self.cfg))

    def test_corrupted_memory_is_rejected(self):
        runner.build("memalpha", synthetic=True)
        directory = self.root / "smoke/memalpha/conversations/one"
        path = directory / "chunks/00000.json"
        saved = read_json(path)
        saved["result"]["memory"]["core"] = "corruption"
        atomic_json(path, saved)
        with self.assertRaisesRegex(RuntimeError, "corrupted"):
            runner.checkpoint_state(self.conversations[0], directory, digest(self.cfg))

    def test_invalid_saved_reader_attempt_is_retried(self):
        for arm in runner.ARMS:
            runner.build(arm, synthetic=True)
        runner.answer(synthetic=True)
        path = self.root / "smoke/memalpha/answers/one/00000.json"
        saved = read_json(path)
        saved.update(status="error", result={"raw_response": "", "errors": []})
        atomic_json(path, saved)
        FakeReader.calls = 0
        runner.answer(synthetic=True)
        self.assertEqual(FakeReader.calls, 1, "Only the invalid saved attempt should be requeued")
        self.assertEqual(read_json(path)["status"], "complete")

    def test_freeze_rejects_smoke_validation_from_different_sources(self):
        self.conversations = [{}] * 10
        self.questions = [{}] * 1986
        for name in ("conversations.json", "questions.json", "manifest.json"):
            atomic_json(self.root / "data/prepared" / name, {})
        atomic_json(self.root / "smoke/validation.json", {
            "passed": True, "sources": {"bridge.py": "old", "upstream_commit": "pinned"}})
        with patch.object(runner, "source_manifest", lambda: {"bridge.py": "new", "upstream_commit": "pinned"}):
            with self.assertRaises(RuntimeError):
                runner.freeze()


if __name__ == "__main__":
    unittest.main()
