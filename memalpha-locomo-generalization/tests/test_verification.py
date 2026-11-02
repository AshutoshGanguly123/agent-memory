"""Standalone audit checks use synthetic receipts, never benchmark answers."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "verification/verify_run.py"
spec = importlib.util.spec_from_file_location("verification_under_test", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class VerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.cfg = {"reader": {"max_model_len": 32768}}
        self.sources = {"bridge.py": "frozen"}
        self.conversations = [{"sample_id": "sample", "chunks": [{"text": "first"}, {"text": "second"}]}]
        self.questions = [{"sample_id": "sample", "qa_index": i, "question": f"question{i}",
                           "reader_prompt": f"task prompt {i}", "category": category, "anomaly": []}
                          for i, category in enumerate([4, 5])]
        for arm in audit.ARMS:
            memory = {"core": "", "semantic": [], "episodic": []}
            for index, chunk in enumerate(self.conversations[0]["chunks"]):
                previous = copy.deepcopy(memory)
                memory["core"] += chunk["text"]
                memory["semantic"] = [{"abcd": "a fact"}]
                saved = {"sample_id": "sample", "chunk_index": index, "config_hash": audit.digest(self.cfg),
                         "source_hash": audit.digest(self.sources), "chunk_hash": audit.digest(chunk),
                         "input_memory_hash": audit.digest(previous), "output_memory_hash": audit.digest(memory),
                         "result": {"memory": copy.deepcopy(memory), "raw_response": "tool call",
                                    "prompt": "memorize " + chunk["text"], "finish_reason": "stop"}}
                save(self.base / arm / "conversations/sample/chunks" / f"{index:05d}.json", saved)
            frozen = {"arm": arm, "sample_id": "sample", "config_hash": audit.digest(self.cfg),
                      "source_hash": audit.digest(self.sources), "memory": memory, "memory_hash": audit.digest(memory),
                      "completed_chunks": 2, "frozen": True}
            save(self.base / arm / "conversations/sample/memory_frozen.json", frozen)
            for q in self.questions:
                saved = {"arm": arm, "sample_id": "sample", "qa_index": q["qa_index"],
                         "question_hash": audit.digest(q), "config_hash": audit.digest(self.cfg),
                         "source_hash": audit.digest(self.sources), "memory_hash": audit.digest(memory),
                         "attempt": 1, "status": "complete", "result": {
                             "raw_response": "synthetic response", "prompt": "<|im_start|>system\nfacts<|im_end|>\n<|im_start|>user\n" + q["reader_prompt"] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                             "finish_reason": "stop", "usage": {"prompt_tokens": 10},
                             "retrieval": {"query": q["question"], "method": "bm25", "top_k_per_store": 20,
                                           "memory": memory, "full_prompt_tokens": 10, "system_prompt_tokens": 5}}}
                save(self.base / arm / "answers/sample" / f"{q['qa_index']:05d}.json", saved)

    def verify(self):
        return audit.verify_receipts(self.base, self.cfg, self.sources, self.conversations, self.questions)

    def edit(self, relative, mutate):
        path = self.base / relative
        saved = audit.read(path)
        mutate(saved)
        save(path, saved)

    def test_full_artifact_graph_passes_without_scoring(self):
        result = self.verify()
        self.assertEqual(result["issues"], [])
        self.assertEqual(result["counts"]["frozen_memories"], 2)
        self.assertEqual(result["counts"]["latest_answers"], 4)
        self.assertEqual(result["counts"]["chunk_receipts"], 4)
        self.assertEqual(result["completed_categories"]["memalpha"], {4: 1, 5: 1})

    def test_changed_checkpoint_input_hash_is_detected(self):
        self.edit("memalpha/conversations/sample/chunks/00001.json", lambda value: value.update(input_memory_hash="wrong"))
        self.assertTrue(any("input_memory_hash" in x["message"] for x in self.verify()["issues"]))

    def test_final_memory_must_equal_last_chunk_even_with_rehashed_content(self):
        def corrupt(value):
            value["memory"]["core"] = "uncommitted"
            value["memory_hash"] = audit.digest(value["memory"])
        self.edit("memalpha/conversations/sample/memory_frozen.json", corrupt)
        self.assertTrue(any("final committed chunk" in x["message"] for x in self.verify()["issues"]))

    def test_swapped_question_and_prompt_are_detected(self):
        def swap(value):
            value["question_hash"] = audit.digest(self.questions[1])
            value["result"]["prompt"] = self.questions[1]["reader_prompt"]
            value["result"]["retrieval"]["query"] = self.questions[1]["question"]
        self.edit("memalpha/answers/sample/00000.json", swap)
        messages = [x["message"] for x in self.verify()["issues"]]
        self.assertTrue(any("question_hash" in value for value in messages))
        self.assertTrue(any("misaligned" in value for value in messages))
        self.assertTrue(any("shared question" in value for value in messages))

    def test_missing_and_extra_latest_answers_are_detected(self):
        source = self.base / "memalpha/answers/sample/00000.json"
        shutil.copyfile(source, source.with_name("99999.json"))
        source.unlink()
        messages = [x["message"] for x in self.verify()["issues"]]
        self.assertIn("Missing expected artifact", messages)
        self.assertIn("Unexpected/duplicate latest answer path", messages)

    def test_interrupted_retry_archive_copy_is_allowed_but_conflict_is_not(self):
        source = self.base / "memalpha/answers/sample/00000.json"
        archive = source.parent / "attempts/00000_1.json"
        save(archive, audit.read(source))
        self.assertFalse(self.verify()["issues"])
        self.edit("memalpha/answers/sample/attempts/00000_1.json", lambda value: value.update(utc="different"))
        self.assertTrue(any("Conflicting duplicate" in x["message"] for x in self.verify()["issues"]))

    def test_unrelated_retrieved_memory_is_detected(self):
        def corrupt(value):
            value["result"]["retrieval"]["memory"]["semantic"] = [{"beef": "not in frozen memory"}]
        self.edit("memalpha/answers/sample/00000.json", corrupt)
        self.assertTrue(any("bounded subset" in x["message"] for x in self.verify()["issues"]))

    def test_complete_receipt_with_empty_output_is_invalid(self):
        self.edit("memalpha/answers/sample/00000.json", lambda value: value["result"].update(raw_response=""))
        self.assertTrue(any("invalid/empty" in x["message"] for x in self.verify()["issues"]))

    def test_report_duplicate_rows_and_denominator_mismatch_are_detected(self):
        rows = [{"arm": arm, "sample_id": "sample", "qa_index": q["qa_index"], "category": q["category"]}
                for arm in audit.ARMS for q in self.questions]
        rows.append(copy.deepcopy(rows[0]))
        save(self.base / "report/per_question.json", rows)
        save(self.base / "report/metrics.json", {"metrics": {}})
        result = audit._verify_report(self.base, self.questions, self.verify()["completed_categories"])
        self.assertTrue(any("duplicate report rows" in x["message"] for x in result["issues"]))
        self.assertTrue(any("denominator" in x["message"] for x in result["issues"]))

    def test_correct_report_alignment_and_denominators_pass(self):
        rows = [{"arm": arm, "sample_id": "sample", "qa_index": q["qa_index"],
                 "category": q["category"], "question": q["question"],
                 "status": "complete", "raw_response": "synthetic response"}
                for arm in audit.ARMS for q in self.questions]
        values = {"count": 2, "completed": 2, "adversarial_clean_count": 1,
                  "categories": {name: {"count": int(category in (4, 5)), "completed": int(category in (4, 5))}
                                 for category, name in audit.CATEGORY_NAMES.items()}}
        save(self.base / "report/per_question.json", rows)
        save(self.base / "report/metrics.json", {"complete": True, "metrics": {arm: values for arm in audit.ARMS}})
        completed = self.verify()["completed_categories"]
        self.assertEqual(audit._verify_report(self.base, self.questions, completed)["issues"], [])
        rows[0]["raw_response"] = "wrong receipt copied into report"
        save(self.base / "report/per_question.json", rows)
        self.assertTrue(any("alignment mismatch" in x["message"] for x in audit._verify_report(self.base, self.questions, completed)["issues"]))


if __name__ == "__main__":
    unittest.main()
