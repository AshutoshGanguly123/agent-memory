"""CPU checks for contracts, replay, capacity, and actual patched memory CRUD.

The upstream memory/parser classes are loaded from their AST with dependency
stubs, so these checks do not require downloading GPU inference libraries.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import uuid

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bridge


def source_class(path, name, globals_):
    tree = ast.parse(path.read_text())
    node = next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name)
    selected = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node], type_ignores=[])
    ast.fix_missing_locations(selected)
    exec(compile(selected, str(path), "exec"), globals_)
    return globals_[name]


Memory = source_class(bridge.UPSTREAM / "memory.py", "Memory", {
    "np": np, "json": json, "hashlib": hashlib, "uuid": uuid, "re": re,
    "count_tokens": lambda text: len(text.split()),
    "openai": types.SimpleNamespace(OpenAI=lambda: (_ for _ in ()).throw(AssertionError("Embedding API called"))),
})


def insert(memory, arguments):
    item = memory.new_memory_insert(arguments["memory_type"].replace("_memory", ""), arguments["content"])
    return {"new_memory": item}


def update(memory, arguments):
    return memory.memory_update(arguments["memory_type"].replace("_memory", ""),
                                arguments["new_content"], arguments.get("memory_id"))


def delete(memory, arguments):
    return memory.memory_delete(arguments["memory_type"].replace("_memory", ""), arguments.get("memory_id"))


RealAgent = source_class(bridge.UPSTREAM / "agent.py", "MemoryAgent", {
    "json": json, "re": re, "repair_json": lambda text: text,
    "FN_NAME": "✿FUNCTION✿", "FN_ARGS": "✿ARGS✿", "FN_RESULT": "✿RESULT✿", "FN_EXIT": "✿RETURN✿",
    "FUNCTION_IMPLS": {"new_memory_insert": insert, "memory_update": update, "memory_delete": delete},
})


class Agent(RealAgent):
    @staticmethod
    def process_text_with_qwen_pipeline(**kwargs):
        assert kwargs["enable_thinking"] is False
        assert kwargs["max_num_of_recent_chunks"] == 5
        assert kwargs["status"] == "memorie"
        return kwargs["memory"].render_system_prompt(status="memorie", max_num_of_recent_chunks=5)[0]["content"] + kwargs["text"]


class Tokenizer:
    eos_token_id = 151645

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        assert kwargs == {"local_files_only": True}
        return cls()

    def encode(self, text, **kwargs):
        return text.split()

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "SYSTEM " + messages[0]["content"] + " USER " + messages[1]["content"] + " ASSISTANT"


class Sampling:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class Model:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.responses = []
        self.calls = []

    def generate(self, prompts, params, **kwargs):
        self.calls.append((prompts, params, kwargs))
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(
            text=(self.responses[i] if i < len(self.responses) else "answer"), token_ids=[1, 2],
            finish_reason="stop", stop_reason=None)]) for i, _ in enumerate(prompts)]


class Processor:
    def _tokenize(self, text):
        return re.findall(r"\b\w+\b", text.lower())

    def search_memories(self, memory, query, top_k):
        self.last_query = query
        return {"core": memory["core"], "semantic": memory["semantic"][:top_k],
                "episodic": memory["episodic"][:top_k]}

    def construct_system_prompt(self, memory, original):
        return memory["core"] + " " + " ".join(next(iter(item.values())) for kind in ("semantic", "episodic") for item in memory[kind])


def runtime():
    return {"Memory": Memory, "MemoryAgent": Agent, "MemoryProcessor": Processor,
            "get_memory_tool_schemas": lambda memory: [{"function": {"name": "new_memory_insert"}}],
            "AutoTokenizer": Tokenizer, "LLM": Model, "SamplingParams": Sampling,
            "np": np, "unified_prompt": "Remember <new_chunk>{context}</new_chunk> limit {max_new_tokens}"}


def tool(name, **arguments):
    return f"✿FUNCTION✿: {name}\n✿ARGS✿: {json.dumps(arguments)}\n"


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = self.temp.name
        Path(self.path, "generation_config.json").write_text(json.dumps({"top_p": 0.95, "top_k": 20}))
        self.mock = patch.object(bridge, "_load_runtime", runtime)
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def test_synthetic_writer_batch_calls_upstream_parser_and_replays(self):
        writer = bridge.WriterBridge(self.path)
        response = tool("new_memory_insert", memory_type="episodic_memory", content="At 2023-05-01 Caroline visited Paris.")
        writer.model.responses = [response, response]
        initial = bridge.empty_memory()
        outputs = writer.complete_batch([
            {"chunk_text": "Caroline visited Paris.", "memory_state": initial},
            {"chunk_text": "Caroline visited Paris.", "memory_state": initial}])
        self.assertEqual(len(writer.model.calls), 1)
        self.assertEqual(initial, bridge.empty_memory())
        self.assertEqual(outputs[0]["memory"], outputs[1]["memory"])
        self.assertEqual(outputs[0]["memory"], writer.replay(initial, response)["memory"])
        self.assertFalse(outputs[0]["errors"])
        self.assertIn("limit 1638", outputs[0]["prompt"])
        self.assertEqual(writer.sampling_params.temperature, 0.0)
        self.assertEqual(writer.sampling_params.max_tokens, 2048)
        self.assertEqual(writer.sampling_params.seed, 42)

    def test_restored_memory_can_update_delete_and_insert_without_embeddings(self):
        writer = bridge.WriterBridge(self.path)
        initial = {"core": "", "semantic": [{"abcd": "old"}], "episodic": [{"def0": "event"}]}
        response = (tool("memory_update", memory_type="semantic_memory", memory_id="abcd", new_content="new")
                    + tool("memory_delete", memory_type="episodic_memory", memory_id="def0")
                    + tool("new_memory_insert", memory_type="semantic_memory", content="another"))
        output = writer.replay(initial, response)
        self.assertFalse(output["errors"])
        self.assertEqual(output["memory"]["semantic"][0], {"abcd": "new"})
        self.assertEqual(output["memory"]["episodic"], [])
        self.assertEqual(len(output["memory"]["semantic"]), 2)
        self.assertEqual(initial["semantic"][0], {"abcd": "old"})

    def test_invalid_tool_content_is_recorded_and_does_not_corrupt_state(self):
        writer = bridge.WriterBridge(self.path)
        output = writer.replay(None, tool("new_memory_insert", memory_type="semantic_memory", content=123))
        self.assertEqual(output["memory"], bridge.empty_memory())
        self.assertEqual(len(output["errors"]), 1)

    def test_error_text_inside_legitimate_memory_is_not_a_tool_failure(self):
        writer = bridge.WriterBridge(self.path)
        output = writer.replay(None, tool("new_memory_insert", memory_type="semantic_memory", content="The log prints a tool error] marker."))
        self.assertFalse(output["errors"])
        self.assertEqual(len(output["memory"]["semantic"]), 1)

    def test_legacy_cross_store_same_id_is_valid_and_store_scoped(self):
        state = {"core": "", "semantic": [{"abcd": "fact"}], "episodic": [{"abcd": "event"}]}
        writer = bridge.WriterBridge(self.path)
        output = writer.replay(state, tool("memory_delete", memory_type="episodic_memory", memory_id="abcd"))
        self.assertFalse(output["errors"])
        self.assertEqual(output["memory"]["semantic"], [{"abcd": "fact"}])
        self.assertEqual(output["memory"]["episodic"], [])

    def test_scheduler_token_budget_forwarded(self):
        writer = bridge.WriterBridge(self.path, {"max_num_batched_tokens": 8192})
        self.assertEqual(writer.model.kwargs["max_num_batched_tokens"], 8192)
        self.assertTrue(writer.model.kwargs["enable_chunked_prefill"])
        reader = bridge.ReaderBridge(self.path, {"max_num_batched_tokens": 8192})
        self.assertTrue(reader.model.kwargs["enable_chunked_prefill"])
        self.assertEqual(reader.model.kwargs["tensor_parallel_size"], 4)

    def test_unique_four_hex_ids_guard_collision_and_exhaustion(self):
        memory = Memory(including_core=True, embedding_enabled=False, id_seed="42")
        original = memory._generate_memory_id()
        # Force the same hash-derived candidate so the linear collision guard is exercised.
        fake_hash = types.SimpleNamespace(hexdigest=lambda: original + "0" * 60)
        memory.semantic = [{original: "existing"}]
        with patch.object(hashlib, "sha256", return_value=fake_hash):
            second = memory._generate_memory_id()
        self.assertNotEqual(original, second)
        self.assertRegex(second, r"^[0-9a-f]{4}$")
        memory.semantic = [{f"{i:04x}": "x"} for i in range(65536)]
        with self.assertRaisesRegex(ValueError, "exhausted"):
            memory._generate_memory_id()

    def test_reader_raw_question_is_separate_and_memory_is_frozen(self):
        reader = bridge.ReaderBridge(self.path)
        memory = {"core": "User likes cycling.", "semantic": [{"abcd": "Paris"}], "episodic": []}
        before = copy.deepcopy(memory)
        output = reader.answer(memory, "Where did she go?", "Choose an option: Where did she go? A Paris B None")
        self.assertEqual(output["retrieval"]["query"], "Where did she go?")
        self.assertEqual(reader.processor.last_query, "Where did she go?")
        self.assertIn("Choose an option", output["prompt"])
        self.assertEqual(memory, before)
        self.assertEqual(reader.sampling_params.top_p, 0.95)
        self.assertEqual(reader.sampling_params.top_k, 20)
        self.assertEqual(reader.sampling_params.temperature, 0.7)
        self.assertEqual(reader.sampling_params.seed, 42)

    def test_reader_reduces_k_for_full_question_overhead(self):
        reader = bridge.ReaderBridge(self.path, {"max_model_len": 2070})
        memory = {"core": "", "semantic": [{f"{i:04x}": "fact " * 5} for i in range(20)], "episodic": []}
        output = reader.answer(memory, "fact?", "user instruction " * 4)
        self.assertLess(output["retrieval"]["top_k_per_store"], 20)
        self.assertLessEqual(output["usage"]["prompt_tokens"] + 2048, 2070)

    def test_overflow_is_explicit_and_does_not_call_model(self):
        reader = bridge.ReaderBridge(self.path, {"max_model_len": 2060})
        with self.assertRaises(bridge.ContextOverflowError):
            reader.answer({"core": "core " * 30, "semantic": [], "episodic": []}, "q", "q")
        self.assertFalse(reader.model.calls)
        writer = bridge.WriterBridge(self.path, {"max_model_len": 2060})
        with self.assertRaises(bridge.ContextOverflowError):
            writer.complete("some conversation")
        self.assertFalse(writer.model.calls)

    def test_duplicate_ids_and_wrong_store_shape_rejected(self):
        for state in [
            {"core": "", "semantic": ["text"], "episodic": []},
            {"core": "", "semantic": [{"abcd": "a"}, {"abcd": "b"}], "episodic": []},
        ]:
            with self.assertRaises(ValueError):
                bridge.canonical_memory(state)

    def test_empty_batches_are_noops(self):
        writer = bridge.WriterBridge(self.path)
        reader = bridge.ReaderBridge(self.path)
        self.assertEqual(writer.complete_batch([]), [])
        self.assertEqual(reader.answer_batch([]), [])
        self.assertFalse(writer.model.calls)
        self.assertFalse(reader.model.calls)

    def test_reader_empty_token_corpus_and_batch_alignment(self):
        reader = bridge.ReaderBridge(self.path)
        reader.model.responses = ["first", "second"]
        memory = {"core": "", "semantic": [{"abcd": "!!!"}], "episodic": []}
        output = reader.answer_batch([{"memory": memory, "raw_question": q, "reader_prompt": q} for q in ["q1", "q2"]])
        self.assertEqual([x["raw_response"] for x in output], ["first", "second"])
        self.assertEqual(output[0]["retrieval"]["empty_token_corpus_types"], ["semantic"])
        self.assertEqual(len(reader.model.calls), 1)


if __name__ == "__main__":
    unittest.main()
