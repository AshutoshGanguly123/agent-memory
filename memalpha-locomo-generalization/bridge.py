"""Frozen-model inference using the pinned Mem-α prompt, CRUD, and BM25 code.

No model, tokenizer, or GPU dependency is imported until a bridge is created.
Model paths must be predownloaded local directories. All public results are JSON
serializable and input memory dictionaries are never mutated.
"""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sys
import time
from typing import Any

UPSTREAM = Path(__file__).resolve().parent / "upstream"
MAX_OUTPUT_TOKENS = 2048


class ContextOverflowError(ValueError):
    pass


def empty_memory() -> dict:
    return {"core": "", "semantic": [], "episodic": []}


def canonical_memory(state: dict | None) -> dict:
    """Validate and copy the canonical stores, ignoring auxiliary saved metadata."""
    state = empty_memory() if state is None else state
    if not isinstance(state, dict):
        raise ValueError("Memory state must be a dictionary")
    result = {"core": state.get("core", "")}
    if not isinstance(result["core"], str):
        raise ValueError("Core memory must be a string")
    for kind in ("semantic", "episodic"):
        # Operations identify a store and an ID. A legacy snapshot can safely
        # reuse an ID across the two stores, but not within one store.
        ids: set[str] = set()
        items = state.get(kind, [])
        if not isinstance(items, list):
            raise ValueError(f"{kind} memory must be a list of single-entry dictionaries")
        for item in items:
            if not isinstance(item, dict) or len(item) != 1:
                raise ValueError(f"Invalid {kind} item: expected one memory ID and content")
            key, value = next(iter(item.items()))
            if not isinstance(key, str) or len(key) != 4 or any(c not in "0123456789abcdef" for c in key):
                raise ValueError("Memory IDs must contain four lowercase hexadecimal characters")
            if key in ids:
                raise ValueError(f"Duplicate memory ID: {key}")
            if not isinstance(value, str):
                raise ValueError("Memory content must be a string")
            ids.add(key)
        result[kind] = copy.deepcopy(items)
    return result


def _load_runtime() -> dict[str, Any]:
    if str(UPSTREAM) not in sys.path:
        sys.path.insert(0, str(UPSTREAM))
    modules = {name: importlib.import_module(name) for name in
               ("memory", "agent", "functions", "memory_server")}
    for name, module in modules.items():
        if Path(module.__file__).resolve().parent != UPSTREAM:
            raise ImportError(f"Unexpected module named {name}: {module.__file__}")
    import numpy as np
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    import yaml
    with (UPSTREAM / "config/prompts_wrt_datasource.yaml").open() as handle:
        unified_prompt = yaml.safe_load(handle)["unified_prompt"]
    return {"Memory": modules["memory"].Memory,
            "MemoryAgent": modules["agent"].MemoryAgent,
            "MemoryProcessor": modules["memory_server"].MemoryProcessor,
            "get_memory_tool_schemas": modules["functions"].get_memory_tool_schemas,
            "AutoTokenizer": AutoTokenizer, "LLM": LLM,
            "SamplingParams": SamplingParams, "np": np,
            "unified_prompt": unified_prompt}


def _restore(runtime: dict, state: dict | None, seed: int):
    state = canonical_memory(state)
    memory = runtime["Memory"](including_core=True, embedding_enabled=False, id_seed=str(seed))
    memory.core = state["core"]
    for kind in ("semantic", "episodic"):
        setattr(memory, kind, state[kind])
        setattr(memory, f"{kind}_embedding_ids", [next(iter(item)) for item in state[kind]])
        # Upstream CRUD expects a correctly shaped row for every existing ID.
        setattr(memory, f"{kind}_embedding_matrix", runtime["np"].zeros((len(state[kind]), 1536)))
    return memory


def _snapshot(memory) -> dict:
    return canonical_memory({kind: getattr(memory, kind) for kind in ("core", "semantic", "episodic")})


def _token_ids(tokenizer, prompt: str) -> list[int]:
    return tokenizer.encode(prompt, add_special_tokens=False)


def _check_capacity(prompt_tokens: int, context_limit: int) -> None:
    if prompt_tokens + MAX_OUTPUT_TOKENS > context_limit:
        raise ContextOverflowError(
            f"Full prompt ({prompt_tokens}) plus output reserve ({MAX_OUTPUT_TOKENS}) "
            f"exceeds served context ({context_limit}); input was not truncated")


class _Bridge:
    def __init__(self, model_path: str, config: dict | None, default_tp: int):
        self.config = dict(config or {})
        self.model_path = str(Path(model_path).expanduser().resolve())
        if not Path(self.model_path).is_dir():
            raise ValueError(f"Model must be a local directory: {self.model_path}")
        self.runtime = _load_runtime()
        self.seed = int(self.config.get("seed", 42))
        self.max_model_len = int(self.config.get("max_model_len", 32768))
        self.tokenizer = self.runtime["AutoTokenizer"].from_pretrained(
            self.model_path, local_files_only=True)
        model_config_path = Path(self.model_path) / "config.json"
        if model_config_path.exists():
            model_config = json.loads(model_config_path.read_text())
            model_limit = model_config.get("max_position_embeddings")
            if model_limit and self.max_model_len > int(model_limit):
                raise ValueError("Served context exceeds the checkpoint's position limit")
        if self.max_model_len <= MAX_OUTPUT_TOKENS:
            raise ValueError("Served context must leave room for an input prompt")
        kwargs = {"model": self.model_path, "tokenizer": self.model_path,
                  "dtype": "bfloat16", "seed": self.seed,
                  "tensor_parallel_size": int(self.config.get("tensor_parallel_size", default_tp)),
                  "max_model_len": self.max_model_len,
                  "enable_chunked_prefill": True,
                  "gpu_memory_utilization": float(self.config.get("gpu_memory_utilization", 0.90)),
                  "max_num_seqs": int(self.config.get("max_num_seqs", 8)),
                  "trust_remote_code": False}
        if "enforce_eager" in self.config:
            kwargs["enforce_eager"] = bool(self.config["enforce_eager"])
        if "max_num_batched_tokens" in self.config:
            kwargs["max_num_batched_tokens"] = int(self.config["max_num_batched_tokens"])
        self.model = self.runtime["LLM"](**kwargs)
        self.engine_config = kwargs

    def _generate(self, prompts: list[str], sampling_params) -> list[dict]:
        if not prompts:
            return []
        lengths = [len(_token_ids(self.tokenizer, prompt)) for prompt in prompts]
        for size in lengths:
            _check_capacity(size, self.max_model_len)
        start = time.monotonic()
        outputs = self.model.generate(prompts, sampling_params, use_tqdm=False)
        elapsed = time.monotonic() - start
        if len(outputs) != len(prompts):
            raise RuntimeError("vLLM returned an unexpected number of completions")
        records = []
        for output, prompt, prompt_size in zip(outputs, prompts, lengths):
            if len(output.outputs) != 1:
                raise RuntimeError("Expected exactly one candidate for each independent prompt")
            candidate = output.outputs[0]
            if not isinstance(candidate.text, str):
                raise RuntimeError("Completion text is missing")
            completion_size = len(candidate.token_ids)
            finish_reason = candidate.finish_reason
            if finish_reason not in {"stop", "length"}:
                raise RuntimeError(f"Incomplete generation: {finish_reason}")
            records.append({"raw_response": candidate.text,
                            "prompt": prompt,
                            "usage": {"prompt_tokens": prompt_size,
                                      "completion_tokens": completion_size,
                                      "total_tokens": prompt_size + completion_size,
                                      "batch_wall_seconds": elapsed,
                                      "batch_size": len(prompts)},
                            "finish_reason": finish_reason,
                            "stop_reason": getattr(candidate, "stop_reason", None),
                            "truncated": finish_reason == "length"})
        return records


class WriterBridge(_Bridge):
    """One completion and ordered upstream CRUD execution for each input chunk."""
    def __init__(self, model_path: str, config: dict | None = None):
        super().__init__(model_path, config, default_tp=1)
        # Bypass upstream __init__: it otherwise creates another model or clients.
        self.agent = self.runtime["MemoryAgent"].__new__(self.runtime["MemoryAgent"])
        self.sampling_params = self.runtime["SamplingParams"](
            temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS, seed=self.seed,
            stop_token_ids=[self.tokenizer.eos_token_id])

    def _prompt(self, chunk_text: str, memory_state: dict | None) -> str:
        if not isinstance(chunk_text, str) or not chunk_text.strip():
            raise ValueError("Writer requires a nonempty string chunk")
        memory = _restore(self.runtime, memory_state, self.seed)
        text = self.runtime["unified_prompt"].format(
            context=chunk_text, max_new_tokens=int(MAX_OUTPUT_TOKENS * 0.8))
        return self.runtime["MemoryAgent"].process_text_with_qwen_pipeline(
            text=text, tokenizer=self.tokenizer,
            functions=[tool["function"] for tool in self.runtime["get_memory_tool_schemas"](memory)],
            status="memorie", enable_thinking=False, return_text=True,
            memory=memory, max_num_of_recent_chunks=5)

    def replay(self, memory_state: dict | None, raw_response: str) -> dict:
        """Apply a journaled completion to its PRE-chunk state without generation.

        Deterministic opaque IDs make this operation exactly replayable. Calling
        it on already updated memory is not the resume protocol.
        """
        memory = _restore(self.runtime, memory_state, self.seed)
        tool_calls, errors = [], []
        messages = self.agent._parse_response(raw_response.strip())
        for message in messages:
            function_call = message.get("function_call")
            if not function_call:
                continue
            before = _snapshot(memory)
            result = self.agent._run_tool_from_function_call(function_call, memory)
            error = None
            try:
                _snapshot(memory)
                if result.startswith(f"[tool {function_call['name']} error]"):
                    error = result
            except (ValueError, TypeError) as exc:
                error = str(exc)
            if error is not None:
                # Bad model tool calls are outcomes, not poisoned saved states.
                memory = _restore(self.runtime, before, self.seed)
                errors.append({"function_call": copy.deepcopy(function_call), "error": error})
            tool_calls.append({"function_call": copy.deepcopy(function_call),
                               "tool_result": result, "error": error})
        if "✿FUNCTION✿" in raw_response and not tool_calls:
            errors.append({"error": "Function marker present but no function call could be parsed"})
        return {"memory": _snapshot(memory), "tool_calls": tool_calls, "errors": errors}

    def complete_batch(self, items: list[dict]) -> list[dict]:
        prompts = [self._prompt(item["chunk_text"], item.get("memory_state")) for item in items]
        generations = self._generate(prompts, self.sampling_params)
        return [{**record, **self.replay(item.get("memory_state"), record["raw_response"])}
                for item, record in zip(items, generations)]

    def complete(self, chunk_text: str, memory_state: dict | None = None) -> dict:
        return self.complete_batch([{"chunk_text": chunk_text, "memory_state": memory_state}])[0]


class ReaderBridge(_Bridge):
    """Independent reader queries against a frozen canonical memory snapshot."""
    def __init__(self, model_path: str, config: dict | None = None):
        super().__init__(model_path, config, default_tp=4)
        processor_type = self.runtime["MemoryProcessor"]
        self.processor = processor_type.__new__(processor_type)
        self.processor.tokenizer = self.tokenizer
        self.processor.model = "qwen3-32b"
        # vLLM's OpenAI server resolves unspecified sampling fields against the
        # checkpoint generation config. Apply the same defaults for direct LLM.
        path = Path(self.model_path) / "generation_config.json"
        generation = json.loads(path.read_text()) if path.exists() else {}
        self.generation_defaults = {"top_p": generation.get("top_p", 1.0),
                                    "top_k": generation.get("top_k", -1),
                                    "repetition_penalty": generation.get("repetition_penalty", 1.0),
                                    "min_p": generation.get("min_p", 0.0)}
        self.sampling_params = self.runtime["SamplingParams"](
            temperature=0.7, max_tokens=MAX_OUTPUT_TOKENS, seed=self.seed,
            **self.generation_defaults)

    def _prepare(self, memory: dict, raw_question: str, reader_prompt: str) -> tuple[str, dict]:
        memory = canonical_memory(memory)
        if not isinstance(raw_question, str) or not raw_question.strip():
            raise ValueError("Reader requires a nonempty raw question")
        if not isinstance(reader_prompt, str) or not reader_prompt.strip():
            raise ValueError("Reader requires a nonempty rendered user prompt")
        # rank_bm25 cannot initialize when every document contains no tokens.
        searchable = copy.deepcopy(memory)
        empty_corpus_types = []
        for kind in ("semantic", "episodic"):
            if searchable[kind] and not any(self.processor._tokenize(next(iter(item.values())))
                                            for item in searchable[kind]):
                searchable[kind] = []
                empty_corpus_types.append(kind)
        for top_k in range(20, 0, -1):
            retrieved = self.processor.search_memories(searchable, raw_question, top_k=top_k)
            system = self.processor.construct_system_prompt(retrieved, memory)
            system_tokens = len(_token_ids(self.tokenizer, system))
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": reader_prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            full_tokens = len(_token_ids(self.tokenizer, prompt))
            if system_tokens < 30000 and full_tokens + MAX_OUTPUT_TOKENS <= self.max_model_len:
                return prompt, {"query": raw_question, "method": "bm25", "top_k_per_store": top_k,
                                "memory": retrieved, "system_prompt_tokens": system_tokens,
                                "full_prompt_tokens": full_tokens,
                                "empty_token_corpus_types": empty_corpus_types}
        raise ContextOverflowError("Reader context does not fit even at BM25 top-k=1; no input was truncated")

    def answer_batch(self, items: list[dict]) -> list[dict]:
        prepared = [self._prepare(item["memory"], item["raw_question"], item["reader_prompt"])
                    for item in items]
        generations = self._generate([prompt for prompt, _ in prepared], self.sampling_params)
        return [{**record, "retrieval": retrieval, "errors": []}
                for record, (_, retrieval) in zip(generations, prepared)]

    def answer(self, memory: dict, raw_question: str, reader_prompt: str) -> dict:
        return self.answer_batch([{"memory": memory, "raw_question": raw_question,
                                   "reader_prompt": reader_prompt}])[0]
