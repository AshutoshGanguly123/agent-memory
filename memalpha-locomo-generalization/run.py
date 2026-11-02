"""Resumable, inference-only LoCoMo experiment runner."""
from __future__ import annotations

import argparse
import copy
import fcntl
import os
import signal
import sys
import time
from pathlib import Path

from common import ARMS, ROOT, atomic_json, config, digest, file_hash, machine, now, read_json, run_root, source_manifest, verify_freeze

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def synthetic_data():
    return [{
        "sample_id": "synthetic-1",
        "conversation": {
            "speaker_a": "Maya", "speaker_b": "Leo",
            "session_1_date_time": "10:00 am on 1 January, 2024",
            "session_1": [
                {"dia_id": "D1:1", "speaker": "Maya", "text": "My dog is named Pixel. I live in Paris."},
                {"dia_id": "D1:2", "speaker": "Leo", "text": "I live in Rome. I enjoy painting landscapes."}],
            "session_2_date_time": "10:00 am on 3 January, 2024",
            "session_2": [
                {"dia_id": "D2:1", "speaker": "Maya", "text": "Yesterday I adopted a cat named Luna. My dog Pixel is happy."},
                {"dia_id": "D2:2", "speaker": "Leo", "text": "I bought blue paint for a landscape yesterday.", "blip_caption": "A blue landscape painting."}],
            "session_3_date_time": "10:00 am on 5 January, 2024",
            "session_3": [
                {"dia_id": "D3:1", "speaker": "Maya", "text": "Pixel and Luna are my two pets. Luna loves the garden."},
                {"dia_id": "D3:2", "speaker": "Leo", "text": "My new landscape painting is blue."}]
        },
        "qa": [
            {"question": "What is Maya's dog named?", "answer": "Pixel", "category": 4, "evidence": ["D1:1"]},
            {"question": "What are the names of Maya's two pets?", "answer": "Pixel, Luna", "category": 1, "evidence": ["D1:1", "D2:1"]},
            {"question": "When did Maya adopt Luna?", "answer": "2 January 2024", "category": 2, "evidence": ["D2:1"]},
            {"question": "Which country does Maya live in?", "answer": "France", "category": 3, "evidence": ["D1:1"]},
            {"question": "What is Maya's dog's birthday?", "adversarial_answer": "June 15", "category": 5, "evidence": []}
        ]
    }]


def prepare(synthetic=False):
    from transformers import AutoTokenizer
    from locomo import load_dataset, prepare_conversations, prepare_questions
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models" / "reader"), local_files_only=True)
    data = synthetic_data() if synthetic else load_dataset(ROOT / "data" / "locomo10.json")
    conversations = prepare_conversations(data, tokenizer)
    if synthetic:
        # Exercise restoration across several actual model calls. Benchmark
        # chunking is unchanged; tiny smoke sessions are deliberately separate.
        for original, prepared in zip(data, conversations):
            chunks = []
            for session in sorted(k for k in original["conversation"] if k.startswith("session_") and not k.endswith("_date_time")):
                one = {"sample_id": original["sample_id"], "conversation": {
                    session: original["conversation"][session],
                    session + "_date_time": original["conversation"][session + "_date_time"]}}
                for chunk in prepare_conversations([one], tokenizer)[0]["chunks"]:
                    chunk["chunk_index"] = len(chunks)
                    chunks.append(chunk)
            prepared["chunks"] = chunks
            prepared["input_tokens"] = sum(c["token_count"] for c in chunks)
    questions = prepare_questions(data)
    target = ROOT / "data" / ("synthetic_prepared" if synthetic else "prepared")
    atomic_json(target / "conversations.json", conversations)
    atomic_json(target / "questions.json", questions)
    receipt = {"created_utc": now(), "conversations": len(conversations), "questions": len(questions),
               "chunks": sum(len(c["chunks"]) for c in conversations), "config_hash": digest(config()),
               "conversations_hash": digest(conversations), "questions_hash": digest(questions)}
    atomic_json(target / "manifest.json", receipt)
    print(receipt, flush=True)


def inputs(synthetic=False):
    target = ROOT / "data" / ("synthetic_prepared" if synthetic else "prepared")
    return read_json(target / "conversations.json"), read_json(target / "questions.json")


def freeze():
    target = run_root() / "freeze.json"
    if target.exists():
        verify_freeze()
        print("Existing freeze verified", flush=True)
        return
    smoke = read_json(run_root(True) / "validation.json")
    if not smoke["passed"] or smoke["sources"] != source_manifest():
        raise RuntimeError("Synthetic validation has not passed")
    conversations, questions = inputs()
    assert len(conversations) == 10 and len(questions) == 1986
    assets = read_json(ROOT / "models" / "manifest.json")
    for alias, expected in config()["models"].items():
        actual = assets["models"][alias]
        if actual["state"] != "ready" or actual["repo_id"] != expected["repo"] or actual["revision"] != expected["revision"]:
            raise RuntimeError("Pinned model asset verification is incomplete: " + alias)
    environment = read_json(ROOT / "results/setup/environment.exit.json")
    if environment["exit_code"] != 0:
        raise RuntimeError("Environment setup did not finish successfully")
    result = {"created_utc": now(), "sources": source_manifest(), "config": config(), "machine": machine(),
              "prepared_hashes": {n: file_hash(ROOT / "data" / "prepared" / n) for n in ("conversations.json", "questions.json", "manifest.json")},
              "synthetic_validation": smoke, "training": False, "assets": assets,
              "resolved_dependencies": (ROOT / "results/setup/requirements.resolved.txt").read_text()}
    if result["sources"]["upstream_commit"] != config()["source_commit"]:
        raise RuntimeError("Unexpected upstream source revision")
    atomic_json(target, result)
    print("Protocol frozen: " + str(target), flush=True)


def checkpoint_state(conv, directory, cfg_hash, source_hash=None):
    memory = {"core": "", "semantic": [], "episodic": []}
    completed = 0
    for i, chunk in enumerate(conv["chunks"]):
        path = directory / "chunks" / f"{i:05d}.json"
        if not path.exists():
            break
        saved = read_json(path)
        if saved["config_hash"] != cfg_hash or saved["chunk_hash"] != digest(chunk) or saved["input_memory_hash"] != digest(memory):
            raise RuntimeError("Checkpoint provenance mismatch: " + str(path))
        if source_hash is not None and saved.get("source_hash") != source_hash:
            raise RuntimeError("Checkpoint source hash mismatch: " + str(path))
        memory = saved["result"]["memory"]
        if digest(memory) != saved["output_memory_hash"]:
            raise RuntimeError("Checkpoint memory corrupted: " + str(path))
        completed = i + 1
    if any(int(p.stem) >= completed for p in (directory / "chunks").glob("*.json")):
        raise RuntimeError("Noncontiguous checkpoint chain: " + str(directory))
    return memory, completed


def build(arm, synthetic=False):
    from bridge import WriterBridge
    if not synthetic:
        verify_freeze()
    cfg = config()
    cfg_hash = digest(cfg)
    source_hash = digest(source_manifest())
    conversations, _ = inputs(synthetic)
    base = run_root(synthetic) / arm
    started = time.monotonic()
    states = []
    for conv in conversations:
        directory = base / "conversations" / conv["sample_id"]
        memory, index = checkpoint_state(conv, directory, cfg_hash, source_hash)
        states.append({"conv": conv, "directory": directory, "memory": memory, "index": index})
    status_path = base / "build_status.json"
    atomic_json(status_path, {"status": "loading", "machine": machine(), "started_utc": now()})
    engine = None
    while True:
        active = [s for s in states if s["index"] < len(s["conv"]["chunks"])][:cfg["writer_batch_size"]]
        if not active:
            break
        if STOP:
            atomic_json(status_path, {"status": "stopped", "utc": now()})
            return 75
        if engine is None:
            engine = WriterBridge(str(ROOT / "models" / arm), cfg["writer"])
            atomic_json(base / "writer_engine.json", {"engine": getattr(engine, "engine_config", {}),
                        "sampling": str(getattr(engine, "sampling_params", "")), "machine": machine()})
        items = [{"chunk_text": s["conv"]["chunks"][s["index"]]["text"], "memory_state": copy.deepcopy(s["memory"])} for s in active]
        before_hashes = [digest(s["memory"]) for s in active]
        started_batch = time.monotonic()
        results = engine.complete_batch(items)
        if len(results) != len(active):
            raise RuntimeError("Writer output count mismatch")
        for s, result, input_hash in zip(active, results, before_hashes):
            if not result.get("raw_response", "").strip() or "memory" not in result:
                raise RuntimeError("Writer emitted no usable completion")
            i = s["index"]
            receipt = {"sample_id": s["conv"]["sample_id"], "chunk_index": i, "config_hash": cfg_hash, "source_hash": source_hash,
                       "chunk_hash": digest(s["conv"]["chunks"][i]), "input_memory_hash": input_hash,
                       "output_memory_hash": digest(result["memory"]), "result": result,
                       "batch_wall_seconds": time.monotonic() - started_batch, "utc": now()}
            atomic_json(s["directory"] / "chunks" / f"{i:05d}.json", receipt)
            s["memory"] = result["memory"]
            s["index"] += 1
        progress = {"status": "running", "utc": now(), "elapsed_seconds": time.monotonic() - started,
                    "completed_chunks": sum(s["index"] for s in states), "total_chunks": sum(len(s["conv"]["chunks"]) for s in states),
                    "conversations": {s["conv"]["sample_id"]: s["index"] for s in states}}
        atomic_json(status_path, progress)
        print(progress, flush=True)
    for s in states:
        frozen = {"sample_id": s["conv"]["sample_id"], "arm": arm, "config_hash": cfg_hash, "source_hash": source_hash,
                  "memory": s["memory"], "memory_hash": digest(s["memory"]), "completed_chunks": s["index"], "frozen": True}
        target = s["directory"] / "memory_frozen.json"
        if target.exists() and read_json(target) != frozen:
            raise RuntimeError("Refusing to replace inconsistent frozen memory")
        atomic_json(target, frozen)
    elapsed = time.monotonic() - started
    gpu_count = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) if os.environ.get("CUDA_VISIBLE_DEVICES") else 0
    atomic_json(status_path, {"status": "complete", "utc": now(), "elapsed_seconds": elapsed, "gpu_count": gpu_count,
                              "gpu_allocated_seconds": elapsed * gpu_count, "machine": machine(),
                              "conversations": len(states), "completed_chunks": sum(s["index"] for s in states)})
    return 0


def answer(synthetic=False):
    from bridge import ReaderBridge
    if not synthetic:
        verify_freeze()
    cfg = config()
    cfg_hash = digest(cfg)
    source_hash = digest(source_manifest())
    _, questions = inputs(synthetic)
    base = run_root(synthetic)
    memories = {}
    tasks = []
    for arm in ARMS:
        for q in questions:
            key = (arm, q["sample_id"])
            if key not in memories:
                saved = read_json(base / arm / "conversations" / q["sample_id"] / "memory_frozen.json")
                if not saved["frozen"] or saved["config_hash"] != cfg_hash or saved.get("source_hash") != source_hash or digest(saved["memory"]) != saved["memory_hash"]:
                    raise RuntimeError("Frozen memory integrity check failed")
                memories[key] = saved
            path = base / arm / "answers" / q["sample_id"] / f'{q["qa_index"]:05d}.json'
            expected = {"arm": arm, "sample_id": q["sample_id"], "qa_index": q["qa_index"], "question_hash": digest(q),
                        "memory_hash": memories[key]["memory_hash"], "config_hash": cfg_hash, "source_hash": source_hash}
            attempt = 1
            if path.exists():
                saved = read_json(path)
                if any(saved.get(k) != v for k, v in expected.items()):
                    raise RuntimeError("Answer checkpoint mismatch: " + str(path))
                if saved["status"] == "complete" or saved.get("attempt", 1) >= 3:
                    continue
                attempt = saved.get("attempt", 1) + 1
                atomic_json(path.parent / "attempts" / f'{q["qa_index"]:05d}_{attempt - 1}.json', saved)
            tasks.append((q, memories[key]["memory"], path, expected, attempt))
    status_path = base / "answer_status.json"
    started = time.monotonic()
    atomic_json(status_path, {"status": "loading", "machine": machine(), "pending": len(tasks), "utc": now()})
    engine = ReaderBridge(str(ROOT / "models" / "reader"), cfg["reader"]) if tasks else None
    if engine is not None:
        atomic_json(base / "reader_engine.json", {"engine": getattr(engine, "engine_config", {}),
                    "sampling": str(getattr(engine, "sampling_params", "")), "machine": machine()})
    for offset in range(0, len(tasks), cfg["reader_batch_size"]):
        if STOP:
            atomic_json(status_path, {"status": "stopped", "utc": now()})
            return 75
        batch = tasks[offset:offset + cfg["reader_batch_size"]]
        items = [{"memory": copy.deepcopy(m), "raw_question": q["question"], "reader_prompt": q["reader_prompt"]} for q, m, _, _, _ in batch]
        start_batch = time.monotonic()
        results = engine.answer_batch(items)
        if len(results) != len(batch):
            raise RuntimeError("Reader output count mismatch")
        for (q, memory, path, expected, attempt), result in zip(batch, results):
            if digest(memory) != expected["memory_hash"]:
                raise RuntimeError("QA mutated frozen memory")
            raw = result.get("raw_response", "")
            valid = isinstance(raw, str) and bool(raw.strip()) and not raw.startswith(("Error:", "No response from server"))
            receipt = dict(expected, status="complete" if valid else "error", result=result, attempt=attempt,
                           batch_wall_seconds=time.monotonic() - start_batch, utc=now())
            atomic_json(path, receipt)
        progress = {"status": "running", "utc": now(), "elapsed_seconds": time.monotonic() - started,
                    "completed": 2 * len(questions) - len(tasks) + min(offset + len(batch), len(tasks)), "total": 2 * len(questions)}
        atomic_json(status_path, progress)
        print(progress, flush=True)
    failures = sum(read_json(p).get("status") != "complete" for p in base.glob("*/answers/*/*.json"))
    elapsed = time.monotonic() - started
    gpu_count = len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) if os.environ.get("CUDA_VISIBLE_DEVICES") else 0
    atomic_json(status_path, {"status": "complete" if not failures else "complete_with_errors", "utc": now(),
                              "elapsed_seconds": elapsed, "gpu_count": gpu_count, "gpu_allocated_seconds": elapsed * gpu_count,
                              "machine": machine(), "errors": failures, "total": 2 * len(questions)})
    return 0 if not failures else 2


def validate_smoke():
    conversations, questions = inputs(True)
    checks = []
    for arm in ARMS:
        for c in conversations:
            directory = run_root(True) / arm / "conversations" / c["sample_id"]
            memory = read_json(directory / "memory_frozen.json")
            checks.append({"check": arm + " memory integrity", "passed": memory["memory_hash"] == digest(memory["memory"])})
            calls = [read_json(p)["result"] for p in sorted((directory / "chunks").glob("*.json"))]
            checks.append({"check": arm + " writer responses", "passed": all(bool(r["raw_response"].strip()) for r in calls)})
            if arm == "memalpha":
                checks.append({"check": "trained writer populated memory", "passed": bool(memory["memory"]["core"] or memory["memory"]["semantic"] or memory["memory"]["episodic"])})
        for q in questions:
            response = read_json(run_root(True) / arm / "answers" / q["sample_id"] / f'{q["qa_index"]:05d}.json')
            checks.append({"check": arm + " answer " + str(q["qa_index"]), "passed": response["status"] == "complete"})
    result = {"passed": all(c["passed"] for c in checks), "checks": checks, "created_utc": now(), "sources": source_manifest()}
    atomic_json(run_root(True) / "validation.json", result)
    print(result, flush=True)
    if not result["passed"]:
        raise RuntimeError("Synthetic preflight failed")


def status(synthetic=False):
    base = run_root(synthetic)
    result = {"run": str(base), "utc": now(), "frozen_memories": len(list(base.glob("*/conversations/*/memory_frozen.json"))),
              "saved_answers": len(list(base.glob("*/answers/*/*.json")))}
    for p in sorted(base.glob("**/*_status.json")):
        result[str(p.relative_to(base))] = read_json(p)
    print(result, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "build", "answer", "freeze", "validate-smoke", "score", "report", "status"])
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    lock_path = run_root(args.synthetic) / ".locks" / f'{args.stage}_{args.arm or "all"}.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stage_lock = lock_path.open("a")
    try:
        fcntl.flock(stage_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("This stage is already running: " + str(lock_path))
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    if args.stage == "prepare": prepare(args.synthetic)
    elif args.stage == "build":
        if not args.arm: parser.error("build requires --arm")
        return build(args.arm, args.synthetic)
    elif args.stage == "answer": return answer(args.synthetic)
    elif args.stage == "freeze": freeze()
    elif args.stage == "validate-smoke": validate_smoke()
    elif args.stage in ("score", "report"):
        from report import generate_report
        generate_report(args.synthetic)
    elif args.stage == "status": status(args.synthetic)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        atomic_json(run_root("--synthetic" in sys.argv) / "failures" / f"{os.getpid()}.json",
                    {"utc": now(), "error": repr(exc), "argv": sys.argv, "machine": machine()})
        sys.exit(1)
