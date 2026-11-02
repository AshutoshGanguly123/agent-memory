"""Standalone, read-only integrity audit of the frozen Mem-alpha LoCoMo run.

No predictions are scored, evaluated, regenerated, or sent to a model. The
optional output path creates a separate verification artifact; no frozen source
or inference receipt is changed. Only Python's standard library is required.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

ARMS = ("memalpha", "qwen3_base")
CATEGORY_COUNTS = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
DATASET_HASH = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def memory_valid(memory):
    if not isinstance(memory, dict) or not isinstance(memory.get("core"), str):
        return False
    for kind in ("semantic", "episodic"):
        items = memory.get(kind)
        if not isinstance(items, list):
            return False
        seen = set()
        for item in items:
            if not isinstance(item, dict) or len(item) != 1:
                return False
            key, value = next(iter(item.items()))
            if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{4}", key) or key in seen or not isinstance(value, str):
                return False
            seen.add(key)
    return True


def verify_receipts(base, cfg, sources, conversations, questions):
    """Check the entire expected artifact graph without examining answer quality."""
    base = Path(base)
    issues = []
    checks = Counter()
    cfg_hash, source_hash = digest(cfg), digest(sources)
    expected_memories, expected_chunks, expected_answers = set(), set(), set()
    frozen_memories = {}
    completed = {arm: Counter() for arm in ARMS}
    archived_attempts = 0

    def fail(path, message):
        issues.append({"path": str(Path(path).relative_to(base)), "message": message})

    def load(path):
        if not path.is_file():
            fail(path, "Missing expected artifact")
            return None
        try:
            return read(path)
        except (ValueError, OSError) as exc:
            fail(path, "Unreadable JSON: " + str(exc))
            return None

    def matches(path, saved, expected):
        bad = [key for key, value in expected.items() if saved.get(key) != value]
        if bad:
            fail(path, "Mismatched fields: " + ", ".join(bad))
        return not bad

    for arm in ARMS:
        for conversation in conversations:
            sample = conversation["sample_id"]
            directory = base / arm / "conversations" / sample
            previous = {"core": "", "semantic": [], "episodic": []}
            valid_chain = True
            for index, chunk in enumerate(conversation["chunks"]):
                path = directory / "chunks" / f"{index:05d}.json"
                expected_chunks.add(path)
                saved = load(path)
                if saved is None:
                    valid_chain = False
                    continue
                checks["chunk_receipts"] += 1
                matches(path, saved, {"sample_id": sample, "chunk_index": index,
                                      "config_hash": cfg_hash, "source_hash": source_hash,
                                      "chunk_hash": digest(chunk)})
                if valid_chain:
                    matches(path, saved, {"input_memory_hash": digest(previous)})
                result = saved.get("result", {})
                memory = result.get("memory")
                if not memory_valid(memory):
                    fail(path, "Invalid canonical memory state")
                    valid_chain = False
                    continue
                matches(path, saved, {"output_memory_hash": digest(memory)})
                if not isinstance(result.get("raw_response"), str) or not result["raw_response"].strip():
                    fail(path, "Writer completion missing")
                prompt = result.get("prompt", "")
                if not isinstance(prompt, str) or chunk["text"] not in prompt:
                    fail(path, "Writer prompt does not contain its source chunk")
                if result.get("finish_reason") not in ("stop", "length"):
                    fail(path, "Writer completion did not finish")
                previous = memory
            path = directory / "memory_frozen.json"
            expected_memories.add(path)
            frozen = load(path)
            if frozen is None:
                continue
            checks["frozen_memories"] += 1
            matches(path, frozen, {"arm": arm, "sample_id": sample, "frozen": True,
                                   "config_hash": cfg_hash, "source_hash": source_hash,
                                   "completed_chunks": len(conversation["chunks"])})
            memory = frozen.get("memory")
            if not memory_valid(memory):
                fail(path, "Invalid frozen memory state")
                continue
            matches(path, frozen, {"memory_hash": digest(memory)})
            if valid_chain and memory != previous:
                fail(path, "Frozen memory differs from the final committed chunk")
            frozen_memories[(arm, sample)] = frozen

    for arm in ARMS:
        for question in questions:
            sample, index = question["sample_id"], question["qa_index"]
            path = base / arm / "answers" / sample / f"{index:05d}.json"
            expected_answers.add(path)
            latest = load(path)
            frozen = frozen_memories.get((arm, sample))
            if latest is not None:
                checks["latest_answers"] += 1
            attempts = [(path, latest)] if latest is not None else []
            for old_path in sorted((path.parent / "attempts").glob(f"{index:05d}_*.json")):
                old = load(old_path)
                if old is not None:
                    attempts.append((old_path, old))
                    archived_attempts += 1
            numbers = {}
            for attempt_path, saved in attempts:
                matches(attempt_path, saved, {"arm": arm, "sample_id": sample, "qa_index": index,
                                             "question_hash": digest(question), "config_hash": cfg_hash,
                                             "source_hash": source_hash})
                if frozen is not None:
                    matches(attempt_path, saved, {"memory_hash": frozen["memory_hash"]})
                else:
                    fail(attempt_path, "Answer has no valid corresponding frozen memory")
                number = saved.get("attempt", 1)
                if not isinstance(number, int) or not 1 <= number <= 3:
                    fail(attempt_path, "Invalid reader attempt number")
                    continue
                if attempt_path != path and attempt_path.name != f"{index:05d}_{number}.json":
                    fail(attempt_path, "Archived attempt filename/number mismatch")
                if number in numbers and digest(numbers[number]) != digest(saved):
                    fail(attempt_path, "Conflicting duplicate reader attempt")
                numbers[number] = saved
                if saved.get("status") not in ("complete", "error"):
                    fail(attempt_path, "Unknown reader receipt status")
                result = saved.get("result", {})
                raw = result.get("raw_response")
                valid_output = isinstance(raw, str) and bool(raw.strip()) and not raw.startswith(("Error:", "No response from server"))
                if saved.get("status") == "complete" and not valid_output:
                    fail(attempt_path, "Complete status contains an invalid/empty output")
                if result.get("finish_reason") not in ("stop", "length"):
                    fail(attempt_path, "Reader completion did not finish")
                retrieval = result.get("retrieval", {})
                if retrieval.get("query") != question["question"] or retrieval.get("method") != "bm25":
                    fail(attempt_path, "Retrieval query/method is misaligned")
                prompt = result.get("prompt", "")
                expected_user_suffix = ("<|im_start|>user\n" + question["reader_prompt"] +
                                        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
                if not isinstance(prompt, str) or not prompt.endswith(expected_user_suffix):
                    fail(attempt_path, "Reader prompt is missing the shared question/option prompt")
                selected = retrieval.get("memory")
                top_k = retrieval.get("top_k_per_store")
                if not memory_valid(selected) or not isinstance(top_k, int) or not 1 <= top_k <= 20:
                    fail(attempt_path, "Invalid retrieval memory or top-k")
                elif frozen is not None:
                    original = frozen["memory"]
                    if selected["core"] != original["core"]:
                        fail(attempt_path, "Reader did not receive the frozen core")
                    for kind in ("semantic", "episodic"):
                        if len(selected[kind]) > top_k or any(item not in original[kind] for item in selected[kind]):
                            fail(attempt_path, "Retrieved items are not a bounded subset of the frozen " + kind + " store")
                usage = result.get("usage", {})
                if (not isinstance(usage.get("prompt_tokens"), int) or
                        usage["prompt_tokens"] + 2048 > cfg["reader"]["max_model_len"] or
                        retrieval.get("full_prompt_tokens") != usage.get("prompt_tokens") or
                        not isinstance(retrieval.get("system_prompt_tokens"), int) or
                        retrieval["system_prompt_tokens"] >= 30000):
                    fail(attempt_path, "Reader context accounting is invalid")
            if numbers and set(numbers) != set(range(1, max(numbers) + 1)):
                fail(path, "Reader attempt history has gaps")
            if numbers and latest is not None and latest.get("attempt", 1) != max(numbers):
                fail(path, "Latest reader receipt is older than its archived history")
            if latest is not None and latest.get("status") == "complete":
                completed[arm][question["category"]] += 1
            elif latest is not None:
                fail(path, "Latest answer is an error; final coverage is incomplete")

    for pattern, expected, label in (
        ("*/conversations/*/memory_frozen.json", expected_memories, "frozen memory"),
        ("*/conversations/*/chunks/*.json", expected_chunks, "chunk receipt"),
        ("*/answers/*/*.json", expected_answers, "latest answer"),
    ):
        for extra in set(base.glob(pattern)) - expected:
            fail(extra, "Unexpected/duplicate " + label + " path")
    known_attempt_parents = {path.parent for path in expected_answers}
    for path in base.glob("*/answers/*/attempts/*.json"):
        match = re.fullmatch(r"(\d{5})_([123])\.json", path.name)
        main = path.parent.parent / ((match[1] + ".json") if match else "invalid")
        if path.parent.parent not in known_attempt_parents or main not in expected_answers:
            fail(path, "Unexpected archived answer path")
    checks["archived_attempt_files"] = archived_attempts
    return {"issues": issues, "counts": dict(checks),
            "expected": {"frozen_memories": len(expected_memories), "chunk_receipts": len(expected_chunks),
                         "latest_answers": len(expected_answers)},
            "completed_categories": {arm: dict(completed[arm]) for arm in ARMS}}


def _verify_report(base, questions, completed):
    """Validate report coverage and declared denominators; do not rescore outputs."""
    issues = []
    metrics_path = base / "report/metrics.json"
    rows_path = base / "report/per_question.json"
    if not metrics_path.exists() and not rows_path.exists():
        return {"present": False, "issues": [{"path": "report", "message": "Final report artifacts are missing"}]}
    if not metrics_path.exists() or not rows_path.exists():
        return {"present": True, "issues": [{"path": "report", "message": "Incomplete report artifacts"}]}
    metrics, rows = read(metrics_path), read(rows_path)
    expected = {(arm, q["sample_id"], q["qa_index"]): q for arm in ARMS for q in questions}
    seen = Counter((row["arm"], row["sample_id"], row["qa_index"]) for row in rows)
    if set(seen) != set(expected) or any(count != 1 for count in seen.values()):
        issues.append({"path": "report/per_question.json", "message": "Missing/extra/duplicate report rows"})
    for row in rows:
        key = (row["arm"], row["sample_id"], row["qa_index"])
        if key in expected and row["category"] != expected[key]["category"]:
            issues.append({"path": "report/per_question.json", "message": "Report category misalignment"})
        if key in expected:
            receipt_path = base / key[0] / "answers" / key[1] / f"{key[2]:05d}.json"
            receipt = read(receipt_path) if receipt_path.exists() else None
            status = receipt["status"] if receipt else "missing"
            raw = receipt["result"]["raw_response"] if receipt and status == "complete" else ""
            if row.get("status") != status or row.get("raw_response") != raw or row.get("question") != expected[key]["question"]:
                issues.append({"path": "report/per_question.json", "message": "Report question/status/raw-output alignment mismatch"})
    counts = Counter(q["category"] for q in questions)
    for arm in ARMS:
        values = metrics.get("metrics", {}).get(arm, {})
        if values.get("count") != len(questions) or values.get("completed") != sum(completed[arm].values()):
            issues.append({"path": "report/metrics.json", "message": arm + " overall denominator/completion mismatch"})
        for category, name in CATEGORY_NAMES.items():
            item = values.get("categories", {}).get(name, {})
            if item.get("count") != counts[category] or item.get("completed") != completed[arm].get(category, 0):
                issues.append({"path": "report/metrics.json", "message": arm + " " + name + " denominator mismatch"})
        clean = sum(q["category"] == 5 and not q.get("anomaly") for q in questions)
        if values.get("adversarial_clean_count") != clean:
            issues.append({"path": "report/metrics.json", "message": arm + " clean adversarial denominator mismatch"})
    complete = all(sum(completed[arm].values()) == len(questions) for arm in ARMS)
    if metrics.get("complete") != complete:
        issues.append({"path": "report/metrics.json", "message": "Report completion flag disagrees with exact receipt coverage"})
    return {"present": True, "issues": issues,
            "artifact_sha256": {"metrics.json": file_hash(metrics_path), "per_question.json": file_hash(rows_path)}}


def verify(root):
    root = Path(root).resolve()
    cfg = read(root / "config.json")
    base = root / "runs" / cfg["run_id"]
    frozen = read(base / "freeze.json")
    issues = []
    if cfg != frozen.get("config") or cfg.get("training_enabled") is not False or frozen.get("training") is not False:
        issues.append({"path": "freeze.json", "message": "Configuration/no-training freeze mismatch"})
    sources = frozen["sources"]
    for name, expected in sources.items():
        if name == "upstream_commit":
            actual = subprocess.check_output(["git", "-C", str(root / "upstream"), "rev-parse", "HEAD"], text=True).strip()
        elif name == "upstream_worktree_diff":
            actual = hashlib.sha256(subprocess.check_output(["git", "-C", str(root / "upstream"), "diff", "HEAD"])).hexdigest()
        else:
            actual = file_hash(root / name)
        if actual != expected:
            issues.append({"path": name, "message": "Frozen source hash mismatch"})
    for name, expected in frozen["prepared_hashes"].items():
        if file_hash(root / "data/prepared" / name) != expected:
            issues.append({"path": "data/prepared/" + name, "message": "Frozen prepared-input hash mismatch"})
    source_path = root / "data/locomo10.json"
    actual_dataset_hash = file_hash(source_path)
    if actual_dataset_hash != DATASET_HASH or actual_dataset_hash != cfg.get("dataset_sha256"):
        issues.append({"path": "data/locomo10.json", "message": "Pinned dataset hash mismatch"})
    conversations = read(root / "data/prepared/conversations.json")
    questions = read(root / "data/prepared/questions.json")
    manifest = read(root / "data/prepared/manifest.json")
    if manifest.get("conversations_hash") != digest(conversations) or manifest.get("questions_hash") != digest(questions) or manifest.get("config_hash") != digest(cfg):
        issues.append({"path": "data/prepared/manifest.json", "message": "Prepared manifest content/config mismatch"})
    if (manifest.get("conversations") != len(conversations) or manifest.get("questions") != len(questions) or
            manifest.get("chunks") != sum(len(c["chunks"]) for c in conversations)):
        issues.append({"path": "data/prepared/manifest.json", "message": "Prepared manifest counts mismatch"})
    if (len(conversations) != 10 or len({c["sample_id"] for c in conversations}) != 10 or
            len(questions) != 1986 or dict(Counter(q["category"] for q in questions)) != CATEGORY_COUNTS or
            len({(q["sample_id"], q["qa_index"]) for q in questions}) != 1986):
        issues.append({"path": "data/prepared", "message": "Expected ten unique conversations and all 1986 unique categorized questions"})
    # The frozen protocol reconstructs only prompt/option metadata, never reads
    # outputs or invokes any scoring/model function.
    spec = importlib.util.spec_from_file_location("frozen_locomo_for_integrity", root / "locomo.py")
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    original = protocol.load_dataset(source_path)
    if protocol.prepare_questions(original) != questions:
        issues.append({"path": "data/prepared/questions.json", "message": "Shared question/option manifest differs from pinned source and seed"})
    original_by_id = {sample["sample_id"]: sample for sample in original}
    for conversation in conversations:
        sample = conversation["sample_id"]
        source_turns = protocol.conversation_turns(original_by_id[sample])
        expected_turns = {turn["source_id"]: turn for turn in source_turns}
        restored = {}
        order = []
        for index, chunk in enumerate(conversation["chunks"]):
            if chunk["chunk_index"] != index or not 0 <= chunk["token_count"] <= 2048:
                issues.append({"path": "data/prepared/conversations.json", "message": sample + " chunk index/token budget mismatch"})
            for fragment in chunk["segments"]:
                source = fragment["source_id"]
                previous = restored.get(source, "")
                original_turn = expected_turns.get(source)
                if (original_turn is None or fragment["char_start"] != len(previous) or
                        fragment["char_end"] != len(previous) + len(fragment["text"]) or
                        any(fragment[key] != original_turn[key] for key in ("speaker", "timestamp", "session"))):
                    issues.append({"path": "data/prepared/conversations.json", "message": sample + " source-fragment provenance mismatch"})
                if source not in restored:
                    order.append(source)
                restored[source] = previous + fragment["text"]
        if restored != {turn["source_id"]: turn["text"] for turn in source_turns} or order != [turn["source_id"] for turn in source_turns]:
            issues.append({"path": "data/prepared/conversations.json", "message": sample + " missing/duplicated/reordered conversation content"})
    result = verify_receipts(base, cfg, sources, conversations, questions)
    issues.extend(result.pop("issues"))
    report = _verify_report(base, questions, result["completed_categories"])
    issues.extend(report.pop("issues"))
    return {"passed": not issues, "checked_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": str(root), "run": str(base), "issues": issues,
            "verifier_sha256": file_hash(__file__),
            "dataset_sha256": actual_dataset_hash, "source_hash": digest(sources), "config_hash": digest(cfg),
            "shared_question_option_manifest_sha256": file_hash(root / "data/prepared/questions.json"),
            "expected_categories_per_arm": CATEGORY_COUNTS,
            "expected_nonadversarial_per_arm": 1540, "expected_clean_adversarial_per_arm": 442,
            "report": report, "prediction_scoring_performed": False, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    parser.add_argument("--output", type=Path, help="Write a separate integrity report; otherwise print only")
    args = parser.parse_args()
    try:
        result = verify(args.experiment)
    except Exception as exc:
        result = {"passed": False, "issues": [{"path": str(args.experiment), "message": str(exc)}],
                  "prediction_scoring_performed": False}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    summary = {key: value for key, value in result.items() if key != "issues"}
    summary["issue_count"] = len(result["issues"])
    summary["first_issues"] = result["issues"][:20]
    print(json.dumps(summary, sort_keys=True, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
