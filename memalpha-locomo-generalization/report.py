"""Official-compatible scoring, paired comparisons, and complete denominators."""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from common import ARMS, ROOT, atomic_json, config, digest, file_hash, now, read_json, run_root, source_manifest, verify_freeze
from locomo import CATEGORY_NAMES, score_prediction


def mean(values):
    return sum(values) / len(values) if values else None


def number(value, decimals=0):
    return "—" if value is None else f"{value:,.{decimals}f}"


def memory_summary(memory_metrics, conversations):
    measured = [row for row in memory_metrics if row["memory_tokens"] is not None]
    stored = sum(row["memory_tokens"] for row in measured) if measured else None
    matched_input = sum(row["input_tokens"] for row in measured) if measured else None
    return {"scheduled_conversations": len(conversations), "frozen_conversations": len(memory_metrics),
            "measured_conversations": len(measured),
            "scheduled_input_tokens": sum(row["input_tokens"] for row in conversations),
            "matched_input_tokens": matched_input, "stored_tokens_total": stored,
            "stored_tokens_mean": mean([row["memory_tokens"] for row in measured]),
            "token_reduction_fraction": 1 - stored / matched_input if matched_input else None,
            "store_tokens_total": {kind: sum(row["store_tokens"][kind] for row in measured)
                                   for kind in ("core", "semantic", "episodic")} if measured else {}}


def phase_allocation(phase, status, status_path):
    """Stage status covers its latest invocation, not all historical leases."""
    elapsed, count = status.get("elapsed_seconds"), status.get("gpu_count")
    allocated = status.get("gpu_allocated_seconds")
    if allocated is None and elapsed is not None and count is not None:
        allocated = elapsed * count
    return {"phase": phase, "status": status.get("status", "unavailable"),
            "status_path": status_path, "elapsed_seconds": elapsed, "gpu_count": count,
            "allocated_gpu_seconds": allocated,
            "allocated_gpu_hours": allocated / 3600 if allocated is not None else None}


def exact_choice_diagnostic(raw, question):
    """Interpret an explicit choice or unique exact option text, never paraphrases.

    This diagnostic deliberately does not use or change LoCoMo's legacy parser.
    Labels accompanied by text must agree with that exact offered option. Case,
    whitespace, and terminal sentence punctuation are insignificant.
    """
    def normalize(text):
        return " ".join(text.casefold().split()).rstrip(".!?").rstrip()

    options = {key: normalize(value) for key, value in question["option_map"].items()}
    text = normalize(raw)
    exact = {key for key, value in options.items() if text == value}
    # Accept (a), a, a., a), a: and analogous b labels, optionally followed
    # by the option's text. Arbitrary explanatory prose does not count.
    labelled = re.fullmatch(r"\s*(?:\(([ab])\)|([ab])(?:[.):]|(?=\s|$)))\s*(.*?)\s*",
                            raw, flags=re.IGNORECASE | re.DOTALL)
    selected, interpretation = None, "unresolved"
    if labelled:
        label = (labelled.group(1) or labelled.group(2)).lower()
        remainder = normalize(labelled.group(3))
        # If the whole response also names a different option literally, the
        # label/text interpretations conflict (e.g. an option itself is "a").
        if exact and exact != {label}:
            pass
        elif not remainder:
            selected, interpretation = label, "explicit_label"
        elif remainder == options[label]:
            selected, interpretation = label, "label_and_matching_option_text"
    elif len(exact) == 1:
        selected, interpretation = next(iter(exact)), "unique_exact_option_text"
    return {"strict_option": selected,
            "strict_option_correct": selected == question["abstention_option"],
            "strict_option_interpretation": interpretation}


def aggregate(rows):
    by_category = {}
    for category, name in CATEGORY_NAMES.items():
        subset = [r for r in rows if r["category"] == category]
        by_category[name] = {"count": len(subset), "completed": sum(r["status"] == "complete" for r in subset),
                             "score": mean([r["score"] for r in subset])}
    clean = [r for r in rows if r["category"] == 5 and r.get("adversarial_clean_eligible")]
    adversarial = [r for r in rows if r["category"] == 5]
    return {"count": len(rows), "completed": sum(r["status"] == "complete" for r in rows),
            "overall_micro": mean([r["score"] for r in rows]),
            "category_macro": mean([v["score"] for v in by_category.values() if v["score"] is not None]),
            "nonadversarial_micro": mean([r["score"] for r in rows if r["category"] != 5]),
            "adversarial_strict": mean([float(r.get("strict_option_correct") or False) for r in adversarial]),
            "adversarial_clean_strict": mean([float(r.get("strict_option_correct") or False) for r in clean]),
            "adversarial_clean_count": len(clean), "categories": by_category}


def generate_report(synthetic=False):
    from run import inputs
    if not synthetic:
        verify_freeze()
    cfg = config()
    cfg_hash = digest(cfg)
    source_hash = digest(source_manifest())
    conversations, questions = inputs(synthetic)
    base = run_root(synthetic)
    target = base / "report"
    target.mkdir(parents=True, exist_ok=True)
    rows, metrics, operational = [], {}, {}
    tokenizer = None
    if (ROOT / "models" / "reader" / "tokenizer.json").exists():
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(ROOT / "models" / "reader"), local_files_only=True)
    for arm in ARMS:
        arm_rows = []
        usage = Counter()
        memory_metrics = []
        frozen_by_sample = {}
        for conv in conversations:
            directory = base / arm / "conversations" / conv["sample_id"]
            frozen_path = directory / "memory_frozen.json"
            if frozen_path.exists():
                saved = read_json(frozen_path)
                expected_memory = {"arm": arm, "sample_id": conv["sample_id"], "frozen": True,
                                   "config_hash": cfg_hash, "source_hash": source_hash}
                if any(saved.get(key) != value for key, value in expected_memory.items()):
                    raise RuntimeError("Frozen memory provenance mismatch in report: " + str(frozen_path))
                if digest(saved["memory"]) != saved["memory_hash"]:
                    raise RuntimeError("Corrupted frozen memory in report")
                frozen_by_sample[conv["sample_id"]] = saved
                contents = {"core": [saved["memory"]["core"]],
                            **{kind: [next(iter(item.values())) for item in saved["memory"][kind]] for kind in ("semantic", "episodic")}}
                counts = {kind: sum(len(tokenizer.encode(text)) for text in values) for kind, values in contents.items()} if tokenizer else {}
                size = sum(counts.values()) if counts else None
                memory_metrics.append({"sample_id": conv["sample_id"], "store_tokens": counts,
                                       "memory_tokens": size, "input_tokens": conv["input_tokens"],
                                       "compression_fraction": 1 - size / conv["input_tokens"] if size is not None and conv["input_tokens"] else None,
                                       "semantic_entries": len(contents["semantic"]), "episodic_entries": len(contents["episodic"])})
            for path in sorted((directory / "chunks").glob("*.json")):
                result = read_json(path)["result"]
                usage["writer_calls"] += 1
                usage["tool_calls"] += len(result.get("tool_calls", []))
                usage["tool_errors"] += len(result.get("errors", []))
                usage["writer_truncated"] += result.get("finish_reason") == "length"
                for key in ("prompt_tokens", "completion_tokens"):
                    usage["writer_" + key] += result.get("usage", {}).get(key, 0)
                u = result.get("usage", {})
                usage["writer_generation_wall_seconds"] += u.get("batch_wall_seconds", 0) / max(1, u.get("batch_size", 1))
                if u.get("batch_wall_seconds") is not None:
                    usage["writer_latency_observed_calls"] += 1
                    usage["writer_observed_batch_latency_seconds"] += u["batch_wall_seconds"]
        for q in questions:
            path = base / arm / "answers" / q["sample_id"] / f'{q["qa_index"]:05d}.json'
            saved = read_json(path) if path.exists() else None
            receipts = [(path, saved)] if saved is not None else []
            receipts += [(attempt_path, read_json(attempt_path)) for attempt_path in
                         sorted((path.parent / "attempts").glob(f'{q["qa_index"]:05d}_*.json'))]
            attempts = {}
            if receipts:
                frozen = frozen_by_sample.get(q["sample_id"])
                if frozen is None:
                    raise RuntimeError("Reader receipt has no corresponding frozen memory: " + str(path))
                expected_answer = {"arm": arm, "sample_id": q["sample_id"], "qa_index": q["qa_index"],
                                   "question_hash": digest(q), "memory_hash": frozen["memory_hash"],
                                   "config_hash": cfg_hash, "source_hash": source_hash}
                for receipt_path, receipt in receipts:
                    if any(receipt.get(key) != value for key, value in expected_answer.items()):
                        raise RuntimeError("Reader receipt provenance mismatch in report: " + str(receipt_path))
                    attempt = receipt.get("attempt", 1)
                    if not isinstance(attempt, int) or attempt < 1:
                        raise RuntimeError("Invalid reader attempt number: " + str(receipt_path))
                    if attempt in attempts and digest(receipt) != digest(attempts[attempt]):
                        raise RuntimeError("Conflicting duplicate reader attempt: " + str(receipt_path))
                    # A retry is archived before generation; interruption can
                    # leave the same attempt in both archive and latest paths.
                    attempts[attempt] = receipt
            complete = saved is not None and saved["status"] == "complete"
            raw = saved["result"]["raw_response"] if complete else ""
            scored = score_prediction(raw, q)
            if q["category"] == 5:
                scored.update(exact_choice_diagnostic(raw, q))
            if not complete:
                scored["score"] = 0.0
                if q["category"] == 5:
                    scored["strict_option_correct"] = False
            row = {"arm": arm, "sample_id": q["sample_id"], "qa_index": q["qa_index"], "category": q["category"],
                   "category_name": CATEGORY_NAMES[q["category"]], "question": q["question"],
                   "raw_response": raw, "status": saved["status"] if saved else "missing", **scored}
            arm_rows.append(row)
            usage["reader_retry_calls"] += max(0, len(attempts) - 1)
            if attempts and set(attempts) != set(range(1, max(attempts) + 1)):
                usage["reader_usage_incomplete_questions"] += 1
            for receipt in attempts.values():
                result = receipt["result"]
                usage["reader_calls"] += 1
                usage["reader_truncated"] += result.get("finish_reason") == "length"
                for key in ("prompt_tokens", "completion_tokens"):
                    usage["reader_" + key] += result.get("usage", {}).get(key, 0)
                u = result.get("usage", {})
                usage["reader_generation_wall_seconds"] += u.get("batch_wall_seconds", 0) / max(1, u.get("batch_size", 1))
                if u.get("batch_wall_seconds") is not None:
                    usage["reader_latency_observed_calls"] += 1
                    usage["reader_observed_batch_latency_seconds"] += u["batch_wall_seconds"]
        rows.extend(arm_rows)
        metrics[arm] = aggregate(arm_rows)
        metrics[arm]["conversations"] = {c["sample_id"]: aggregate([r for r in arm_rows if r["sample_id"] == c["sample_id"]]) for c in conversations}
        latencies = {}
        for phase in ("writer", "reader"):
            observed = usage.get(phase + "_latency_observed_calls", 0)
            calls = usage.get(phase + "_calls", 0)
            latencies[phase] = {"calls": calls, "observed_calls": observed,
                                "mean_batch_generation_seconds": usage[phase + "_observed_batch_latency_seconds"] / observed if observed else None,
                                "allocated_batch_generation_seconds": usage[phase + "_generation_wall_seconds"] if observed or not calls else None}
        operational[arm] = {"usage": dict(usage), "memory": memory_metrics,
                            "memory_summary": memory_summary(memory_metrics, conversations),
                            "latency": latencies,
                            "token_totals": {"input_tokens": usage["writer_prompt_tokens"] + usage["reader_prompt_tokens"],
                                             "output_tokens": usage["writer_completion_tokens"] + usage["reader_completion_tokens"]}}
        status_path = base / arm / "build_status.json"
        if status_path.exists(): operational[arm]["build_status"] = read_json(status_path)
    paired = {}
    for metric in ("overall_micro", "category_macro", "nonadversarial_micro", "adversarial_strict", "adversarial_clean_strict"):
        a, b = metrics[ARMS[0]][metric], metrics[ARMS[1]][metric]
        paired[metric] = a - b if a is not None and b is not None else None
    paired["categories"] = {name: metrics[ARMS[0]]["categories"][name]["score"] - metrics[ARMS[1]]["categories"][name]["score"] for name in CATEGORY_NAMES.values()}
    paired["conversations"] = {c["sample_id"]: metrics[ARMS[0]]["conversations"][c["sample_id"]]["overall_micro"] - metrics[ARMS[1]]["conversations"][c["sample_id"]]["overall_micro"] for c in conversations}
    result = {"created_utc": now(), "run_id": cfg["run_id"], "synthetic": synthetic,
              "complete": all(v["completed"] == len(questions) for v in metrics.values()),
              "metrics": metrics, "paired_memalpha_minus_base": paired, "operational": operational,
              "protocol": "Official LoCoMo category-specific task prompts and compatible scoring applied to Mem-alpha structured memory",
              "diagnostic_definitions": {
                  "adversarial_strict": "Exact unambiguous choice accuracy: explicit option label, label plus matching exact option text, or unique exact option text; case, whitespace and terminal sentence punctuation normalized. Conflicts, unresolved responses and missing/error outputs score zero.",
                  "adversarial_clean_strict": "The same exact unambiguous choice accuracy, excluding the four flagged annotation anomalies."},
              "no_training": True, "reader_seed": 42, "option_seed": 42}
    if (base / "answer_status.json").exists(): result["answer_status"] = read_json(base / "answer_status.json")
    result["phase_allocations"] = [phase_allocation("writer_" + arm, operational[arm].get("build_status", {}),
                                    str((base / arm / "build_status.json").relative_to(ROOT))) for arm in ARMS]
    result["phase_allocations"].append(phase_allocation("reader_shared", result.get("answer_status", {}),
                                            str((base / "answer_status.json").relative_to(ROOT))))
    result["allocation_definition"] = "Latest recorded phase invocation elapsed wall time multiplied by visible GPU count, including model loading. Not measured GPU compute time or cumulative lease time across restarts; the reader row is shared by both arms."
    result["token_totals"] = {key: sum(operational[arm]["token_totals"][key] for arm in ARMS)
                              for key in ("input_tokens", "output_tokens")}
    result["report_provenance"] = {"source_path": "report.py", "source_sha256": file_hash(Path(__file__).resolve()),
                                    "outside_scientific_source_freeze": True, "lease_receipts_path": "results/operations"}
    runtime_path = ROOT / "results/setup/reader_runtime.json"
    if runtime_path.exists():
        result["report_provenance"]["reader_runtime"] = {"path": str(runtime_path.relative_to(ROOT)),
            "sha256": file_hash(runtime_path), "environment": read_json(runtime_path).get("environment", {})}
    atomic_json(target / "metrics.json", result)
    atomic_json(target / "per_question.json", rows)
    columns = ["arm", "sample_id", "qa_index", "category", "category_name", "status", "score", "strict_option_correct", "strict_option_interpretation", "adversarial_clean_eligible", "anomaly", "question", "prediction", "raw_response"]
    with (target / "per_question.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = [{"arm": arm, "category": name, **values} for arm in ARMS for name, values in metrics[arm]["categories"].items()]
    with (target / "per_category.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["arm", "category", "count", "completed", "score"])
        writer.writeheader(); writer.writerows(summary_rows)
    conversation_rows = []
    for conversation in conversations:
        sample_id = conversation["sample_id"]
        a, b = (metrics[arm]["conversations"][sample_id] for arm in ARMS)
        conversation_rows.append({"sample_id": sample_id, "questions_per_arm": a["count"],
                                  "memalpha_completed": a["completed"], "base_completed": b["completed"],
                                  "memalpha_score": a["overall_micro"], "base_score": b["overall_micro"],
                                  "difference": paired["conversations"][sample_id]})
    if conversation_rows:
        with (target / "per_conversation.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(conversation_rows[0]))
            writer.writeheader(); writer.writerows(conversation_rows)
    anomalies = [r for r in rows if r.get("anomaly")]
    atomic_json(target / "annotation_anomalies.json", anomalies)
    lines = ["# Mem-α → LoCoMo inference-only evaluation", "", result["protocol"] + ".", "",
             f"Coverage: {sum(m['completed'] for m in metrics.values())}/{2 * len(questions)} completed answers; "
             f"{len(list(base.glob('*/conversations/*/memory_frozen.json')))}/{2 * len(conversations)} frozen memories.", "",
             "| Category | Questions per arm | Mem-α | Base Qwen3-4B | Difference |", "|---|---:|---:|---:|---:|"]
    for category, name in CATEGORY_NAMES.items():
        a, b = metrics[ARMS[0]]["categories"][name], metrics[ARMS[1]]["categories"][name]
        lines.append(f"| {name} | {a['count']} | {a['score']:.4f} | {b['score']:.4f} | {a['score'] - b['score']:+.4f} |")
    for label, metric in [("Overall micro", "overall_micro"), ("Category macro", "category_macro"), ("Nonadversarial micro", "nonadversarial_micro"), ("Exact unambiguous adversarial choice accuracy", "adversarial_strict"), ("Exact unambiguous choice accuracy excluding four anomalies", "adversarial_clean_strict")]:
        a, b = metrics[ARMS[0]][metric], metrics[ARMS[1]][metric]
        if a is not None and b is not None: lines.append(f"| {label} | — | {a:.4f} | {b:.4f} | {a-b:+.4f} |")
    lines += ["", "## Paired conversation scores", "",
              "Each row uses every scheduled question in that conversation, including missing/error outputs at zero.", "",
              "| Conversation | Questions per arm | Completed Mem-α / base | Mem-α | Base Qwen3-4B | Difference |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in conversation_rows:
        lines.append(f"| {row['sample_id']} | {row['questions_per_arm']} | {row['memalpha_completed']} / {row['base_completed']} | {row['memalpha_score']:.4f} | {row['base_score']:.4f} | {row['difference']:+.4f} |")
    lines += ["", "## Interpretation and protocol", "",
              "The comparison changes only the frozen memory-writing checkpoint. Both arms use identical source chunks, memory tools, BM25 retrieval, category-specific task prompts, adversarial option order, and frozen Qwen3-32B reader settings. No parameters are trained or selected on LoCoMo.", "",
              "Scores use the official category-specific token-F1 rules and legacy adversarial parser/scorer. Adversarial scores are binary, so the overall number is a mixed task score, not a uniform token-F1 metric. Missing/error outputs score zero and remain in the denominator.", "",
              "The additional exact unambiguous choice diagnostic accepts an explicit a/b label, a label followed by its matching exact option text, or a response equal to one distinct option text. It normalizes case, whitespace and terminal sentence punctuation. Conflicting label/text and unresolved or missing responses score zero. This reporting-only diagnostic does not alter the official headline score or legacy prediction mapping.", "",
              "The reader uses Mem-α’s structured-memory system prompt, temperature 0.7, 2,048 output tokens and seed 42. LoCoMo’s original reader generation used different settings. This is a memory-transfer evaluation with official task prompts, not an exact end-to-end reproduction of the original LoCoMo models. One seed per arm does not quantify run-to-run sampling variability.", "",
              "The adversarial distractor comes from adversarial_answer, fixing the stale missing-answer interface. conv-26 questions 167/178 contain answer=No and distractor=Yes; conv-30 questions 79/103 have a distractor reading Not mentioned. All four stay in the headline score; the additional clean strict diagnostic excludes them. Question indices are zero-based. Invalid short responses accepted by the legacy parser are flagged in per_question.json.", "",
              "Memory construction uses only dialogue text, dates, speakers and supplied image captions. Dialogue IDs remain in provenance sidecars. No gold evidence, questions, supplied summaries or observations enter construction. QA cannot update frozen memory. Retrieval evidence recall is not reported because generated memory entries do not provide reliable exact source attribution.", "", "## Memory size and compression", "",
              "| Arm | Measured / frozen / scheduled memories | Matched source tokens | Stored tokens total | Mean stored tokens | Token reduction |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        m = operational[arm]["memory_summary"]
        reduction = f"{m['token_reduction_fraction']:.1%}" if m["token_reduction_fraction"] is not None else "—"
        lines.append(f"| {arm} | {m['measured_conversations']} / {m['frozen_conversations']} / {m['scheduled_conversations']} | {number(m['matched_input_tokens'])} | {number(m['stored_tokens_total'])} | {number(m['stored_tokens_mean'], 1)} | {reduction} |")
    lines += ["", "Memory sizes count stored core, semantic and episodic content with the reader tokenizer, excluding entry IDs and prompt wrappers. Reduction is one minus total stored tokens divided by source-chunk tokens for the same measured conversations. Missing measurements remain unavailable.", "",
              "## Model tokens and generation latency", "",
              "| Arm | Phase | Calls / retries | Model input tokens | Model output tokens | Allocated batch generation (s) | Mean batch latency (s) | Timed calls |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        u = operational[arm]["usage"]
        for phase in ("writer", "reader"):
            latency = operational[arm]["latency"][phase]
            retries = str(u.get("reader_retry_calls", 0)) if phase == "reader" else "—"
            lines.append(f"| {arm} | {phase} | {latency['calls']} / {retries} | {number(u.get(phase + '_prompt_tokens', 0))} | {number(u.get(phase + '_completion_tokens', 0))} | {number(latency['allocated_batch_generation_seconds'], 2)} | {number(latency['mean_batch_generation_seconds'], 2)} | {latency['observed_calls']} / {latency['calls']} |")
        total = operational[arm]["token_totals"]
        lines.append(f"| {arm} | **Total** | — | **{number(total['input_tokens'])}** | **{number(total['output_tokens'])}** | — | — | — |")
    lines.append(f"| **Both arms** | **Total** | — | **{number(result['token_totals']['input_tokens'])}** | **{number(result['token_totals']['output_tokens'])}** | — | — | — |")
    lines += ["", "Model input totals include repeated context and instructions. Reader usage includes unique archived retries plus the latest attempt; only the latest receipt is scored. Mean batch latency is the observed generation-call wall time for each timed request, excluding model loading, prompt preparation and tool execution. Allocated batch generation divides each batch's wall time evenly among its requests; it is not individual request latency. Gaps in archived attempts are flagged as reader_usage_incomplete_questions.", "",
              "## Latest phase GPU allocation", "",
              "| Phase | Status | Elapsed wall time (s) | Visible GPUs | Allocated GPU seconds | Allocated GPU hours |",
              "|---|---|---:|---:|---:|---:|"]
    for phase in result["phase_allocations"]:
        lines.append(f"| {phase['phase']} | {phase['status']} | {number(phase['elapsed_seconds'], 2)} | {number(phase['gpu_count'])} | {number(phase['allocated_gpu_seconds'], 2)} | {number(phase['allocated_gpu_hours'], 4)} |")
    lines += ["", result["allocation_definition"] + " Complete lease and hardware receipts are under [results/operations](../../../results/operations/).", "",
              "Per-store memory sizes, tool-error counts, truncation counts, detailed usage and source status paths are retained in metrics.json.", "",
              "## Artifacts and reproducibility", "",
              f"Reporting code SHA-256: `{result['report_provenance']['source_sha256']}`. This identifies report.py separately from the immutable scientific source freeze.", "",
              "- metrics.json: aggregates, paired differences, per-conversation results, and operational measurements.",
              "- per_question.csv / per_question.json: all scheduled questions including failures.", "- per_category.csv: comparison table.",
              "- per_conversation.csv: paired scores and completion coverage for every conversation.",
              "- annotation_anomalies.json: explicitly flagged annotation cases.", "- ../freeze.json: frozen source, configuration and prepared-input hashes.",
              "- ../<arm>/conversations: raw writer prompts/completions, tool traces, atomic checkpoints, and frozen memory.",
              "- ../<arm>/answers: raw reader prompts/completions and retrieval traces.", "", "See the experiment README.md for restart commands.", ""]
    if "reader_runtime" in result["report_provenance"]:
        runtime = result["report_provenance"]["reader_runtime"]
        lines += ["Reader communication/runtime settings: [results/setup/reader_runtime.json](../../../results/setup/reader_runtime.json), "
                  f"SHA-256 `{runtime['sha256']}`. The saved environment is included in metrics.json provenance.", ""]
    (target / "report.md").write_text("\n".join(lines))
    print(json.dumps({"report": str(target / "report.md"), "complete": result["complete"], "metrics": {arm: {k: v for k, v in metrics[arm].items() if k != "conversations"} for arm in ARMS}}, indent=2), flush=True)
    return result
