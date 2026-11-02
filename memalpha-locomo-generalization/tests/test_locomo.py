"""Meaningful protocol tests against the pinned official implementation."""
import ast
import contextlib
import hashlib
import io
import json
import random
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import regex
from nltk.stem import PorterStemmer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import locomo


class CharacterTokenizer:
    """Deliberately harsh tokenizer exercises all overflow branches."""
    def encode(self, text):
        return list(text)


@pytest.fixture(scope="module")
def data():
    return locomo.load_dataset(ROOT / "data/locomo10.json")


@pytest.fixture(scope="module")
def questions(data):
    return locomo.prepare_questions(data)


@pytest.fixture(scope="module")
def official():
    tree = ast.parse((ROOT / "vendor/locomo_evaluation.py").read_text())
    names = {"normalize_answer", "f1_score", "f1", "eval_question_answering"}
    module = ast.Module(body=[node for node in tree.body
                             if isinstance(node, ast.FunctionDef) and node.name in names], type_ignores=[])
    namespace = {"np": np, "regex": regex, "string": string,
                 "Counter": Counter, "ps": PorterStemmer()}
    exec(compile(module, "<pinned official scorer>", "exec"), namespace)
    return namespace


def test_dataset_counts_checksums_and_corruption(data, tmp_path):
    assert len(data) == 10
    assert Counter(qa["category"] for sample in data for qa in sample["qa"]) == locomo.CATEGORY_COUNTS
    assert sum(len(locomo.conversation_turns(sample)) for sample in data) == 5882
    provenance = json.loads((ROOT / "vendor/locomo_sources.json").read_text())
    for item in provenance["files"]:
        assert hashlib.sha256((ROOT / item["path"]).read_bytes()).hexdigest() == item["sha256"]
    corrupt = tmp_path / "locomo.json"
    corrupt.write_bytes((ROOT / "data/locomo10.json").read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum"):
        locomo.load_dataset(corrupt)


def test_all_categories_match_official_scorer(questions, official):
    reference_rows, actual = [], []
    for qa in questions:
        raw_candidates = [str(qa.get("answer", "")), "", "the, Paris and Berlin", "No information available"]
        if qa["category"] == 5:
            raw_candidates += ["a", "b", "(a)", "(b)", "c", "yes", "a.", "(x)"]
        for raw in raw_candidates:
            result = locomo.score_prediction(raw, qa)
            prediction = locomo.official_get_cat_5_answer(raw, qa["option_map"]).strip() if qa["category"] == 5 else raw.strip()
            # Sole compatibility change: supply an unused answer slot so the
            # legacy evaluator can reach its category-5 branch on current data.
            reference_rows.append({**qa, "answer": qa.get("answer", ""), "prediction": prediction})
            actual.append(result["score"])
    with contextlib.redirect_stdout(io.StringIO()):
        expected = official["eval_question_answering"](reference_rows)[0]
    assert actual == pytest.approx(expected, abs=1e-15)


def test_official_prompt_constants_date_suffix_and_shared_options(data, questions):
    source = ast.parse((ROOT / "vendor/locomo_gpt_utils.py").read_text())
    constants = {}
    for node in source.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    random.seed(799)
    state = random.getstate()
    assert locomo.prepare_questions(data) == questions
    assert random.getstate() == state
    rng = random.Random(42)
    for row in questions:
        question = row["question"]
        if row["category"] == 2:
            question += " Use DATE of CONVERSATION to answer with an approximate date."
        if row["category"] == 5:
            abstention_option = "a" if rng.random() < .5 else "b"
            assert row["abstention_option"] == abstention_option
            assert row["option_map"][abstention_option] == locomo.ABSTENTION
            assert row["option_map"]["b" if abstention_option == "a" else "a"] == row["adversarial_answer"]
            question += " Select the correct answer: (a) {} (b) {}. ".format(row["option_map"]["a"], row["option_map"]["b"])
        prompt = constants["QA_PROMPT_CAT_5" if row["category"] == 5 else "QA_PROMPT"]
        assert row["reader_prompt"] == prompt.format(question)
        assert 'json dictionary' not in row["reader_prompt"]
    assert Counter(q["abstention_option"] for q in questions if q["category"] == 5) == {"a": 220, "b": 226}
    assert questions[152]["sample_id"] == "conv-26" and questions[152]["abstention_option"] == "b"


def test_adversarial_missing_answer_and_four_annotation_anomalies(questions):
    adversarial = [q for q in questions if q["category"] == 5]
    assert sum("answer" not in q for q in adversarial) == 444
    anomalies = {(q["sample_id"], q["qa_index"]): q for q in adversarial if q["anomaly"]}
    assert set(anomalies) == {("conv-26", 167), ("conv-26", 178), ("conv-30", 79), ("conv-30", 103)}
    for qa in adversarial:
        correct = qa["abstention_option"]
        wrong = "b" if correct == "a" else "a"
        good, bad = locomo.score_prediction(correct, qa), locomo.score_prediction(wrong, qa)
        assert good["score"] == 1 and good["strict_option_correct"] is True
        assert bad["strict_option_correct"] is False
        if qa["adversarial_answer"] == "Not mentioned":
            assert bad["score"] == 1  # Preserve and expose the upstream metric defect.
            assert bad["anomaly"] and not bad["adversarial_clean_eligible"]
        else:
            assert bad["score"] == 0
        if "answer" in qa:
            assert qa["answer"] == "No"
            assert qa["option_map"][wrong] == "Yes"
    assert sum(locomo.score_prediction("a", q)["adversarial_clean_eligible"] for q in adversarial) == 442


def test_legacy_option_parser_quirks_stay_visible(questions):
    qa = next(q for q in questions if q["category"] == 5)
    for raw in ("A", "(A)", "b", "(b)", "c", "yes", "(x)", "a.", "No information available"):
        result = locomo.score_prediction(raw, qa)
        assert result["prediction"] == locomo.official_get_cat_5_answer(raw, qa["option_map"]).strip()
        assert result["legacy_parser_accepted_invalid_option"] == (raw in ("c", "yes", "(x)"))
    assert locomo.score_prediction("yes", qa)["strict_option_correct"] is None


def _synthetic_sample():
    return {"sample_id": "synthetic", "conversation": {
        "speaker_a": "MiXeDName", "speaker_b": "Alex",
        "session_10_date_time": "10 September, 2026",
        "session_10": [{"dia_id": "D10:1", "speaker": "Alex", "text": "Final turn."}],
        "session_2_date_time": "2 September, 2026",
        "session_2": [{"dia_id": "D2:1", "speaker": "MiXeDName",
                       "text": "  First sentence.\n\nSecond sentence?  " + "A long word " * 400 + "世" * 5000,
                       "blip_caption": "a picture, even without an image URL"},
                      {"dia_id": "D2:2", "speaker": "Alex", "text": "", "blip_caption": ""}]},
        "qa": [{"question": "HIDDEN_QUESTION", "answer": "GOLD_SECRET", "evidence": ["HIDDEN_EVIDENCE"], "category": 4}],
        "event_summary": "EVENT_SECRET", "session_summary": "SUMMARY_SECRET", "observation": "OBSERVATION_SECRET"}


def test_lossless_chunking_unicode_whitespace_captions_order_and_qa_isolation():
    sample = _synthetic_sample()
    result = locomo.prepare_conversations([sample], CharacterTokenizer())[0]
    chunks = result["chunks"]
    assert all(0 < chunk["token_count"] <= 2048 for chunk in chunks)
    assert all(chunk["chunk_index"] == i for i, chunk in enumerate(chunks))
    assert result["input_tokens"] == sum(len(chunk["text"]) for chunk in chunks)
    assert result["num_turns"] == 3 and result["num_sessions"] == 2 and result["num_captions"] == 1
    all_fragments = [fragment for chunk in chunks for fragment in chunk["segments"]]
    assert all_fragments[0]["source_id"] == "D2:1" and all_fragments[-1]["source_id"] == "D10:1"
    expected = {row["source_id"]: row["text"] for row in locomo.conversation_turns(sample)}
    for source_id, text in expected.items():
        pieces = [fragment for fragment in all_fragments if fragment["source_id"] == source_id]
        assert "".join(piece["text"] for piece in pieces) == text
        assert pieces[0]["char_start"] == 0 and pieces[-1]["char_end"] == len(text)
        for left, right in zip(pieces, pieces[1:]):
            assert left["char_end"] == right["char_start"]
        assert all(piece["char_end"] - piece["char_start"] == len(piece["text"]) for piece in pieces)
    assert all(len(f'<{fragment["speaker"]}>{fragment["text"]}') <= 1898 for fragment in all_fragments)
    rendered = json.dumps(result, ensure_ascii=False)
    for secret in ("HIDDEN_QUESTION", "GOLD_SECRET", "HIDDEN_EVIDENCE", "EVENT_SECRET", "SUMMARY_SECRET", "OBSERVATION_SECRET"):
        assert secret not in rendered
    assert "<MiXeDName>" in rendered and "<Mixedname>" not in rendered


def test_adjacent_short_sessions_pack_with_headers_and_threshold():
    conversation = {}
    for i, size in enumerate((1000, 600, 200), 1):
        conversation[f"session_{i}_date_time"] = f"day {i}"
        conversation[f"session_{i}"] = [{"dia_id": f"D{i}:1", "speaker": "A", "text": "x" * size}]
    result = locomo.prepare_conversations([{"sample_id": "packing", "conversation": conversation}], CharacterTokenizer())[0]
    assert len(result["chunks"]) == 1
    chunk = result["chunks"][0]
    assert 1800 <= chunk["token_count"] <= 2048
    assert chunk["text"].count("[Dialogue at timestamp") == 3
    assert chunk["source_ids"] == ["D1:1", "D2:1", "D3:1"]


def test_all_real_conversations_have_complete_source_coverage(data):
    # Character tokens force more splits than Qwen, exercising coverage broadly.
    result = locomo.prepare_conversations(data, CharacterTokenizer())
    assert len(result) == 10
    assert sum(item["num_turns"] for item in result) == 5882
    assert sum(item["num_sessions"] for item in result) == 272
    assert sum(item["num_captions"] for item in result) == 1226
    assert all(chunk["token_count"] <= 2048 for item in result for chunk in item["chunks"])
