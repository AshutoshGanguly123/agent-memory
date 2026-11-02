"""Pinned LoCoMo QA protocol and lossless Mem-alpha inference inputs.

Construction receives only conversation turns. Questions (including the official
adversarial distractor options) are prepared independently, after construction.
The upstream single-question task templates and legacy option parser are loaded
from checksum-verified vendor source without importing its API clients.
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import string
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
GPTY_UTILS_SHA256 = "5fc977375878199735acd28fba5ae6f4d657fa0e000c0d2918a90c07b6035793"
CATEGORY_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}
CATEGORY_COUNTS = {1: 282, 2: 321, 3: 96, 4: 841, 5: 446}
OPTION_SEED = 42
ABSTENTION = "Not mentioned in the conversation"
MAX_CHUNK_TOKENS = 2048
MAX_TURN_TOKENS = 1898
PACK_FLUSH_TOKENS = 1800


def _upstream_prompts():
    source = (ROOT / "vendor/locomo_gpt_utils.py").read_bytes()
    if hashlib.sha256(source).hexdigest() != GPTY_UTILS_SHA256:
        raise ValueError("Vendored LoCoMo prompt source checksum mismatch")
    tree = ast.parse(source)
    names = {"QA_PROMPT", "QA_PROMPT_CAT_5", "QA_PROMPT_BATCH"}
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    constants[target.id] = ast.literal_eval(node.value)
    parser = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "get_cat_5_answer")
    namespace = {}
    exec(compile(ast.Module(body=[parser], type_ignores=[]), "<pinned LoCoMo option parser>", "exec"), namespace)
    return constants, namespace["get_cat_5_answer"]


_PROMPTS, official_get_cat_5_answer = _upstream_prompts()
QA_PROMPT = _PROMPTS["QA_PROMPT"]
QA_PROMPT_CAT_5 = _PROMPTS["QA_PROMPT_CAT_5"]


def load_dataset(path):
    """Require the complete, byte-identical pinned public 10-conversation release."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != DATASET_SHA256:
        raise ValueError("LoCoMo dataset checksum mismatch")
    data = json.loads(raw)
    counts = Counter(qa["category"] for sample in data for qa in sample["qa"])
    if len(data) != 10 or dict(counts) != CATEGORY_COUNTS:
        raise ValueError("Expected 10 conversations / 1,986 QA across all five categories")
    if len({sample["sample_id"] for sample in data}) != 10:
        raise ValueError("Duplicate conversation identifiers")
    if sum(len(conversation_turns(sample)) for sample in data) != 5882:
        raise ValueError("Expected all 5,882 conversation turns")
    return data


def _session_keys(conversation):
    return sorted((key for key in conversation if re.fullmatch(r"session_\d+", key)),
                  key=lambda key: int(key.split("_")[1]))


def conversation_turns(sample):
    """Explicit allowlist: no QA, evidence, observations, or supplied summaries."""
    conversation = sample["conversation"]
    rows = []
    for session in _session_keys(conversation):
        timestamp = conversation[session + "_date_time"]
        for turn in conversation[session]:
            text = turn["text"]
            caption = turn.get("blip_caption")
            if caption:
                text += "\n[Image caption: " + caption + "]"
            rows.append({"source_id": turn["dia_id"], "session": session,
                         "timestamp": timestamp, "speaker": turn["speaker"],
                         "text": text, "has_caption": bool(caption)})
    if len({row["source_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate turn identifiers in " + sample["sample_id"])
    return rows


def _count(tokenizer, text):
    # Match process_data.py's count_tokens_qwen; Qwen3 adds no default BOS/EOS.
    return len(tokenizer.encode(text))


def _render_fragment(speaker, text):
    # Keep original named speakers rather than assigning user/assistant roles.
    return f"<{speaker}>{text}"


def _render_session(timestamp, fragments):
    return f"[Dialogue at timestamp {timestamp}]\n" + "\n".join(
        _render_fragment(fragment["speaker"], fragment["text"]) for fragment in fragments)


def _lossless_units(text, mode):
    if mode == "sentence":
        ends = [match.end() for match in re.finditer(r"(?<=[.!?])\s+", text)]
        starts = [0] + ends
        ends.append(len(text))
        return [text[start:end] for start, end in zip(starts, ends) if end > start]
    return re.findall(r"\s+|\S+\s*", text)


def _split_text(text, speaker, tokenizer, limit=MAX_TURN_TOKENS):
    """Sentence-first, then whitespace, then character slicing; never normalize."""
    fits = lambda value: _count(tokenizer, _render_fragment(speaker, value)) <= limit
    if fits(text):
        return [text]
    if not fits(""):
        raise ValueError("Speaker marker exceeds turn budget")

    def subdivide(unit, mode):
        if fits(unit):
            return [unit]
        if mode == "sentence":
            result = []
            for word in _lossless_units(unit, "word"):
                result.extend(subdivide(word, "character"))
            return result
        parts = []
        while unit:
            # Slice Python characters rather than token IDs, preserving Unicode.
            low, high, valid = 1, len(unit), 0
            while low <= high:
                middle = (low + high) // 2
                if fits(unit[:middle]):
                    valid, low = middle, middle + 1
                else:
                    high = middle - 1
            if valid == 0:
                raise ValueError("A single character exceeds the available turn budget")
            parts.append(unit[:valid])
            unit = unit[valid:]
        return parts

    units = []
    for sentence in _lossless_units(text, "sentence"):
        units.extend(subdivide(sentence, "sentence"))
    parts, current = [], ""
    for unit in units:
        if current and not fits(current + unit):
            parts.append(current)
            current = ""
        current += unit
    if current:
        parts.append(current)
    if "".join(parts) != text or any(not fits(part) for part in parts):
        raise AssertionError("Lossless turn splitting invariant failed")
    return parts


def prepare_conversations(data, tokenizer):
    """Construct bounded input chunks, with source-character coverage sidecars."""
    prepared = []
    for sample in data:
        turns = conversation_turns(sample)
        by_session = {}
        for turn in turns:
            by_session.setdefault(turn["session"], []).append(turn)
        session_segments = []
        unsegmented = []
        for session in _session_keys(sample["conversation"]):
            timestamp = sample["conversation"][session + "_date_time"]
            session_turns = by_session.get(session, [])
            unsegmented.append(_render_session(timestamp, session_turns))
            current = []

            def flush_segment():
                if current:
                    text = _render_session(timestamp, current)
                    session_segments.append({"text": text, "fragments": current.copy()})
                    current.clear()

            if not session_turns:
                session_segments.append({"text": _render_session(timestamp, []), "fragments": []})
            for turn in session_turns:
                # The normal 150-token reserve handles LoCoMo headers. A long
                # synthetic timestamp receives an exact smaller safe budget.
                header_tokens = _count(tokenizer, _render_session(timestamp, []))
                limit = min(MAX_TURN_TOKENS, MAX_CHUNK_TOKENS - header_tokens - 8)
                if limit <= 0:
                    raise ValueError("Session timestamp exceeds chunk token budget")
                pieces = _split_text(turn["text"], turn["speaker"], tokenizer, limit)
                offset = 0
                for piece in pieces:
                    fragment = {**turn, "text": piece, "char_start": offset,
                                "char_end": offset + len(piece),
                                "source_text_length": len(turn["text"])}
                    offset += len(piece)
                    if current and _count(tokenizer, _render_session(timestamp, current + [fragment])) > MAX_CHUNK_TOKENS:
                        flush_segment()
                    if _count(tokenizer, _render_session(timestamp, [fragment])) > MAX_CHUNK_TOKENS:
                        raise ValueError("A turn fragment plus header exceeds chunk limit")
                    current.append(fragment)
            flush_segment()

        chunks, pending = [], []

        def flush_chunk():
            if not pending:
                return
            text = "\n\n".join(segment["text"] for segment in pending)
            fragments = [fragment for segment in pending for fragment in segment["fragments"]]
            count = _count(tokenizer, text)
            if count > MAX_CHUNK_TOKENS:
                raise AssertionError("Chunk token budget exceeded")
            chunks.append({"chunk_index": len(chunks), "text": text, "token_count": count,
                           "source_ids": list(dict.fromkeys(fragment["source_id"] for fragment in fragments)),
                           "segments": fragments})
            pending.clear()

        for segment in session_segments:
            candidate = "\n\n".join(item["text"] for item in pending + [segment])
            if _count(tokenizer, candidate) > MAX_CHUNK_TOKENS:
                flush_chunk()
                pending.append(segment)
            else:
                pending.append(segment)
                if _count(tokenizer, candidate) >= PACK_FLUSH_TOKENS:
                    flush_chunk()
        flush_chunk()
        # Prove contiguous, ordered coverage of every source character.
        fragments = [fragment for chunk in chunks for fragment in chunk["segments"]]
        reconstructed = {}
        for fragment in fragments:
            source = fragment["source_id"]
            previous = reconstructed.get(source, "")
            if fragment["char_start"] != len(previous):
                raise AssertionError("Missing, reordered, or duplicated source content")
            reconstructed[source] = previous + fragment["text"]
        if reconstructed != {turn["source_id"]: turn["text"] for turn in turns}:
            raise AssertionError("Conversation source coverage mismatch")
        prepared.append({"sample_id": sample["sample_id"], "chunks": chunks,
                         "input_tokens": sum(chunk["token_count"] for chunk in chunks),
                         "unsegmented_input_tokens": _count(tokenizer, "\n\n".join(unsegmented)),
                         "num_turns": len(turns), "num_sessions": len(unsegmented),
                         "num_captions": sum(turn["has_caption"] for turn in turns)})
    return prepared


def prepare_questions(data):
    """Prepare the entire shared question manifest BEFORE sharding or resuming."""
    rng = random.Random(OPTION_SEED)
    rows = []
    for sample in data:
        for qa_index, qa in enumerate(sample["qa"]):
            category = int(qa["category"])
            if category not in CATEGORY_NAMES:
                raise ValueError("Unknown LoCoMo category")
            question = qa["question"]
            option_map, abstention_option, anomaly = None, None, []
            if category == 2:
                question += " Use DATE of CONVERSATION to answer with an approximate date."
            elif category == 5:
                distractor = qa["adversarial_answer"]
                abstention_option = "a" if rng.random() < .5 else "b"
                distractor_option = "b" if abstention_option == "a" else "a"
                option_map = {abstention_option: ABSTENTION, distractor_option: distractor}
                question += " Select the correct answer: (a) {} (b) {}. ".format(option_map["a"], option_map["b"])
                if "answer" in qa:
                    anomaly.append("adversarial_row_has_answer")
                if _abstention_matches(distractor):
                    anomaly.append("distractor_matches_official_abstention_scorer")
            row = {"sample_id": sample["sample_id"], "qa_index": qa_index,
                   "category": category, "question": qa["question"],
                   "reader_prompt": (QA_PROMPT_CAT_5 if category == 5 else QA_PROMPT).format(question),
                   "option_map": option_map, "abstention_option": abstention_option,
                   "anomaly": anomaly, "evidence": list(qa.get("evidence", []))}
            for key in ("answer", "adversarial_answer"):
                if key in qa:
                    row[key] = qa[key]
            rows.append(row)
    return rows


def _abstention_matches(text):
    lower = text.lower()
    return "no information available" in lower or "not mentioned" in lower


def normalize_answer(text):
    import regex
    text = str(text).replace(",", "").lower()
    text = "".join(char for char in text if char not in string.punctuation)
    return " ".join(regex.sub(r"\b(a|an|the|and)\b", " ", text).split())


_STEMMER = None


def f1_score(prediction, answer):
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem import PorterStemmer
        _STEMMER = PorterStemmer()
    predicted = [_STEMMER.stem(word) for word in normalize_answer(prediction).split()]
    reference = [_STEMMER.stem(word) for word in normalize_answer(answer).split()]
    overlap = sum((Counter(predicted) & Counter(reference)).values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(reference)
    return 2 * precision * recall / (precision + recall)


def score_prediction(raw, question_record):
    """Official metric plus strict option diagnostic; gold is never reader input."""
    if not isinstance(raw, str):
        raise TypeError("Reader output must be a string")
    category = question_record["category"]
    strict_option, strict_correct, invalid_legacy = None, None, False
    prediction = raw.strip()
    if category == 5:
        normalized = prediction.lower()
        if normalized in ("a", "b", "(a)", "(b)"):
            strict_option = normalized.strip("()")
            strict_correct = strict_option == question_record["abstention_option"]
        elif len(normalized) in (1, 3):
            invalid_legacy = True
        prediction = official_get_cat_5_answer(raw, question_record["option_map"]).strip()
        score = float(_abstention_matches(prediction))
    else:
        answer = str(question_record["answer"])
        if category == 3:
            answer = answer.split(";")[0].strip()
        if category == 1:
            choices, references = prediction.split(","), answer.split(",")
            score = sum(max(f1_score(choice.strip(), reference.strip()) for choice in choices)
                        for reference in references) / len(references)
        elif category in (2, 3, 4):
            score = f1_score(prediction, answer)
        else:
            raise ValueError("Unknown LoCoMo category")
    anomaly = list(question_record.get("anomaly", []))
    return {"prediction": prediction, "score": float(score),
            "strict_option": strict_option, "strict_option_correct": strict_correct,
            "legacy_parser_accepted_invalid_option": invalid_legacy,
            "anomaly": anomaly, "adversarial_clean_eligible": category == 5 and not anomaly}
