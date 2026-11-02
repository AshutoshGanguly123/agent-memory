"""Scientific denominators and reader-receipt binding in generated reports."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import locomo
import report
import run
from common import ARMS, digest


def perfect_raw(question):
    if question["category"] == 5:
        return question["abstention_option"]
    answer = str(question["answer"])
    return answer.split(";")[0].strip() if question["category"] == 3 else answer


@pytest.fixture(scope="module")
def all_questions():
    return locomo.prepare_questions(locomo.load_dataset(ROOT / "data/locomo10.json"))


def test_complete_and_failed_full_benchmark_denominators(all_questions):
    missing_qa = next(q for q in all_questions if q["category"] == 5 and not q["anomaly"])
    failed_qa = next(q for q in all_questions if q["category"] == 4)
    rows = []
    for question in all_questions:
        scored = locomo.score_prediction(perfect_raw(question), question)
        status = "complete"
        if question is missing_qa or question is failed_qa:
            status = "missing" if question is missing_qa else "error"
            scored["score"] = 0.0
            scored["strict_option_correct"] = False
        rows.append({**question, **scored, "status": status})
    values = report.aggregate(rows)
    assert values["count"] == 1986 and values["completed"] == 1984
    assert values["overall_micro"] == pytest.approx(1984 / 1986)
    assert values["nonadversarial_micro"] == pytest.approx(1539 / 1540)
    assert values["adversarial_strict"] == pytest.approx(445 / 446)
    assert values["adversarial_clean_count"] == 442
    assert values["adversarial_clean_strict"] == pytest.approx(441 / 442)
    assert values["category_macro"] == pytest.approx(1 - (1 / 841 + 1 / 446) / 5)
    assert values["categories"]["single_hop"] == {"count": 841, "completed": 840, "score": pytest.approx(840 / 841)}
    assert values["categories"]["adversarial"]["count"] == 446


@pytest.fixture
def report_environment(tmp_path, monkeypatch, all_questions):
    questions = [next(q for q in all_questions if q["category"] == category) for category in (1, 2, 3, 4)]
    questions += [next(q for q in all_questions if q["category"] == 5 and not q["anomaly"])]
    questions += [q for q in all_questions if q["anomaly"]]
    assert len(questions) == 9
    conversations = [{"sample_id": sample_id, "input_tokens": 100, "chunks": []}
                     for sample_id in dict.fromkeys(q["sample_id"] for q in questions)]
    cfg = {"run_id": "test-report", "training_enabled": False, "reader_batch_size": 3,
           "reader": {"seed": 42, "temperature": .7, "max_tokens": 2048}}
    sources = {"test-source": "pinned"}
    base = tmp_path / "runs" / "synthetic_smoke"
    monkeypatch.setattr(report, "ROOT", tmp_path)
    monkeypatch.setattr(report, "config", lambda: cfg)
    monkeypatch.setattr(report, "run_root", lambda synthetic=False: base)
    monkeypatch.setattr(report, "source_manifest", lambda: sources, raising=False)
    monkeypatch.setattr(report, "verify_freeze", lambda: {"sources": sources}, raising=False)
    monkeypatch.setattr(run, "inputs", lambda synthetic=False: (conversations, questions))

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    memories = {}
    for arm in ARMS:
        for conversation in conversations:
            memory = {"core": arm + " memory", "semantic": [], "episodic": []}
            frozen = {"arm": arm, "sample_id": conversation["sample_id"], "frozen": True,
                      "memory": memory, "memory_hash": digest(memory),
                      "config_hash": digest(cfg), "source_hash": digest(sources)}
            memories[(arm, conversation["sample_id"])] = frozen
            write(base / arm / "conversations" / conversation["sample_id"] / "memory_frozen.json", frozen)
        for question in questions:
            receipt = {"arm": arm, "sample_id": question["sample_id"], "qa_index": question["qa_index"],
                       "question_hash": digest(question), "memory_hash": memories[(arm, question["sample_id"])]["memory_hash"],
                       "config_hash": digest(cfg), "source_hash": digest(sources), "status": "complete", "attempt": 1,
                       "result": {"raw_response": perfect_raw(question), "usage": {"completion_tokens": 4, "prompt_tokens": 30}}}
            write(base / arm / "answers" / question["sample_id"] / f'{question["qa_index"]:05d}.json', receipt)
    return {"base": base, "questions": questions, "write": write, "cfg": cfg, "sources": sources}


def answer_path(environment, arm, question):
    return environment["base"] / arm / "answers" / question["sample_id"] / f'{question["qa_index"]:05d}.json'


def test_generate_complete_report_and_four_anomalies_per_arm(report_environment):
    result = report.generate_report(synthetic=True)
    assert result["complete"] is True
    for arm in ARMS:
        values = result["metrics"][arm]
        assert values["count"] == values["completed"] == 9
        for name in ("overall_micro", "category_macro", "nonadversarial_micro", "adversarial_strict", "adversarial_clean_strict"):
            assert values[name] == 1
        assert values["adversarial_clean_count"] == 1
    assert result["paired_memalpha_minus_base"]["overall_micro"] == 0
    target = report_environment["base"] / "report"
    rows = json.loads((target / "per_question.json").read_text())
    assert len(rows) == 18
    assert len(json.loads((target / "annotation_anomalies.json").read_text())) == 8
    assert "mixed task score" in (target / "report.md").read_text()


def test_report_keeps_missing_and_error_outputs_in_both_denominators(report_environment):
    questions = report_environment["questions"]
    failed = next(q for q in questions if q["category"] == 4)
    missing_clean = next(q for q in questions if q["category"] == 5 and not q["anomaly"])
    missing_base = next(q for q in questions if q["category"] == 1)
    path = answer_path(report_environment, ARMS[0], failed)
    receipt = json.loads(path.read_text())
    # Even a coincidentally correct error payload must count zero.
    receipt["status"] = "error"
    report_environment["write"](path, receipt)
    answer_path(report_environment, ARMS[0], missing_clean).unlink()
    answer_path(report_environment, ARMS[1], missing_base).unlink()
    result = report.generate_report(synthetic=True)
    assert result["complete"] is False
    trained, base = (result["metrics"][arm] for arm in ARMS)
    assert trained["count"] == base["count"] == 9
    assert trained["completed"] == 7 and base["completed"] == 8
    assert trained["overall_micro"] == pytest.approx(7 / 9)
    assert base["overall_micro"] == pytest.approx(8 / 9)
    assert trained["nonadversarial_micro"] == .75
    assert trained["adversarial_strict"] == .8
    assert trained["adversarial_clean_count"] == 1 and trained["adversarial_clean_strict"] == 0
    assert result["paired_memalpha_minus_base"]["overall_micro"] == pytest.approx(-1 / 9)
    rows = json.loads((report_environment["base"] / "report/per_question.json").read_text())
    assert len(rows) == 18 and sum(row["status"] != "complete" for row in rows) == 3
    assert all(row["score"] == 0 for row in rows if row["status"] != "complete")


@pytest.mark.parametrize("field", ["arm", "sample_id", "qa_index", "question_hash", "memory_hash", "config_hash", "source_hash"])
def test_report_rejects_stale_or_misbound_reader_receipt(report_environment, field):
    question = report_environment["questions"][0]
    path = answer_path(report_environment, ARMS[0], question)
    receipt = json.loads(path.read_text())
    receipt[field] = "WRONG_BINDING"
    report_environment["write"](path, receipt)
    with pytest.raises(RuntimeError):
        report.generate_report(synthetic=True)


def test_report_rejects_answer_without_its_frozen_memory(report_environment):
    question = report_environment["questions"][0]
    path = report_environment["base"] / ARMS[0] / "conversations" / question["sample_id"] / "memory_frozen.json"
    path.unlink()
    with pytest.raises(RuntimeError):
        report.generate_report(synthetic=True)


def test_unstarted_run_retains_all_zero_denominators_without_memories(report_environment):
    for arm in ARMS:
        for path in (report_environment["base"] / arm).glob("answers/*/*.json"):
            path.unlink()
        for path in (report_environment["base"] / arm).glob("conversations/*/memory_frozen.json"):
            path.unlink()
    result = report.generate_report(synthetic=True)
    assert not result["complete"]
    for values in result["metrics"].values():
        assert values["count"] == 9 and values["completed"] == 0
        assert values["overall_micro"] == 0
        assert values["adversarial_clean_count"] == 1
        assert values["adversarial_clean_strict"] == 0


@pytest.mark.parametrize("field", ["arm", "sample_id", "frozen", "config_hash", "source_hash", "memory_hash"])
def test_report_checks_frozen_memory_provenance(report_environment, field):
    question = report_environment["questions"][0]
    path = report_environment["base"] / ARMS[0] / "conversations" / question["sample_id"] / "memory_frozen.json"
    frozen = json.loads(path.read_text())
    frozen[field] = "WRONG_BINDING"
    report_environment["write"](path, frozen)
    with pytest.raises(RuntimeError):
        report.generate_report(synthetic=True)


def test_non_synthetic_report_requires_freeze_verification(report_environment, monkeypatch):
    def reject_modified_freeze():
        raise RuntimeError("Freeze mismatch")
    monkeypatch.setattr(report, "verify_freeze", reject_modified_freeze)
    with pytest.raises(RuntimeError, match="Freeze mismatch"):
        report.generate_report(synthetic=False)


def test_runner_binds_reader_inputs_and_receipts_without_evaluator_leakage(report_environment, monkeypatch):
    environment = report_environment
    for arm in ARMS:
        for path in (environment["base"] / arm).glob("answers/*/*.json"):
            path.unlink()
    monkeypatch.setattr(run, "ROOT", environment["base"].parents[1])
    monkeypatch.setattr(run, "config", lambda: environment["cfg"])
    monkeypatch.setattr(run, "run_root", lambda synthetic=False: environment["base"])
    monkeypatch.setattr(run, "source_manifest", lambda: environment["sources"])
    monkeypatch.setattr(run, "STOP", False)
    seen = []

    class Reader:
        def __init__(self, *args):
            pass

        def answer_batch(self, items):
            for item in items:
                assert set(item) == {"memory", "raw_question", "reader_prompt"}
                assert "answer" not in item and "category" not in item and "evidence" not in item
                seen.append({**item, "memory": dict(item["memory"])})
                # The request receives its own copy, never a mutable live store.
                item["memory"]["core"] = "MUTATION_INSIDE_REQUEST"
            return [{"raw_response": "A valid response", "usage": {}} for _ in items]

    monkeypatch.setitem(sys.modules, "bridge", SimpleNamespace(ReaderBridge=Reader))
    assert run.answer(synthetic=True) == 0
    assert len(seen) == 2 * len(environment["questions"])
    for arm_index, arm in enumerate(ARMS):
        for index, question in enumerate(environment["questions"]):
            item = seen[arm_index * len(environment["questions"]) + index]
            assert item["reader_prompt"] == question["reader_prompt"]
            assert item["raw_question"] == question["question"]
            assert item["memory"]["core"] == arm + " memory"
            receipt = json.loads(answer_path(environment, arm, question).read_text())
            frozen = json.loads((environment["base"] / arm / "conversations" / question["sample_id"] / "memory_frozen.json").read_text())
            assert receipt["question_hash"] == digest(question)
            assert receipt["memory_hash"] == frozen["memory_hash"] == digest(frozen["memory"])
            assert frozen["memory"]["core"] == arm + " memory"
            assert receipt["status"] == "complete"
    seen.clear()
    assert run.answer(synthetic=True) == 0
    assert seen == []  # Existing complete receipts resume without new requests.


def test_retry_usage_counts_all_attempts_but_scores_only_latest(report_environment):
    question = report_environment["questions"][0]
    path = answer_path(report_environment, ARMS[0], question)
    latest = json.loads(path.read_text())
    archived = {**latest, "status": "error", "attempt": 1,
                "result": {"raw_response": "An old incorrect output", "finish_reason": "length",
                           "usage": {"prompt_tokens": 100, "completion_tokens": 10,
                                     "batch_wall_seconds": 8, "batch_size": 2}}}
    latest["attempt"] = 2
    latest["result"]["usage"].update({"batch_wall_seconds": 4, "batch_size": 1})
    report_environment["write"](path.parent / "attempts" / f'{question["qa_index"]:05d}_1.json', archived)
    report_environment["write"](path, latest)
    result = report.generate_report(synthetic=True)
    assert result["complete"]
    assert result["metrics"][ARMS[0]]["overall_micro"] == 1
    usage = result["operational"][ARMS[0]]["usage"]
    assert usage["reader_calls"] == 10 and usage["reader_retry_calls"] == 1
    assert usage["reader_prompt_tokens"] == 370
    assert usage["reader_completion_tokens"] == 46
    assert usage["reader_truncated"] == 1
    assert usage["reader_generation_wall_seconds"] == 8


def test_interrupted_retry_does_not_double_count_archived_latest_receipt(report_environment):
    question = report_environment["questions"][0]
    path = answer_path(report_environment, ARMS[0], question)
    receipt = json.loads(path.read_text())
    receipt["status"] = "error"
    report_environment["write"](path, receipt)
    report_environment["write"](path.parent / "attempts" / f'{question["qa_index"]:05d}_1.json', receipt)
    result = report.generate_report(synthetic=True)
    usage = result["operational"][ARMS[0]]["usage"]
    assert usage["reader_calls"] == 9 and usage["reader_retry_calls"] == 0
    assert usage["reader_prompt_tokens"] == 270 and usage["reader_completion_tokens"] == 36
    assert result["metrics"][ARMS[0]]["completed"] == 8


@pytest.mark.parametrize("raw,option,correct", [
    ("(b) Not mentioned in the conversation.", "b", True),
    ("(B)  NOT mentioned\n in the conversation!", "b", True),
    ("b", "b", True),
    ("b.", "b", True),
    ("b: Not mentioned in the conversation", "b", True),
    ("Not mentioned in the conversation.", "b", True),
    ("(a) Yes.", "a", False),
    ("Yes", "a", False),
    ("yes", "a", False),
    ("c", None, False),
    ("(x)", None, False),
    ("(a) Not mentioned in the conversation.", None, False),
    ("(b) Yes", None, False),
    ("(a) Yes or (b) Not mentioned in the conversation", None, False),
    ("b or a", None, False),
    ("I think it is not mentioned", None, False),
    ("", None, False),
])
def test_exact_choice_diagnostic_accepts_unambiguous_labels_and_text(raw, option, correct):
    question = {"option_map": {"a": "Yes", "b": locomo.ABSTENTION}, "abstention_option": "b"}
    result = report.exact_choice_diagnostic(raw, question)
    assert result["strict_option"] == option
    assert result["strict_option_correct"] is correct


def test_exact_choice_rejects_ambiguous_standalone_text_or_label_text_conflict():
    duplicated = {"option_map": {"a": "Not mentioned", "b": "Not mentioned."}, "abstention_option": "b"}
    assert report.exact_choice_diagnostic("Not mentioned", duplicated)["strict_option_correct"] is False
    assert report.exact_choice_diagnostic("Not mentioned", duplicated)["strict_option"] is None
    assert report.exact_choice_diagnostic("(b) Not mentioned", duplicated)["strict_option_correct"] is True
    label_conflict = {"option_map": {"a": locomo.ABSTENTION, "b": "a"}, "abstention_option": "a"}
    assert report.exact_choice_diagnostic("a", label_conflict)["strict_option"] is None


def test_report_extended_choice_diagnostic_does_not_change_official_score(report_environment):
    question = next(q for q in report_environment["questions"] if q["category"] == 5 and not q["anomaly"])
    assert question["abstention_option"] == "b"
    path = answer_path(report_environment, ARMS[0], question)
    receipt = json.loads(path.read_text())
    receipt["result"]["raw_response"] = "(b) Not mentioned in the conversation."
    report_environment["write"](path, receipt)
    official = locomo.score_prediction(receipt["result"]["raw_response"], question)
    assert official["score"] == 1 and official["strict_option_correct"] is None
    result = report.generate_report(synthetic=True)
    assert result["metrics"][ARMS[0]]["adversarial_strict"] == 1
    rows = json.loads((report_environment["base"] / "report/per_question.json").read_text())
    row = next(r for r in rows if r["arm"] == ARMS[0] and r["sample_id"] == question["sample_id"] and r["qa_index"] == question["qa_index"])
    assert row["score"] == official["score"]
    assert row["prediction"] == official["prediction"]
    assert row["strict_option_correct"] is True
    assert row["strict_option_interpretation"] == "label_and_matching_option_text"
    assert "Exact unambiguous choice accuracy" in result["diagnostic_definitions"]["adversarial_strict"]
