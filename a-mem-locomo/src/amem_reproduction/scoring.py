"""LoCoMo's released scoring rules, without unused BERTScore imports.

Source: benchmark/upstream/locomo_evaluation.py (pinned in sources.json).
"""
import re
import string
from collections import Counter
from nltk.stem import PorterStemmer

_stemmer = PorterStemmer()


def normalize_answer(s):
    s = str(s).replace(",", "").lower()
    s = "".join(c for c in s if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the|and)\b", " ", s).split())


def f1_score(prediction, ground_truth):
    predicted = [_stemmer.stem(w) for w in normalize_answer(prediction).split()]
    gold = [_stemmer.stem(w) for w in normalize_answer(ground_truth).split()]
    overlap = sum((Counter(predicted) & Counter(gold)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def score_answer(prediction, qa):
    category = int(qa["category"])
    if category == 5:
        text = prediction.lower()
        return float("no information available" in text or "not mentioned" in text)
    answer = str(qa["answer"])
    if category == 3:
        answer = answer.split(";")[0].strip()
    if category == 1:
        predictions = [p.strip() for p in prediction.split(",")]
        golds = [g.strip() for g in answer.split(",")]
        return sum(max(f1_score(p, g) for p in predictions) for g in golds) / len(golds)
    return f1_score(prediction, answer)
