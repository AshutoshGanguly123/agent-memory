"""Execution Accuracy / Jaccard against BiomedSQL gold execution_results."""

from __future__ import annotations

import ast
import json
import math
from typing import Any


class GoldParseError(ValueError):
    """Raised when gold execution_results cannot be parsed as a list of dict rows."""


def _parse_gold(raw: str, *, strict: bool = False) -> list[dict]:
    if raw is None or not str(raw).strip():
        if strict:
            raise GoldParseError("empty gold execution_results")
        return []
    text = str(raw).strip()
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            if strict:
                raise GoldParseError(f"unparseable gold: {exc}") from exc
            return []
    if isinstance(value, list):
        rows = [item for item in value if isinstance(item, dict)]
        if strict and len(rows) != len(value):
            raise GoldParseError("gold list contains non-dict rows")
        return rows
    if strict:
        raise GoldParseError(f"gold must be list[dict], got {type(value).__name__}")
    return []


def parse_gold_strict(raw: str) -> list[dict]:
    return _parse_gold(raw, strict=True)


def _normalize_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return round(value, 8)
    if isinstance(value, int):
        return value
    text = str(value).strip()
    try:
        if "." in text or "e" in text.lower():
            return round(float(text), 8)
        return int(text)
    except ValueError:
        return text


def result_keyset(rows: list[dict]) -> set[tuple]:
    """Paper-style comparison: UUID set when present, else full-row tuples."""
    if not rows:
        return set()
    if all("UUID" in row or "uuid" in row for row in rows):
        keys = set()
        for row in rows:
            uid = row.get("UUID", row.get("uuid"))
            keys.add((_normalize_cell(uid),))
        return keys
    keys = set()
    for row in rows:
        items = tuple(sorted((k, _normalize_cell(v)) for k, v in row.items()))
        keys.add(items)
    return keys


def execution_accuracy(pred_rows: list[dict], gold_raw: str, *, strict_gold: bool = False) -> float:
    gold_rows = _parse_gold(gold_raw, strict=strict_gold)
    pred = result_keyset(pred_rows)
    gold = result_keyset(gold_rows)
    return 1.0 if pred == gold else 0.0


def jaccard(pred_rows: list[dict], gold_raw: str, *, strict_gold: bool = False) -> float:
    gold_rows = _parse_gold(gold_raw, strict=strict_gold)
    pred = result_keyset(pred_rows)
    gold = result_keyset(gold_rows)
    if not pred and not gold:
        return 1.0
    union = pred | gold
    if not union:
        return 1.0
    return len(pred & gold) / len(union)
