"""SQLite execution + BigQuery→SQLite rewriting for BiomedSQL gold/model SQL."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

ROW_LIMIT = 100          # default when the query has no LIMIT
ROW_HARD_CAP = 5000      # safety cap even when LIMIT is larger
STATEMENT_TIMEOUT_MS = 60_000

# BigQuery-style: FROM `{project_id}.{dataset_name}.TableName`
_BQ_TABLE = re.compile(
    r"`\{project_id\}\.\{dataset_name\}\.([A-Za-z0-9_]+)`"
    r"|`([A-Za-z0-9_\.]+)`"
    r"|\{project_id\}\.\{dataset_name\}\.([A-Za-z0-9_]+)",
    re.IGNORECASE,
)

# Double-quoted string literals (BigQuery / MySQL) → single quotes for SQLite.
# Do not touch already-single-quoted strings; scan outside them.
_DQ_STRING = re.compile(r'"([^"\\]*(?:\\.[^"\\]*)*)"')


def rewrite_sql(sql: str) -> str:
    """Make BiomedSQL / BigQuery-ish SQL runnable on our SQLite file."""
    text = (sql or "").strip().rstrip(";")
    if not text:
        return text

    def table_sub(match: re.Match) -> str:
        name = match.group(1) or match.group(3)
        if name:
            return name
        # bare `foo.bar.baz` or `Table` — take last segment
        raw = match.group(2) or ""
        return raw.split(".")[-1]

    text = _BQ_TABLE.sub(table_sub, text)

    # Replace "string" literals with 'string', but leave "Identifier" alone
    # when it looks like a column (no spaces, already a known pattern is hard).
    # BiomedSQL gold uses "rs..." and "%GENE %" as string values — convert all
    # double-quoted spans that are not empty.
    def dq_sub(match: re.Match) -> str:
        inner = match.group(1).replace("\\'", "'").replace("'", "''")
        return f"'{inner}'"

    text = _DQ_STRING.sub(dq_sub, text)

    # TRUE/FALSE → 1/0 for SQLite
    text = re.sub(r"\bTRUE\b", "1", text, flags=re.IGNORECASE)
    text = re.sub(r"\bFALSE\b", "0", text, flags=re.IGNORECASE)

    return text


def execute_select(db_path: Path, sql: str, row_limit: int = ROW_LIMIT) -> dict:
    """Run one read-only SELECT. Returns rows as list[dict], or error."""
    original = (sql or "").strip()
    rewritten = rewrite_sql(original)
    if not rewritten:
        return {"sql": original, "rewritten": rewritten, "rows": [], "error": "empty SQL", "kind": "sql"}
    lowered = rewritten.lstrip().lower()
    if not lowered.startswith("select") and not lowered.startswith("with"):
        return {
            "sql": original,
            "rewritten": rewritten,
            "rows": [],
            "error": "only SELECT/WITH allowed",
            "kind": "sql",
        }
    limit_match = re.search(r"\blimit\s+(\d+)\b", lowered)
    if limit_match:
        fetch_n = min(int(limit_match.group(1)), ROW_HARD_CAP)
    else:
        rewritten = f"{rewritten.rstrip()} LIMIT {row_limit}"
        fetch_n = row_limit

    try:
        connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {STATEMENT_TIMEOUT_MS}")
        cursor = connection.execute(rewritten)
        fetched = cursor.fetchmany(fetch_n)
        rows = [dict(row) for row in fetched]
        connection.close()
        return {"sql": original, "rewritten": rewritten, "rows": rows, "error": "", "kind": "sql"}
    except Exception as exc:  # noqa: BLE001 — surface any SQL failure to the agent
        return {"sql": original, "rewritten": rewritten, "rows": [], "error": str(exc), "kind": "sql"}


def rows_preview(rows: list[dict], max_rows: int = 10, max_chars: int = 4000) -> str:
    if not rows:
        return "(no rows)"
    lines = []
    keys = list(rows[0].keys())
    lines.append(" | ".join(keys))
    for row in rows[:max_rows]:
        lines.append(" | ".join(str(row.get(k, ""))[:120] for k in keys))
    text = "\n".join(lines)
    if len(text) > max_chars:
        return text[:max_chars] + "\n...(truncated)"
    return text


def list_tables(db_path: Path) -> list[str]:
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    names = [
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]
    connection.close()
    return names
