"""DB-agent ReAct SQL loop over BiomedSQL SQLite."""

from __future__ import annotations

from pathlib import Path

from db import execute_select, rows_preview
from llm import as_lines, ask, ask_json, parse_json_object
from prompts import answer_prompt, db_query_prompt, db_round_prompt

DB_ROUNDS = 6
N_QUERIES = 5


def outcome_line(label: str, result: dict) -> str:
    if result["error"]:
        return f"{label}: ERROR {result['error']} | sql={result.get('rewritten') or result['sql']}"
    return f"{label}: {len(result['rows'])} rows | sql={result.get('rewritten') or result['sql']}"


def db_react(args, log, qid: str, brief: str, db_path: Path) -> dict:
    notes = ""
    attempts_text = ""
    last_rows = ""
    best_sql = ""
    best_rows: list[dict] = []
    whole_log: list[str] = []
    rounds_used = 0
    n_queries = sql_errors = 0

    for round_n in range(1, DB_ROUNDS + 1):
        rounds_used = round_n
        query_text = ask(
            args,
            log,
            qid,
            "db_query",
            db_query_prompt(brief, notes, attempts_text, last_rows, round_n, DB_ROUNDS),
            max_tokens=args.query_tokens,
            temperature=args.reason_temperature,
            extra={"round": round_n},
        )
        parsed = parse_json_object(query_text) or {}
        queries = [item.strip() for item in as_lines(parsed.get("queries"), N_QUERIES)]
        lines = []
        round_best_rows: list[dict] = []
        round_best_sql = ""

        for index, statement in enumerate(queries, start=1):
            n_queries += 1
            result = execute_select(db_path, statement)
            if result["error"]:
                sql_errors += 1
            else:
                round_best_rows = result["rows"]
                round_best_sql = result.get("rewritten") or result["sql"]
                last_rows = rows_preview(result["rows"])
                if not best_sql or (result["rows"] and not best_rows):
                    best_sql = round_best_sql
                    best_rows = result["rows"]
                elif result["rows"]:
                    best_sql = round_best_sql
                    best_rows = result["rows"]
            line = outcome_line(f"SQL {index}", result)
            lines.append(line)
            whole_log.append(f"round {round_n} {line}")
            log.write(
                {
                    "uuid": qid,
                    "action": "sql_result",
                    "round": round_n,
                    "query": index,
                    "sql": result["sql"],
                    "rewritten": result.get("rewritten"),
                    "error": result["error"],
                    "n_rows": len(result["rows"]),
                }
            )
            print(
                f"[{qid}] round {round_n} sql {index} "
                f"{result['error'] or str(len(result['rows'])) + ' rows'}",
                flush=True,
            )

        attempts_text = "\n".join(lines) or "no queries"
        if round_best_rows or round_best_sql:
            last_rows = rows_preview(round_best_rows) if round_best_rows else last_rows

        round_parsed, _raw = ask_json(
            args,
            log,
            qid,
            "db_round",
            db_round_prompt(brief, round_n, DB_ROUNDS, attempts_text, last_rows),
            max_tokens=args.observe_tokens,
            temperature=args.reason_temperature,
            keys=("observation",),
            extra={"round": round_n},
        )
        enough = str(round_parsed.get("enough") or "").strip().lower().startswith("yes")
        notes = str(round_parsed.get("next_focus") or "").strip()
        nominated = str(round_parsed.get("best_sql") or "").strip()
        if nominated:
            check = execute_select(db_path, nominated)
            if not check["error"]:
                best_sql = check.get("rewritten") or check["sql"]
                best_rows = check["rows"]
                last_rows = rows_preview(best_rows)

        log.write(
            {
                "uuid": qid,
                "action": "db_round_done",
                "round": round_n,
                "enough": enough,
                "observation": str(round_parsed.get("observation") or "").strip(),
                "best_sql": best_sql,
                "n_rows": len(best_rows),
            }
        )
        print(
            f"[{qid}] round {round_n} rows={len(best_rows)} enough={enough}",
            flush=True,
        )
        if enough:
            break

    return {
        "rounds": rounds_used,
        "n_queries": n_queries,
        "sql_errors": sql_errors,
        "best_sql": best_sql,
        "rows": best_rows,
        "log": whole_log,
    }


def write_answer(args, log, qid: str, question: str, brief: str, search: dict) -> str:
    sql = search.get("best_sql") or ""
    rows_text = rows_preview(search.get("rows") or [])
    parsed, _raw = ask_json(
        args,
        log,
        qid,
        "answer",
        answer_prompt(question, brief, sql, rows_text),
        max_tokens=args.answer_tokens,
        temperature=args.reason_temperature,
        keys=("answer",),
    )
    return str(parsed.get("answer") or "").strip()
