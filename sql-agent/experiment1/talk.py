"""Controller ↔ DB-agent dialogue channel (exp9-style dual scores)."""

from __future__ import annotations

from llm import as_lines, as_rating, ask_json
from prompts import (
    controller_seed_prompt,
    controller_talk_prompt,
    db_talk_prompt,
    talk_window_prompt,
)

TALK_MAX = 6
TALK_PASS = 8.0
TALK_WINDOW = 2


def format_transcript(turns: list[dict]) -> str:
    blocks = []
    for turn in turns:
        lines = [
            f"Turn {turn['turn']}",
            f"controller asked for:\n{turn['need']}",
            f"database agent understood it as:\n{turn['restatement']}",
            f"database agent would search:\n{turn['plan']}",
        ]
        if turn["questions"]:
            lines.append(
                "database agent asks:\n" + "\n".join(f"- {q}" for q in turn["questions"])
            )
        if turn.get("reply"):
            lines.append(f"controller answered:\n{turn['reply']}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def seed_need(args, log, qid: str, question: str) -> str:
    parsed, _raw = ask_json(
        args,
        log,
        qid,
        "controller_seed",
        controller_seed_prompt(question),
        max_tokens=args.talk_tokens,
        temperature=args.reason_temperature,
        keys=("need",),
    )
    need = str(parsed.get("need") or "").strip()
    return need or question


def talk(args, log, qid: str, need: str) -> dict:
    turns: list[dict] = []
    windows: list[str] = []
    stop = "turn cap"
    for turn_n in range(1, TALK_MAX + 1):
        seen = "\n\n".join(windows + [format_transcript(turns[-TALK_WINDOW:])]).strip()
        db_parsed, _raw = ask_json(
            args,
            log,
            qid,
            "db_talk",
            db_talk_prompt(need, seen, turn_n, TALK_MAX),
            max_tokens=args.talk_tokens,
            temperature=args.reason_temperature,
            keys=("restatement",),
            extra={"turn": turn_n},
        )
        understanding = as_rating(db_parsed.get("understanding"))
        questions = as_lines(db_parsed.get("questions"), 2)
        pending = {
            "turn": turn_n,
            "need": need,
            "restatement": str(db_parsed.get("restatement") or "").strip(),
            "plan": str(db_parsed.get("plan") or "").strip(),
            "questions": questions,
            "reply": "",
        }
        shown = "\n\n".join(
            windows + [format_transcript((turns + [pending])[-TALK_WINDOW:])]
        ).strip()
        ctrl, _raw = ask_json(
            args,
            log,
            qid,
            "controller_talk",
            controller_talk_prompt(need, shown, turn_n, TALK_MAX),
            max_tokens=args.talk_tokens,
            temperature=args.reason_temperature,
            keys=("reply",),
            extra={"turn": turn_n},
        )
        satisfaction = as_rating(ctrl.get("satisfaction"))
        refined = str(ctrl.get("need") or "").strip()
        record = {
            **pending,
            "understanding": understanding,
            "reply": str(ctrl.get("reply") or "").strip(),
            "satisfaction": satisfaction,
            "need_after": refined or need,
        }
        turns.append(record)
        log.write({"uuid": qid, "action": "talk_turn", **record})
        print(
            f"[{qid}] talk {turn_n} understanding {understanding} satisfaction {satisfaction}",
            flush=True,
        )
        need = refined or need
        ready = (
            understanding is not None
            and satisfaction is not None
            and understanding >= TALK_PASS
            and satisfaction >= TALK_PASS
        )
        if turn_n % TALK_WINDOW == 0 or ready:
            window = turns[-TALK_WINDOW:]
            ratings = "; ".join(
                f"turn {item['turn']} understanding {item['understanding']} satisfaction {item['satisfaction']}"
                for item in window
            )
            summary, _raw = ask_json(
                args,
                log,
                qid,
                "talk_window",
                talk_window_prompt(need, format_transcript(window), ratings),
                max_tokens=args.summary_tokens,
                temperature=args.reason_temperature,
                keys=("summary",),
                extra={"through_turn": turn_n},
            )
            text = str(summary.get("summary") or "").strip()
            if text:
                windows.append(
                    f"turns {window[0]['turn']} to {window[-1]['turn']}: {text}"
                )
        if ready:
            stop = "both sides ready"
            break

    session = {
        "brief": need,
        "turns": turns,
        "n_turns": len(turns),
        "stop_reason": stop,
        "windows": windows,
        "summary": "\n".join(windows),
    }
    log.write(
        {
            "uuid": qid,
            "action": "talk_done",
            "brief": need,
            "n_turns": len(turns),
            "stop_reason": stop,
            "summary": session["summary"],
        }
    )
    return session
