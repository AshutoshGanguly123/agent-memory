"""OpenAI-compatible chat client (same shape as PABLO exp7/exp9)."""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

CONTEXT_LIMIT = 32768


def chat(
    base_url: str,
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int,
) -> tuple[str, str | None]:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=1800) as response:
        body = json.loads(response.read().decode())
    choice = body["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    if not content.strip():
        content = message.get("reasoning") or message.get("reasoning_content") or ""
    return content.strip(), choice.get("finish_reason")


def prompt_tokens(base_url: str, model: str, text: str) -> int | None:
    payload = {"model": model, "prompt": text}
    request = urllib.request.Request(
        base_url.rstrip("/") + "/tokenize",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode())
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError, KeyError):
        return None
    count = body.get("count")
    if isinstance(count, int):
        return count
    tokens = body.get("tokens")
    return len(tokens) if isinstance(tokens, list) else None


def parse_json_object(text: str) -> dict | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    blob = fenced.group(1) if fenced else None
    if blob is None:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            blob = text[start : end + 1]
    if blob is None:
        return None
    try:
        payload = json.loads(blob)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value)
        if match:
            try:
                return float(match.group(0))
            except ValueError:
                return None
    return None


def as_rating(value: object) -> float | None:
    number = as_float(value)
    if number is None:
        return None
    return max(0.0, min(10.0, number))


def as_lines(value: object, limit: int) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out[:limit]


class Log:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            self._handle.write(line)
            self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def model_call(args, log: Log, qid: str, action: str, messages: list[dict], max_tokens: int, temperature: float, extra: dict | None = None):
    prompt = "\n\n".join(message["content"] for message in messages)
    n_tokens = prompt_tokens(args.model_url, args.model_name, prompt)
    if n_tokens is not None and n_tokens + max_tokens > CONTEXT_LIMIT:
        log.write(
            {
                "uuid": qid,
                "action": "context_stop",
                "stopped_action": action,
                "prompt_tokens": n_tokens,
                "max_tokens": max_tokens,
                "limit": CONTEXT_LIMIT,
            }
        )
        raise SystemExit(
            f"[{qid}] CONTEXT {action}: prompt {n_tokens} + {max_tokens} exceeds {CONTEXT_LIMIT}."
        )
    started = time.time()
    text, finish = chat(
        args.model_url,
        args.model_name,
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    record = {
        "uuid": qid,
        "action": action,
        "messages": messages,
        "prompt": prompt,
        "prompt_chars": len(prompt),
        "completion": text,
        "finish_reason": finish,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "seconds": round(time.time() - started, 3),
        "model": args.model_name,
    }
    if n_tokens is not None:
        record["prompt_tokens"] = n_tokens
    if extra:
        record.update(extra)
    log.write(record)
    token_note = f", {n_tokens} prompt tokens" if n_tokens is not None else ""
    length_note = ", TRUNCATED" if finish == "length" else ""
    print(f"[{qid}] {action} ({len(text)} chars{token_note}{length_note})", flush=True)
    return text, finish, n_tokens


def ask(args, log, qid, action, prompt, max_tokens, temperature, extra=None):
    text, _finish, _n = model_call(
        args,
        log,
        qid,
        action,
        [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
        extra=extra,
    )
    return text


def ask_json(args, log, qid, action, prompt, max_tokens, temperature, keys, tries=2, extra=None):
    parsed: dict = {}
    text = ""
    for attempt in range(1, tries + 1):
        text = ask(
            args,
            log,
            qid,
            action,
            prompt,
            max_tokens,
            temperature,
            extra={**(extra or {}), "parse_try": attempt},
        )
        parsed = parse_json_object(text) or {}
        if all(str(parsed.get(key) or "").strip() for key in keys):
            return parsed, text
        print(f"[{qid}] {action} JSON missing {list(keys)}", flush=True)
    return parsed, text
