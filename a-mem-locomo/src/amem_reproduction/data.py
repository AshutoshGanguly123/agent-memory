import json
import re
from datetime import datetime
from pathlib import Path
from .storage import file_digest


def load_dataset(path, expected_sha=None):
    if expected_sha and file_digest(path) != expected_sha:
        raise ValueError("Dataset checksum mismatch")
    samples = json.loads(Path(path).read_text())
    if len(samples) != 10 or sum(len(c["qa"]) for c in samples) != 1986:
        raise ValueError("Expected complete LoCoMo: 10 conversations / 1,986 questions")
    ids = [c["sample_id"] for c in samples]
    assert len(set(ids)) == len(ids)
    assert sum(len(turns(c)) for c in samples) == 5882
    return samples


def turns(sample):
    conversation = sample["conversation"]
    keys = sorted((k for k in conversation if re.fullmatch(r"session_\d+", k)),
                  key=lambda k: int(k.split("_")[1]))
    result = []
    for key in keys:
        date = conversation[key + "_date_time"]
        for raw in conversation[key]:
            result.append({"dia_id": raw["dia_id"], "speaker": raw["speaker"],
                           "text": raw["text"], "timestamp": date, "session": key,
                           "has_caption": bool(raw.get("blip_caption"))})
    assert len({t["dia_id"] for t in result}) == len(result)
    return result


def evidence_ids(values):
    # Bad references remain visible in the QA record and denominator.
    return sorted(set(str(x).strip() for x in values if str(x).strip()))
