"""Release a GPU only after a proven conversation-local schema exhaustion."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_overlay(root):
    code = root/'operations/direct_reuse_v2'
    manifest = json.loads((root/'results/operations/direct_reuse_v2/manifest.json').read_text())
    actual = {p.name: sha(p) for p in code.glob('*.py')}
    if actual != manifest['source_sha256']:
        raise RuntimeError('GPU reuse adapter changed after validation')
    return manifest


def schema_exhaustion(ops, out, config, uuid, entry):
    """Return auditable evidence, or None for every unproven/unknown failure.

    Check both process records and the full completed-attempt journal. Never
    remove a failure marker, change retry counts, or modify a conversation.
    """
    if entry.get('exit_code') != 1:
        return None
    try:
        attempt = entry['attempt']
        worker_path = ops/'workers'/(attempt+'.json')
        worker = json.loads(worker_path.read_text())
        exit_record = json.loads((ops/'exits'/(attempt+'.json')).read_text())
        sid = worker['conversation']
        prefix = f"StageFailure('Schema failure after {config['max_attempts']} attempts: "
        if not worker['error'].startswith(prefix) or not worker['error'].endswith("')"):
            return None
        logical = worker['error'][len(prefix):-2]
        failure_path = ops/'failures'/(sid+'.json')
        failure = json.loads(failure_path.read_text())
        expected = f"StageFailure('Schema failure after {config['max_attempts']} attempts: {logical}')"
        if (worker['state'] != 'failed' or worker['attempt'] != attempt
                or worker['gpu_uuid'] != uuid or exit_record['gpu_uuid'] != uuid
                or exit_record['pid'] != worker['pid'] or exit_record['exit_code'] != 1
                or failure['attempt'] != attempt or failure['sample_id'] != sid
                or worker['error'] != expected or failure['error'] != expected
                or logical.rsplit('/', 1)[-1] not in ('note', 'evolution')
                or not logical.startswith(sid+'/')):
            return None
        completed = []
        for line in (out/'conversations'/sid/'calls.jsonl').read_text().splitlines():
            call = json.loads(line)
            if call.get('event') == 'call_complete' and call.get('logical_id') == logical:
                completed.append(call)
        count = config['max_attempts']
        if (len(completed) != count
                or {c.get('attempt') for c in completed} != set(range(count))
                or len({c.get('call_id') for c in completed}) != 1
                or any(c.get('status') != 'schema_error' or c.get('parsed') is not None
                       or c.get('context_overflow') for c in completed)):
            return None
        return dict(conversation=sid, logical_id=logical, call_id=completed[0]['call_id'],
                    completed_schema_attempts=count, gpu_uuid=uuid, attempt=attempt,
                    failure_sha256=sha(failure_path), worker_sha256=sha(worker_path),
                    disposition='preserve_failed_conversation_and_reuse_gpu')
    except (OSError, KeyError, TypeError, ValueError):
        return None

