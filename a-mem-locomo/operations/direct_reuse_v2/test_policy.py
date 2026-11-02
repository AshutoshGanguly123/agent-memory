import json
from pathlib import Path
import pytest
from policy import schema_exhaustion, sha
from common import claim


@pytest.fixture
def evidence(tmp_path):
    ops, out = tmp_path/'ops', tmp_path/'out'
    attempt, uuid, sid = 'full_fixture_gpu1', 'GPU-test', 'conv-failed'
    logical = sid+'/D3:11/evolution'
    error = f"StageFailure('Schema failure after 3 attempts: {logical}')"
    data = {
        ops/'workers'/(attempt+'.json'): dict(attempt=attempt, state='failed', gpu_uuid=uuid,
            conversation=sid, current_call=logical, current_call_phase='evolution', pid=123, error=error),
        ops/'exits'/(attempt+'.json'): dict(exit_code=1, gpu_uuid=uuid, pid=123),
        ops/'failures'/(sid+'.json'): dict(attempt=attempt, sample_id=sid, error=error),
    }
    for path, value in data.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    journal = out/'conversations'/sid/'calls.jsonl'
    journal.parent.mkdir(parents=True)
    calls = [dict(event='call_complete', logical_id=logical, call_id='digest', attempt=i,
                  status='schema_error', parsed=None, context_overflow=False) for i in range(3)]
    journal.write_text(''.join(json.dumps(c)+'\n' for c in calls))
    return ops, out, dict(max_attempts=3), uuid, dict(attempt=attempt, exit_code=1), journal, calls


def test_proven_schema_failure_releases_gpu_but_preserves_conversation(evidence):
    ops, out, config, uuid, entry, journal, calls = evidence
    before = {str(p): sha(p) for root in (ops, out) for p in root.rglob('*') if p.is_file()}
    decision = schema_exhaustion(ops, out, config, uuid, entry)
    assert decision['conversation'] == 'conv-failed'
    assert decision['completed_schema_attempts'] == 3
    after = {str(p): sha(p) for root in (ops, out) for p in root.rglob('*') if p.is_file()}
    assert before == after
    sid, handle = claim(['conv-failed', 'conv-queued'], out, ops)
    assert sid == 'conv-queued'
    handle.close()


def test_exhaustion_reconstructed_from_cache_needs_no_new_call_callback(evidence):
    ops, out, config, uuid, entry, journal, calls = evidence
    path = ops/'workers'/(entry['attempt']+'.json')
    worker = json.loads(path.read_text())
    worker.pop('current_call')
    worker.pop('current_call_phase')
    path.write_text(json.dumps(worker))
    assert schema_exhaustion(ops, out, config, uuid, entry)['completed_schema_attempts'] == 3


@pytest.mark.parametrize('change', ['backend_error', 'context_overflow', 'incomplete_budget',
                                  'later_success', 'wrong_gpu', 'missing_exit', 'corrupt_journal'])
def test_unknown_or_inconsistent_failure_never_releases_gpu(evidence, change):
    ops, out, config, uuid, entry, journal, calls = evidence
    if change == 'backend_error': calls[-1]['status'] = 'backend_error'
    elif change == 'context_overflow': calls[-1]['context_overflow'] = True
    elif change == 'incomplete_budget': calls.pop()
    elif change == 'later_success': calls.append(dict(calls[-1], status='ok', parsed={}))
    elif change == 'wrong_gpu': uuid = 'GPU-other'
    elif change == 'missing_exit': (ops/'exits'/(entry['attempt']+'.json')).unlink()
    journal.write_text(''.join(json.dumps(c)+'\n' for c in calls))
    if change == 'corrupt_journal': journal.write_text('broken\n')
    assert schema_exhaustion(ops, out, config, uuid, entry) is None

