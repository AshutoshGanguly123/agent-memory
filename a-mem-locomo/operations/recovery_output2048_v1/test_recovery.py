import json
from pathlib import Path
import numpy as np
import pytest
import common
from prepare import seed_subset
from supervise import card_eligible
from amem_reproduction.calls import CallJournal, StageFailure
from amem_reproduction.storage import file_digest, read_jsonl


def test_config_changes_only_cap_and_run_id():
    common.validate_config()
    assert common.CONFIG['max_new_tokens'] == 2048
    assert common.CONFIG['max_attempts'] == 3


@pytest.mark.parametrize('key,value', [('temperature', 0.0), ('max_attempts', 4),
                                     ('max_new_tokens', 4096), ('run_id', 'other'), ('read_k', 20)])
def test_unapproved_scientific_changes_rejected(key, value):
    with pytest.raises(RuntimeError):
        common.validate_config(dict(common.CONFIG, **{key: value}))


def test_seed_scope_immutable_parent_and_separate_retry_journals(tmp_path):
    parent, out = tmp_path / 'parent', tmp_path / 'out'
    for sid in [*common.TARGETS, 'conv-48']:
        folder = parent / 'conversations' / sid
        folder.mkdir(parents=True)
        (folder / 'memory.json').write_text(json.dumps(dict(notes=[dict(id='D1:1')], frozen=False,
            index_documents=['deliberately stale'], embeddings=[[1., 2.]], evolution_count=7)))
        (folder / 'memory_events.jsonl').write_text('{"event_id":"old-event"}\n')
        (folder / 'calls.jsonl').write_text('{"call_id":"parent-failed-attempt"}\n')
    before = {str(p): file_digest(p) for p in parent.rglob('*') if p.is_file()}
    assert seed_subset(parent, out) == {sid: 1 for sid in common.TARGETS}
    assert {p.name for p in (out / 'conversations').iterdir()} == set(common.TARGETS)
    for sid in common.TARGETS:
        assert not (out / 'conversations' / sid / 'calls.jsonl').exists()
        assert (out / 'lineage/conversations' / sid / 'calls.jsonl').read_bytes() == (parent / 'conversations' / sid / 'calls.jsonl').read_bytes()
        assert (out / 'conversations' / sid / 'memory.json').read_bytes() == (parent / 'conversations' / sid / 'memory.json').read_bytes()
    assert before == {str(p): file_digest(p) for p in parent.rglob('*') if p.is_file()}
    with pytest.raises(RuntimeError):
        seed_subset(parent, tmp_path / 'forbidden', (*common.TARGETS, 'conv-48'))


def test_dispatch_and_worker_reject_completed_conversations():
    import worker
    with pytest.raises(RuntimeError):
        common.claim([*common.TARGETS, 'conv-48'])
    implementation = worker.implementation()
    assert implementation.CONFIG is common.CONFIG and implementation.OUT == common.OUT
    with pytest.raises(RuntimeError, match='outside'):
        implementation.run_conversation({'sample_id': 'conv-48'}, None, None, None,
                                        common.CONFIG, common.OUT, None)


def test_physical_gpu_lock_is_shared_with_original_adapter(tmp_path, monkeypatch):
    monkeypatch.setattr(common, 'OPS', tmp_path / 'recovery')
    monkeypatch.setattr(common.legacy, 'OPS', tmp_path / 'baseline')
    first = common.lock(common.OPS / 'locks/gpu_GPU-test.lock')
    assert first is not None
    assert common.legacy.lock(common.legacy.OPS / 'locks/gpu_GPU-test.lock') is None
    first.close()
    other = common.legacy.lock(common.legacy.OPS / 'locks/gpu_GPU-test.lock')
    assert other is not None
    other.close()


@pytest.mark.parametrize('condition', ['foreign_pid', 'low_vram', 'low_ram', 'scheduled', 'cooldown', 'four_workers', 'all_claimed', 'blocked'])
def test_busy_or_unavailable_cards_never_launch(condition):
    card = dict(uuid='GPU-x', pids=[], free_mib=32000)
    children, blocked, cooldown, available, jobs, ram = {}, {}, {}, list(common.TARGETS), [], 64 * 1024**3
    if condition == 'foreign_pid': card['pids'] = [99]
    if condition == 'low_vram': card['free_mib'] = 1000
    if condition == 'low_ram': ram = 10 * 1024**3
    if condition == 'scheduled': jobs = ['123|other']
    if condition == 'cooldown': cooldown['GPU-x'] = 100
    if condition == 'four_workers': children = {str(i): {} for i in range(4)}
    if condition == 'all_claimed': children, available = {'GPU-y': {}}, ['conv-41']
    if condition == 'blocked': blocked['GPU-x'] = {'exit_code': 1}
    assert not card_eligible(card, children, blocked, cooldown, available, jobs, ram, 50)


def test_free_card_launches_and_recovery_retry_budget_survives_restart(tmp_path):
    assert card_eligible(dict(uuid='GPU-x', pids=[], free_mib=32000), {}, {}, {},
                         list(common.TARGETS), [], 64 * 1024**3, 50)
    class Bad:
        calls = 0
        def encode(self, prompt): return np.zeros((1, 20), dtype=int)
        def generate(self, tokens, schema, seed):
            self.calls += 1
            return dict(parsed=None, raw_response='{', output_tokens=2048,
                        latency_seconds=.1, stop_reason='output_limit')
    backend = Bad()
    path = tmp_path / 'calls.jsonl'
    for _ in range(2):
        with pytest.raises(StageFailure):
            CallJournal(backend, common.CONFIG, path).complete('evolution', 'conv-41/D3:11/evolution', 'p', {})
    assert backend.calls == 3
    completed = [r for r in read_jsonl(path) if r['event'] == 'call_complete']
    assert [r['attempt'] for r in completed] == [0, 1, 2]
    assert all(r['output_reserve'] == 2048 for r in completed)


def test_checkpoint_resume_preserves_stale_prefix_and_only_processes_remaining_turn(tmp_path):
    from amem_reproduction.memory import Memory
    from amem_reproduction.prompts import Prompts
    from amem_reproduction.runner import run_conversation
    fixtures = common.load_module('amem_original_test_fixtures', common.ROOT / 'tests/test_protocol.py')
    prompts, embedder = Prompts(common.ROOT), fixtures.Embedder()
    memory = Memory(embedder, prompts, fixtures.Journal(), common.legacy.CONFIG)
    dialogs = [dict(dia_id=f'D1:{i}', speaker='Maya', text=f'Fact {i}') for i in range(1, 4)]
    for row in dialogs[:2]:
        memory.add(dict(row, timestamp='10:00 am on 1 January, 2025', session='session_1'), 'conv-41/' + row['dia_id'])
    stale = list(memory.state['index_documents'])
    assert 'changed old context' not in stale[0]
    folder = tmp_path / 'conversations/conv-41'
    folder.mkdir(parents=True)
    memory.save(folder / 'memory.json')
    class Controlled:
        calls = 0
        def encode(self, prompt): return np.zeros((1, 20), dtype=int)
        def generate(self, tokens, schema, seed):
            self.calls += 1
            keys = set(schema['properties'])
            if keys == {'keywords', 'context', 'tags'}:
                value = dict(keywords=['fact'], context='Fact three', tags=['fact'])
            elif 'should_evolve' in keys:
                value = dict(should_evolve=False, actions=[], suggested_connections=[], tags_to_update=[],
                             new_context_neighborhood=[], new_tags_neighborhood=[])
            elif keys == {'keywords'}:
                value = dict(keywords='fact')
            else:
                value = dict(answer='Fact 3')
            return dict(parsed=value, raw_response=json.dumps(value), output_tokens=5,
                        latency_seconds=.01, stop_reason='json_complete')
    sample = dict(sample_id='conv-41', conversation={'session_1_date_time': '10:00 am on 1 January, 2025',
                  'session_1': dialogs}, qa=[dict(question='What is the third fact?', answer='Fact 3', category=4, evidence=['D1:3'])])
    backend = Controlled()
    run_conversation(sample, backend, embedder, prompts, common.CONFIG, tmp_path, lambda **v: None)
    assert backend.calls == 4  # only third note/evolution, then one rewrite/answer
    final = json.loads((folder / 'memory.json').read_text())
    assert final['index_documents'][:2] == stale and final['frozen']
    assert [n['id'] for n in final['notes']] == ['D1:1', 'D1:2', 'D1:3']
    calls = [r for r in read_jsonl(folder / 'calls.jsonl') if r['event'] == 'call_complete']
    assert not any('/D1:1/' in r['logical_id'] or '/D1:2/' in r['logical_id'] for r in calls)
    run_conversation(sample, backend, embedder, prompts, common.CONFIG, tmp_path, lambda **v: None)
    assert backend.calls == 4
