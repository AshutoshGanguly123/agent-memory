import pytest
import common

def record(used='gpu:nvidia-rtx-5000-ada-generation:3(IDX:0-2)', state='MIXED', drain='N/A'):
    return f'NodeName=gpu-02 Arch=x86_64\n GresUsed={used}\n GresDrain={drain}\n State={state}\n'

def test_only_reserved_cards_are_unavailable():
    snapshot = common.parse_reservations(record(), 'gpu-02')
    assert [common.reserved({'index': i}, snapshot) for i in range(8)] == [True]*3 + [False]*5

@pytest.mark.parametrize('used,expected', [
    ('gpu:ada:0(IDX:N/A)', []), ('gpu:ada:2(IDX:1,6)', [1,6]),
    ('gpu:ada:4(IDX:0-1,6-7)', [0,1,6,7]), ('N/A', []),
    ('gpu:ada:1(IDX:2),gpu:other:1(IDX:5)', [2,5])])
def test_index_forms(used, expected):
    assert common.parse_reservations(record(used), 'gpu-02')['reserved_indices'] == expected

@pytest.mark.parametrize('used', ['gpu:ada:3(IDX:0-1)', 'gpu:ada:1', 'gpu:ada:2(IDX:2-1)', 'gpu:ada:1(IDX:x)'])
def test_unknown_mapping_rejected(used):
    with pytest.raises(RuntimeError):
        common.parse_reservations(record(used), 'gpu-02')

@pytest.mark.parametrize('state,drain', [('MIXED+DRAIN','N/A'), ('DOWN','N/A'), ('MIXED','gpu:ada:1(IDX:7)')])
def test_node_unavailability_blocks_all_cards(state, drain):
    assert common.reserved({'index': 7}, common.parse_reservations(record(state=state,drain=drain), 'gpu-02'))

def test_wrong_node_rejected():
    with pytest.raises(RuntimeError):
        common.parse_reservations(record(), 'gpu-01')

def test_scheduler_error_blocks_all_cards(monkeypatch):
    monkeypatch.setattr(common, 'node_name', lambda: 'gpu-02')
    def fail(*args, **kwargs):
        raise OSError('unavailable')
    monkeypatch.setattr(common.subprocess, 'run', fail)
    assert common.reserved({'index': 7}, common.scheduled_jobs())

def test_worker_keeps_exact_recovery_scope():
    assert common.CONFIG['max_new_tokens'] == 2048
    assert set(common.TARGETS) == {'conv-26','conv-41','conv-44','conv-50'}
    with pytest.raises(RuntimeError):
        common.require_target('conv-48')

@pytest.mark.parametrize('bad_key', ['caps_equal','repeats_equal','previous_probes_equal','matches_previous_probe'])
def test_unstable_reference_rejected(bad_key):
    evidence={k: True for k in ('caps_equal','repeats_equal','previous_probes_equal','matches_previous_probe')}
    evidence[bad_key]=False
    with pytest.raises(RuntimeError):
        common.reference_result(evidence)

def test_actual_reference_results_must_match():
    evidence={k: True for k in ('caps_equal','repeats_equal','previous_probes_equal','matches_previous_probe')}
    result=dict(output_token_ids=[1,2], parsed={'answer':'x'}, stop_reason='json_complete')
    evidence['results']=dict(cap1000=result,cap2048=result,cap2048_repeat=dict(result,output_token_ids=[1,3]))
    with pytest.raises(RuntimeError):
        common.reference_result(evidence)
