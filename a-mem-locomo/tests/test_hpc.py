import ast
import json
import pytest
from pathlib import Path
from amem_reproduction.calls import CallJournal,StageFailure,CheckpointStop
from amem_reproduction.storage import read_jsonl
from test_protocol import Backend,CONFIG,ROOT,Embedder


def test_retry_cap_persists_across_restart(tmp_path):
    backend=Backend();path=tmp_path/'calls.jsonl'
    with pytest.raises(StageFailure):CallJournal(backend,CONFIG,path).complete('note','id','prompt',{})
    with pytest.raises(StageFailure):CallJournal(backend,CONFIG,path).complete('note','id','prompt',{})
    assert backend.calls==3
    assert [e['attempt'] for e in read_jsonl(path) if e['event']=='call_complete']==[0,1,2]


def test_planned_stop_after_durable_call_reuses_response(tmp_path):
    backend=Backend(True);path=tmp_path/'calls.jsonl'
    def stop():
        if backend.calls:raise CheckpointStop('allocation slice')
    backend.stop_check=stop
    with pytest.raises(CheckpointStop):CallJournal(backend,CONFIG,path).complete('note','id','prompt',{})
    backend.stop_check=lambda:None
    parsed,key=CallJournal(backend,CONFIG,path).complete('note','id','prompt',{})
    assert backend.calls==1 and parsed is not None


def test_original_prompt_strings_match_upstream_ast():
    tree=ast.parse((ROOT/'benchmark/upstream/amem_memory_layer.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MemoryNote')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='analyze_content')
    assignment=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='prompt' for t in n.targets))
    # The author's prompt expression appends content; extract only its constant prefix.
    expression=assignment.value
    if isinstance(expression,ast.BinOp):expression=expression.left
    assert ast.literal_eval(expression)==(ROOT/'configs/prompts/note.txt').read_text()
    assignments=[n for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='evolution_system_prompt' for t in n.targets)]
    assert len(assignments)==1
    assert ast.literal_eval(assignments[0].value)==(ROOT/'configs/prompts/evolution.txt').read_text()


def test_interior_journal_corruption_fails_closed(tmp_path):
    path=tmp_path/'calls.jsonl';path.write_text('{"ok":1}\nBROKEN\n{"ok":2}\n')
    with pytest.raises(json.JSONDecodeError):read_jsonl(path,repair=True)


def test_backend_error_retains_partial_output(tmp_path):
    class Broken(Backend):
        def generate(self,*args):
            self.partial_generation=dict(raw_response='{',output_token_ids=[123],output_tokens=1)
            raise RuntimeError('synthetic CUDA OOM')
    path=tmp_path/'calls.jsonl'
    with pytest.raises(RuntimeError):CallJournal(Broken(),CONFIG,path).complete('note','id','prompt',{})
    last=read_jsonl(path)[-1]
    assert last['status']=='backend_error' and last['output_tokens']==1 and last['raw_response']=='{'


def test_gpu_backend_refuses_login_node(monkeypatch):
    from amem_reproduction.cuda_backend import CudaLlama
    monkeypatch.delenv('SLURM_JOB_ID',raising=False)
    with pytest.raises(RuntimeError,match='Slurm'):CudaLlama(CONFIG,ROOT,0)


def test_analysis_counts_unfinished_attempt_as_unknown(tmp_path):
    from amem_reproduction.analyze import call_attempts
    path=tmp_path/'calls.jsonl'
    events=[dict(call_id='one',attempt=0,time_utc='a',worker_pid=1,event='call_started'),
            dict(call_id='two',attempt=0,time_utc='b',worker_pid=1,event='call_started'),
            dict(call_id='two',attempt=0,time_utc='b',worker_pid=1,event='call_complete',status='ok',output_tokens=4)]
    path.write_text(''.join(json.dumps(e)+'\n' for e in events))
    calls=call_attempts(path)
    assert len(calls)==2 and calls[-1]['status']=='interrupted_unknown' and calls[-1]['output_tokens'] is None


def test_interrupted_utf8_journal_tail_is_preserved_and_repaired(tmp_path):
    path=tmp_path/'calls.jsonl'
    partial=b'{"response":"'+b'\xf0\x9f'
    path.write_bytes(b'{"ok":1}\n'+partial)
    assert read_jsonl(path,repair=True)==[{'ok':1}]
    assert path.read_bytes()==b'{"ok":1}\n'
    backups=list(tmp_path.glob('calls.jsonl.incomplete-*'))
    assert len(backups)==1 and backups[0].read_bytes()==partial


@pytest.mark.parametrize('free,holders,blocked',[(32000,'',False),(32000,'GPU-test, 999999\n',True),(1000,'',True)])
def test_allocated_gpu_requires_actual_free_memory_and_no_other_compute(monkeypatch,free,holders,blocked):
    from types import SimpleNamespace
    from amem_reproduction import device_guard
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','2')
    def query(args,**kwargs):
        text=f'GPU-test, {free}, 32760\n' if '-i' in args else holders
        return SimpleNamespace(returncode=0,stdout=text,stderr='')
    monkeypatch.setattr(device_guard.subprocess,'run',query)
    if blocked:
        with pytest.raises(device_guard.DeviceBusy):device_guard.allocated_gpu()
    else:assert device_guard.allocated_gpu()['uuid']=='GPU-test'


def test_torch_and_nvml_uuid_forms_identify_the_same_card():
    from amem_reproduction.device_guard import same_gpu
    value='1b2a14b7-b84f-4031-0751-f4b5400631bd'
    assert same_gpu(value,'GPU-'+value)
    assert not same_gpu(value,'GPU-de37a606-2151-0bde-91cd-d26a1f3c4567')
