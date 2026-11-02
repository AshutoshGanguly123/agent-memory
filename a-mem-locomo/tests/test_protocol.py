import ast
import json
import string
from collections import Counter
from pathlib import Path
import numpy as np
import pytest
import regex
from nltk.stem import PorterStemmer
from amem_reproduction.scoring import score_answer
from amem_reproduction.data import load_dataset, turns
from amem_reproduction.runner import coverage
from amem_reproduction.storage import read_jsonl
from amem_reproduction.memory import Memory
from amem_reproduction.prompts import Prompts
from amem_reproduction.calls import CallJournal, StageFailure

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / 'configs/experiment.json').read_text())


def test_scoring_matches_official_source_all_nonadversarial_questions():
    tree = ast.parse((ROOT / 'benchmark/upstream/locomo_evaluation.py').read_text())
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and
                             n.name in ('normalize_answer', 'f1_score', 'f1')], type_ignores=[])
    namespace = dict(regex=regex, string=string, Counter=Counter, ps=PorterStemmer(), np=np)
    exec(compile(module, '<official scoring>', 'exec'), namespace)
    for sample in load_dataset(ROOT / CONFIG['dataset'], CONFIG['dataset_sha256']):
        for qa in sample['qa']:
            if qa['category'] == 5:
                assert score_answer('No information available', qa) == 1
                assert score_answer('Paris', qa) == 0
                continue
            gold = str(qa['answer'])
            if qa['category'] == 3: gold = gold.split(';')[0].strip()
            for prediction in (gold, '', 'the, Paris and Berlin', gold[:len(gold)//2]):
                fn = namespace['f1' if qa['category'] == 1 else 'f1_score']
                assert score_answer(prediction, qa) == pytest.approx(fn(prediction, gold))


def test_dataset_input_has_no_evaluation_fields_and_coverage_missing_is_null():
    data = load_dataset(ROOT / CONFIG['dataset'], CONFIG['dataset_sha256'])
    for sample in data:
        assert all(set(t) == {'dia_id', 'speaker', 'text', 'timestamp', 'session', 'has_caption'} for t in turns(sample))
    assert coverage({'evidence': []}, {}, [])['evidence_coverage'] is None
    c = coverage({'evidence': ['D1:1', 'D9:9']}, {'supplied_ids': ['D1:1', 'D1:1']}, ['D1:1'])
    assert c['evidence_coverage'] == .5 and c['invalid_evidence_ids'] == ['D9:9']
    prompt, _ = Prompts(ROOT).answer('Where?', 5, 'Known memory')
    assert 'No information available' in prompt


class Embedder:
    def encode(self, docs):
        return np.array([[len(d), sum(map(ord, d)) % 97 + 1, 1] for d in docs], dtype=np.float32)


class Journal:
    def __init__(self): self.count = 0
    def complete(self, phase, key, prompt, schema):
        if phase == 'note': return {'context': 'old context', 'keywords': ['original'], 'tags': ['old']}, key
        self.count += 1
        return {'should_evolve': True, 'actions': ['strengthen', 'update_neighbor'],
                'suggested_connections': [0] if self.count > 1 else [], 'tags_to_update': ['new note tags'],
                'new_context_neighborhood': ['changed old context'] if self.count > 1 else [],
                'new_tags_neighborhood': [['changed old tags']] if self.count > 1 else []}, key


def test_evolution_and_resume_preserve_stale_index_then_refresh(tmp_path):
    cfg = dict(CONFIG, evolution_refresh_interval=3)
    memory = Memory(Embedder(), Prompts(ROOT), Journal(), cfg)
    def add(i): return memory.add(dict(dia_id=f'D1:{i}', speaker='Maya', text=f'Fact {i}',
                                      timestamp='today', session='session_1'), str(i))
    add(1)
    first_doc = memory.state['index_documents'][0]
    add(2)
    assert memory.notes[1]['links'] == [0]
    assert memory.notes[1]['tags'] == ['new note tags']
    assert memory.notes[0]['context'] == 'changed old context'
    assert memory.notes[0]['keywords'] == ['original']
    assert memory.state['index_documents'][0] == first_doc
    memory.save(tmp_path / 'memory.json')
    resumed = Memory(Embedder(), Prompts(ROOT), memory.journal, cfg, json.loads((tmp_path/'memory.json').read_text()))
    assert resumed.retrieve('Fact') == memory.retrieve('Fact')
    event = add(3)
    assert event['index_refreshed'] and memory.state['index_documents'][0] != first_doc
    memory.state['frozen'] = True
    with pytest.raises(RuntimeError): add(4)


def test_retrieval_exactly_matches_released_counter_behavior():
    tree = ast.parse((ROOT / 'benchmark/upstream/amem_memory_layer.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'AgenticMemorySystem')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'find_related_memories_raw')
    fn.returns = None
    for a in fn.args.args: a.annotation = None
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<official retrieval>', 'exec'), ns)
    from types import SimpleNamespace
    notes = [dict(id=str(i), timestamp='date', content=str(i), context='context', keywords=['k'], tags=['t'],
                  links=list(range(3))) for i in range(4)]
    m = Memory(Embedder(), Prompts(ROOT), None, dict(CONFIG, read_k=2),
               dict(notes=notes, index_documents=[], embeddings=[], frozen=True, evolution_count=0))
    m.search = lambda query, k: [3, 2]
    official = SimpleNamespace(memories={i: SimpleNamespace(**n) for i, n in enumerate(notes)},
                               retriever=SimpleNamespace(search=lambda query, k: [3, 2]))
    context, trace = m.retrieve('q')
    assert context == ns['find_related_memories_raw'](official, 'q', 2)
    assert trace['expanded_indices'] == [3, 0, 1, 2, 2, 0]


class Backend:
    def __init__(self, valid=False): self.calls=0; self.valid=valid
    def encode(self, prompt): return np.zeros((1, 20), dtype=int)
    def generate(self, tokens, schema, seed):
        self.calls += 1
        return dict(raw_response='{}', parsed={'answer': 'yes'} if self.valid else None,
                    output_tokens=1, latency_seconds=.1)


def test_calls_fail_closed_overflow_and_success_resume(tmp_path):
    bad = Backend()
    journal = CallJournal(bad, CONFIG, tmp_path/'bad.jsonl')
    with pytest.raises(StageFailure): journal.complete('note', 'id', 'prompt', {})
    assert bad.calls == 3
    overflow = CallJournal(bad, dict(CONFIG, max_context=1000), tmp_path/'overflow.jsonl')
    with pytest.raises(StageFailure): overflow.complete('note', 'id', 'prompt', {})
    assert bad.calls == 3
    good = Backend(True)
    path = tmp_path/'good.jsonl'
    result = CallJournal(good, CONFIG, path).complete('answer', 'id', 'prompt', {})
    assert CallJournal(good, CONFIG, path).complete('answer', 'id', 'prompt', {}) == result
    assert good.calls == 1


def test_monitor_does_not_truncate_active_partial_journal(tmp_path):
    path = tmp_path/'events.jsonl'
    raw = b'{"ok":true}\n{"pending":'
    path.write_bytes(raw)
    assert read_jsonl(path) == [{'ok': True}]
    assert path.read_bytes() == raw
    assert read_jsonl(path, repair=True) == [{'ok': True}]
    assert path.read_bytes() == b'{"ok":true}\n'


def test_real_tokenizer_json_constraint_contract():
    from transformers import AutoTokenizer
    from lmformatenforcer import TokenEnforcer, JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
    path=ROOT/CONFIG['model_path']
    if not path.exists(): pytest.skip('Model cache only exists on TT')
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    token_data=build_token_enforcer_tokenizer_data(tokenizer)
    prompts=Prompts(ROOT)
    schema=prompts.evolution({'context':'x','content':'y','keywords':[]},'neighbor 17',[17])[1]
    example={'should_evolve':True,'actions':['strengthen','update_neighbor'],'suggested_connections':[17],
             'tags_to_update':['bicycle'],'new_context_neighborhood':['new'],'new_tags_neighborhood':[['ride']]}
    parser=TokenEnforcer(token_data,JsonSchemaParser(schema))
    prefix=[]
    for token in tokenizer.encode(json.dumps(example),add_special_tokens=False):
        allowed=parser.get_allowed_tokens(prefix).allowed_tokens
        assert token in allowed, (tokenizer.decode(prefix),tokenizer.decode([token]))
        prefix.append(token)


def test_end_to_end_checkpoint_qa_failure_denominators_and_no_gold_leakage(tmp_path):
    from amem_reproduction.runner import run_conversation
    class ControlledBackend:
        def __init__(self): self.prompts=[]; self.calls=0
        def encode(self,prompt):
            self.prompts.append(prompt)
            return np.zeros((1,20),dtype=int)
        def generate(self,tokens,schema,seed):
            self.calls+=1
            keys=set(schema['properties'])
            if keys=={'keywords','context','tags'}:
                value={'keywords':['bicycle'],'context':'A bicycle','tags':['travel']}
            elif 'should_evolve' in keys:
                value={'should_evolve':False,'actions':[],'suggested_connections':[],'tags_to_update':[],
                       'new_context_neighborhood':[],'new_tags_neighborhood':[]}
            elif keys=={'keywords'}: value={'keywords':'bicycle'}
            else: value=None  # Exhaust retries, but preserve both question records with zero score.
            return dict(parsed=value,raw_response=json.dumps(value),output_tokens=5,latency_seconds=.01)
    sample={'sample_id':'synthetic','conversation':{
        'session_1_date_time':'10:00 am on 1 January, 2025',
        'session_1':[{'dia_id':'D1:1','speaker':'Maya','text':'My bicycle is red.'}]},
        'qa':[{'question':'What color is the bicycle?','answer':'SECRET_GOLD_ANSWER','evidence':['D1:1'],'category':4},
              {'question':'Who made it?','adversarial_answer':'SECRET_DISTRACTOR','evidence':[],'category':5}]}
    b=ControlledBackend()
    run_conversation(sample,b,Embedder(),Prompts(ROOT),CONFIG,tmp_path,lambda **v:None)
    answers=read_jsonl(tmp_path/'conversations/synthetic/answers.jsonl')
    assert len(answers)==2 and all(a['status']=='failed' and a['score']==0 for a in answers)
    assert all('SECRET_' not in p for p in b.prompts)
    assert json.loads((tmp_path/'conversations/synthetic/memory.json').read_text())['frozen']
    before=b.calls
    run_conversation(sample,b,Embedder(),Prompts(ROOT),CONFIG,tmp_path,lambda **v:None)
    assert b.calls==before
    assert len(read_jsonl(tmp_path/'conversations/synthetic/answers.jsonl'))==2
