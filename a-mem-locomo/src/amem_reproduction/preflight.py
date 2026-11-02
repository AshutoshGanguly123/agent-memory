"""Synthetic CUDA verification is separate from LoCoMo output and scores."""
import json
import time
from .storage import atomic_json,read_jsonl,utc_now
from .runner import run_conversation
from .provenance import source_hashes


def exercise(root,out,backend,embedder,prompts,config,update):
    before = source_hashes(root)
    update(phase='chunked_prefill_parity')
    parity = backend.prefill_parity_probe()
    sample = dict(sample_id='synthetic',conversation={
        'session_1_date_time':'10:00 am on 1 January, 2025',
        'session_1':[dict(dia_id='D1:1',speaker='Maya',text='My bicycle is red.'),
                     dict(dia_id='D1:2',speaker='Alex',text='I live in Berlin.'),
                     dict(dia_id='D1:3',speaker='Maya',text='I live in Paris and work as a teacher.')],
        'session_2_date_time':'10:00 am on 2 February, 2025',
        'session_2':[dict(dia_id='D2:1',speaker='Alex',text='I bought a blue bicycle yesterday.'),
                     dict(dia_id='D2:2',speaker='Maya',text='Today I moved from Paris to Rome.'),
                     dict(dia_id='D2:3',speaker='Alex',text='My bicycle is blue; Maya has a red one.')]},
        qa=[dict(question='What color is Maya\'s bicycle?',answer='red',evidence=['D1:1'],category=4),
            dict(question='When did Maya move to Rome?',answer='2 February 2025',evidence=['D2:2'],category=2),
            dict(question='What colors are the bicycles?',answer='red, blue',evidence=['D1:1','D2:1'],category=1),
            dict(question='What is Maya\'s favorite song?',adversarial_answer='SECRET_DISTRACTOR',evidence=[],category=5)])
    update(phase='synthetic_end_to_end')
    run_conversation(sample,backend,embedder,prompts,config,out,update)
    folder = out/'conversations/synthetic'
    answers = read_jsonl(folder/'answers.jsonl')
    assert len(answers) == 4 and all(a['status']=='ok' for a in answers), answers
    completed_calls = [c for c in read_jsonl(folder/'calls.jsonl') if c['event']=='call_complete']
    assert all('SECRET_DISTRACTOR' not in c['prompt'] for c in completed_calls)
    before_bytes = (folder/'calls.jsonl').read_bytes()
    run_conversation(sample,backend,embedder,prompts,config,out,update)
    assert (folder/'calls.jsonl').read_bytes() == before_bytes, 'Resume invoked completed calls'
    update(phase='context_capacity_probe',context_input_tokens=config['max_context']-1)
    capacity = backend.context_probe(config['max_context']-1)
    assert capacity['cache_tokens'] == config['max_context']
    atomic_json(out/'context_capacity.json',capacity)
    after = source_hashes(root)
    assert before == after, 'Source changed during preflight'
    atomic_json(out/'passed.json',dict(passed_utc=utc_now(),source_sha256=after,device=backend.device_info,
        context_capacity=capacity,prefill_parity=parity,synthetic_notes=6,synthetic_questions=4,stages=['note','evolution','rewrite','answer'],
        resume_reused_all_calls=True,embedding_max_seq_length=embedder.max_seq_length,
        total_generated_tokens=sum(c.get('output_tokens') or 0 for c in completed_calls),
        measured_call_seconds=sum(c.get('latency_seconds',0) for c in completed_calls)))
