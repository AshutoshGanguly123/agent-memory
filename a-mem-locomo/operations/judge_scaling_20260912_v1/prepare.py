import csv
import json
import os
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from protocol import *
sys.path.insert(0, str(ROOT / 'src'))
from amem_reproduction.storage import atomic_json, digest
from amem_reproduction.data import load_dataset


def main():
    assert os.environ.get('TMUX'), 'Prepare in tmux'
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'manifest.json').exists(), 'Refuse to replace frozen experiment'
    config = json.loads((ROOT / 'configs/experiment.json').read_text())
    dataset = {x['sample_id']: x for x in load_dataset(ROOT / config['dataset'], config['dataset_sha256'])}
    score_files = [ROOT / 'results/insights/six_completed_comparison_20260911_verified/questions.csv',
                   ROOT / 'results/insights/recovery_status_20260912_v1/questions.csv']
    scores = {}
    for path in score_files:
        for r in csv.DictReader(path.open()):
            assert r['question_id'] not in scores
            scores[r['question_id']] = r
    parents = [('locomo_llama32_3b_amem_cuda_v1', ['conv-30','conv-42','conv-43','conv-47','conv-48','conv-49']),
               ('locomo_llama32_3b_amem_cuda_output2048_recovery_v1', ['conv-26','conv-44','conv-50'])]
    prepared, failures, provenance = [], [], {}
    for run, sids in parents:
        for sid in sids:
            folder = ROOT / 'results/raw_results' / run / 'conversations' / sid
            complete = json.loads((folder / 'complete.json').read_text())
            state = json.loads((folder / 'memory.json').read_text())
            frozen = json.loads((folder / 'memory_frozen.json').read_text())
            assert state['frozen'] and digest({k:v for k,v in state.items() if k != 'build_seconds'}) == frozen['fingerprint']
            answers = rows(folder / 'answers.jsonl')
            assert len(answers) == complete['questions'] == len(dataset[sid]['qa'])
            for name in ['answers.jsonl', 'complete.json', 'memory_frozen.json']:
                provenance[str(folder / name)] = sha(folder / name)
            for answer in answers:
                qid = answer['question_id']
                qa = dataset[sid]['qa'][answer['question_index']]
                assert answer['question'] == qa['question'] and answer['gold_answer'] == qa.get('answer')
                assert answer['memory_fingerprint'] == frozen['fingerprint']
                assert answer['prediction'] == scores[qid]['prediction']
                if answer['status'] != 'ok':
                    failures.append(answer)
                    continue
                cat = int(answer['category'])
                prepared.append(dict(question_id=qid, sample_id=sid, category=cat, question=answer['question'],
                    gold_answer='No information available' if cat == 5 else answer['gold_answer'],
                    original_gold_answer=answer['gold_answer'], stored_answer=answer['prediction'],
                    f1=float(scores[qid]['amem_f1']) if cat != 5 else None, original_score=answer['score'],
                    source_answers_path=str(folder / 'answers.jsonl')))
    prepared.sort(key=lambda r: r['question_id'])
    assert len(prepared) == 1792 and len(failures) == 1
    assert len({r['question_id'] for r in prepared}) == len(prepared)
    inputs = {r['question_id']: r for r in prepared}
    audit_path = ROOT / 'results/insights/semantic_review_20260911_v1/manual_audit/all_reviewed.jsonl'
    reference = []
    for case in rows(audit_path):
        a = case['original_answer_record']
        if a['status'] != 'ok':
            continue
        assert a['prediction'] == inputs[a['question_id']]['stored_answer']
        reference.append(dict(question_id=a['question_id'], correct=case['review']['correct'],
                              method='Existing Codex source-based audit, not human ground truth'))
    assert len(reference) == 1231
    for name, data in [('input_answers.jsonl', prepared), ('execution_failures.jsonl', failures), ('reference_labels.jsonl', reference)]:
        (OUT / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in data))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), run_id='judge_scaling_20260912_v1',
        questions=len(prepared), preserved_execution_failures=1, missing_conv41_questions=193,
        nonadversarial_questions=sum(r['category'] != 5 for r in prepared), category_counts=dict(Counter(r['category'] for r in prepared)),
        model=MODEL, model_revision=REVISION, model_path=MODEL_PATH, temperature=0., seed=42,
        max_new_tokens=MAX_NEW_TOKENS, precision='bfloat16', batch_size=BATCH_SIZE, max_padded_tokens=MAX_PADDED_TOKENS,
        decoding='Greedy JSON-constrained batched CUDA; fresh model context for each call',
        shards=SHARDS, total_planned_calls=len(prepared)*7, calls_by_setting=CALLS,
        schema=SCHEMA, system_prompt=SYSTEM, rubric=RUBRIC, base_openings=OPENINGS, meta_template=META, meta_verbs=VERBS,
        inputs_to_meta='One earlier verdict for L1_V1; all three earlier verdicts for each L1_V3 reviewer; no layer 2.',
        code_sha256=source_hashes(), input_sha256={name:sha(OUT/name) for name in ['input_answers.jsonl','execution_failures.jsonl','reference_labels.jsonl']},
        baseline_artifacts_sha256=provenance, dataset_sha256=config['dataset_sha256'], audit_sha256=sha(audit_path),
        scores_sha256={str(p):sha(p) for p in score_files},
        graph_definition='Percent of fixed saved answers marked correct, versus unchanged mean F1 on the same non-adversarial subset. Separate Codex-reference agreement.',
        limitations=['Nine completed conversations combine baseline and checkpoint-recovery answer generations.',
          'Codex reference labels are provisional and sometimes use source context absent from the gold-only judge.',
          'Prompt variants from one model have correlated errors; increased acceptance is not evidence of improved judging.'])
    atomic_json(OUT / 'manifest.json', manifest)
    (OUT / 'prompts.md').write_text('# Judge scaling prompts\n\n' + '\n\n'.join(OPENINGS) + '\n\n' + RUBRIC + '\n\n' + META + '\n')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    OPS.mkdir(parents=True, exist_ok=True)
    rc = 1
    try:
        main()
        rc = 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        atomic_json(OPS / 'prepare_exit.json', {'exit_code':rc})
