import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/insights/semantic_review_20260911_v1/boolean_judge'
IDS = ['conv-30/q0001', 'conv-43/q0089', 'conv-48/q0073', 'conv-42/q0064', 'conv-30/q0095']


def read(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


if __name__ == '__main__':
    rc = 1
    try:
        judged = {r['question_id']: r for r in read(OUT / 'full_8b_v3_verdicts.jsonl')}
        manual = {r['original_answer_record']['question_id']: r for r in read(OUT.parent / 'manual_audit/all_reviewed.jsonl')}
        examples = []
        for qid in IDS:
            j, m = judged[qid], manual[qid]
            assert j['correct'] != m['review']['correct']
            row = {'question_id': qid, 'question': j['question'], 'gold': j['gold_answer'],
                'prediction': j['prediction'], 'judge_correct': j['correct'],
                'codex_correct': m['review']['correct'], 'codex_note': m['review']['review_note'],
                'source_evidence_turns': m['source_evidence_turns']}
            examples.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        (OUT / 'five_disagreement_examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2))
        rc = 0
    finally:
        (OUT / 'inspect_disagreements_exit.json').write_text(json.dumps({'exit_code': rc}))
