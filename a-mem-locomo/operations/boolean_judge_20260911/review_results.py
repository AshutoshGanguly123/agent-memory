import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/insights/semantic_review_20260911_v1/boolean_judge'


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def compare():
    labels = {r['question_id']: r for r in json.loads((OUT / 'pilot_review.json').read_text())['labels']}
    comparisons = {}
    for tag in ('pilot', 'pilot_v2', 'pilot_8b', 'pilot_8b_v3'):
        if not (OUT / (tag + '_exit.json')).exists():
            continue
        verdicts = rows(OUT / (tag + '_verdicts.jsonl'))
        errors = [{k: r[k] for k in ('question_id', 'question', 'gold_answer', 'prediction', 'correct')}
                  | {'review_correct': labels[r['question_id']]['correct'], 'reason': labels[r['question_id']]['reason']}
                  for r in verdicts if r['correct'] != labels[r['question_id']]['correct']]
        comparisons[tag] = {'agreement_count': len(verdicts) - len(errors), 'total': len(verdicts), 'disagreements': errors}
    (OUT / 'pilot_comparison.json').write_text(json.dumps(comparisons, indent=2))
    print(json.dumps(comparisons, indent=2), flush=True)


def fresh():
    target = OUT / 'fresh_check'
    if (target / 'pilot_inputs.jsonl').exists():
        return
    target.mkdir(exist_ok=True)
    used = {r['question_id'] for r in rows(OUT / 'pilot_inputs.jsonl')}
    all_rows = rows(OUT / 'input_answers.jsonl')
    rng = random.Random(101)
    selected = []
    for category in range(1, 6):
        candidates = [r for r in all_rows if int(r['category']) == category and r['question_id'] not in used]
        selected.extend(rng.sample(candidates, 2))
    (target / 'pilot_inputs.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in selected))
    for name in ('input_answers.jsonl', 'prompt_v2.txt'):
        (target / name).symlink_to(OUT / name)
    for r in selected:
        print(json.dumps({k: r[k] for k in ('question_id', 'category', 'question', 'gold_answer', 'prediction')}), flush=True)


if __name__ == '__main__':
    rc = 1
    try:
        compare()
        fresh()
        rc = 0
    finally:
        (OUT / 'review_results_exit.json').write_text(json.dumps({'exit_code': rc}))
