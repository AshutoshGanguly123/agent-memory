import hashlib
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/insights/semantic_review_20260911_v1/boolean_judge'


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def agreement(verdict_path, labels_path):
    predicted = rows(verdict_path)
    labels = {r['question_id']: r for r in json.loads(labels_path.read_text())['labels']}
    assert len(predicted) == len(labels)
    errors = [{k: r[k] for k in ('question_id', 'question', 'gold_answer', 'prediction', 'correct')}
              | {'review_correct': labels[r['question_id']]['correct'], 'review_reason': labels[r['question_id']]['reason']}
              for r in predicted if r['correct'] != labels[r['question_id']]['correct']]
    return {'agreed': len(predicted) - len(errors), 'reviewed': len(predicted), 'disagreements': errors}


def main():
    pilot = agreement(OUT / 'pilot_8b_v3_verdicts.jsonl', OUT / 'pilot_review.json')
    fresh = agreement(OUT / 'fresh_check/fresh_8b_v3_verdicts.jsonl', OUT / 'fresh_check/fresh_review.json')
    summary = json.loads((OUT / 'full_8b_v3_summary.json').read_text())
    verdicts = rows(OUT / 'full_8b_v3_verdicts.jsonl')
    original = rows(OUT / 'input_answers.jsonl')
    assert len(verdicts) == len(original) == 1231
    assert {r['question_id'] for r in verdicts} == {r['question_id'] for r in original}
    assert all(type(r['correct']) is bool for r in verdicts)
    assert sum(r['correct'] for r in verdicts) == summary['correct']
    summary['pilot_check'] = pilot
    summary['fresh_check'] = fresh
    summary['execution_failures_excluded'] = 1
    summary['scope'] = '1,231 successfully generated answers from 1,232 saved records across six completed conversations; one execution failure separately preserved'
    summary['status'] = 'provisional gold-referenced judge score; small Codex sanity check, not human-validated accuracy'
    summary['known_reference_issue'] = 'conv-30/q0095 is an adversarial duplicate of q0061 with source-supported answer trophy; adversarial judge protocol still uses abstention target consistently'
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2))
    compact = [{k: r[k] for k in ('question_id', 'sample_id', 'category', 'correct')} for r in verdicts]
    (OUT / 'boolean_scores.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in compact))
    manual_path = OUT.parent / 'manual_audit/all_reviewed.jsonl'
    if manual_path.exists():
        manual = {r['original_answer_record']['question_id']: r['review']['correct'] for r in rows(manual_path)
                  if r['original_answer_record'].get('status') == 'ok' and type(r['review']['correct']) is bool}
        assert set(manual) == {r['question_id'] for r in verdicts}
        mismatches = [r['question_id'] for r in verdicts if r['correct'] != manual[r['question_id']]]
        comparison = {'agreed': len(verdicts) - len(mismatches), 'compared': len(verdicts),
            'disagreement_question_ids': mismatches,
            'note': 'Independent source-based Codex review; not human labels. Final judge/prompt unchanged after comparison.',
            'manual_source_sha256': hashlib.sha256(manual_path.read_bytes()).hexdigest()}
        (OUT / 'comparison_with_manual_audit.json').write_text(json.dumps(comparison, indent=2))
    table = '\n'.join(f"| {name} | {s['correct']}/{s['total']} | {s['accuracy_percent']:.2f}% |" for name, s in summary['by_category'].items())
    report = f'''# Boolean judge: focused first pass

**{summary['correct']}/{summary['total']} answers graded correct ({summary['accuracy_percent']:.2f}%).** This is a provisional gold-referenced score from Llama-3.1-8B-Instruct, not a human-validated accuracy estimate.

| Category | Correct / judged | Correct |
|---|---:|---:|
{table}

Scope: 1,231 generated answers in the fixed six-conversation snapshot. One additional saved question failed to generate an answer and is recorded separately in `execution_failures.jsonl`.

The judge receives only the question, generated answer, and gold answer, and returns `{{"correct": true}}` or `{{"correct": false}}`. Equivalent wording is accepted; required facts must agree. Adversarial questions use the expected abstention response; dataset distractors are never treated as gold.

The final 8B prompt agreed with my direct inspection on **{pilot['agreed']}/40 pilot cases and {fresh['agreed']}/10 fresh cases**. These are Codex checks, not human labels. Remaining pilot mistakes were two rejected paraphrases and one accepted wrong date. The initial 3B judge was less reliable, especially on abstentions; its checks are retained in `pilot_comparison.json`. Treat this first-pass score as provisional.

The final prompt explicitly treats the gold as authoritative and ignores the candidate's claims about missing context. It is saved in `prompt_v3.txt`; model revision is `0e9e39f249a16976918f6564b8830bc894c89659`. Inference used BF16 CUDA on gpu-02 GPU 1 with greedy, schema-constrained boolean output. Full prompts, raw outputs, token IDs, manifests, device checks, logs, and exits are preserved.

This simple judge sees no conversation evidence, so it can reject a relative date such as "next month" even when the source conversation anchors it correctly. The independent source-based audit can also accept answers where the dataset reference/category is defective; one known example is adversarial `conv-30/q0095`, which asks about Gina's source-supported trophy.

Artifacts: `boolean_scores.jsonl` is the compact per-question output; `full_8b_v3_verdicts.jsonl` contains complete judge calls; `summary.json` contains totals and sanity-check disagreements. Original memory and answer runs were unchanged.
'''
    (OUT / 'report.md').write_text(report)
    manifest = json.loads((OUT / 'full_8b_v3_manifest.json').read_text())
    assert manifest['prompt_sha256'] == hashlib.sha256((OUT / 'prompt_v3.txt').read_bytes()).hexdigest()
    (OUT / 'verification.json').write_text(json.dumps({'passed': True, 'unique_question_ids': len(compact),
        'all_boolean': True, 'counts_reconciled': True, 'prompt_hash_verified': True}, indent=2))
    print(report, flush=True)


if __name__ == '__main__':
    rc = 1
    try:
        main()
        rc = 0
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        (OUT / 'finalize_exit.json').write_text(json.dumps({'exit_code': rc}))
