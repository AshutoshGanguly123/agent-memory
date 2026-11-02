import csv
import json
import os
import statistics
import sys
import traceback
from collections import Counter
from protocol import *
sys.path.insert(0,str(ROOT/'src'))
from amem_reproduction.storage import atomic_json, utc_now


def main():
    assert os.environ.get('TMUX')
    manifest=verify()
    for i in range(SHARDS):
        assert json.loads((OPS/f'shard_{i}_status.json').read_text())['state']=='complete'
        assert json.loads((OPS/f'shard_{i}_exit.json').read_text())['exit_code']==0
    inputs=rows(OUT/'input_answers.jsonl')
    reference={r['question_id']:r['correct'] for r in rows(OUT/'reference_labels.jsonl')}
    calls=[]
    for i in range(SHARDS):
        calls.extend(rows(OUT/f'calls_shard_{i}.jsonl'))
    assert len(calls)==7*len(inputs)
    by_id={c['call_id']:c for c in calls}
    assert len(by_id)==len(calls)
    records=[]
    for row in inputs:
        qid=row['question_id']
        for kind in KINDS:
            c=by_id[qid+'/'+kind]
            assert c['question_id']==qid and c['kind']==kind
            deps=[] if kind.startswith('base_') else [qid+'/'+k for k in (['base_0'] if kind=='meta_single_0' else KINDS[:3])]
            assert c['previous_call_ids']==deps
            prior=[by_id[k]['result']['parsed'] for k in deps]
            assert c['prompt']==prompt(row,kind,prior or None)
            assert c['result']['stop_reason']=='json_complete'
            assert type(c['result']['parsed']['correct']) is bool
        r=dict(row)
        r['reference_correct']=reference.get(qid)
        for setting,kinds in SETTINGS.items():
            r[setting]=verdict([by_id[qid+'/'+k]['result']['parsed']['correct'] for k in kinds])
        records.append(r)
    plotted=[r for r in records if r['category']!=5]
    assert len(plotted)==1387
    f1=100*statistics.mean(r['f1'] for r in plotted)
    summary={}
    for setting in SETTINGS:
        audited=[r for r in records if r['reference_correct'] is not None]
        summary[setting]=dict(calls_per_answer=CALLS[setting],
            answer_accuracy_percent=100*statistics.mean(r[setting] for r in plotted),
            nonadversarial_correct=sum(r[setting] for r in plotted),nonadversarial_total=len(plotted),
            all_answer_accuracy_percent=100*statistics.mean(r[setting] for r in records),
            all_correct=sum(r[setting] for r in records),all_total=len(records),
            reference_agreement_percent=100*statistics.mean(r[setting]==r['reference_correct'] for r in audited),
            reference_agreed=sum(r[setting]==r['reference_correct'] for r in audited),reference_total=len(audited),
            categories={str(cat):dict(total=sum(r['category']==cat for r in records),
                correct=sum(r[setting] for r in records if r['category']==cat)) for cat in range(1,6)})
    transitions={}
    for a,b in [('L0_V1','L0_V3'),('L0_V1','L1_V1'),('L0_V3','L1_V3')]:
        flips=[r for r in records if r[a]!=r[b]]
        audited=[r for r in flips if r['reference_correct'] is not None]
        transitions[a+'__'+b]=dict(changed=len(flips),false_to_true=sum(not r[a] and r[b] for r in flips),
            true_to_false=sum(r[a] and not r[b] for r in flips),
            improved_vs_reference=sum(r[b]==r['reference_correct'] for r in audited),
            worsened_vs_reference=sum(r[b]!=r['reference_correct'] for r in audited),
            reference_comparable_changes=len(audited),question_ids=[r['question_id'] for r in flips])
    for path,expected in manifest['baseline_artifacts_sha256'].items():
        assert sha(path)==expected, 'Source changed: '+path
    metrics=dict(generated_utc=utc_now(),settings=summary,f1_percent=f1,plot_questions=len(plotted),
        questions=len(records),reference_questions=len(reference),total_calls=len(calls),unique_calls_per_answer=len(calls)/len(records),
        input_tokens=sum(c['result']['input_tokens'] for c in calls),output_tokens=sum(c['result']['output_tokens'] for c in calls),
        summed_worker_inference_seconds=sum(c['result']['amortized_seconds'] for c in calls),
        transitions=transitions,validation=dict(prompt_and_dependency_checks=len(calls),baseline_artifacts_unchanged=True),
        limitations=manifest['limitations'])
    atomic_json(OUT/'metrics.json',metrics)
    with (OUT/'per_answer.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'axes.titleweight':'bold','savefig.facecolor':'white'})
    def chart(name,x,series,xlabel,title,subtitle):
        fig,ax=plt.subplots(figsize=(8.4,5.3))
        colors=['#136FAD','#D46A32']
        values=[f1]
        for (label,keys),color in zip(series,colors):
            ys=[summary[k]['answer_accuracy_percent'] for k in keys];values+=ys
            ax.plot(x,ys,'o-',color=color,lw=2.5,ms=8,label=label)
            for xx,yy in zip(x,ys):
                ax.annotate(f'{yy:.2f}%',(xx,yy),xytext=(0,9 if color==colors[0] else -18),textcoords='offset points',ha='center',color=color)
        ax.axhline(f1,color='#63717D',ls='--',lw=1.8,label=f'Fixed token F1: {f1:.2f}')
        ax.set_xticks(x);ax.set_xlabel(xlabel);ax.set_ylabel('Score (0–100)')
        ax.set_ylim(max(0,min(values)-8),min(100,max(values)+10));ax.set_xlim(min(x)-.15,max(x)+.15)
        ax.grid(axis='y',alpha=.18);ax.legend(loc='best',frameon=False,fontsize=10)
        ax.set_title(title,pad=32,loc='left')
        ax.text(0,1.035,subtitle,transform=ax.transAxes,color='#536170',fontsize=10)
        fig.text(.02,.035,'Same 1,387 non-adversarial saved answers. Accuracy = fraction marked correct by the judge.\nF1 and judged accuracy measure different things; this plot does not measure judge reliability.',fontsize=9,color='#536170')
        fig.tight_layout(rect=(0,.10,1,1))
        fig.savefig(OUT/(name+'.png'),dpi=180);fig.savefig(OUT/(name+'.pdf'));plt.close(fig)
    chart('voting_curve',[1,3],[('Layer 0',['L0_V1','L0_V3']),('Layer 1',['L1_V1','L1_V3'])],
          'Votes per layer','Voting: one versus three judgments','Llama-3.1-8B-Instruct · layer 0 uses 1 / 3 calls; layer 1 uses 2 / 6 calls')
    chart('layering_curve',[0,1],[('One vote per layer',['L0_V1','L1_V1']),('Three votes per layer',['L0_V3','L1_V3'])],
          'Review layer','Layering: initial judgment versus one review','Same judge and rubric · layer 0 = initial grading; layer 1 = meta-review')
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    for label,keys,color in [('One vote',['L0_V1','L1_V1'],'#136FAD'),('Three votes',['L0_V3','L1_V3'],'#D46A32')]:
        x=[CALLS[k] for k in keys];y=[summary[k]['reference_agreement_percent'] for k in keys]
        ax.plot(x,y,'o-',lw=2.5,color=color,label=label)
        for xx,yy in zip(x,y):
            ax.annotate(f'{yy:.2f}%',(xx,yy),xytext=(0,9),textcoords='offset points',ha='center',color=color)
    ax.set_xlabel('Cumulative judge calls per answer');ax.set_ylabel('Agreement with Codex audit (%)')
    ax.set_title('Judge reliability against the existing audit',loc='left',pad=15)
    ax.set_xticks([1,2,3,6]);ax.grid(axis='y',alpha=.18);ax.legend(frameon=False)
    ax.set_ylim(0,100)
    fig.text(.02,.02,'1,231 previously reviewed answers. Codex labels are a provisional reference, not human ground truth.',fontsize=9,color='#536170')
    fig.tight_layout(rect=(0,.06,1,1));fig.savefig(OUT/'reference_agreement.png',dpi=180);fig.savefig(OUT/'reference_agreement.pdf');plt.close(fig)
    lines=['# LLM judge voting and layering — 12 September 2026','',
        f'Completed {len(calls):,} unique calls over {len(records):,} saved answers using pinned Llama-3.1-8B-Instruct. No layer 2 was run.',
        '',f'Plots compare judged answer accuracy with fixed F1 {f1:.2f}, on the same {len(plotted):,} non-adversarial answers.',
        '', '| Setting | Calls/answer | Judged accuracy, non-adversarial | Judged accuracy, all answers | Agreement with Codex audit |',
        '|---|---:|---:|---:|---:|']
    for setting,r in summary.items():
        lines.append(f"| {setting} | {CALLS[setting]} | {r['answer_accuracy_percent']:.2f}% | {r['all_answer_accuracy_percent']:.2f}% | {r['reference_agreement_percent']:.2f}% |")
    lines+=['','![Voting](voting_curve.png)','','![Layering](layering_curve.png)','','![Reference agreement](reference_agreement.png)',
        '', 'Actual experiment cost is seven unique calls per answer: three initial votes, one review of the first vote, and three reviews of all initial votes. Reusing initial votes gives all four settings with 12,544 calls.',
        '', 'The 405 adversarial cases have no comparable F1 and are included in the all-answer column only. One original QA execution failure is preserved separately; it has no generated answer to judge. Conv-41 has 193 unrun questions.',
        '', 'The audit reference comprises 1,231 earlier Codex judgments, not human annotations. Some audit decisions use source evidence/date anchors missing from the judge inputs. Agreement is provisional, and a higher fraction marked correct does not by itself mean more reliable judging.',
        '', 'All prompts, dependencies, output booleans, counts, and frozen source-answer hashes were verified. Raw calls and per-answer verdicts are retained in this directory.']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    marker='## 2026-09-12 — Judge voting and layer-1 experiment (measured)'
    obs=ROOT/'results/further_work/observations.md'
    entry='\n\n'+marker+'\n\n'+lines[2]+' Fixed non-adversarial F1: '+f'{f1:.2f}'+'.\n\n'
    entry+='; '.join(f"{k}: judged accuracy {r['answer_accuracy_percent']:.2f}%, Codex-reference agreement {r['reference_agreement_percent']:.2f}%" for k,r in summary.items())+'.\n\n'
    entry+='Reference is the earlier Codex source-based audit, not human truth. Full results, plots, prompts, raw calls, failure accounting and validation: results/insights/judge_scaling_20260912_v1/. No answer generation, memory ingestion, or layer 2 was run.\n'
    if marker not in obs.read_text():
        with obs.open('a') as h:h.write(entry)
    (OUT/'observation_entry.md').write_text(entry)
    print(json.dumps({k:v for k,v in metrics.items() if k not in ['transitions']},indent=2),flush=True)


if __name__=='__main__':
    rc=1
    try:
        main();rc=0
    except BaseException:
        traceback.print_exc();raise
    finally:
        atomic_json(OPS/'finalize_exit.json',dict(exit_code=rc,time_utc=utc_now()))
