"""Read saved artifacts only. Scores never invoke an LLM judge."""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from .data import load_dataset, turns
from .storage import atomic_json, read_jsonl, utc_now

LABELS = {1:'Multi-hop',2:'Temporal',3:'Open-domain',4:'Single-hop',5:'Adversarial'}


def call_attempts(path):
    events=read_jsonl(path)
    key=lambda c:(c['call_id'],c.get('attempt'),c.get('time_utc'),c.get('worker_pid'))
    finished=[c for c in events if c.get('event')=='call_complete']
    done={key(c) for c in finished}
    unfinished=[dict(c,status='interrupted_unknown',output_tokens=None,latency_seconds=None)
                for c in events if c.get('event')=='call_started' and key(c) not in done]
    return finished+unfinished


def summarize(root):
    config = json.loads((root/'configs/experiment.json').read_text())
    raw = root/'results/raw_results'/config['run_id']
    out = root/'results/insights'/config['run_id']
    vis, report = out/'visualisations', out/'report'
    vis.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    samples = load_dataset(root/config['dataset'], config['dataset_sha256'])
    records, calls, growth, memories = [], [], [], []
    for sample in samples:
        folder = raw/'conversations'/sample['sample_id']
        ordered = turns(sample)
        positions = {t['dia_id']:i for i,t in enumerate(ordered)}
        answers = {a['question_index']:a for a in read_jsonl(folder/'answers.jsonl')}
        for qi, qa in enumerate(sample['qa']):
            a = answers.get(qi)
            if a is None: continue  # Partial metrics are labelled and report completion denominators.
            a = dict(a)
            positions_present = [positions[x] for x in a['evidence_ids'] if x in positions]
            a['evidence_age_turns'] = len(ordered)-1-min(positions_present) if positions_present else None
            a['evidence_span_turns'] = max(positions_present)-min(positions_present) if positions_present else None
            a['category_name'] = LABELS[a['category']]
            records.append(a)
        calls.extend(call_attempts(folder/'calls.jsonl'))
        events = {e['event_id']:e for e in read_jsonl(folder/'memory_events.jsonl')}
        growth.extend(dict(sample_id=sample['sample_id'], note_number=e['note_number'],
                           latency_seconds=e['latency_seconds'], **e['footprint']) for e in events.values())
        p=folder/'memory.json'
        if p.exists():
            state=json.loads(p.read_text())
            memories.append(dict(sample_id=sample['sample_id'],notes=len(state['notes']),
                 directed_edges=sum(len(n['links']) for n in state['notes']),
                 notes_json_bytes=len(json.dumps(state['notes'],ensure_ascii=False).encode()),
                 index_documents_json_bytes=len(json.dumps(state['index_documents'],ensure_ascii=False).encode()),
                 embedding_float32_bytes=np.asarray(state['embeddings'],dtype=np.float32).nbytes,
                 checkpoint_json_bytes=p.stat().st_size,build_seconds=state['build_seconds']))
    rows=pd.DataFrame(records)
    complete=(len(records)==1986 and sum(m['notes'] for m in memories)==5882
              and all((raw/'conversations'/s['sample_id']/'complete.json').exists() for s in samples))
    metric={'run_id':config['run_id'],'generated_utc':utc_now(),'status':'complete' if complete else 'partial',
            'questions_scored':len(records),'questions_expected':1986,'notes_built':sum(m['notes'] for m in memories),
            'notes_expected':5882,'categories':{},'memory_footprint':memories,
            'definitions':{'quality':'Released LoCoMo F1 for categories 1-4; generative abstention accuracy for 5.',
               'evidence_coverage':'Fraction of unique annotated turn IDs supplied in answer context; missing evidence is null.',
               'latency':'Memory build excludes model load/checkpoint writes; QA includes rewrite/retrieval/answer. Resumed cached-call reconstruction can undercount these operation timers; all completed call attempts are separately summed.',
               'tokens':'All completed attempts, including retries. Crashed calls with unknown output token counts are identified.',
               'context_overflow':'Input tokens + 1000 reserved output tokens exceed 131072; never truncate.',
               'footprint':'UTF-8 JSON note/index sizes plus float32 embedding payload; excludes model weights and archival logs.'}}
    for category,name in LABELS.items():
        selected=[a for a in records if a['category']==category]
        expected=sum(q['category']==category for s in samples for q in s['qa'])
        metric['categories'][name]={'scored':len(selected),'expected':expected,
            'mean_score':float(np.mean([a['score'] for a in selected])) if selected else None,
            'failed_questions':sum(a['status']!='ok' for a in selected)}
    with_evidence=[a for a in records if a['evidence_coverage'] is not None]
    metric['evidence']={'annotated_questions_scored':len(with_evidence),
            'mean_coverage':float(np.mean([a['evidence_coverage'] for a in with_evidence])) if with_evidence else None,
            'all_evidence_rate':float(np.mean([a['all_evidence_retrieved'] for a in with_evidence])) if with_evidence else None,
            'invalid_reference_questions':sum(bool(a['invalid_evidence_ids']) for a in records)}
    metric['runtime']={'build_worker_seconds':sum(m['build_seconds'] for m in memories),
            'qa_median_seconds':float(np.median([a['latency_seconds'] for a in records])) if records else None,
            'qa_p95_seconds':float(np.percentile([a['latency_seconds'] for a in records],95)) if records else None,
            'all_completed_call_seconds':sum(c.get('latency_seconds') or 0 for c in calls),
            'all_completed_prefill_seconds':sum(c.get('prefill_seconds',0) for c in calls)}
    metric['calls_by_phase']={}
    for phase in ['note','evolution','rewrite','answer']:
        selected=[c for c in calls if c['phase']==phase]
        metric['calls_by_phase'][phase]={'completed_attempts':sum(c['status']!='interrupted_unknown' for c in selected),
            'interrupted_without_completion':sum(c['status']=='interrupted_unknown' for c in selected),
            'operational_interrupts':sum(c['status']=='operational_interrupt' for c in selected),
            'input_tokens':sum(c['input_tokens'] for c in selected),
            'output_tokens_known':sum(c.get('output_tokens') or 0 for c in selected),
            'unknown_output_attempts':sum(c.get('output_tokens') is None for c in selected),
            'retry_attempts':sum(c.get('attempt',0)>0 for c in selected),
            'schema_errors':sum(c['status']=='schema_error' for c in selected),
            'backend_errors':sum(c['status']=='backend_error' for c in selected),
            'context_overflows':sum(c['context_overflow'] for c in selected),
            'max_input_tokens':max((c['input_tokens'] for c in selected),default=0),
            'p95_context_utilization':float(np.percentile([(c['input_tokens']+c['output_reserve'])/c['max_context']
                                                        for c in selected],95)) if selected else None}
    atomic_json(out/'metrics.json',metric)
    atomic_json(out/'question_traces.json',records)
    pd.DataFrame(growth).to_csv(out/'memory_growth.csv',index=False)
    if records:
        flat=rows.drop(columns=['retrieval','call_ids'],errors='ignore')
        flat.to_csv(out/'questions.csv',index=False)
        flat['raw_trace_path'] = flat.sample_id.map(lambda sid: str(raw/'conversations'/sid/'calls.jsonl'))
        flat[flat.score<.5].sort_values(['category','score']).to_csv(out/'failure_cases.csv',index=False)
    sns.set_theme(style='whitegrid',font_scale=.9)
    if records:
        means=rows.pivot_table(index='sample_id',columns='category_name',values='score',aggfunc='mean')
        counts=rows.pivot_table(index='sample_id',columns='category_name',values='score',aggfunc='count')
        annotation=means.copy().astype(object)
        for i in means.index:
            for j in means.columns:
                annotation.loc[i,j]=f'{means.loc[i,j]:.2f}\n(n={int(counts.loc[i,j])})' if pd.notna(means.loc[i,j]) else ''
        fig,ax=plt.subplots(figsize=(9,7))
        sns.heatmap(means,annot=annotation,fmt='',vmin=0,vmax=1,cmap='viridis',ax=ax)
        ax.set(title='Answer quality by conversation and category',xlabel='',ylabel='Conversation')
        fig.tight_layout();fig.savefig(vis/'category_conversation.png',dpi=180);plt.close(fig)
        subset=rows[(rows.category!=5)&rows.evidence_coverage.notna()].copy()
        if not subset.empty:
            subset['Evidence supplied']=pd.cut(subset.evidence_coverage,[-.01,0,.9999,1],labels=['None','Some','All'])
            fig,ax=plt.subplots(figsize=(9,5))
            sns.boxplot(data=subset,x='Evidence supplied',y='score',hue='category_name',ax=ax,fliersize=2)
            ax.set(title='Retrieved evidence versus answer F1',ylabel='F1 score');ax.legend(title='Category')
            fig.tight_layout();fig.savefig(vis/'evidence_quality.png',dpi=180);plt.close(fig)
            fig,axes=plt.subplots(2,4,figsize=(9,7),sharey=True)
            for column,category in enumerate([1,2,3,4]):
                sub=subset[subset.category==category]
                for row,x in enumerate(['evidence_age_turns','evidence_span_turns']):
                    ax=axes[row,column]
                    ax.scatter(sub[x],sub.score,alpha=.4,s=12)
                    ax.set(title=LABELS[category] if row==0 else '',xlabel='Age (turns)' if row==0 else 'Span (turns)',ylabel='F1')
                    ax.tick_params(labelsize=7)
            fig.tight_layout();fig.savefig(vis/'evidence_distance.png',dpi=180);plt.close(fig)
    if growth:
        g=pd.DataFrame(growth);fig,axes=plt.subplots(1,2,figsize=(12,5))
        for sid,sub in g.groupby('sample_id'):
            sub=sub.sort_values('note_number')
            memory_mb=(sub.notes_json_bytes+sub.index_documents_json_bytes+sub.embedding_float32_bytes)/1e6
            axes[0].plot(sub.note_number,memory_mb,label=sid)
            axes[1].plot(sub.note_number,sub.latency_seconds.cumsum()/3600,label=sid)
        axes[0].set(xlabel='Notes added',ylabel='Memory payload (MB)',title='Memory growth')
        axes[1].set(xlabel='Notes added',ylabel='Build worker hours',title='Cumulative build cost')
        axes[1].legend(fontsize=7);fig.tight_layout();fig.savefig(vis/'memory_cost.png',dpi=180);plt.close(fig)
    write_report(report,vis,metric,records,root)
    print(json.dumps(metric,indent=2),flush=True)
    return metric


def tex_escape(value):
    mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}'}
    return ''.join(mapping.get(c,c) for c in str(value))


def write_report(report,vis,m,records,root):
    status=m['status'].upper()
    table='\n'.join(f"{name} & {r['scored']}/{r['expected']} & " +
                     (f"{r['mean_score']:.3f}" if r['mean_score'] is not None else '--') + r' \\'
                     for name,r in m['categories'].items())
    failures=sorted((a for a in records if a['score']<.5),key=lambda a:(a['score'],a['question_id']))[:8]
    examples='\n'.join(r'\item '+tex_escape(a['question_id']+': '+a['question'])+
                       ' Gold: '+tex_escape(a['gold_answer'])+'. Output: '+tex_escape(a['prediction'])+
                       f". Score {a['score']:.3f}; evidence coverage {a['evidence_coverage']}." for a in failures)
    fmt=lambda x: '--' if x is None else f'{x:.3f}'
    costs='\n'.join(tex_escape(phase)+f" & {c['input_tokens']:,} & {c['output_tokens_known']:,} & {c['context_overflows']}"+r' \\'
                    for phase,c in m['calls_by_phase'].items())
    runtime=m['runtime']; evidence=m['evidence']
    metrics_text=(f"Mean annotated-evidence coverage is {fmt(evidence['mean_coverage'])}; the all-evidence rate is "
        f"{fmt(evidence['all_evidence_rate'])} across {evidence['annotated_questions_scored']} annotated questions. "
        f"{evidence['invalid_reference_questions']} questions contain invalid evidence references. "
        f"Cumulative memory-build worker time is {runtime['build_worker_seconds']/3600:.3f} hours. "
        f"QA median/p95 latency is {fmt(runtime['qa_median_seconds'])}/{fmt(runtime['qa_p95_seconds'])} seconds. "
        f"Memory payload across the conversations is {sum(n['notes_json_bytes']+n['index_documents_json_bytes']+n['embedding_float32_bytes'] for n in m['memory_footprint'])/1e6:.3f} MB. "
        f"There are {sum(c['schema_errors'] for c in m['calls_by_phase'].values())} schema failures and "
        f"{sum(c['backend_errors'] for c in m['calls_by_phase'].values())} recorded backend failures.")
    figures='\n'.join(r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{../visualisations/'+p.name+
                      r'}\caption{'+tex_escape(p.stem.replace('_',' '))+r'.}\end{figure}' for p in sorted(vis.glob('*.png')))
    source=r'''\documentclass{article}
\usepackage[preprint,nonatbib]{neurips_2026}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{booktabs,graphicx,url}
\title{A-Mem on LoCoMo with NVIDIA CUDA Llama-3.2-3B: An Experimental Audit}
\author{Ashutosh Ganguly}
\begin{document}\maketitle
\begin{abstract}
This report records a single-model, single-benchmark audit of A-Mem. Status: STATUS.
We have scored NQUEST of 1,986 questions and built NNOTES of 5,882 notes.
The analysis separates answer quality, evidence retrieval, runtime, tokens, memory footprint and context overflow.
A single run cannot establish an improvement over a baseline or identify causal failure mechanisms.
\end{abstract}
\section{Experimental protocol}
We use all ten LoCoMo conversations. Each dialogue turn becomes one note, in chronological order.
We exclude images, image captions, question text, gold answers, evidence annotations and supplied summaries from memory construction.
Llama-3.2-3B-Instruct performs note enrichment, evolution, query keyword generation and answering.
Four independent logical shards use Slurm-allocated NVIDIA RTX 5000 Ada GPUs within the account quota; mutable conversation memory is never shared.
The 131,072-token window includes a 1,000-token output reserve. Temperature is 0.7 and the run seed is 42.
All language-model forwards use CUDA BF16 with PyTorch SDPA, preserving full context through 4,096-token prefill chunks. Host MiniLM embeddings, tokenization and sampling are allowed. CPU language-model fallback is forbidden.
\section{Fidelity and implementation differences}
We preserve released A-Mem prompts and directed integer links, write retrieval of five neighbors, and the released expanded-retrieval loop (including repeated notes).
Read retrieval uses a fixed ten neighbors. The vector index refreshes existing metadata every 100 positive evolution decisions.
The backend uses BF16 CUDA precision and local JSON-constrained sampling. Its numeric behavior differs from the earlier TT backend, whose results are excluded. The Python hash seed is fixed at 42.
Constraints restrict action names and connection indices to valid supplied neighbors. JSON generation stops when a complete schema-valid object is obtained.
Malformed output is retained and retried at most three times. A failed memory write stops that worker; a failed QA receives zero and remains in the denominator.
Adversarial questions use generative abstention without exposing distractor answers, differing from the authors' multiple-choice evaluation path.
This is a documented adaptation, not an exact numerical replication of the paper's model conditions.
\section{Metrics and results}
Scores for categories 1--4 use released LoCoMo stemmed F1; adversarial scores use the released abstention-string rule.
Missing evidence annotations are excluded from evidence-coverage means and retained as missing values.
Partial results describe only questions processed so far, with completed and expected counts shown.
\begin{center}\begin{tabular}{lrr}\toprule Category & Scored/expected & Score\\\midrule
TABLE
\bottomrule\end{tabular}\end{center}
The machine-readable metrics and complete per-question traces are stored alongside this report.
METRICSTEXT
\begin{center}\begin{tabular}{lrrr}\toprule Phase & Input tokens & Output tokens & Overflows\\\midrule
COSTTABLE
\bottomrule\end{tabular}\end{center}
\section{Diagnostic views}
FIGURES
\section{Failure examples and research leads}
Examples below are selected by low automatic score and question ID. They require manual semantic review: lexical F1 can penalize a correct paraphrase.
\begin{itemize}EXAMPLES\end{itemize}
Untested hypotheses include stale embedding metadata, duplicate expanded context, temporal ambiguity, and imperfect neighbor selection.
Retrieval evidence coverage alone cannot establish that generation or evolution caused an error. Controlled interventions are required before claiming a research contribution.
\section{Limitations and reproducibility}
One stochastic run and one model cannot establish statistical robustness or a benefit over a no-memory baseline.
Image-dependent questions remain in this text-only benchmark evaluation. Category-aware prompts and the generative adversarial protocol must be considered when comparing external scores.
The raw-results manifest pins dataset/model revisions, prompts, code, dependencies, assignments and device evidence.
Prompt logs contain every completed response and its validation status; interrupted CUDA calls may have unknown generated-token counts.
Resume checkpoints preserve both evolving notes and stale embedding state. Archived call traces allow successful requests to be reused after an interruption.
\begin{thebibliography}{9}
\bibitem{amem} Xu et al. A-Mem: Agentic Memory for LLM Agents. Author code: \url{https://github.com/WujiangXu/A-mem}.
\bibitem{locomo} Maharana et al. Evaluating Very Long-Term Conversational Memory of LLM Agents. LoCoMo: \url{https://github.com/snap-research/locomo}.
\end{thebibliography}
\end{document}
'''
    for key,value in {'STATUS':status,'NQUEST':str(m['questions_scored']),'NNOTES':str(m['notes_built']),
                      'COSTTABLE':costs,'METRICSTEXT':metrics_text,'TABLE':table,'FIGURES':figures,'EXAMPLES':examples or r'\item No low-scoring completed examples are available yet.'}.items():
        source=source.replace(key,value)
    (report/'report.tex').write_text(source)
    shutil.copy2(root/'benchmark/report_template/neurips_2026.sty',report/'neurips_2026.sty')
    compiler=shutil.which('tectonic') or (str(root/'tools/tectonic') if (root/'tools/tectonic').exists() else None)
    if compiler:
        proc=subprocess.run([compiler,'report.tex'],cwd=report,capture_output=True,text=True)
        (report/'compile.log').write_text(proc.stdout+proc.stderr)
        if proc.returncode: raise RuntimeError('Report compilation failed; inspect compile.log')
    else:
        (report/'PDF_PENDING.md').write_text('LaTeX source is ready. Install/run Tectonic or pdflatex to compile, then visually inspect every page before calling the report final.\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    summarize(parser.parse_args().root.resolve())
