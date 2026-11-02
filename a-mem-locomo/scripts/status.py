import json
from pathlib import Path
from amem_reproduction.storage import read_jsonl,utc_now
root=Path(__file__).resolve().parents[1]
c=json.loads((root/'configs/experiment.json').read_text())
out=root/'results/raw_results'/c['run_id']
rows=[]
for p in sorted((out/'conversations').glob('*/memory.json')):
    m=json.loads(p.read_text())
    answers=read_jsonl(p.parent/'answers.jsonl')
    rows.append(dict(conversation=p.parent.name,notes=len(m['notes']),questions=len(answers),frozen=m['frozen'],complete=(p.parent/'complete.json').exists()))
s=root/'results/operations/full_coordinator.json'
print(json.dumps(dict(time_utc=utc_now(),run_id=c['run_id'],notes=sum(r['notes'] for r in rows),expected_notes=5882,
    questions=sum(r['questions'] for r in rows),expected_questions=1986,conversations=rows,
    scheduler=json.loads(s.read_text()) if s.exists() else None),indent=2))
