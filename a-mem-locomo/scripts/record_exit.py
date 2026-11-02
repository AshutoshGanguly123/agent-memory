import sys
from pathlib import Path
from amem_reproduction.storage import atomic_json,utc_now
root=Path(__file__).resolve().parents[1]
job,code=sys.argv[1],int(sys.argv[2])
atomic_json(root/'results/operations'/f'job_{job}.exit.json',dict(job=job,exit_code=code,time_utc=utc_now()))
print(f'Slurm worker step {job} finished with exit {code}',flush=True)
