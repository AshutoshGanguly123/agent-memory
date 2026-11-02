"""Persistent tmux Slurm coordinator: one GPU quota, four isolated shards."""
import argparse,fcntl,json,os,subprocess,time
from pathlib import Path
from amem_reproduction.provenance import freeze,verify_manifest
from amem_reproduction.storage import atomic_json,append_jsonl,utc_now

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'configs/experiment.json').read_text())
OPS=ROOT/'results/operations'

def queue(job):
    p=subprocess.run(['squeue','--noheader','--jobs',job,'--format=%T|%R'],capture_output=True,text=True)
    if p.returncode:raise RuntimeError('Cannot read Slurm job status: '+p.stderr)
    return p.stdout.strip()

def submit(mode,worker):
    args=['sbatch','--parsable','--job-name',f'amem-{mode}-{worker}',
          '--output',str(OPS/'slurm_%j.log'),'--chdir',str(ROOT),
          '--export',f'ALL,AMEM_MODE={mode},AMEM_WORKER={worker},AMEM_PROJECT_ROOT={ROOT},AMEM_LAUNCHED_FROM_TMUX=1',str(ROOT/'scripts/slurm_job.sh')]
    p=subprocess.run(args,capture_output=True,text=True)
    if p.returncode:raise RuntimeError('Slurm submission failed: '+p.stderr)
    return p.stdout.strip().split(';')[0]

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['preflight','full']);a=p.parse_args()
    if not os.environ.get('TMUX'):raise RuntimeError('Coordinator requires tmux')
    OPS.mkdir(parents=True,exist_ok=True)
    lock=(OPS/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OPS/f'{a.mode}_coordinator.json'
    state=json.loads(path.read_text()) if path.exists() else dict(mode=a.mode,created_utc=utc_now(),job=None,worker=0,cycles=0)
    if state.get('state')=='failed':raise RuntimeError('Prior failure requires inspection; archive coordinator state before intentional retry')
    manifest=freeze(ROOT) if a.mode=='full' else None
    def save(**values):
        state.update(values,updated_utc=utc_now());atomic_json(path,state)
    try:
        while True:
            if not state['job']:
                if a.mode=='full':
                    verify_manifest(ROOT)
                    out=ROOT/'results/raw_results'/CONFIG['run_id']
                    remaining=[i for i in CONFIG['worker_ids'] if not all((out/'conversations'/s/'complete.json').exists() for s in manifest['assignments'][str(i)])]
                    if not remaining:break
                    worker=next((i for i in remaining if i>=state['worker']),remaining[0])
                else:worker=0
                job=submit(a.mode,worker)
                save(job=job,worker=worker,cycles=state['cycles']+1,state='submitted')
                append_jsonl(OPS/'scheduler_events.jsonl',dict(event='submitted',mode=a.mode,worker=worker,job=job,time_utc=utc_now()))
            job=state['job']
            exit_path=OPS/f'job_{job}.exit.json'
            current=queue(job)
            save(state='running' if current.startswith('RUNNING') else 'waiting',slurm=current)
            if current:
                time.sleep(30);continue
            if not exit_path.exists():
                raise RuntimeError(f'Slurm job {job} disappeared without owned exit record; inspect scheduler/preemption/OOM before retry')
            rc=int(json.loads(exit_path.read_text())['exit_code'])
            append_jsonl(OPS/'scheduler_events.jsonl',dict(event='finished',job=job,worker=state['worker'],exit_code=rc,time_utc=utc_now()))
            if rc==78:
                save(job=None,state='waiting_for_physical_gpu',retry_after_seconds=900)
                time.sleep(900)
                continue
            if rc not in [0,75]:raise RuntimeError(f'Worker job {job} failed with {rc}; no blind restart')
            if a.mode=='preflight':
                if rc:raise RuntimeError('Preflight stopped early; inspect')
                break
            save(job=None,worker=(state['worker']+1)%4,state='between_slices')
        save(state='complete',finished_utc=utc_now())
        if a.mode=='full':
            result=subprocess.run([str(ROOT/'.venv/bin/python'),'-m','amem_reproduction.analyze','--root',str(ROOT)],cwd=ROOT)
            atomic_json(OPS/'analysis_exit.json',dict(exit_code=result.returncode,time_utc=utc_now()))
            if result.returncode:raise RuntimeError('Analysis failed')
    except BaseException as exc:
        save(state='failed',error=repr(exc));raise

if __name__=='__main__':main()
