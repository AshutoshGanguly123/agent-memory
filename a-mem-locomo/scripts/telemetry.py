"""Read only our allocated CUDA GPU and owned worker tree every 30 seconds."""
import argparse,json,os,subprocess,time
from pathlib import Path
import psutil
from amem_reproduction.storage import append_jsonl,utc_now
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--worker',required=True);p.add_argument('--mode',required=True);a=p.parse_args()
c=json.loads((root/'configs/experiment.json').read_text())
kind='preflight' if a.mode=='preflight' else 'raw_results'
status=root/'results'/kind/c['run_id']/f'worker_{a.worker}.json'
out=root/'results/operations'/f'job_{a.job}.telemetry.jsonl'
end=root/'results/operations'/f'job_{a.job}.exit.json'
while not end.exists() and not (root/'results/operations'/f'job_{a.job}.telemetry.stop').exists():
    record=dict(time_utc=utc_now(),job=a.job,node=os.uname().nodename,host_memory_available_bytes=psutil.virtual_memory().available)
    try:
        s=json.loads(status.read_text()) if status.exists() else {}
        if s.get('slurm_job_id')==a.job:
            proc=psutil.Process(s['pid'])
            if proc.uids().real!=os.getuid():raise RuntimeError('Worker ownership mismatch')
            record.update(worker_pid=proc.pid,worker_rss_bytes=proc.memory_info().rss,worker_cpu_times=proc.cpu_times()._asdict(),phase=s.get('phase'))
            uuid=s.get('device',{}).get('uuid')
            if uuid:
                q=subprocess.run(['nvidia-smi','-i',uuid,'--query-gpu=uuid,memory.used,memory.total,utilization.gpu,utilization.memory,power.draw,temperature.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
                record.update(gpu_csv=q.stdout.strip(),gpu_query_error=q.stderr.strip() if q.returncode else None)
    except (OSError,ValueError,psutil.Error,subprocess.TimeoutExpired) as exc:record['sample_error']=repr(exc)
    append_jsonl(out,record)
    time.sleep(30)
