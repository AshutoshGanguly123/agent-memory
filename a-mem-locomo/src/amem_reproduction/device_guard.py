"""Read physical occupancy before allocating model memory on Slurm's GPU."""
import json,os,subprocess
from uuid import UUID


class DeviceBusy(RuntimeError): pass


def same_gpu(cuda_uuid,nvml_uuid):
    return UUID(str(cuda_uuid).lower().removeprefix('gpu-')) == UUID(str(nvml_uuid).lower().removeprefix('gpu-'))


def allocated_gpu():
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible or ',' in visible:raise RuntimeError('Expected one allocated GPU identifier')
    query=subprocess.run(['nvidia-smi','-i',visible,'--query-gpu=uuid,memory.free,memory.total',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20)
    if query.returncode:raise RuntimeError('Cannot verify allocated GPU: '+query.stderr)
    uuid,free,total=[s.strip() for s in query.stdout.strip().split(',')]
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
                        capture_output=True,text=True,timeout=20)
    if apps.returncode:raise RuntimeError('Cannot verify GPU ownership: '+apps.stderr)
    holders=[]
    for line in apps.stdout.splitlines():
        fields=[s.strip() for s in line.split(',')]
        if len(fields)==2 and fields[0]==uuid and fields[1].isdigit() and int(fields[1])!=os.getpid():
            holders.append(int(fields[1]))
    result=dict(visible=visible,uuid=uuid,free_mib=float(free),total_mib=float(total),other_compute_pids=holders)
    if holders or float(free)<28*1024:
        raise DeviceBusy('Allocated GPU is physically occupied or lacks 28 GiB free: '+json.dumps(result))
    return result
