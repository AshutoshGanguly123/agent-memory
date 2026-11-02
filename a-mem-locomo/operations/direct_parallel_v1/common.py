"""Operational scheduling only; frozen scientific modules remain untouched."""
import fcntl
import json
import os
import subprocess
from pathlib import Path
from amem_reproduction.storage import atomic_json, file_digest
from amem_reproduction.provenance import verify_manifest

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OPS = ROOT / 'results/operations/direct_parallel_v1'
CONFIG = json.loads((ROOT / 'configs/experiment.json').read_text())
OUT = ROOT / 'results/raw_results' / CONFIG['run_id']


def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def verify():
    original = verify_manifest(ROOT)
    operational = json.loads((OPS / 'manifest.json').read_text())
    if operational['original_manifest_sha256'] != file_digest(OUT / 'manifest.json'):
        raise RuntimeError('Original manifest changed')
    actual = {p.name: file_digest(p) for p in CODE.glob('*.py')}
    if actual != operational['adapter_source_sha256']:
        raise RuntimeError('Operational adapter changed after freeze')
    return original, operational


def inventory():
    command = ['nvidia-smi', '--query-gpu=index,uuid,name,memory.free,memory.total,memory.used,utilization.gpu',
               '--format=csv,noheader,nounits']
    output = subprocess.run(command, capture_output=True, text=True, check=True, timeout=15).stdout
    cards = []
    for line in output.splitlines():
        i, uuid, name, free, total, used, util = [v.strip() for v in line.split(',')]
        cards.append(dict(index=int(i), uuid=uuid, name=name, free_mib=float(free),
                          total_mib=float(total), used_mib=float(used), utilization=int(util), pids=[]))
    output = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, check=True, timeout=15).stdout
    by_uuid = {c['uuid']: c for c in cards}
    for line in output.splitlines():
        uuid, pid = [v.strip() for v in line.split(',')]
        if uuid in by_uuid and pid.isdigit():
            by_uuid[uuid]['pids'].append(int(pid))
    return cards


def scheduled_jobs():
    # Read-only: direct workers yield to scheduler-owned work on this node.
    result = subprocess.run(['squeue', '-h', '-w', 'gpu-04', '-t', 'RUNNING,COMPLETING,CONFIGURING',
                             '-o', '%i|%u'], capture_output=True, text=True, check=True, timeout=15)
    return result.stdout.splitlines()


def claim(order, out=OUT, ops=OPS):
    for sid in order:
        if (out / 'conversations' / sid / 'complete.json').exists():
            continue
        if (ops / 'failures' / (sid + '.json')).exists():
            continue
        handle = lock(ops / 'locks' / ('conversation_' + sid + '.lock'))
        if handle is None:
            continue
        if (out / 'conversations' / sid / 'complete.json').exists() or (ops / 'failures' / (sid + '.json')).exists():
            handle.close()
            continue
        return sid, handle
    return None, None


def pending(order):
    return [sid for sid in order if not (OUT / 'conversations' / sid / 'complete.json').exists()
            and not (OPS / 'failures' / (sid + '.json')).exists()]


def validate_context():
    if os.uname().nodename.split('.')[0] != 'gpu-04':
        raise RuntimeError('Direct execution is restricted to gpu-04')
    if not os.environ.get('TMUX') or os.environ.get('AMEM_EXECUTION_MODE') != 'direct_ssh':
        raise RuntimeError('Direct execution requires the dedicated tmux launcher')
    if os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Direct execution must not impersonate or reuse a Slurm allocation')


def atomic_stop(path, reason):
    atomic_json(path, {'reason': reason})
