"""Append operational provenance without changing the scientific freeze."""
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import atomic_json, file_hash, now, read_json, run_root, verify_freeze

NODE_CODE = r'''
import csv, io, json, os, pathlib, subprocess, sys
import psutil, torch, pynvml
spec = json.loads(sys.argv[1])
allowed = {'CUDA_VISIBLE_DEVICES','CUDA_DEVICE_ORDER','MEMALPHA_GPU_UUIDS','MEMALPHA_GPU_LEASE',
           'OMP_NUM_THREADS','PYTHONHASHSEED','TOKENIZERS_PARALLELISM','HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE',
           'NLTK_DATA','TIKTOKEN_CACHE_DIR','NCCL_DEBUG','NCCL_P2P_DISABLE','NCCL_IB_DISABLE',
           'NCCL_SOCKET_IFNAME','GLOO_SOCKET_IFNAME','VLLM_WORKER_MULTIPROC_METHOD','VLLM_USE_V1',
           'VLLM_CACHE_ROOT','TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR'}
processes = []
for pid in spec['child_pids']:
    try:
        proc = psutil.Process(pid)
        if proc.uids().real != os.getuid():
            raise RuntimeError('Process owner changed')
        environ = proc.environ()
        processes.append(dict(pid=pid, parent_pid=proc.ppid(), created_unix=proc.create_time(),
            command=proc.cmdline(), environment={k:environ.get(k) for k in sorted(allowed)},
            cpu_times=proc.cpu_times()._asdict(), rss_bytes=proc.memory_info().rss))
    except psutil.NoSuchProcess:
        processes.append(dict(pid=pid, state='already_exited'))
fields = ['index','uuid','name','pci.bus_id','driver_version','compute_cap','memory.total']
raw = subprocess.check_output(['nvidia-smi','--query-gpu='+','.join(fields),'--format=csv,noheader,nounits'],text=True)
cards = [dict(zip(fields,[x.strip() for x in row])) for row in csv.reader(io.StringIO(raw))]
gpu_processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader,nounits'],text=True)
lock_lines = pathlib.Path('/proc/locks').read_text().splitlines()
locks = []
for item in spec['locks']:
    inode = pathlib.Path(item['path']).stat().st_ino
    matches = [line for line in lock_lines if len(line.split()) >= 6 and
               line.split()[4] == str(item['launcher_pid']) and line.split()[5].split(':')[-1] == str(inode)]
    locks.append(dict(item, inode=inode, kernel_lock_records=matches))
pynvml.nvmlInit()
try:
    driver_cuda_api = pynvml.nvmlSystemGetCudaDriverVersion_v2()
finally:
    pynvml.nvmlShutdown()
assert not torch.cuda.is_initialized(), 'Provenance collection must not initialize a CUDA context'
print(json.dumps(dict(hostname=os.uname().nodename, gpu_cards=cards, gpu_processes=gpu_processes,
    processes=processes, gpu_locks=locks, torch_version=torch.__version__,
    torch_cuda_runtime=torch.version.cuda, torch_nccl_version=torch.cuda.nccl.version(),
    nvml_cuda_driver_api_version=driver_cuda_api, available_host_ram_bytes=psutil.virtual_memory().available,
    collector_initialized_cuda=False)))
'''


def main():
    if not os.environ.get('TMUX'):
        raise RuntimeError('Run provenance collection in server tmux')
    verify_freeze()
    freeze_path = run_root() / 'freeze.json'
    runtime_path = ROOT / 'results/setup/reader_runtime.json'
    runtime = read_json(runtime_path)
    freeze = read_json(freeze_path)
    if runtime.get('state') != 'validated' or runtime['method_sources'] != freeze['sources']:
        raise RuntimeError('Reader runtime evidence does not match the scientific freeze')
    operations = {p.parent.name: read_json(p) for p in sorted((ROOT / 'results/operations').glob('full_*/status.json'))}
    per_node = {}
    rank_evidence = {}
    for name, state in operations.items():
        if state.get('state') != 'running':
            continue
        if state['operations_sha256'] != file_hash(ROOT / 'ops.py'):
            raise RuntimeError('An admitted operation uses a different operations source')
        spec = per_node.setdefault(state['node'], {'child_pids': [], 'locks': []})
        spec['child_pids'].append(state['child_pid'])
        log_path = ROOT / 'results/operations' / name / 'attempts' / state['attempt'] / 'worker.log'
        if name.startswith('full_reader_') and log_path.exists():
            log_text = log_path.read_text(errors='replace')
            ranks = sorted({(int(rank), int(pid)) for rank, pid in re.findall(r'VllmWorker rank=(\d+) pid=(\d+)', log_text)})
            spec['child_pids'].extend(pid for _, pid in ranks)
            rank_evidence[name] = {'log_path': str(log_path.relative_to(ROOT)),
                                  'captured_log_sha256': file_hash(log_path),
                                  'rank_pids': [{'rank': rank, 'pid': pid} for rank, pid in ranks],
                                  'startup_lines': [line for line in log_text.splitlines() if any(marker in line for marker in ('Loading weights took', 'Model loading took', 'init engine (profile', 'using nccl==', 'Init COMPLETE'))]}
        for gpu in state['gpus']:
            path = ROOT.parent / 'A_Mem_reproduction/results/operations/direct_parallel_v1/locks' / f'gpu_{gpu}.lock'
            spec['locks'].append({'operation': name, 'gpu_uuid': gpu, 'path': str(path), 'launcher_pid': state['launcher_pid']})
    nodes = {}
    for node, spec in per_node.items():
        command = [str(ROOT / '.venv/bin/python'), '-c', NODE_CODE, json.dumps(spec)]
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', node, shlex.join(command)],
                                check=True, capture_output=True, text=True, timeout=90)
        nodes[node] = json.loads(result.stdout)
    for name, evidence in rank_evidence.items():
        node = nodes[operations[name]['node']]
        compute = [line.split(',') for line in node['gpu_processes'].splitlines() if line.strip()]
        for rank in evidence['rank_pids']:
            rank['live_gpu_records'] = [{'gpu_uuid': row[0].strip(), 'pid': int(row[1]), 'used_memory_mib': row[2].strip()} for row in compute if int(row[1]) == rank['pid']]
            rank['included_in_owned_pids'] = rank['pid'] in operations[name]['owned_pids']
    verify_freeze()
    record = {'recorded_utc': now(), 'scope': 'Supplemental operational provenance; scientific files unchanged',
              'scientific_freeze_sha256': file_hash(freeze_path),
              'operational_source_sha256': {name: file_hash(ROOT / name) for name in ('ops.py', 'pipeline.py', 'scripts/capture_runtime_provenance.py')},
              'reader_runtime_sha256': file_hash(runtime_path), 'validated_reader_environment': runtime['environment'],
              'reader_runtime_receipt': str(runtime_path.relative_to(ROOT)),
              'reader_consumption': 'pipeline.dispatch reads this receipt and supplies its whitelisted environment settings to the four-GPU reader command. Full reader is not dispatched until all 20 memories are frozen.',
              'environment_verification_sha256': file_hash(ROOT / 'results/setup/environment_verified.json'),
              'operations': operations, 'node_runtime': nodes,
              'reader_rank_evidence': rank_evidence,
              'pipeline_state': read_json(ROOT / 'results/pipeline/status.json')}
    path = ROOT / 'results/operations/provenance' / (now().replace(':','').replace('+','_') + '.json')
    atomic_json(path, record)
    print(json.dumps({'provenance': str(path), 'running_operations': list(operations), 'nodes': list(nodes)}))


if __name__ == '__main__':
    main()
