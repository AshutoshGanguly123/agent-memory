"""Node scheduling overlay; the frozen four-conversation recovery is unchanged."""
import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path
from amem_reproduction.storage import file_digest

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
ADAPTER_OPS = ROOT / 'results/operations/recovery_gpu_nodes_v3'
PREVIOUS_OPS = ROOT / 'results/operations/recovery_gpu_nodes_v2'
spec = importlib.util.spec_from_file_location('recovery_parent_common', ROOT / 'operations/recovery_output2048_v1/common.py')
parent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parent)
for name in ('OPS', 'OUT', 'CONFIG', 'TARGETS', 'legacy', 'load_module', 'inventory', 'lock',
             'claim', 'pending', 'atomic_stop', 'schema_exhaustion', 'require_target',
             'assert_baseline_unchanged', 'assert_lineage_unchanged'):
    globals()[name] = getattr(parent, name)


def node_name():
    node = os.uname().nodename.split('.')[0]
    if not re.fullmatch(r'gpu-\d{2}', node):
        raise RuntimeError('User permits execution only on gpu-xx nodes: ' + node)
    return node


def parse_reservations(raw, node):
    if not re.search(r'\bNodeName=' + re.escape(node) + r'\s', raw):
        raise RuntimeError('Scheduler returned another node')
    state = re.search(r'\bState=(\S+)', raw)
    used = re.search(r'\bGresUsed=(\S+)', raw)
    drained = re.search(r'\bGresDrain=(\S+)', raw)
    if not state or not used or not drained:
        raise RuntimeError('Incomplete scheduler resource evidence')
    blocked = any(x in state[1] for x in ('DOWN', 'DRAIN', 'FAIL', 'MAINT', 'UNKNOWN', 'NOT_RESPONDING'))
    blocked = blocked or drained[1] not in ('N/A', '(null)')
    indices = set()
    # Each GPU resource must disclose physical indices, including zero allocations.
    resource = used[1]
    if resource not in ('N/A', '(null)'):
        matches = list(re.finditer(r'gpu(?::[^:,()]+)*:(\d+)\(IDX:([^)]*)\)', resource))
        if not matches or ''.join(m[0] for m in matches) != resource.replace('),gpu', ')gpu'):
            raise RuntimeError('Unrecognized scheduler GPU mapping: ' + resource)
        for match in matches:
            group = set()
            if match[2] not in ('', 'N/A'):
                for segment in match[2].split(','):
                    if not re.fullmatch(r'\d+(?:-\d+)?', segment):
                        raise RuntimeError('Invalid scheduler GPU index: ' + segment)
                    bounds = [int(x) for x in segment.split('-')]
                    if len(bounds) == 2 and bounds[1] < bounds[0]:
                        raise RuntimeError('Reversed GPU range')
                    group.update(range(bounds[0], bounds[-1] + 1))
            if len(group) != int(match[1]):
                raise RuntimeError('Scheduler GPU count/index mismatch')
            indices.update(group)
    return dict(node=node, reserved_indices=sorted(indices), blocked=blocked, raw=raw)


def scheduled_jobs():
    node = node_name()
    try:
        raw = subprocess.run(['scontrol', 'show', 'node', '-d', node], capture_output=True,
                             text=True, check=True, timeout=15).stdout
        return parse_reservations(raw, node)
    except Exception as exc:
        # Uncertain scheduler ownership prevents launches and yields our workers.
        return dict(node=node, reserved_indices=[], blocked=True, error=repr(exc))


def reserved(card, snapshot):
    return snapshot['blocked'] or card['index'] in snapshot['reserved_indices']


def validate_context():
    node_name()
    if not os.environ.get('TMUX') or os.environ.get('AMEM_EXECUTION_MODE') != 'direct_ssh':
        raise RuntimeError('Dedicated direct-SSH tmux launcher required')
    if os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Direct execution must not reuse a Slurm allocation')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible:
        from amem_reproduction.device_guard import DeviceBusy
        cards = [c for c in inventory() if c['uuid'] == visible]
        if len(cards) != 1:
            raise RuntimeError('Expected exactly one verified physical GPU UUID')
        if reserved(cards[0], scheduled_jobs()):
            raise DeviceBusy('GPU is reserved or scheduler availability is uncertain')


def source_hashes():
    return {p.name: file_digest(p) for p in sorted(CODE.glob('*.py'))}


def verify():
    result = parent.verify()
    manifest = json.loads((ADAPTER_OPS / 'manifest.json').read_text())
    if manifest['source_sha256'] != source_hashes():
        raise RuntimeError('Node scheduling overlay changed after freeze')
    if manifest['parent_manifest_sha256'] != file_digest(OPS / 'manifest.json'):
        raise RuntimeError('Parent recovery manifest changed')
    evidence = json.loads((PREVIOUS_OPS / 'replay_gpu3_diagnosis.json').read_text())
    reference_result(evidence)
    if manifest['reference_diagnosis_sha256'] != file_digest(PREVIOUS_OPS / 'replay_gpu3_diagnosis.json'):
        raise RuntimeError('Node replay reference changed after freeze')
    previous_manifest = json.loads((PREVIOUS_OPS / 'manifest.json').read_text())
    previous_hashes = {p.name: file_digest(p) for p in (ROOT / 'operations/recovery_gpu_nodes_v2').glob('*.py')}
    if previous_manifest['source_sha256'] != previous_hashes:
        raise RuntimeError('Historical node adapter was changed')
    return result

def reference_result(evidence):
    required = ('caps_equal', 'repeats_equal', 'previous_probes_equal', 'matches_previous_probe')
    if not all(evidence.get(k) is True for k in required):
        raise RuntimeError('Node replay/cap stability was not demonstrated')
    results = evidence['results']
    reference = results['cap2048']
    if not reference['parsed'] or reference['stop_reason'] != 'json_complete':
        raise RuntimeError('Node replay is not schema-valid JSON')
    for label in ('cap1000', 'cap2048_repeat'):
        if (results[label]['output_token_ids'] != reference['output_token_ids']
                or results[label]['parsed'] != reference['parsed']):
            raise RuntimeError('Replay comparison contradicts its claimed stability')
    return reference
