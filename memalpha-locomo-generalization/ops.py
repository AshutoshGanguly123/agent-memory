"""Direct gpu-xx/tmux operations with physical and scheduler GPU leases.

dispatch runs on the login node and selects gpu-04 then gpu-03. launch and
locked run on the selected node. No training, scheduler submissions, GPU resets,
or signals to processes outside this invocation's verified child tree.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
OPS = ROOT / 'results/operations'
SHARED_LOCKS = ROOT.parent / 'A_Mem_reproduction/results/operations/direct_parallel_v1/locks'
OPS_VERSION = '1.1'


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as out:
        json.dump(value, out, sort_keys=True, indent=2)
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def command(argv):
    return subprocess.run(argv, check=True, text=True, capture_output=True, timeout=30).stdout


def node_name():
    node = socket.gethostname().split('.')[0]
    if not re.fullmatch(r'gpu-\d{2}', node):
        raise RuntimeError('GPU execution is limited to gpu-xx nodes')
    return node


def parse_reservations(raw, node):
    if not re.search(r'\bNodeName=' + re.escape(node) + r'\s', raw):
        raise ValueError('Scheduler node identity mismatch')
    fields = {k: re.search(r'\b' + k + r'=(\S+)', raw) for k in ('State', 'GresUsed', 'GresDrain')}
    if not all(fields.values()):
        raise ValueError('Incomplete scheduler GPU evidence')
    if any(s in fields['State'][1] for s in ('DOWN', 'DRAIN', 'FAIL', 'MAINT', 'UNKNOWN', 'NOT_RESPONDING')):
        raise ValueError('Node is unavailable')
    if fields['GresDrain'][1] not in ('N/A', '(null)'):
        raise ValueError('Drained GPUs')
    used, result = fields['GresUsed'][1], set()
    if used in ('N/A', '(null)'):
        return result
    groups = list(re.finditer(r'gpu(?::[^:,()]+)*:(\d+)\(IDX:([^)]*)\)', used))
    if not groups or ''.join(g[0] for g in groups) != used.replace('),gpu', ')gpu'):
        raise ValueError('Unrecognized GPU reservation indices')
    for match in groups:
        indices = set()
        if match[2] not in ('', 'N/A'):
            for segment in match[2].split(','):
                if not re.fullmatch(r'\d+(?:-\d+)?', segment):
                    raise ValueError('Invalid GPU reservation range')
                bounds = [int(x) for x in segment.split('-')]
                if bounds[-1] < bounds[0]:
                    raise ValueError('Reversed GPU reservation range')
                indices.update(range(bounds[0], bounds[-1] + 1))
        if len(indices) != int(match[1]):
            raise ValueError('GPU reservation count/index mismatch')
        result.update(indices)
    return result


def inventory():
    import psutil
    node = node_name()
    reserved = parse_reservations(command(['scontrol', 'show', 'node', '-d', node]), node)
    rows = command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.free,memory.total,pci.bus_id',
                    '--format=csv,noheader,nounits'])
    cards = []
    for row in rows.splitlines():
        index, gpu, name, free, total, pci = [x.strip() for x in row.split(',')]
        cards.append(dict(index=int(index), uuid=gpu, name=name, free_mib=float(free),
                          total_mib=float(total), pci_bus_id=pci, reserved=int(index) in reserved, pids=[]))
    by_uuid = {c['uuid']: c for c in cards}
    rows = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    for row in rows.splitlines():
        gpu, pid = [x.strip() for x in row.split(',')]
        if gpu not in by_uuid or not pid.isdigit():
            raise ValueError('Unrecognized physical GPU process evidence')
        by_uuid[gpu]['pids'].append(int(pid))
    return dict(node=node, checked_utc=now(), host_available_bytes=psutil.virtual_memory().available, cards=cards)


def cuda_indices(snapshot, gpus):
    """vLLM 0.8.5 requires numeric visibility, with proven PCI/NVML parity."""
    cards = sorted(snapshot['cards'], key=lambda card: card['index'])
    if [card['index'] for card in cards] != list(range(len(cards))):
        raise RuntimeError('Physical GPU indices are incomplete')

    def pci_key(card):
        value = card.get('pci_bus_id', '')
        match = re.fullmatch(r'([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}):([0-9a-fA-F]{2})\.([0-7])', value)
        if not match:
            raise RuntimeError('Cannot establish the physical PCI ordering')
        return tuple(int(x, 16) for x in match.groups())

    ordered = sorted(cards, key=pci_key)
    if len(set(pci_key(card) for card in cards)) != len(cards) or ordered != cards:
        raise RuntimeError('NVML GPU indices do not match CUDA PCI_BUS_ID ordering')
    by_uuid = {card['uuid']: card['index'] for card in cards}
    if any(gpu not in by_uuid for gpu in gpus):
        raise RuntimeError('Selected physical GPU UUID is absent')
    return [by_uuid[gpu] for gpu in gpus]


def select_cards(snapshot, count, min_host_gib):
    if snapshot['host_available_bytes'] < min_host_gib * 1024**3:
        raise RuntimeError(f'Need {min_host_gib:g} GiB available host memory')
    cards = [c for c in snapshot['cards'] if not c['reserved'] and not c['pids'] and c['free_mib'] >= 28 * 1024]
    if len(cards) < count:
        raise RuntimeError(f'Need {count} physically empty, unreserved GPUs')
    return [c['uuid'] for c in cards[:count]]


@contextmanager
def lease(paths):
    """All locks are acquired before model launch; partial acquisitions roll back."""
    handles = []
    try:
        for path in sorted(set(map(Path, paths))):
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = path.open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                handle.close()
                raise RuntimeError(f'Lease is already held: {path.name}')
            handles.append(handle)
        yield
    finally:
        for handle in reversed(handles):
            handle.close()


class OwnedProcesses:
    def __init__(self, child=None):
        import psutil
        self.psutil = psutil
        self.pgid = None
        self.root_birth = None
        self.launch_window = (time.time() - 1, time.time() + 1)
        self.identities = {}
        if child is not None:
            self.attach(child)

    def attach(self, child, before=None, after=None):
        # Assign the unique new-session identity before any /proc lookup can
        # fail. The tracker itself is constructed before Popen in locked().
        self.pgid = child.pid
        if before is not None and after is not None:
            self.launch_window = (before - 0.1, after + 0.1)
        try:
            self.root_birth = self.psutil.Process(child.pid).create_time()
            self.identities[child.pid] = self.root_birth
        except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
            # A fast parent can exit while its descendants retain this newly
            # created process group. refresh() recovers those group members.
            self.root_birth = None

    def is_owned(self, pid):
        try:
            process = self.psutil.Process(pid)
            if process.uids().real != os.getuid():
                return False
            if self.identities.get(pid) == process.create_time():
                return True
            for parent in process.parents():
                if self.identities.get(parent.pid) == parent.create_time():
                    self.identities[pid] = process.create_time()
                    return True
            return False
        except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
            return False

    def refresh(self):
        if self.pgid is None:
            return
        # Catch children that were reparented after a fast parent exit. The
        # process group was created uniquely by Popen(start_new_session=True).
        # A reused group-leader PID with a different birth time invalidates this
        # fallback; ancestry/recorded identities remain required in that case.
        try:
            observed_birth = self.psutil.Process(self.pgid).create_time()
            if self.root_birth is None:
                root_reused = not (self.launch_window[0] <= observed_birth <= self.launch_window[1])
                if not root_reused:
                    self.root_birth = observed_birth
                    self.identities[self.pgid] = observed_birth
            else:
                root_reused = observed_birth != self.root_birth
        except self.psutil.NoSuchProcess:
            root_reused = False
        except self.psutil.AccessDenied:
            root_reused = True
        if not root_reused:
            for process in self.psutil.process_iter(['pid', 'uids', 'create_time']):
                try:
                    if (process.info['uids'].real == os.getuid() and
                            process.info['create_time'] >= (self.root_birth or self.launch_window[0]) and
                            os.getpgid(process.pid) == self.pgid):
                        self.identities[process.pid] = process.info['create_time']
                except (self.psutil.NoSuchProcess, self.psutil.AccessDenied, ProcessLookupError):
                    pass
        for pid in list(self.identities):
            if not self.is_owned(pid):
                continue
            try:
                for child in self.psutil.Process(pid).children(recursive=True):
                    if child.uids().real == os.getuid():
                        self.identities[child.pid] = child.create_time()
            except (self.psutil.NoSuchProcess, self.psutil.AccessDenied):
                continue

    def living(self):
        self.refresh()
        result = []
        for pid in list(self.identities):
            if self.is_owned(pid):
                try:
                    if self.psutil.Process(pid).status() != self.psutil.STATUS_ZOMBIE:
                        result.append(pid)
                except self.psutil.NoSuchProcess:
                    pass
        return result

    def signal(self, sig):
        living = self.living()
        group_members = []
        for pid in living:
            try:
                if os.getpgid(pid) == self.pgid:
                    group_members.append(pid)
            except ProcessLookupError:
                pass
        if group_members:
            try:
                os.killpg(self.pgid, sig)
            except ProcessLookupError:
                pass
        # A backend may create a separate child process group. Signal it only
        # after verifying its recorded PID birth time and ownership ancestry.
        for pid in living:
            if pid not in group_members and self.is_owned(pid):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass


@contextmanager
def child_lease(paths, get_owned):
    # The cleanup runs inside the lease even when an unexpected exception
    # escapes the worker monitor. Never release while a known child is running.
    with lease(paths):
        try:
            yield
        finally:
            owned = get_owned()
            if owned is not None and owned.living():
                owned.signal(signal.SIGTERM)
                deadline = time.monotonic() + 30
                while owned.living() and time.monotonic() < deadline:
                    time.sleep(1)
                if owned.living():
                    owned.signal(signal.SIGKILL)
                # A kernel-stalled owned process may take longer to disappear.
                # Preserve its GPU lease until it actually leaves the device.
                while owned.living():
                    time.sleep(5)


def locked(args):
    if not os.environ.get('TMUX') or os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('Use a direct-node tmux session outside a Slurm allocation')
    node = node_name()
    gpus = parse_gpus(args.gpus)
    if args.attempt is None:
        raise RuntimeError('Missing launch attempt identity')
    attempt = OPS / args.name / 'attempts' / args.attempt
    attempt.mkdir(parents=True, exist_ok=True)
    state = dict(name=args.name, attempt=args.attempt, state='starting', node=node,
                 gpus=gpus, command=args.command, started_utc=now(), launcher_pid=os.getpid(),
                 operations_version=OPS_VERSION, operations_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    latest = OPS / args.name / 'status.json'
    stop = {'reason': None, 'time': None}
    child = owned = None
    code = 1

    def request_stop(*_):
        if stop['reason'] is None:
            stop.update(reason='requested_stop', time=time.monotonic())

    def persist():
        save(attempt / 'status.json', state)
        save(latest, state)

    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, request_stop)
    paths = [OPS / 'locks' / f'operation_{args.name}.lock'] + [SHARED_LOCKS / f'gpu_{g}.lock' for g in gpus]
    try:
        with child_lease(paths, lambda: owned):
            snapshot = inventory()
            select_cards(snapshot, len(gpus), args.min_host_gib)
            by_uuid = {c['uuid']: c for c in snapshot['cards']}
            if any(g not in by_uuid or by_uuid[g]['reserved'] or by_uuid[g]['pids']
                   or by_uuid[g]['free_mib'] < 28*1024 for g in gpus):
                raise RuntimeError('Selected GPU lease is no longer physically/scheduler available')
            if stop['reason'] or (OPS / 'STOP').exists() or (OPS / args.name / 'STOP').exists():
                raise RuntimeError('Stop requested before child launch')
            indices = cuda_indices(snapshot, gpus)
            state['cuda_mapping'] = [{'uuid': gpu, 'physical_index': index} for gpu, index in zip(gpus, indices)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, indices)), CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       MEMALPHA_GPU_UUIDS=','.join(gpus), PYTHONHASHSEED='42',
                       OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                       TRANSFORMERS_OFFLINE='1', NLTK_DATA=str(ROOT / '.venv/nltk_data'),
                       TIKTOKEN_CACHE_DIR=str(ROOT / 'models/tiktoken_cache'),
                       MEMALPHA_GPU_LEASE=args.attempt, VLLM_WORKER_MULTIPROC_METHOD='spawn')
            with (attempt / 'worker.log').open('a') as output:
                owned = OwnedProcesses()
                before_launch = time.time()
                child = subprocess.Popen(args.command, cwd=ROOT, env=env, stdout=output,
                                         stderr=subprocess.STDOUT, start_new_session=True)
                after_launch = time.time()
                state['child_pid'] = child.pid
                owned.attach(child, before_launch, after_launch)
                state.update(state='running', child_pid=child.pid, physical_inventory=snapshot)
                persist()
                sent_stop = False
                while child.poll() is None:
                    owned.refresh()
                    if (OPS / 'STOP').exists() or (OPS / args.name / 'STOP').exists():
                        request_stop()
                    try:
                        snapshot = inventory()
                        cards = {c['uuid']: c for c in snapshot['cards']}
                        competing = any(g not in cards or cards[g]['reserved'] or
                                        any(not owned.is_owned(pid) for pid in cards[g]['pids']) for g in gpus)
                        state['last_resource_check'] = snapshot
                        if competing and stop['reason'] is None:
                            stop.update(reason='resource_unavailable', time=time.monotonic())
                    except Exception as exc:
                        state['resource_error'] = repr(exc)
                        if stop['reason'] is None:
                            stop.update(reason='resource_evidence_unavailable', time=time.monotonic())
                    if stop['reason'] and not sent_stop:
                        owned.signal(signal.SIGTERM)
                        sent_stop = True
                    if stop['time'] and time.monotonic() - stop['time'] > 120:
                        owned.signal(signal.SIGKILL)
                    state.update(owned_pids=owned.living(), stop_reason=stop['reason'])
                    persist()
                    time.sleep(5)
                code = child.wait()
                # Never release the lease while owned backend descendants live.
                if owned.living():
                    owned.signal(signal.SIGTERM)
                    deadline = time.monotonic() + 30
                    while owned.living() and time.monotonic() < deadline:
                        time.sleep(1)
                    if owned.living():
                        owned.signal(signal.SIGKILL)
                    deadline = time.monotonic() + 10
                    while owned.living() and time.monotonic() < deadline:
                        time.sleep(1)
                    if owned.living():
                        raise RuntimeError('Owned backend descendants did not stop; inspect physical occupancy')
                state['state'] = 'stopped' if stop['reason'] else ('complete' if code == 0 else 'failed')
                if stop['reason']:
                    code = 75
    except BaseException as exc:
        state.update(state='failed', error=repr(exc))
        if owned is not None:
            owned.signal(signal.SIGTERM)
        code = 1
    finally:
        state.update(exit_code=code, finished_utc=now(), owned_pids=owned.living() if owned is not None else [])
        persist()
        save(attempt / 'exit.json', state)
    return code


def parse_gpus(value):
    gpus = value.split(',') if value else []
    if not gpus or len(set(gpus)) != len(gpus) or any(not re.fullmatch(r'GPU-[0-9a-fA-F-]+', g) for g in gpus):
        raise ValueError('Use unique physical GPU UUIDs')
    return gpus


def launch(args):
    node = node_name()
    if not args.command:
        raise ValueError('Provide the child command after --')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,70}', args.name):
        raise ValueError('Invalid operation name')
    if args.gpus:
        gpus = parse_gpus(args.gpus)
    else:
        snapshot = inventory()
        for card in snapshot['cards']:
            try:
                with lease([SHARED_LOCKS / f"gpu_{card['uuid']}.lock"]):
                    pass
            except RuntimeError:
                card['pids'].append(-1)
        gpus = select_cards(snapshot, args.gpu_count, args.min_host_gib)
    session = 'memalpha_' + args.name
    if subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0:
        raise RuntimeError('Named tmux operation is already active')
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    argv = [str(ROOT / '.venv/bin/python'), str(ROOT / 'ops.py'), 'locked', '--name', args.name,
            '--gpus', ','.join(gpus), '--min-host-gib', str(args.min_host_gib), '--attempt', attempt,
            '--', *args.command]
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, '-c', str(ROOT), shlex.join(argv)], check=True)
    # Confirm the child/lease exists before another caller chooses free cards.
    # This is process dispatch evidence, not a claim that model loading passed.
    status = OPS / args.name / 'attempts' / attempt / 'status.json'
    deadline = time.monotonic() + 35
    while time.monotonic() < deadline:
        if status.exists():
            state = json.loads(status.read_text())
            if state['state'] in ('failed', 'stopped'):
                if state.get('child_pid'):
                    # A child already ran: never turn a scientific/runtime
                    # failure into a blind second-node retry.
                    return dict(state=state['state'], node=node, tmux_session=session,
                                attempt=attempt, gpus=gpus, child_pid=state['child_pid'],
                                error=state.get('error'), exit_code=state.get('exit_code'))
                raise RuntimeError(json.dumps(state))
            if state['state'] in ('running', 'complete'):
                return dict(state=state['state'], node=node, tmux_session=session,
                            attempt=attempt, gpus=gpus, child_pid=state.get('child_pid'))
        time.sleep(0.5)
    return dict(state='unconfirmed', node=node, tmux_session=session, attempt=attempt,
                gpus=gpus, error='Child lease was not confirmed; inspect this attempt before retry')


def dispatch(args):
    failures = []
    for node in args.nodes.split(','):
        if not re.fullmatch(r'gpu-\d{2}', node):
            raise ValueError('Only gpu-xx nodes are permitted')
        argv = [str(ROOT / '.venv/bin/python'), str(ROOT / 'ops.py'), 'launch', '--name', args.name,
                '--gpu-count', str(args.gpu_count), '--min-host-gib', str(args.min_host_gib), '--', *args.command]
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', node, shlex.join(argv)],
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            value = json.loads(result.stdout)
            value['previous_node_failures'] = failures
            return value
        failures.append(dict(node=node, error=result.stderr.strip()[-2000:]))
    raise RuntimeError(json.dumps({'state': 'no_available_node', 'failures': failures}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inventory', 'dispatch', 'launch', 'locked', 'status'])
    parser.add_argument('--name')
    parser.add_argument('--gpus')
    parser.add_argument('--gpu-count', type=int, default=1)
    parser.add_argument('--min-host-gib', type=float, default=64)
    parser.add_argument('--nodes', default='gpu-04,gpu-03')
    parser.add_argument('--attempt')
    # Split manually: argparse.REMAINDER otherwise swallows options after action.
    argv = sys.argv[1:]
    child = []
    if '--' in argv:
        split = argv.index('--')
        argv, child = argv[:split], argv[split+1:]
    args = parser.parse_args(argv)
    args.command = child
    if args.action in ('launch', 'dispatch', 'locked') and not args.name:
        parser.error('--name is required')
    if args.gpu_count < 1 or args.min_host_gib <= 0:
        parser.error('GPU count and host memory must be positive')
    if args.action == 'locked':
        return locked(args)
    if args.action == 'inventory':
        result = inventory()
    elif args.action == 'status':
        result = [json.loads(p.read_text()) for p in sorted(OPS.glob('*/status.json'))]
    else:
        result = {'launch': launch, 'dispatch': dispatch}[args.action](args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
