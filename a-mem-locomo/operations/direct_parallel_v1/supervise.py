"""Persistent direct-node tmux supervisor. Never submits Slurm jobs."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
import psutil
from common import ROOT, CODE, OPS, OUT, CONFIG, verify, inventory, scheduled_jobs, lock, pending, validate_context, atomic_stop
from amem_reproduction.storage import atomic_json, append_jsonl, utc_now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    validate_context()
    _, manifest = verify()
    held = []
    for path in [ROOT / 'results/operations/coordinator.lock'] + [OUT / f'worker_{i}.lock' for i in CONFIG['worker_ids']]:
        handle = lock(path)
        if handle is None:
            raise RuntimeError('A legacy or parallel coordinator/worker is still active: ' + str(path))
        held.append(handle)
    mode = 'probe' if args.probe else 'full'
    state_path = OPS / (mode + '_coordinator.json')
    if not args.probe and state_path.exists():
        prior = json.loads(state_path.read_text())
        if prior.get('state') not in ['complete']:
            raise RuntimeError('Prior direct coordinator requires inspection and an archived state before intentional resume')
    if (OPS / 'STOP').exists():
        raise RuntimeError('Intentional global STOP is present; inspect before removing')
    if not args.probe:
        validation = json.loads((OPS / 'validation.json').read_text())
        if not validation['passed']:
            raise RuntimeError('Direct GPU validation is not passed')
        approved_uuids = set(validation['gpu_uuids'])
    else:
        approved_uuids = None
    children, completed, blocked, cooldown = {}, set(), {}, {}
    stopping = {'value': False}
    def request_stop(signum, frame):
        stopping['value'] = True
    for signum in [signal.SIGTERM, signal.SIGINT, signal.SIGUSR1]:
        signal.signal(signum, request_stop)
    state = dict(mode=mode, pid=os.getpid(), started_utc=utc_now(), state='starting',
                 execution_mode='direct_ssh', node=os.uname().nodename, max_workers=8)
    first_probe_set = None
    try:
        while True:
            cards = inventory()
            jobs = scheduled_jobs()
            if first_probe_set is None and args.probe:
                first_probe_set = {c['uuid'] for c in cards if not c['pids'] and c['free_mib'] >= 28*1024}
                if not first_probe_set or jobs:
                    raise RuntimeError('No clean node/GPU available for direct validation')
            for uuid, entry in list(children.items()):
                rc = entry['process'].poll()
                if rc is None:
                    continue
                entry['log'].close()
                atomic_json(OPS / 'exits' / (entry['attempt'] + '.json'), dict(
                    attempt=entry['attempt'], pid=entry['process'].pid, gpu_uuid=uuid, exit_code=rc, time_utc=utc_now()))
                append_jsonl(OPS / 'events.jsonl', dict(event='worker_exit', attempt=entry['attempt'],
                    gpu_uuid=uuid, exit_code=rc, time_utc=utc_now()))
                del children[uuid]
                if rc == 0:
                    completed.add(uuid)
                elif rc in [75, 78]:
                    cooldown[uuid] = time.time() + 900
                else:
                    blocked[uuid] = dict(attempt=entry['attempt'], exit_code=rc)
            stop_all = stopping['value'] or (OPS / 'STOP').exists()
            for card in cards:
                entry = children.get(card['uuid'])
                if entry is None:
                    continue
                foreign = [pid for pid in card['pids'] if pid != entry['process'].pid]
                if stop_all or jobs or foreign:
                    reason = 'user_stop' if stop_all else ('scheduled_node_use' if jobs else 'foreign_gpu_process')
                    path = OPS / 'stops' / (entry['attempt'] + '.json')
                    if not path.exists():
                        atomic_stop(path, reason)
                        append_jsonl(OPS / 'events.jsonl', dict(event='boundary_stop_requested',
                            attempt=entry['attempt'], reason=reason, foreign_pids=foreign, time_utc=utc_now()))
            available_work = first_probe_set - completed - set(blocked) if args.probe else pending(manifest['conversation_order'])
            if not stop_all and not jobs:
                for card in cards:
                    uuid = card['uuid']
                    if uuid in children or uuid in blocked or time.time() < cooldown.get(uuid, 0):
                        continue
                    if args.probe:
                        if uuid not in available_work:
                            continue
                    elif uuid not in approved_uuids or len(available_work) <= len(children):
                        continue
                    if card['pids'] or card['free_mib'] < 28*1024 or psutil.virtual_memory().available < 32*1024**3:
                        continue
                    if len(children) >= 8:
                        break
                    attempt = f'{mode}_{time.time_ns()}_gpu{card["index"]}'
                    log_path = OPS / 'logs' / (attempt + '.log')
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log = log_path.open('a')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
                    env.pop('AMEM_STOP_FILE', None)
                    env.pop('AMEM_STOP_EPOCH', None)
                    command = [str(ROOT / '.venv/bin/python'), '-u', str(CODE / 'worker.py'),
                               '--uuid', uuid, '--index', str(card['index']), '--attempt', attempt]
                    if args.probe:
                        command.append('--probe')
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    children[uuid] = dict(process=process, attempt=attempt, log=log)
                    append_jsonl(OPS / 'events.jsonl', dict(event='worker_started', attempt=attempt,
                        gpu_uuid=uuid, gpu_index=card['index'], pid=process.pid, time_utc=utc_now()))
            live = []
            for uuid, entry in children.items():
                try:
                    proc = psutil.Process(entry['process'].pid)
                    rss, cpu = proc.memory_info().rss, proc.cpu_times()._asdict()
                except psutil.NoSuchProcess:
                    rss, cpu = None, None
                live.append(dict(gpu_uuid=uuid, pid=entry['process'].pid, attempt=entry['attempt'], host_rss_bytes=rss, cpu_times=cpu))
            state.update(state='stopping' if stop_all else ('yielding_to_scheduled_work' if jobs else 'running'),
                         updated_utc=utc_now(), workers=live, blocked_gpus=blocked, scheduled_jobs=jobs,
                         remaining_conversations=pending(manifest['conversation_order']), inventory=cards)
            atomic_json(state_path, state)
            append_jsonl(OPS / (mode + '_telemetry.jsonl'), dict(time_utc=utc_now(), cards=cards, workers=live,
                host_available_bytes=psutil.virtual_memory().available, scheduled_jobs=jobs))
            if not children:
                if stop_all:
                    state.update(state='checkpointed', finished_utc=utc_now())
                    atomic_json(state_path, state)
                    return 75
                if args.probe and (first_probe_set <= completed | set(blocked)):
                    passed = not blocked and bool(completed)
                    atomic_json(OPS / 'validation.json', dict(passed=passed, gpu_uuids=sorted(completed),
                        blocked=blocked, time_utc=utc_now(), criterion='Exact token IDs and parsed JSON match a saved original-run note call on every tested card'))
                    state.update(state='complete' if passed else 'failed', finished_utc=utc_now())
                    atomic_json(state_path, state)
                    return 0 if passed else 1
                if not args.probe and not available_work:
                    failed = sorted(p.stem for p in (OPS / 'failures').glob('*.json'))
                    state.update(state='failed' if failed else 'complete', failed_conversations=failed, finished_utc=utc_now())
                    atomic_json(state_path, state)
                    if failed:
                        return 1
                    result = subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'amem_reproduction.analyze', '--root', str(ROOT)], cwd=ROOT)
                    atomic_json(OPS / 'analysis_exit.json', dict(exit_code=result.returncode, time_utc=utc_now()))
                    return result.returncode
                if blocked and len(blocked) == len(cards):
                    raise RuntimeError('All devices failed; inspect rather than retry')
            time.sleep(10)
    except BaseException as exc:
        for entry in children.values():
            atomic_stop(OPS / 'stops' / (entry['attempt'] + '.json'), 'supervisor_error')
        state.update(state='failed', error=repr(exc), finished_utc=utc_now())
        atomic_json(state_path, state)
        raise
    finally:
        # Never terminate a different user's process; owned workers finish durable boundaries.
        for entry in children.values():
            atomic_stop(OPS / 'stops' / (entry['attempt'] + '.json'), 'supervisor_exit')
        for handle in held:
            handle.close()


if __name__ == '__main__':
    raise SystemExit(main())
