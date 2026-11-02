"""Wait for available cards, then finish only the authorized recovery subset."""
import os
import signal
import subprocess
import time
import psutil
from common import (ROOT, CODE, OPS, OUT, CONFIG, TARGETS, verify, inventory,
                    scheduled_jobs, lock, pending, validate_context, atomic_stop,
                    schema_exhaustion, assert_baseline_unchanged, assert_lineage_unchanged,
                    ADAPTER_OPS, reserved, node_name)
from amem_reproduction.storage import atomic_json, append_jsonl, utc_now


def card_eligible(card, children, blocked, cooldown, available, jobs, host_available, now):
    return (not jobs and card['uuid'] not in children and card['uuid'] not in blocked
            and now >= cooldown.get(card['uuid'], 0) and not card['pids']
            and card['free_mib'] >= 28 * 1024 and host_available >= 32 * 1024**3
            and len(children) < min(4, len(available)))


def main():
    validate_context()
    _, manifest = verify()
    held = []
    paths = [ROOT / 'results/operations/coordinator.lock', ADAPTER_OPS / 'coordinator.lock']
    for path in paths:
        handle = lock(path)
        if handle is None:
            raise RuntimeError('Another coordinator holds ' + str(path))
        held.append(handle)
    state_path = ADAPTER_OPS / 'coordinator.json'
    if state_path.exists():
        raise RuntimeError('Recovery already started; inspect existing state before resuming')
    if (OPS / 'STOP').exists():
        raise RuntimeError('Recovery STOP is present')
    assert_baseline_unchanged()
    assert_lineage_unchanged()
    import json
    validation = json.loads((ADAPTER_OPS / 'validation.json').read_text())
    if not validation['passed'] or validation['node'] != node_name():
        raise RuntimeError('This node has not passed operational GPU validation')
    approved = set(validation['gpu_uuids'])
    previous = json.loads((OPS / 'coordinator.json').read_text())
    if previous['state'] != 'checkpointed' or previous['workers']:
        raise RuntimeError('Previous recovery supervisor must finish its cooperative handover')
    atomic_json(ADAPTER_OPS / 'handover.json', dict(previous_state=previous, time_utc=utc_now(),
                destination_node=node_name(), reason='User authorized available gpu-xx nodes'))
    children, blocked, cooldown = {}, {}, {}
    stopping = {'value': False}
    def request_stop(signum, frame):
        stopping['value'] = True
    for signum in [signal.SIGTERM, signal.SIGINT, signal.SIGUSR1]:
        signal.signal(signum, request_stop)
    state = dict(run_id=CONFIG['run_id'], supervisor_version='recovery_gpu_nodes_v2',
                 pid=os.getpid(), started_utc=utc_now(), state='starting', node=os.uname().nodename,
                 execution_mode='direct_ssh', max_workers=4, conversation_order=list(TARGETS),
                 max_new_tokens=2048, inherited_notes=manifest['inherited_notes'])
    try:
        while True:
            cards, jobs = inventory(), scheduled_jobs()
            for uuid, entry in list(children.items()):
                rc = entry['process'].poll()
                if rc is None:
                    continue
                entry['log'].close()
                atomic_json(OPS / 'exits' / (entry['attempt'] + '.json'), dict(
                    attempt=entry['attempt'], pid=entry['process'].pid,
                    gpu_uuid=uuid, exit_code=rc, time_utc=utc_now()))
                append_jsonl(OPS / 'events.jsonl', dict(event='worker_exit', attempt=entry['attempt'],
                    gpu_uuid=uuid, exit_code=rc, time_utc=utc_now()))
                del children[uuid]
                if rc in (75, 78):
                    cooldown[uuid] = time.time() + 900
                elif rc != 0:
                    failed_entry = dict(attempt=entry['attempt'], exit_code=rc)
                    evidence = schema_exhaustion(OPS, OUT, CONFIG, uuid, failed_entry)
                    if evidence is None:
                        blocked[uuid] = failed_entry
                    else:
                        append_jsonl(OPS / 'events.jsonl', dict(event='conversation_skipped_gpu_released',
                            evidence=evidence, time_utc=utc_now()))
            stop_all = stopping['value'] or (OPS / 'STOP').exists()
            for card in cards:
                entry = children.get(card['uuid'])
                if entry is None:
                    continue
                foreign = [pid for pid in card['pids'] if pid != entry['process'].pid]
                if stop_all or reserved(card, jobs) or foreign:
                    reason = 'user_stop' if stop_all else ('scheduled_gpu_use' if reserved(card, jobs) else 'foreign_gpu_process')
                    path = OPS / 'stops' / (entry['attempt'] + '.json')
                    if not path.exists():
                        atomic_stop(path, reason)
                        append_jsonl(OPS / 'events.jsonl', dict(event='boundary_stop_requested',
                            attempt=entry['attempt'], reason=reason, foreign_pids=foreign, time_utc=utc_now()))
            available = pending(manifest['conversation_order'])
            if not stop_all:
                for card in cards:
                    uuid = card['uuid']
                    if uuid not in approved or not card_eligible(card, children, blocked, cooldown,
                            available, reserved(card, jobs), psutil.virtual_memory().available, time.time()):
                        continue
                    attempt = f'recovery_nodes_{node_name()}_{time.time_ns()}_gpu{card["index"]}'
                    log_path = OPS / 'logs' / (attempt + '.log')
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    log = log_path.open('a')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=uuid)
                    env.pop('AMEM_STOP_FILE', None)
                    env.pop('AMEM_STOP_EPOCH', None)
                    process = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', str(CODE / 'worker.py'),
                        '--uuid', uuid, '--index', str(card['index']), '--attempt', attempt],
                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    children[uuid] = dict(process=process, attempt=attempt, log=log)
                    append_jsonl(OPS / 'events.jsonl', dict(event='worker_started', attempt=attempt,
                        pid=process.pid, gpu_uuid=uuid, gpu_index=card['index'], time_utc=utc_now()))
            live = []
            for uuid, entry in children.items():
                live.append(dict(gpu_uuid=uuid, pid=entry['process'].pid, attempt=entry['attempt']))
            phase = ('stopping' if stop_all else 'yielding_to_scheduled_work' if jobs['blocked']
                     else 'running' if live else 'waiting_for_gpus')
            state.update(state=phase, updated_utc=utc_now(), workers=live, blocked_gpus=blocked,
                         scheduled_jobs=jobs, remaining_conversations=pending(TARGETS), inventory=cards,
                         completed_conversations=[sid for sid in TARGETS if (OUT / 'conversations' / sid / 'complete.json').exists()],
                         failed_conversations=sorted(p.stem for p in (OPS / 'failures').glob('*.json')))
            atomic_json(state_path, state)
            append_jsonl(OPS / 'telemetry.jsonl', dict(time_utc=utc_now(), cards=cards, workers=live,
                host_available_bytes=psutil.virtual_memory().available, scheduled_jobs=jobs))
            if not children:
                if stop_all:
                    state.update(state='checkpointed', finished_utc=utc_now())
                    atomic_json(state_path, state)
                    return 75
                if not available:
                    assert_baseline_unchanged()
                    assert_lineage_unchanged()
                    failed = state['failed_conversations']
                    state.update(state='failed' if failed else 'complete', finished_utc=utc_now())
                    atomic_json(state_path, state)
                    # The frozen full-dataset analyzer must not treat this subset as a fresh full run.
                    return 1 if failed else 0
                if approved.issubset(blocked):
                    raise RuntimeError('All devices blocked by unknown errors; inspect before retrying')
            time.sleep(10)
    except BaseException as exc:
        for entry in children.values():
            atomic_stop(OPS / 'stops' / (entry['attempt'] + '.json'), 'supervisor_error')
        state.update(state='failed', error=repr(exc), finished_utc=utc_now())
        atomic_json(state_path, state)
        raise
    finally:
        for handle in held:
            handle.close()


if __name__ == '__main__':
    raise SystemExit(main())
