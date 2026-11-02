"""One intentional, audited handover. All gates pass before signaling our owner."""
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
import psutil

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'operations/direct_parallel_v1'))
from common import verify, lock, inventory, scheduled_jobs, OUT, OPS, CONFIG
from policy import schema_exhaustion, sha, verify_overlay
from amem_reproduction.storage import atomic_json, utc_now

NEWOPS = ROOT/'results/operations/direct_reuse_v2'


def load(path):
    return json.loads(path.read_text())


def progress():
    return {p.parent.name: len(load(p)['notes']) for p in (OUT/'conversations').glob('*/memory.json')}


def main():
    if not os.environ.get('TMUX') or os.uname().nodename.split('.')[0] != 'gpu-04':
        raise RuntimeError('Run in gpu-04 tmux using the project environment')
    guard = lock(NEWOPS/'rollout.lock')
    if guard is None:
        raise RuntimeError('Another handover is active')
    audit = NEWOPS/'migrations'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    audit.mkdir(parents=True, exist_ok=False)
    state_path = OPS/'full_coordinator.json'
    initial = load(state_path)
    if initial['state'] != 'running' or initial.get('supervisor_version'):
        raise RuntimeError('This rollout only replaces the original running supervisor')
    if (OPS/'STOP').exists():
        raise RuntimeError('Intentional STOP is present')
    verify()
    # Review the exact failed GPUs now; no failure is unblocked merely by rc=1.
    evidence = {uuid: schema_exhaustion(OPS, OUT, CONFIG, uuid, entry)
                for uuid, entry in initial['blocked_gpus'].items()}
    if not any(evidence.values()):
        raise RuntimeError('No proven conversation-local schema exhaustion to recover')
    atomic_json(audit/'before.json', dict(coordinator=initial, progress=progress(),
        inventory=inventory(), scheduled_jobs=scheduled_jobs(), release_evidence=evidence))
    tests = [str(ROOT/'.venv/bin/python'), '-m', 'pytest', '-q', 'tests',
             'operations/direct_parallel_v1/test_parallel.py', 'operations/direct_reuse_v2/test_policy.py']
    with (audit/'checks.log').open('w') as log:
        result = subprocess.run(tests, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    atomic_json(audit/'checks.exit.json', dict(exit_code=result.returncode, time_utc=utc_now()))
    if result.returncode:
        raise RuntimeError('Validation failed before handover; running workers untouched')
    print('Software validation passed; preparing checkpointed handover.', flush=True)
    manifest = dict(adapter_version='direct_reuse_v2', created_utc=utc_now(),
        authorization='User requested skipping failed conversation work and reusing the GPU for other data on 2026-09-10.',
        source_sha256={p.name: sha(p) for p in CODE.glob('*.py')},
        original_adapter_manifest_sha256=sha(OPS/'manifest.json'),
        original_scientific_manifest_sha256=sha(OUT/'manifest.json'), checks=str(audit/'checks.log'),
        change='Supervisor dispatch only: proven exhausted-schema conversation remains failed; physical GPU is reusable for other conversations. Unknown errors remain blocked. Frozen worker/backend unchanged.')
    if (NEWOPS/'manifest.json').exists():
        if load(NEWOPS/'manifest.json')['source_sha256'] != manifest['source_sha256']:
            raise RuntimeError('Existing adapter manifest differs')
    else:
        atomic_json(NEWOPS/'manifest.json', manifest)
    verify_overlay(ROOT)
    current = load(state_path)
    if current['pid'] != initial['pid'] or current['state'] != 'running':
        raise RuntimeError('Supervisor changed during validation')
    owner = psutil.Process(current['pid'])
    if owner.uids().real != os.getuid() or str(ROOT/'operations/direct_parallel_v1/supervise.py') not in owner.cmdline():
        raise RuntimeError('Supervisor ownership/identity mismatch')
    if subprocess.run(['tmux', 'has-session', '-t', 'amem_direct_full_v2'], capture_output=True).returncode == 0:
        raise RuntimeError('Replacement supervisor already exists')
    atomic_json(audit/'stop_request.json', dict(pid=owner.pid, create_time=owner.create_time(),
        command=owner.cmdline(), signal='SIGUSR1', time_utc=utc_now()))
    owner.send_signal(signal.SIGUSR1)
    deadline = time.monotonic()+300
    while time.monotonic() < deadline:
        state = load(state_path)
        if state.get('state') == 'checkpointed' and not state.get('workers'):
            break
        if state.get('state') == 'failed':
            raise RuntimeError('Original supervisor failed during graceful handover; inspect logs')
        time.sleep(2)
    else:
        raise RuntimeError('Checkpointed handover timed out; no force kill or replacement launched')
    try:
        owner.wait(timeout=20)
    except psutil.TimeoutExpired:
        raise RuntimeError('Original supervisor still exists; replacement not launched')
    # Require every formerly live worker to have an owned, durable exit.
    for worker in current['workers']:
        exit_record = load(OPS/'exits'/(worker['attempt']+'.json'))
        if exit_record['pid'] != worker['pid']:
            raise RuntimeError('Worker exit identity mismatch')
        if psutil.pid_exists(worker['pid']):
            raise RuntimeError('Old worker still exists; refuse overlap')
    verify()
    stable = progress()
    failure_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (OPS/'failures').glob('*.json')}
    failed_memory_hashes = {sid: sha(OUT/'conversations'/sid/'memory.json')
                            for sid in (p.stem for p in (OPS/'failures').glob('*.json'))}
    with tarfile.open(audit/'pre_handover_raw.tar.gz', 'w:gz', compresslevel=1) as archive:
        archive.add(OUT, arcname=OUT.name)
    shutil.copy2(state_path, audit/'original_full_coordinator.json')
    atomic_json(audit/'checkpointed.json', dict(progress=stable, total_notes=sum(stable.values()),
        failure_sha256=failure_hashes, failed_memory_sha256=failed_memory_hashes,
        raw_backup_sha256=sha(audit/'pre_handover_raw.tar.gz'), time_utc=utc_now()))
    atomic_json(NEWOPS/'migration_ready.json', dict(coordinator_sha256=sha(state_path),
        audit_directory=str(audit), time_utc=utc_now()))
    print('All original workers stopped at durable boundaries; raw backup saved. Starting replacement.', flush=True)
    subprocess.run([str(ROOT/'.venv/bin/python'), str(CODE/'start.py')], check=True, cwd=ROOT)
    deadline = time.monotonic()+180
    while time.monotonic() < deadline:
        new = load(state_path)
        now = progress()
        if (new.get('supervisor_version') == 'direct_reuse_v2'
                and now.get('conv-26', 0) > 0 and now.get('conv-30', 0) > 0):
            break
        if new.get('supervisor_version') == 'direct_reuse_v2' and new.get('state') == 'failed':
            raise RuntimeError('Replacement supervisor failed; inspect its durable log')
        time.sleep(3)
    else:
        atomic_json(audit/'verification_pending.json', dict(coordinator=new, progress=now, time_utc=utc_now()))
        raise RuntimeError('Launch needs follow-up: queued conversation checkpoints not yet observed')
    verify()
    verify_overlay(ROOT)
    assert all(sha(ROOT/p) == h for p, h in failure_hashes.items())
    assert all(sha(OUT/'conversations'/sid/'memory.json') == h for sid, h in failed_memory_hashes.items())
    assert all(now.get(sid, 0) >= count for sid, count in stable.items())
    gpu_state = inventory()
    live = {w['pid']: w['gpu_uuid'] for w in new['workers']}
    assert all(pid in next(c['pids'] for c in gpu_state if c['uuid'] == uuid) for pid, uuid in live.items())
    result = dict(time_utc=utc_now(), coordinator=new, progress=now, total_notes=sum(now.values()),
        inventory=gpu_state, failed_conversations_preserved=sorted(failed_memory_hashes),
        original_and_adapter_hashes_verified=True, audit_directory=str(audit))
    atomic_json(NEWOPS/'launch_verified.json', result)
    print(json.dumps(result, indent=2), flush=True)
    guard.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

