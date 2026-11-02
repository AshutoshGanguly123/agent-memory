"""Freeze a separately labeled recovery without modifying the original run."""
import json
import os
import shutil
import subprocess
from pathlib import Path
import psutil
from common import (ROOT, CODE, OPS, OUT, RUN_ID, TARGETS, CONFIG, legacy, reuse,
                    lock, validate_config, source_hashes, baseline_artifact_hashes,
                    verify, assert_baseline_unchanged, assert_lineage_unchanged)
from amem_reproduction.data import load_dataset, turns
from amem_reproduction.storage import atomic_json, file_digest, digest, utc_now


def seed_subset(parent_out, destination, targets=TARGETS):
    if tuple(targets) != TARGETS:
        raise RuntimeError('Only the authorized four conversations may be seeded')
    inherited = {}
    for sid in targets:
        source = parent_out / 'conversations' / sid
        memory = json.loads((source / 'memory.json').read_text())
        if memory.get('frozen') or (source / 'complete.json').exists():
            raise RuntimeError('Recovery target is already complete/frozen: ' + sid)
        if (source / 'answers.jsonl').exists() and (source / 'answers.jsonl').stat().st_size:
            raise RuntimeError('Unexpected QA in a failed ingestion target: ' + sid)
        lineage = destination / 'lineage/conversations' / sid
        shutil.copytree(source, lineage)
        active = destination / 'conversations' / sid
        active.mkdir(parents=True, exist_ok=False)
        for name in ['memory.json', 'memory_events.jsonl']:
            if (source / name).exists():
                shutil.copy2(source / name, active / name)
        # Parent calls, seeds and failures remain in lineage/original files.
        # New-cap calls get their own journal, call IDs and three-attempt budget.
        inherited[sid] = len(memory['notes'])
    return inherited


def run_checks():
    suites = [
        ('baseline', [ROOT / 'tests', ROOT / 'operations/direct_parallel_v1/test_parallel.py',
                      ROOT / 'operations/direct_reuse_v2/test_policy.py'],
         [ROOT / 'operations/direct_reuse_v2', ROOT / 'operations/direct_parallel_v1', ROOT / 'src']),
        ('recovery', [CODE / 'test_recovery.py'], [CODE, ROOT / 'src']),
    ]
    checks = {}
    for name, paths, pythonpath in suites:
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str, pythonpath)))
        log_path = OPS / ('checks_' + name + '.log')
        with log_path.open('w') as log:
            result = subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'pytest', '-q',
                                     *map(str, paths)], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        checks[name] = dict(exit_code=result.returncode, log=str(log_path.relative_to(ROOT)))
        atomic_json(OPS / 'checks.json', checks)
        if result.returncode:
            raise RuntimeError('Checks failed; inspect ' + str(log_path))
    return checks


def main():
    if not os.environ.get('TMUX'):
        raise RuntimeError('Preparation must run in tmux using the project .venv')
    OPS.mkdir(parents=True, exist_ok=True)
    held = [lock(ROOT / 'results/operations/coordinator.lock'), lock(OPS / 'prepare.lock')]
    if any(h is None for h in held):
        raise RuntimeError('An active coordinator/preparation prevents safe copying')
    try:
        validate_config()
        original, operational = legacy.verify()
        reuse.verify_overlay(ROOT)
        if (OPS / 'ready.json').exists():
            verify()
            assert_baseline_unchanged()
            assert_lineage_unchanged()
            print('Previously prepared recovery verified', flush=True)
            return 0
        if OUT.exists():
            raise RuntimeError('Recovery destination already exists without readiness; inspect it')
        parent_state = json.loads((legacy.OPS / 'full_coordinator.json').read_text())
        if parent_state['workers'] or set(parent_state['failed_conversations']) != set(TARGETS):
            raise RuntimeError('Baseline is not stopped with exactly the four expected failures')
        for proc in psutil.process_iter(['cmdline']):
            try:
                if proc.uids().real != os.getuid():
                    continue
                command = proc.info['cmdline'] or []
                executables = [str(ROOT / 'operations/direct_parallel_v1/worker.py'),
                               str(ROOT / 'operations/direct_reuse_v2/supervise.py')]
                if any(path in command for path in executables):
                    raise RuntimeError('Original worker or supervisor is still active')
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        before = baseline_artifact_hashes()
        checks = run_checks()
        dataset = {s['sample_id']: s for s in load_dataset(ROOT / CONFIG['dataset'], CONFIG['dataset_sha256'])}
        counts = {sid: dict(notes=len(turns(dataset[sid])), questions=len(dataset[sid]['qa'])) for sid in TARGETS}
        if sum(c['notes'] for c in counts.values()) != 2325 or sum(c['questions'] for c in counts.values()) != 754:
            raise RuntimeError('Recovery subset counts differ from the verified dataset')
        stamp = utc_now().replace(':', '').replace('+', '_')
        stage = OUT.with_name(OUT.name + '.preparing.' + stamp)
        stage.mkdir(parents=True, exist_ok=False)
        inherited = seed_subset(legacy.OUT, stage)
        if inherited != {'conv-41': 54, 'conv-50': 91, 'conv-26': 47, 'conv-44': 417}:
            raise RuntimeError('Original checkpoint counts changed; inspect before proceeding')
        for sid in TARGETS:
            expected_ids = [r['dia_id'] for r in turns(dataset[sid])][:inherited[sid]]
            memory = json.loads((stage / 'conversations' / sid / 'memory.json').read_text())
            if [n['id'] for n in memory['notes']] != expected_ids:
                raise RuntimeError('Inherited checkpoint is not the chronological prefix: ' + sid)
            if file_digest(stage / 'conversations' / sid / 'memory.json') != file_digest(legacy.OUT / 'conversations' / sid / 'memory.json'):
                raise RuntimeError('Checkpoint changed during copy: ' + sid)
        lineage_hashes = {str(p.relative_to(stage)): file_digest(p)
                          for p in sorted((stage / 'lineage').rglob('*')) if p.is_file()}
        for name, checksum in lineage_hashes.items():
            original_path = legacy.OUT / Path(name).relative_to('lineage')
            if file_digest(original_path) != checksum:
                raise RuntimeError('Parent journal/checkpoint copy differs: ' + name)
        raw_manifest = dict(run_id=RUN_ID, created_utc=utc_now(), kind='checkpoint_recovery', fresh_run=False,
            config=CONFIG, config_sha256=digest(CONFIG), conversation_order=list(TARGETS), counts=counts,
            expected_conversations=4, expected_notes=2325, expected_questions=754,
            inherited_notes=inherited, remaining_notes=2325-sum(inherited.values()),
            parent_run_id=legacy.CONFIG['run_id'], parent_manifest_sha256=file_digest(legacy.OUT / 'manifest.json'),
            parent_source_sha256=original['source_sha256'], lineage_sha256=lineage_hashes,
            parent_failures={sid: json.loads((legacy.OPS / 'failures' / (sid + '.json')).read_text()) for sid in TARGETS},
            adapter_source_sha256=source_hashes(), checks=checks,
            protocol_change=dict(max_new_tokens=dict(parent=1000, recovery=2048),
                other_scientific_settings='unchanged; distinct run ID changes derived call IDs/seeds',
                inherited_prefix='609 notes built under the parent 1000-token policy; stale embeddings preserved byte-for-byte',
                calls='parent journals retained under lineage; new calls use the recovery config and at most three attempts across restarts',
                uncommitted_parent_note_call='regenerated under the new configuration at the failed turn',
                scope='only conv-26, conv-41, conv-44, conv-50; no execution for the six completed conversations',
                interpretation='checkpoint-derived mixed-policy recovery, not a fresh full-dataset 2048-token benchmark'))
        atomic_json(stage / 'manifest.json', raw_manifest)
        if baseline_artifact_hashes() != before:
            raise RuntimeError('Original results changed during preparation')
        atomic_json(OPS / 'baseline_artifacts.json', before)
        os.rename(stage, OUT)
        manifest = dict(run_id=RUN_ID, prepared_utc=utc_now(), conversation_order=list(TARGETS), counts=counts,
            inherited_notes=inherited, config_sha256=digest(CONFIG), adapter_source_sha256=source_hashes(),
            raw_manifest_sha256=file_digest(OUT / 'manifest.json'),
            baseline_artifacts_sha256=file_digest(OPS / 'baseline_artifacts.json'),
            original_adapter_manifest_sha256=file_digest(legacy.OPS / 'manifest.json'), checks=checks)
        atomic_json(OPS / 'manifest.json', manifest)
        verify()
        assert_baseline_unchanged()
        assert_lineage_unchanged()
        atomic_json(OPS / 'ready.json', dict(ready=True, run_id=RUN_ID, prepared_utc=utc_now(),
            conversation_order=list(TARGETS), inherited_notes=inherited, remaining_notes=1716,
            expected_questions=754, baseline_artifacts_verified=len(before), checks=checks))
        print(json.dumps(json.loads((OPS / 'ready.json').read_text()), indent=2), flush=True)
        return 0
    finally:
        for handle in held:
            if handle is not None:
                handle.close()


if __name__ == '__main__':
    raise SystemExit(main())
