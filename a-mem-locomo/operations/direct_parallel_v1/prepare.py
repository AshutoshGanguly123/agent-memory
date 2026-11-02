"""Snapshot the clean stop, run checks, then freeze the operational adapter."""
import json
import os
import subprocess
import tarfile
from common import ROOT, CODE, OPS, OUT, CONFIG
from amem_reproduction.provenance import verify_manifest
from amem_reproduction.data import load_dataset, turns
from amem_reproduction.storage import atomic_json, file_digest, utc_now


def main():
    if not os.environ.get('TMUX'):
        raise RuntimeError('Preparation requires tmux')
    OPS.mkdir(parents=True, exist_ok=True)
    if (OPS / 'manifest.json').exists():
        raise RuntimeError('Adapter already frozen; no silent replacement')
    original = verify_manifest(ROOT)
    exit_record = json.loads((ROOT / 'results/operations/job_183458.exit.json').read_text())
    if exit_record['exit_code'] != 75:
        raise RuntimeError('Original worker did not confirm durable stop')
    queued = subprocess.run(['squeue', '-h', '-u', 'ashutosh.ganguly'], capture_output=True, text=True, check=True).stdout
    if queued.strip():
        raise RuntimeError('Expected the old account job queue to be empty before migration')
    backup = OPS / 'pre_migration_raw.tar.gz'
    with tarfile.open(backup, 'w:gz') as archive:
        archive.add(OUT, arcname=OUT.name)
    with backup.open('rb') as handle:
        os.fsync(handle.fileno())
    before = {str(f.relative_to(OUT)): file_digest(f) for f in OUT.rglob('*') if f.is_file()}
    check = subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'pytest', '-q', str(ROOT / 'tests'),
                            str(CODE / 'test_parallel.py')], cwd=ROOT)
    atomic_json(OPS / 'checks.json', dict(exit_code=check.returncode, time_utc=utc_now()))
    if check.returncode:
        raise RuntimeError('Parallelism checks failed')
    samples = load_dataset(ROOT / CONFIG['dataset'], CONFIG['dataset_sha256'])
    counts = {s['sample_id']: dict(notes=len(turns(s)), questions=len(s['qa'])) for s in samples}
    order = ['conv-48'] + sorted([sid for sid in counts if sid != 'conv-48'], key=lambda sid: -counts[sid]['notes'])
    if len(order) != 10 or len(set(order)) != 10 or sum(c['notes'] for c in counts.values()) != 5882:
        raise RuntimeError('Conversation dispatch must cover the complete benchmark exactly once')
    if sum(c['questions'] for c in counts.values()) != 1986:
        raise RuntimeError('Question denominator changed')
    manifest = dict(adapter_version='direct_parallel_v1', created_utc=utc_now(),
        authorization='User explicitly requested direct SSH/tmux on gpu-04 without Slurm, up to all idle cards, on 2026-09-10.',
        original_manifest_sha256=file_digest(OUT / 'manifest.json'), original_source_count=len(original['source_sha256']),
        adapter_source_sha256={p.name: file_digest(p) for p in CODE.glob('*.py')},
        backup_sha256=file_digest(backup), preserved_raw_sha256=before,
        preserved_notes=144, preserved_questions=0, original_exit=exit_record,
        conversation_order=order, counts=counts, max_workers=8,
        isolation='One process/model per GPU; exclusive per-conversation file locks; one chronological writer; QA after freeze.',
        unchanged='All original frozen source/config/assets, prompts, call IDs, seeds, retry limits, inference methods, scoring and denominators.',
        dispatch_only='Physical GPU binding, direct tmux lifecycle and whole-conversation assignment; sequential/parallel performance phases reported separately.')
    atomic_json(OPS / 'manifest.json', manifest)
    with tarfile.open(OPS / 'adapter_source.tar.gz', 'w:gz') as archive:
        for name in manifest['adapter_source_sha256']:
            archive.add(CODE / name, arcname=name)
    atomic_json(OPS / 'prepare_exit.json', dict(exit_code=0, time_utc=utc_now()))
    print(json.dumps(dict(preserved_notes=144, adapter_files=len(manifest['adapter_source_sha256']), conversation_order=order, counts=counts)), flush=True)


if __name__ == '__main__':
    main()
