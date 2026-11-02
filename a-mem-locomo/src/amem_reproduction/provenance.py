import json
import tarfile
from pathlib import Path
from .storage import atomic_json,file_digest,digest,utc_now


def source_hashes(root):
    files = [p for folder in ['src','scripts','configs','tests','benchmark'] for p in (root/folder).rglob('*')
             if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.zip']]
    files += [root/n for n in ['requirements.lock','requirements.in','pyproject.toml','models/manifest.json','dataset/manifest.json']]
    return {str(p.relative_to(root)):file_digest(p) for p in sorted(files)}


def freeze(root):
    c = json.loads((root/'configs/experiment.json').read_text())
    out = root/'results/raw_results'/c['run_id']
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): return verify_manifest(root)
    if (root/'results/setup/checks.exit').read_text().strip() != '0': raise RuntimeError('Checks not passed')
    passed = json.loads((root/'results/preflight'/c['run_id']/'passed.json').read_text())
    hashes = source_hashes(root)
    if passed['source_sha256'] != hashes: raise RuntimeError('Sources changed since GPU preflight')
    manifest = dict(run_id=c['run_id'],created_utc=utc_now(),config=c,config_sha256=digest(c),
        source_sha256=hashes,assignments={'0':['conv-48','conv-49','conv-26'],
        '1':['conv-47','conv-50','conv-30'],'2':['conv-43','conv-42'],'3':['conv-44','conv-41']},
        expected_notes=5882,expected_questions=1986,expected_conversations=10,preflight=passed,
        operational_parallelism='Slurm-limited one GPU at a time; four independent logical shards',
        fresh_run=True,tt_checkpoints_imported=False)
    from .data import load_dataset
    ids=[sid for shard in manifest['assignments'].values() for sid in shard]
    samples=load_dataset(root/c['dataset'],c['dataset_sha256'])
    if len(ids)!=len(set(ids)) or set(ids)!={s['sample_id'] for s in samples}:
        raise RuntimeError('Shards must cover the full dataset exactly once')
    atomic_json(out/'manifest.json',manifest)
    with tarfile.open(out/'source_snapshot.tar.gz','w:gz') as archive:
        for name in hashes: archive.add(root/name,arcname=name)
    return manifest


def verify_manifest(root):
    c = json.loads((root/'configs/experiment.json').read_text())
    m = json.loads((root/'results/raw_results'/c['run_id']/'manifest.json').read_text())
    if m['config_sha256'] != digest(c) or m['source_sha256'] != source_hashes(root):
        raise RuntimeError('Frozen source/config changed; do not silently resume')
    return m
