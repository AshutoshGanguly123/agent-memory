"""Acquire original sources on HPC. Never access the TT host or its artifacts."""
import ast
import io
import json
import os
import urllib.request
import zipfile
from pathlib import Path
from huggingface_hub import snapshot_download
from amem_reproduction.storage import file_digest, atomic_json, utc_now
from amem_reproduction.data import load_dataset, turns

ROOT = Path(__file__).resolve().parents[1]
assert os.environ.get('TMUX'), 'Asset setup must run in tmux'
config = json.loads((ROOT/'configs/experiment.json').read_text())
AMEM = 'c3bb1ee6d4825b13460dd26be1444b5f870ddb64'
LOCOMO = '3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376'
known = {}
for relative in ['benchmark/upstream/sources.json','dataset/manifest.json','benchmark/report_template/source.json']:
    p = ROOT/relative
    if not p.exists(): continue
    obj = json.loads(p.read_text())
    records = list(obj.get('files',{}).values()) if 'files' in obj else [obj.get('source',obj)]
    for record in records:
        if isinstance(record,dict) and 'url' in record and 'sha256' in record: known[record['url']] = record['sha256']

def download(url, relative):
    path = ROOT/relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and url in known and file_digest(path) == known[url]:
        print('Verified cached original source:',relative,flush=True)
        return dict(url=url,sha256=known[url])
    print('Downloading original source:',url,flush=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        body = response.read()
    path.write_bytes(body)
    return {'url': url, 'sha256': file_digest(path)}

sources = {}
for local, remote in [('amem_memory_layer.py','memory_layer.py'), ('amem_test_advanced.py','test_advanced.py'), ('AMEM_LICENSE','LICENSE')]:
    sources[local] = download(f'https://raw.githubusercontent.com/WujiangXu/A-mem/{AMEM}/{remote}', 'benchmark/upstream/'+local)
sources['locomo_evaluation.py'] = download(f'https://raw.githubusercontent.com/snap-research/locomo/{LOCOMO}/task_eval/evaluation.py','benchmark/upstream/locomo_evaluation.py')
sources['LOCOMO_LICENSE'] = download(f'https://raw.githubusercontent.com/snap-research/locomo/{LOCOMO}/LICENSE.txt','benchmark/upstream/LOCOMO_LICENSE')
dataset_source = download(f'https://raw.githubusercontent.com/snap-research/locomo/{LOCOMO}/data/locomo10.json', config['dataset'])
samples = load_dataset(ROOT/config['dataset'], config['dataset_sha256'])
atomic_json(ROOT/'dataset/manifest.json', dict(source=dataset_source, sha256=config['dataset_sha256'], conversations=len(samples), turns=sum(len(turns(s)) for s in samples), questions=sum(len(s['qa']) for s in samples), text_only=True, acquired_on='HPC from original LoCoMo repository'))
atomic_json(ROOT/'benchmark/upstream/sources.json', dict(amem_commit=AMEM, locomo_commit=LOCOMO, files=sources))

manifest = {}
for name, revision, relative, patterns in [
    (config['model'], config['model_revision'], config['model_path'], ['*.json','*.safetensors','*.model','*.txt','README.md','LICENSE*','USE_POLICY*']),
    (config['embedding_model'], config['embedding_revision'], config['embedding_path'], ['*.json','*.safetensors','*.txt','*.model','README.md'])]:
    target = ROOT/relative
    if relative == config['model_path'] and not target.exists():
        # This HPC's /l client produced EFAULT on multi-GB model downloads.
        # Keep independently downloaded weights on the HPC home filesystem.
        cache = Path.home()/'.cache/agent_memory_models'/('Llama-3.2-3B-Instruct-'+revision)
        cache.mkdir(parents=True,exist_ok=True)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.symlink_to(cache,target_is_directory=True)
    print('Downloading pinned model:',name,revision,'to',target.resolve(),flush=True)
    snapshot_download(name, revision=revision, local_dir=target, allow_patterns=patterns, max_workers=4)
    files = {str(p.relative_to(target)): file_digest(p) for p in sorted(target.rglob('*')) if p.is_file() and '.cache' not in p.parts}
    manifest[relative] = dict(repository=name, revision=revision, files=files, resolved_path=str(target.resolve()), acquired_on='HPC directly from Hugging Face')
expected = {'model-00001-of-00002.safetensors':'13cbd6d16e927a0c5bad54102514e6e18b4a47b3a6eb911e39d678d328d19f55', 'model-00002-of-00002.safetensors':'7b770216613ac5c34d7c54bdff1fa616bc4e338a9d0b20af6303e48c295ee23c'}
for name, sha in expected.items():
    assert manifest[config['model_path']]['files'][name] == sha, name
atomic_json(ROOT/'models/manifest.json', dict(assets=manifest, verified_utc=utc_now()))

url = 'https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip'
source = download(url,'benchmark/report_template/author_kit.zip')
assert source['sha256'] == '82473931e3ef710fcd3f4a8cd4119b9de32e56825f90f9e5a6d55f2d01b817d9'
with zipfile.ZipFile(ROOT/'benchmark/report_template/author_kit.zip') as z:
    member = next(n for n in z.namelist() if n.endswith('neurips_2026.sty') and '__MACOSX' not in n)
    (ROOT/'benchmark/report_template/neurips_2026.sty').write_bytes(z.read(member))
atomic_json(ROOT/'benchmark/report_template/source.json', source)
print('Independent pinned assets verified.', flush=True)
