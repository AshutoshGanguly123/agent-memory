"""Isolated, checkpoint-derived recovery for exactly four failed conversations."""
import importlib.util
import json
import os
from pathlib import Path
from amem_reproduction.storage import atomic_json, file_digest, digest

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OPS = ROOT / 'results/operations/recovery_output2048_v1'
RUN_ID = 'locomo_llama32_3b_amem_cuda_output2048_recovery_v1'
TARGETS = ('conv-41', 'conv-50', 'conv-26', 'conv-44')
CONFIG = json.loads((CODE / 'config.json').read_text())
OUT = ROOT / 'results/raw_results' / RUN_ID


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_module('amem_original_common', ROOT / 'operations/direct_parallel_v1/common.py')
reuse = load_module('amem_original_reuse_policy', ROOT / 'operations/direct_reuse_v2/policy.py')
inventory = legacy.inventory
scheduled_jobs = legacy.scheduled_jobs
validate_context = legacy.validate_context
atomic_stop = legacy.atomic_stop
schema_exhaustion = reuse.schema_exhaustion


def validate_config(config=CONFIG, baseline=None):
    baseline = legacy.CONFIG if baseline is None else baseline
    expected = dict(baseline, run_id=RUN_ID, max_new_tokens=2048)
    if config != expected:
        raise RuntimeError('Only the recovery run ID and 2048-token output cap may change')


def require_target(sid):
    if sid not in TARGETS:
        raise RuntimeError('Conversation outside the authorized four: ' + str(sid))


def lock(path):
    path = Path(path)
    # Share physical GPU locks with the baseline adapter, across run directories.
    if path.parent == OPS / 'locks' and path.name.startswith('gpu_'):
        path = legacy.OPS / 'locks' / path.name
    return legacy.lock(path)


def claim(order):
    if tuple(order) != TARGETS:
        raise RuntimeError('Recovery dispatch scope or order changed')
    return legacy.claim(order, out=OUT, ops=OPS)


def pending(order):
    if tuple(order) != TARGETS:
        raise RuntimeError('Recovery dispatch scope or order changed')
    return [sid for sid in order if not (OUT / 'conversations' / sid / 'complete.json').exists()
            and not (OPS / 'failures' / (sid + '.json')).exists()]


def source_hashes():
    return {p.name: file_digest(p) for p in sorted(CODE.iterdir())
            if p.is_file() and (p.suffix == '.py' or p.name == 'config.json')}


def baseline_artifact_hashes():
    paths = [p for p in legacy.OUT.rglob('*') if p.is_file() and p.suffix != '.lock']
    paths += list((legacy.OPS / 'failures').glob('*.json'))
    paths += [legacy.OPS / 'full_coordinator.json']
    return {str(p.relative_to(ROOT)): file_digest(p) for p in sorted(paths)}


def assert_baseline_unchanged():
    expected = json.loads((OPS / 'baseline_artifacts.json').read_text())
    if baseline_artifact_hashes() != expected:
        raise RuntimeError('Original experiment artifacts changed after recovery preparation')


def assert_lineage_unchanged():
    expected = json.loads((OUT / 'manifest.json').read_text())['lineage_sha256']
    actual = {str(p.relative_to(OUT)): file_digest(p)
              for p in sorted((OUT / 'lineage').rglob('*')) if p.is_file()}
    if actual != expected:
        raise RuntimeError('Inherited parent journals/checkpoints changed')


def verify():
    validate_config()
    original, _ = legacy.verify()
    reuse.verify_overlay(ROOT)
    manifest = json.loads((OPS / 'manifest.json').read_text())
    if manifest['adapter_source_sha256'] != source_hashes():
        raise RuntimeError('Recovery source or config changed after freeze')
    if (tuple(manifest['conversation_order']) != TARGETS
            or manifest['config_sha256'] != digest(CONFIG)
            or manifest['raw_manifest_sha256'] != file_digest(OUT / 'manifest.json')
            or manifest['baseline_artifacts_sha256'] != file_digest(OPS / 'baseline_artifacts.json')):
        raise RuntimeError('Recovery scope, provenance or manifest changed')
    actual = {p.name for p in (OUT / 'conversations').iterdir() if p.is_dir()}
    if actual != set(TARGETS):
        raise RuntimeError('Recovery output contains an unexpected conversation')
    return original, manifest
