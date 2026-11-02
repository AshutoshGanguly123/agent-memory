"""Validate and freeze this operational adapter, then probe available cards."""
import json
import os
import subprocess
import common
from amem_reproduction.storage import atomic_json, file_digest, utc_now

def main():
    common.validate_context()
    common.parent.verify()
    evidence_path = common.PREVIOUS_OPS / 'replay_gpu3_diagnosis.json'
    evidence = json.loads(evidence_path.read_text())
    common.reference_result(evidence)
    if evidence['device']['node'].split('.')[0] != common.node_name():
        raise RuntimeError('Replay reference belongs to another node')
    if json.loads((common.PREVIOUS_OPS / 'replay_gpu3_diagnosis_exit.json').read_text())['exit_code'] != 0:
        raise RuntimeError('Node replay diagnosis did not complete cleanly')
    common.assert_baseline_unchanged()
    common.assert_lineage_unchanged()
    common.ADAPTER_OPS.mkdir(parents=True, exist_ok=True)
    manifest_path = common.ADAPTER_OPS / 'manifest.json'
    if manifest_path.exists():
        raise RuntimeError('Overlay already prepared; inspect evidence instead of overwriting')
    with (common.ADAPTER_OPS / 'checks.log').open('w') as log:
        result = subprocess.run([str(common.ROOT / '.venv/bin/python'), '-m', 'pytest', '-q',
                                 str(common.CODE / 'test_nodes.py')], cwd=common.ROOT, stdout=log, stderr=subprocess.STDOUT)
    atomic_json(common.ADAPTER_OPS / 'checks_exit.json', dict(exit_code=result.returncode, time_utc=utc_now()))
    result.check_returncode()
    atomic_json(manifest_path, dict(version='recovery_gpu_nodes_v3', source_sha256=common.source_hashes(),
        parent_manifest_sha256=file_digest(common.OPS / 'manifest.json'), time_utc=utc_now(),
        authorization='2026-09-11 user: inspect sinfo; SSH only gpu-xx; deploy available GPUs',
        reference_diagnosis_sha256=file_digest(evidence_path), historical_token_parity=evidence['matches_historical'],
        acceptance='Same frozen inference code; valid JSON and exact node-reference replay on each card; cap and repeat parity verified. Historical cross-node token equality is not established.',
        scientific_changes=[], recovery_run_id=common.CONFIG['run_id'], conversation_order=list(common.TARGETS)))
    cards, reservation = common.inventory(), common.scheduled_jobs()
    candidates = [c for c in cards if not common.reserved(c, reservation) and not c['pids']
                  and c['free_mib'] >= 28 * 1024 and c['name'] == 'NVIDIA RTX 5000 Ada Generation'][:4]
    atomic_json(common.ADAPTER_OPS / 'selection.json', dict(node=common.node_name(), cards=cards,
        reservation=reservation, selected=candidates, time_utc=utc_now()))
    if not candidates:
        raise RuntimeError('No physically idle, unreserved cards for validation')
    children = []
    for card in candidates:
        log = (common.ADAPTER_OPS / ('probe_gpu' + str(card['index']) + '.log')).open('w')
        process = subprocess.Popen([str(common.ROOT / '.venv/bin/python'), '-u', str(common.CODE / 'probe.py'),
            '--uuid', card['uuid'], '--index', str(card['index'])], cwd=common.ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=card['uuid']), stdout=log, stderr=subprocess.STDOUT)
        children.append((card, process, log))
    passed, exits = [], []
    for card, process, log in children:
        rc = process.wait()
        log.close()
        exits.append(dict(gpu_uuid=card['uuid'], exit_code=rc))
        if rc == 0:
            passed.append(card['uuid'])
    common.assert_baseline_unchanged()
    common.assert_lineage_unchanged()
    atomic_json(common.ADAPTER_OPS / 'validation.json', dict(node=common.node_name(), gpu_uuids=passed,
        exits=exits, time_utc=utc_now(), passed=bool(passed)))
    if not passed:
        raise RuntimeError('No new GPU passed exact-token replay')
    print(json.dumps(dict(validated_node=common.node_name(), gpu_uuids=passed)), flush=True)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
