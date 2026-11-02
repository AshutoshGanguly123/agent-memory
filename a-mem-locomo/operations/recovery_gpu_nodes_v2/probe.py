"""Replay one inherited, valid note from an authorized target; never write memory."""
import argparse
import os
import sys
import common
from amem_reproduction.storage import atomic_json, read_jsonl, utc_now

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uuid', required=True)
    parser.add_argument('--index', required=True, type=int)
    args = parser.parse_args()
    common.validate_context()
    common.verify()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.uuid:
        raise RuntimeError('Probe UUID mismatch')
    handle = common.lock(common.OPS / 'locks' / ('gpu_' + args.uuid + '.lock'))
    if handle is None:
        raise RuntimeError('GPU already locked')
    backend = None
    try:
        sys.path.insert(1, str(common.legacy.CODE))
        from backend import DirectCudaLlama
        baseline = next(c for c in read_jsonl(common.OUT / 'lineage/conversations/conv-41/calls.jsonl')
                        if c.get('event') == 'call_complete' and c.get('phase') == 'note' and c.get('status') == 'ok')
        backend = DirectCudaLlama(common.CONFIG, common.ROOT, args.index)
        result = backend.generate(backend.encode(baseline['prompt']), baseline['schema'], baseline['seed'])
        passed = result['output_token_ids'] == baseline['output_token_ids'] and result['parsed'] == baseline['parsed']
        atomic_json(common.ADAPTER_OPS / 'probes' / (args.uuid + '.json'), dict(
            passed=passed, node=common.node_name(), gpu_uuid=args.uuid, gpu_index=args.index,
            baseline_conversation='conv-41', baseline_call_id=baseline['call_id'], baseline_seed=baseline['seed'],
            device=backend.device_info, result=result, time_utc=utc_now(),
            purpose='Operational replay only; no memory/QA writes and not benchmark accounting'))
        if not passed:
            raise RuntimeError('New GPU did not reproduce inherited valid note tokens')
        return 0
    finally:
        if backend is not None:
            backend.close()
        handle.close()

if __name__ == '__main__':
    raise SystemExit(main())
