"""One GPU, one model instance, one conversation at a time."""
import argparse
import json
import os
import signal
import time
import traceback
from pathlib import Path
from common import ROOT, OPS, OUT, CONFIG, claim, lock, verify, validate_context
from amem_reproduction.storage import atomic_json, read_jsonl, utc_now, digest
from amem_reproduction.calls import CheckpointStop
from amem_reproduction.device_guard import DeviceBusy
from amem_reproduction.data import load_dataset
from amem_reproduction.runner import run_conversation
from amem_reproduction.prompts import Prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--uuid', required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    validate_context()
    _, operational = verify()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != args.uuid:
        raise RuntimeError('Worker is not bound to the requested UUID')
    gpu_lock = lock(OPS / 'locks' / ('gpu_' + args.uuid + '.lock'))
    if gpu_lock is None:
        return 78
    status_path = OPS / 'workers' / (args.attempt + '.json')
    stop_path = OPS / 'stops' / (args.attempt + '.json')
    status = dict(attempt=args.attempt, pid=os.getpid(), gpu_index=args.index, gpu_uuid=args.uuid,
                  started_utc=utc_now(), state='running', phase='loading', mode='probe' if args.probe else 'full',
                  execution_mode='direct_ssh', slurm_job_id=None, config_sha256=digest(CONFIG))
    stopping = {'value': False}
    def request_stop(signum, frame):
        stopping['value'] = True
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    def stop_check():
        if stopping['value'] or stop_path.exists() or (OPS / 'STOP').exists():
            raise CheckpointStop('Direct worker requested to stop at durable call boundary')
    def update(**values):
        status.update(values, updated_utc=utc_now())
        atomic_json(status_path, status)
    backend = None
    conversation_lock = None
    sid = None
    update()
    try:
        stop_check()
        from backend import DirectCudaLlama
        backend = DirectCudaLlama(CONFIG, ROOT, args.index)
        backend.stop_check = stop_check
        backend.progress_callback = update
        update(device=backend.device_info, model_load_seconds=backend.load_seconds)
        if args.probe:
            baseline = next(c for c in read_jsonl(OUT / 'conversations/conv-48/calls.jsonl')
                            if c.get('event') == 'call_complete' and c.get('phase') == 'note' and c.get('status') == 'ok')
            result = backend.generate(backend.encode(baseline['prompt']), baseline['schema'], baseline['seed'])
            passed = result['output_token_ids'] == baseline['output_token_ids'] and result['parsed'] == baseline['parsed']
            atomic_json(OPS / 'probes' / (args.uuid + '.json'), dict(
                passed=passed, gpu_uuid=args.uuid, gpu_index=args.index, baseline_call_id=baseline['call_id'],
                baseline_seed=baseline['seed'], token_ids_equal=result['output_token_ids'] == baseline['output_token_ids'],
                device=backend.device_info, result=result, time_utc=utc_now()))
            if not passed:
                raise RuntimeError('Direct GPU probe differs from original successful call')
            update(state='complete', phase='probe_complete', finished_utc=utc_now())
            return 0
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(str(ROOT / CONFIG['embedding_path']), device='cpu', local_files_only=True)
        prompts = Prompts(ROOT)
        samples = {s['sample_id']: s for s in load_dataset(ROOT / CONFIG['dataset'], CONFIG['dataset_sha256'])}
        update(embedding_max_seq_length=embedder.max_seq_length)
        while True:
            stop_check()
            sid, conversation_lock = claim(operational['conversation_order'])
            if sid is None:
                break
            update(conversation=sid, phase='claimed', questions_done=0, notes_done=0)
            try:
                # Original function preserves chronological writes, stale embeddings,
                # call cache/seeds/retry budget, frozen QA, scoring and raw journals.
                run_conversation(samples[sid], backend, embedder, prompts, CONFIG, OUT, update)
            except CheckpointStop:
                raise
            except BaseException as exc:
                atomic_json(OPS / 'failures' / (sid + '.json'), dict(sample_id=sid,
                    attempt=args.attempt, error=repr(exc), traceback=traceback.format_exc(), time_utc=utc_now()))
                raise
            finally:
                conversation_lock.close()
                conversation_lock = None
            update(last_completed_conversation=sid, conversation=None, phase='dispatch')
            sid = None
        update(state='complete', phase='no_unclaimed_work', finished_utc=utc_now())
        return 0
    except DeviceBusy as exc:
        update(state='waiting_for_device', reason=str(exc), finished_utc=utc_now())
        return 78
    except CheckpointStop as exc:
        update(state='checkpointed', reason=str(exc), finished_utc=utc_now())
        return 75
    except BaseException as exc:
        update(state='failed', error=repr(exc), traceback=traceback.format_exc(), finished_utc=utc_now())
        return 1
    finally:
        if conversation_lock is not None:
            conversation_lock.close()
        if backend is not None:
            backend.close()
        gpu_lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
