import argparse
import gc
import json
import os
import signal
import sys
import time
import traceback
from protocol import *
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(1, str(ROOT / 'operations/recovery_gpu_nodes_v3'))
sys.path.insert(2, str(ROOT / 'operations/direct_parallel_v1'))
import common
from amem_reproduction.storage import append_jsonl, atomic_json, utc_now


def generate_batch(backend, jobs, heartbeat):
    """Independent schema state per prompt; left padding is attention-masked."""
    from lmformatenforcer import TokenEnforcer, JsonSchemaParser
    from jsonschema import Draft202012Validator, ValidationError
    import torch
    encoded = [backend.encode(job['prompt'])[0].tolist() for job in jobs]
    lengths = [len(x) for x in encoded]
    width = max(lengths)
    assert width + MAX_NEW_TOKENS <= backend.config['max_context'], 'No silent truncation'
    pad = backend.stop_ids[0]
    ids = torch.tensor([[pad]*(width-len(x))+x for x in encoded], device='cuda:0')
    mask = torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in encoded], device='cuda:0')
    position = (mask.cumsum(-1)-1).clamp(min=0)
    enforcers = [TokenEnforcer(backend.enforcer_data, JsonSchemaParser(SCHEMA)) for _ in jobs]
    validator = Draft202012Validator(SCHEMA)
    tokens = [[] for _ in jobs]
    parsed = [None]*len(jobs)
    reasons = ['output_limit']*len(jobs)
    active = [True]*len(jobs)
    cache = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.inference_mode():
            output = backend.model(input_ids=ids, attention_mask=mask, position_ids=position,
                use_cache=True, logits_to_keep=1, return_dict=True)
            cache = output.past_key_values
            scores = output.logits[:, -1].float().cpu()
            del output, ids, position
            prefill = time.perf_counter()-started
            for step in range(MAX_NEW_TOKENS):
                assert torch.isfinite(scores).all(), 'Non-finite logits'
                selected = [pad]*len(jobs)
                for i in range(len(jobs)):
                    if not active[i]:
                        continue
                    allowed = list(dict.fromkeys(t for t in enforcers[i].get_allowed_tokens(tokens[i]).allowed_tokens if t < scores.shape[1]))
                    if not allowed:
                        active[i] = False
                        reasons[i] = 'no_valid_tokens'
                        continue
                    values = scores[i, torch.tensor(allowed, dtype=torch.long)]
                    token = allowed[int(values.argmax())]
                    selected[i] = token
                    tokens[i].append(token)
                    raw = backend.tokenizer.decode(tokens[i], skip_special_tokens=True)
                    if raw.rstrip().endswith('}'):
                        try:
                            obj = json.loads(raw)
                            validator.validate(obj)
                            parsed[i] = obj
                            reasons[i] = 'json_complete'
                            active[i] = False
                        except (json.JSONDecodeError, ValidationError):
                            pass
                    if token in backend.stop_ids and active[i]:
                        active[i] = False
                        reasons[i] = 'eos'
                if step % 16 == 0:
                    heartbeat()
                if not any(active) or step+1 == MAX_NEW_TOKENS:
                    break
                extra = torch.tensor(active, device='cuda:0', dtype=mask.dtype).reshape(-1,1)
                mask = torch.cat([mask, extra], dim=1)
                position = (mask.sum(-1)-1).clamp(min=0).reshape(-1,1)
                output = backend.model(input_ids=torch.tensor(selected, device='cuda:0').reshape(-1,1),
                    attention_mask=mask, position_ids=position, past_key_values=cache,
                    use_cache=True, logits_to_keep=1, return_dict=True)
                cache = output.past_key_values
                scores = output.logits[:, -1].float().cpu()
                del output
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        return [dict(raw_response=backend.tokenizer.decode(tokens[i], skip_special_tokens=True),
                     parsed=parsed[i], output_token_ids=tokens[i], input_tokens=lengths[i], output_tokens=len(tokens[i]),
                     stop_reason=reasons[i], batch_seconds=elapsed, batch_size=len(jobs), amortized_seconds=elapsed/len(jobs),
                     prefill_seconds=prefill, peak_allocated_bytes=torch.cuda.max_memory_allocated()) for i in range(len(jobs))]
    except BaseException:
        append_jsonl(OPS / ('partial_batch_' + str(os.getpid()) + '.jsonl'), dict(time_utc=utc_now(), jobs=jobs,
            generated_token_ids=tokens, raw_responses=[backend.tokenizer.decode(t,skip_special_tokens=True) for t in tokens]))
        raise
    finally:
        del cache
        gc.collect()


def run(args):
    import psutil
    common.validate_context()
    manifest = verify()
    assert 0 <= args.shard < SHARDS
    stem = 'preflight' if args.preflight else 'shard_' + str(args.shard)
    status_path = OPS / (stem + '_status.json')
    handles = [common.lock(common.OPS / 'locks' / ('gpu_' + args.uuid + '.lock')),
               common.lock(OPS / 'locks' / (stem + '.lock'))]
    if any(h is None for h in handles):
        raise RuntimeError('GPU or experiment shard already locked')
    if psutil.virtual_memory().available < 32*1024**3:
        raise RuntimeError('Insufficient host RAM')
    stop = {'reason': None}
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, lambda *_: stop.update(reason='requested_stop'))
    last_guard = 0

    def guard(force=False):
        nonlocal last_guard
        if (OPS / 'STOP').exists():
            stop['reason'] = 'STOP_file'
        if not force and time.monotonic()-last_guard < 10:
            return
        cards = [c for c in common.inventory() if c['uuid'] == args.uuid]
        scheduled = common.scheduled_jobs()
        if len(cards) != 1 or common.reserved(cards[0], scheduled):
            stop['reason'] = 'GPU_reserved_or_uncertain'
        elif any(pid != os.getpid() for pid in cards[0]['pids']):
            stop['reason'] = 'foreign_gpu_process'
        append_jsonl(OPS / (stem+'_device_checks.jsonl'), dict(time_utc=utc_now(), cards=cards, scheduled=scheduled))
        last_guard = time.monotonic()

    model = None
    try:
        guard(True)
        if stop['reason']:
            raise InterruptedError(stop['reason'])
        from backend import DirectCudaLlama
        cfg = json.loads((ROOT / 'configs/experiment.json').read_text())
        cfg.update(run_id='judge_scaling_20260912_v1', model=MODEL, model_revision=REVISION,
                   model_path=MODEL_PATH, max_new_tokens=MAX_NEW_TOKENS, temperature=0.)
        model = DirectCudaLlama(cfg, ROOT, args.index)
        atomic_json(OPS / (stem+'_device.json'), dict(device=model.device_info, config=cfg, pid=os.getpid(), started_utc=utc_now()))
        if args.preflight:
            cases = [dict(question='What instrument does Jo play?', gold_answer='guitar', stored_answer='Jo plays the guitar.'),
                     dict(question='What instrument does Jo play?', gold_answer='guitar', stored_answer='Jo plays the piano.'),
                     dict(question='Which two sports?', gold_answer='swimming and cycling', stored_answer='swimming'),
                     dict(question='Where was Jo born?', gold_answer='No information available', stored_answer="I don't know.")]
            jobs = [dict(prompt=prompt(r,'base_0'), question_id='synthetic_'+str(i)) for i,r in enumerate(cases)]
            batch = generate_batch(model, jobs, guard)
            singleton = generate_batch(model, [jobs[0]], guard)[0]
            assert all(type(r['parsed']['correct']) is bool for r in batch)
            assert batch[0]['parsed']['correct'] == singleton['parsed']['correct']
            # Check attention-mask/cache equivalence on full vocabulary logits, not semantic labels.
            import torch
            masks = [model.encode(j['prompt'])[0].tolist() for j in jobs[:2]]
            width = max(map(len,masks)); pad=model.stop_ids[0]
            ids=torch.tensor([[pad]*(width-len(t))+t for t in masks],device='cuda:0')
            mask=torch.tensor([[0]*(width-len(t))+[1]*len(t) for t in masks],device='cuda:0')
            with torch.inference_mode():
                full=model.model(input_ids=ids,attention_mask=mask,position_ids=(mask.cumsum(-1)-1).clamp(min=0),use_cache=False,logits_to_keep=1).logits[:,-1].float().cpu()
            solo, cache=model._prefill(torch.tensor(masks[0]).reshape(1,-1)); del cache
            cosine=float(torch.nn.functional.cosine_similarity(full[0],solo,dim=0))
            assert cosine > .999, 'Masked batch prefill differs substantially'
            atomic_json(OPS/'preflight_results.json',dict(batch=batch,singleton=singleton,prefill_logit_cosine=cosine,
                expected_booleans=[True,False,False,True],observed_booleans=[x['parsed']['correct'] for x in batch],
                note='Synthetic check only; prompts are not tuned on benchmark labels.'))
            atomic_json(status_path,dict(state='complete',time_utc=utc_now()))
            return 0
        assert json.loads((OPS/'preflight_exit.json').read_text())['exit_code'] == 0
        inputs = [r for i,r in enumerate(rows(OUT/'input_answers.jsonl')) if i % SHARDS == args.shard]
        journal = OUT / ('calls_shard_'+str(args.shard)+'.jsonl')
        completed = rows(journal) if journal.exists() else []
        assert len({x['call_id'] for x in completed}) == len(completed), 'Duplicate journal call'
        done = {x['call_id']:x for x in completed if x['result']['parsed'] is not None}
        if len(done) != len(completed):
            raise RuntimeError('Preserved failed call requires review before resume')
        target = len(inputs)*7

        def flush_jobs(jobs):
            while jobs:
                guard(True)
                if stop['reason']:
                    raise InterruptedError(stop['reason'])
                batch = []
                width = 0
                for job in jobs[:BATCH_SIZE]:
                    n = int(model.encode(job['prompt']).shape[1])
                    if batch and max(width,n)*(len(batch)+1) > MAX_PADDED_TOKENS:
                        break
                    batch.append(job); width=max(width,n)
                jobs = jobs[len(batch):]
                batch_id = str(args.shard)+'_'+str(len(done))+'_'+str(time.time_ns())
                append_jsonl(OPS/(stem+'_starts.jsonl'),dict(time_utc=utc_now(),batch_id=batch_id,calls=batch))
                results = generate_batch(model,batch,guard)
                for job,result in zip(batch,results):
                    record=dict(**job,result=result,time_utc=utc_now(),batch_id=batch_id,attempt=0)
                    append_jsonl(journal,record)
                    if result['parsed'] is not None:
                        done[job['call_id']]=record
                atomic_json(status_path,dict(state='running',done=len(done),total=target,shard=args.shard,
                    time_utc=utc_now(),last_batch_seconds=results[0]['batch_seconds'],last_batch_size=len(batch),pid=os.getpid()))
                print(f'{stem}: {len(done)}/{target}; batch {len(batch)} in {results[0]["batch_seconds"]:.2f}s',flush=True)
                if any(r['parsed'] is None for r in results):
                    raise RuntimeError('Judge JSON failure; preserved without assigning a false verdict')

        base_jobs=[]
        for r in inputs:
            for kind in KINDS[:3]:
                cid=r['question_id']+'/'+kind
                if cid not in done:
                    base_jobs.append(dict(call_id=cid,question_id=r['question_id'],kind=kind,prompt=prompt(r,kind),previous_call_ids=[]))
        flush_jobs(base_jobs)
        meta_jobs=[]
        for r in inputs:
            for kind in KINDS[3:]:
                cid=r['question_id']+'/'+kind
                if cid in done:
                    continue
                deps=[r['question_id']+'/'+k for k in (['base_0'] if kind=='meta_single_0' else KINDS[:3])]
                previous=[done[k]['result']['parsed'] for k in deps]
                meta_jobs.append(dict(call_id=cid,question_id=r['question_id'],kind=kind,prompt=prompt(r,kind,previous),previous_call_ids=deps))
        flush_jobs(meta_jobs)
        assert len(done)==target
        verify()
        atomic_json(status_path,dict(state='complete',done=len(done),total=target,time_utc=utc_now()))
        return 0
    except InterruptedError as exc:
        atomic_json(status_path,dict(state='checkpointed',reason=str(exc),time_utc=utc_now()))
        return 75
    finally:
        if model is not None:
            model.close()
        for h in handles:
            if h is not None:
                h.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--shard',type=int,default=0); p.add_argument('--index',type=int,required=True)
    p.add_argument('--uuid',required=True); p.add_argument('--preflight',action='store_true')
    args=p.parse_args(); stem='preflight' if args.preflight else 'shard_'+str(args.shard)
    rc=1
    try:
        rc=run(args)
    except BaseException:
        traceback.print_exc()
    finally:
        atomic_json(OPS/(stem+'_exit.json'),dict(exit_code=rc,time_utc=utc_now()))
    raise SystemExit(rc)
