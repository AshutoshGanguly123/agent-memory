import argparse
import fcntl
import faulthandler
import json
import os
import signal
import time
import traceback
from pathlib import Path
from .calls import CallJournal, StageFailure, CheckpointStop
from .data import load_dataset, turns, evidence_ids
from .memory import Memory
from .prompts import Prompts
from .scoring import score_answer
from .storage import append_jsonl, atomic_json, digest, read_jsonl, utc_now


def coverage(qa, retrieved, all_ids):
    evidence = evidence_ids(qa.get("evidence", []))
    supplied = set(retrieved.get("supplied_ids", []))
    hits = sorted(set(evidence) & supplied)
    return {"evidence_ids": evidence, "evidence_hits": hits,
            "invalid_evidence_ids": sorted(set(evidence) - set(all_ids)),
            "evidence_coverage": len(hits) / len(evidence) if evidence else None,
            "all_evidence_retrieved": set(evidence) <= supplied if evidence else None}


def run_conversation(sample, backend, embedder, prompts, config, out, update):
    sid = sample["sample_id"]
    folder = out / "conversations" / sid
    folder.mkdir(parents=True, exist_ok=True)
    checkpoint = folder / "memory.json"
    journal = CallJournal(backend, config, folder / "calls.jsonl", update)
    state = json.loads(checkpoint.read_text()) if checkpoint.exists() else None
    memory = Memory(embedder, prompts, journal, config, state)
    rows = turns(sample)
    all_ids = [t["dia_id"] for t in rows]
    if [n["id"] for n in memory.notes] != all_ids[:len(memory.notes)]:
        raise ValueError("Checkpoint does not match ordered conversation")
    if memory.state['frozen'] and len(memory.notes) != len(rows):
        raise ValueError('Frozen checkpoint is incomplete')
    frozen_path = folder/'memory_frozen.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text())['fingerprint'] != memory.fingerprint():
        raise ValueError('Frozen checkpoint fingerprint changed')
    # Recover any trailing interrupted journal write before this worker appends.
    read_jsonl(folder / "memory_events.jsonl", repair=True)
    update(conversation=sid, phase="memory_build", notes_done=len(memory.notes), notes_total=len(rows),
           questions_done=0, questions_total=len(sample["qa"]))
    for i in range(len(memory.notes), len(rows)):
        event = memory.add(rows[i], sid + "/" + rows[i]["dia_id"])
        event.update(event_id=digest([sid, i]), time_utc=utc_now(), note_number=i+1)
        # An event may precede a crash before checkpoint; event_id supports deduplication.
        append_jsonl(folder / "memory_events.jsonl", event)
        memory.save(checkpoint)
        update(notes_done=i+1, memory_footprint=event["footprint"], build_seconds=memory.state["build_seconds"])
    memory.state["frozen"] = True
    memory.save(checkpoint)
    frozen = memory.fingerprint()
    atomic_json(folder / "memory_frozen.json", {"fingerprint": frozen, "notes": len(memory.notes),
                                               "time_utc": utc_now(), "footprint": memory.footprint()})
    answers_path = folder / "answers.jsonl"
    previous = read_jsonl(answers_path, repair=True)
    completed = {a["question_index"] for a in previous}
    if len(completed) != len(previous) or any(a['memory_fingerprint'] != frozen for a in previous):
        raise ValueError('Duplicate questions or frozen-memory mismatch')
    update(phase="qa", questions_done=len(completed))
    for qi, qa in enumerate(sample["qa"]):
        if qi in completed:
            continue
        started = time.perf_counter()
        result = {"sample_id": sid, "question_index": qi, "question_id": f"{sid}/q{qi:04d}",
                  "question": qa["question"], "category": int(qa["category"]),
                  "gold_answer": qa.get("answer"), "adversarial_answer": qa.get("adversarial_answer"),
                  "memory_fingerprint": frozen, "retrieval": {}, "prediction": "", "call_ids": {}}
        try:
            rewritten, call_id = journal.complete("rewrite", result["question_id"] + "/rewrite",
                                                 *prompts.rewrite(qa["question"]))
            result["call_ids"]["rewrite"] = call_id
            result["rewritten_query"] = rewritten["keywords"]
            context, retrieval = memory.retrieve(rewritten["keywords"])
            result["retrieval"] = retrieval
            response, call_id = journal.complete("answer", result["question_id"] + "/answer",
                                                *prompts.answer(qa["question"], int(qa["category"]), context))
            result["call_ids"]["answer"] = call_id
            result.update(prediction=response["answer"], status="ok")
        except StageFailure as exc:
            result.update(status="failed", error=str(exc))
        result.update(coverage(qa, result["retrieval"], all_ids))
        result.update(score=score_answer(result["prediction"], qa) if result["status"] == "ok" else 0.0,
                      latency_seconds=time.perf_counter()-started, time_utc=utc_now())
        append_jsonl(answers_path, result)
        completed.add(qi)
        update(questions_done=len(completed), last_question_status=result["status"])
    if memory.fingerprint() != frozen:
        raise RuntimeError("Memory changed during QA")
    atomic_json(folder / "complete.json", {"sample_id": sid, "questions": len(completed),
                                          "notes": len(memory.notes), "time_utc": utc_now()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--worker',type=int,required=True)
    parser.add_argument('--preflight',action='store_true')
    args = parser.parse_args()
    from .provenance import verify_manifest
    from .device_guard import DeviceBusy
    root = args.root.resolve()
    config = json.loads((root/'configs/experiment.json').read_text())
    kind = 'preflight' if args.preflight else 'raw_results'
    out = root/'results'/kind/config['run_id']
    out.mkdir(parents=True,exist_ok=True)
    lock = (out/f'worker_{args.worker}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.preflight: manifest = verify_manifest(root)
    stop = {'requested':False}
    deadline = float(os.environ.get('AMEM_STOP_EPOCH','inf'))
    def request_stop(signum,frame): stop['requested'] = True
    signal.signal(signal.SIGUSR1,request_stop)
    signal.signal(signal.SIGTERM,request_stop)
    def stop_check():
        if stop['requested'] or time.time() >= deadline or (os.environ.get('AMEM_STOP_FILE') and Path(os.environ['AMEM_STOP_FILE']).exists()):
            raise CheckpointStop('Slurm slice end: durable call boundary')
    status = dict(worker=args.worker,pid=os.getpid(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        started_utc=utc_now(),state='running',phase='loading',config_sha256=digest(config))
    def update(**values):
        status.update(values,updated_utc=utc_now())
        atomic_json(out/f'worker_{args.worker}.json',status)
    update()
    backend = None
    try:
        print('Loading CPU MiniLM embedder',flush=True)
        from sentence_transformers import SentenceTransformer
        from .cuda_backend import CudaLlama
        embedder = SentenceTransformer(str(root/config['embedding_path']),device='cpu',local_files_only=True)
        print('MiniLM ready; initializing CUDA Llama',flush=True)
        backend = CudaLlama(config,root,args.worker)
        backend.stop_check = stop_check
        update(model_load_seconds=backend.load_seconds,device=backend.device_info,embedding_max_seq_length=embedder.max_seq_length)
        prompts = Prompts(root)
        if args.preflight:
            from .preflight import exercise
            exercise(root,out,backend,embedder,prompts,config,update)
        else:
            samples = {s['sample_id']:s for s in load_dataset(root/config['dataset'],config['dataset_sha256'])}
            for sid in manifest['assignments'][str(args.worker)]:
                stop_check()
                run_conversation(samples[sid],backend,embedder,prompts,config,out,update)
        update(state='complete',phase='complete',finished_utc=utc_now())
    except DeviceBusy as exc:
        update(state='waiting_for_device',reason=str(exc),finished_utc=utc_now())
        return 78
    except CheckpointStop as exc:
        update(state='checkpointed',reason=str(exc),finished_utc=utc_now())
        return 75
    except BaseException as exc:
        update(state='failed',error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        if backend is not None: backend.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
