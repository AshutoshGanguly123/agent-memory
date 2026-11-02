"""Durable calls and retries across preemption; no hidden scientific recovery."""
import json
import os
import time
from pathlib import Path
from .storage import append_jsonl, digest, read_jsonl, utc_now


class StageFailure(RuntimeError): pass
class CheckpointStop(RuntimeError): pass


class CallJournal:
    def __init__(self, backend, config, path, status_callback=None):
        self.backend,self.config,self.path = backend,config,Path(path)
        self.status_callback = status_callback
        backend.progress_callback = status_callback
        self.cached,self.failed_attempts = {},{}
        for event in read_jsonl(path,repair=True):
            if event.get('event') != 'call_complete': continue
            if event['status'] == 'ok': self.cached[event['call_id']] = event
            if event['status'] == 'schema_error':
                self.failed_attempts[event['call_id']] = max(self.failed_attempts.get(event['call_id'],0),event['attempt']+1)

    def boundary(self):
        callback = getattr(self.backend,'stop_check',None)
        if callback: callback()

    def complete(self,phase,logical_id,prompt,schema):
        call_id = digest(dict(logical_id=logical_id,prompt=prompt,schema=schema,config=self.config))
        if call_id in self.cached: return self.cached[call_id]['parsed'],call_id
        self.boundary()
        tokens = self.backend.encode(prompt)
        n = int(tokens.shape[1])
        base = dict(call_id=call_id,logical_id=logical_id,phase=phase,prompt=prompt,
            system_prompt='You must respond with a JSON object.',schema=schema,input_tokens=n,
            output_reserve=self.config['max_new_tokens'],max_context=self.config['max_context'],
            context_overflow=n+self.config['max_new_tokens']>self.config['max_context'],
            worker_pid=os.getpid(),slurm_job_id=os.environ.get('SLURM_JOB_ID'))
        if base['context_overflow']:
            append_jsonl(self.path,dict(base,event='call_complete',status='context_overflow',output_tokens=0,latency_seconds=0,time_utc=utc_now()))
            raise StageFailure(f'Context overflow in {phase}: {n} + output reserve')
        for attempt in range(self.failed_attempts.get(call_id,0),self.config['max_attempts']):
            seed = (self.config['seed']+int(digest([call_id,attempt])[:8],16))%(2**31)
            entry = dict(base,attempt=attempt,seed=seed,time_utc=utc_now())
            append_jsonl(self.path,dict(entry,event='call_started'))
            if self.status_callback:
                self.status_callback(current_call=logical_id,current_call_phase=phase,current_attempt=attempt,
                    current_output_tokens=0,call_started_utc=utc_now())
            started = time.perf_counter()
            try:
                result = self.backend.generate(tokens,schema,seed)
            except BaseException as exc:
                failed = dict(entry,event='call_complete',status='operational_interrupt' if isinstance(exc,CheckpointStop) else 'backend_error',
                    error=repr(exc),latency_seconds=time.perf_counter()-started,output_tokens=None,completed_at=utc_now())
                failed.update(getattr(self.backend,'partial_generation',{}))
                append_jsonl(self.path,failed)
                raise
            entry.update(result,event='call_complete',completed_at=utc_now(),status='ok' if result['parsed'] is not None else 'schema_error')
            append_jsonl(self.path,entry)
            if result['parsed'] is not None: self.cached[call_id] = entry
            else: self.failed_attempts[call_id] = attempt+1
            self.boundary()
            if result['parsed'] is not None: return result['parsed'],call_id
        raise StageFailure(f"Schema failure after {self.config['max_attempts']} attempts: {logical_id}")
