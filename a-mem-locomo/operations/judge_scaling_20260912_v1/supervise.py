"""Durable direct-node execution; yields to physical and scheduler owners."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from protocol import *
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(1,str(ROOT/'operations/recovery_gpu_nodes_v3'))
import common
from amem_reproduction.storage import atomic_json, append_jsonl, utc_now


def main():
    common.validate_context();verify()
    assert common.node_name()=='gpu-02'
    lock=common.lock(OPS/'locks/supervisor.lock')
    assert lock is not None,'Another supervisor is running'
    if (OPS/'STOP').exists():raise RuntimeError('STOP exists')
    stop={'value':False}
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
        signal.signal(sig,lambda *_:stop.update(value=True))
    live={};cooldown={};complete=set()
    started=utc_now()

    def inventory():
        cards=common.inventory();scheduled=common.scheduled_jobs()
        append_jsonl(OPS/'inventory.jsonl',dict(time_utc=utc_now(),cards=cards,scheduled=scheduled))
        return [c for c in cards if not common.reserved(c,scheduled) and not c['pids'] and c['free_mib']>=28672
                and c['utilization']==0 and cooldown.get(c['uuid'],0)<time.time()]

    def launch(card,shard,preflight=False):
        stem='preflight' if preflight else 'shard_'+str(shard)
        for name in [stem+'_exit.json',stem+'_status.json']:
            path=OPS/name
            if path.exists():
                history=OPS/'resumes';history.mkdir(exist_ok=True)
                path.rename(history/(str(time.time_ns())+'_'+name))
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=card['uuid'],PYTHONHASHSEED='42',TOKENIZERS_PARALLELISM='false',
            OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',AMEM_EXECUTION_MODE='direct_ssh')
        cmd=[str(ROOT/'.venv/bin/python'),str(CODE/'worker.py'),'--shard',str(shard),'--index',str(card['index']),'--uuid',card['uuid']]
        if preflight:cmd.append('--preflight')
        log=(OPS/(stem+'.log')).open('a')
        proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        append_jsonl(OPS/'launches.jsonl',dict(time_utc=utc_now(),pid=proc.pid,shard=shard,preflight=preflight,card=card))
        return dict(proc=proc,log=log,card=card)

    try:
        preflight=OPS/'preflight_exit.json'
        if not preflight.exists() or json.loads(preflight.read_text())['exit_code']!=0:
            cards=inventory()
            while not cards:
                atomic_json(OPS/'coordinator.json',dict(state='waiting_for_preflight_gpu',started_utc=started,time_utc=utc_now()))
                if stop['value'] or (OPS/'STOP').exists():return 75
                time.sleep(15);cards=inventory()
            worker=launch(cards[0],0,True)
            atomic_json(OPS/'coordinator.json',dict(state='preflight',pid=worker['proc'].pid,time_utc=utc_now()))
            rc=worker['proc'].wait();worker['log'].close()
            if rc!=0:raise RuntimeError('Preflight stopped: '+str(rc))
        for i in range(SHARDS):
            p=OPS/f'shard_{i}_status.json'
            if p.exists() and json.loads(p.read_text()).get('state')=='complete':complete.add(i)
        while len(complete)<SHARDS:
            if stop['value'] or (OPS/'STOP').exists():
                for job in live.values():job['proc'].send_signal(signal.SIGUSR1)
                for job in live.values():job['proc'].wait();job['log'].close()
                return 75
            for shard,job in list(live.items()):
                rc=job['proc'].poll()
                if rc is None:continue
                job['log'].close();del live[shard]
                if rc==0:complete.add(shard)
                elif rc==75:cooldown[job['card']['uuid']]=time.time()+900
                else:raise RuntimeError(f'Shard {shard} failed with code {rc}; inspect preserved logs')
            cards=inventory()
            occupied={job['card']['uuid'] for job in live.values()}
            cards=[c for c in cards if c['uuid'] not in occupied]
            for shard in range(SHARDS):
                if shard in complete or shard in live or not cards:continue
                live[shard]=launch(cards.pop(0),shard)
            progress={}
            for shard in range(SHARDS):
                p=OPS/f'shard_{shard}_status.json'
                if p.exists():progress[str(shard)]=json.loads(p.read_text())
            atomic_json(OPS/'coordinator.json',dict(state='running' if live else 'waiting_for_gpus',started_utc=started,time_utc=utc_now(),
                complete=sorted(complete),workers={str(k):dict(pid=v['proc'].pid,card=v['card']) for k,v in live.items()},progress=progress))
            time.sleep(10)
        with (OPS/'finalize.log').open('a') as log:
            rc=subprocess.run([str(ROOT/'.venv/bin/python'),str(CODE/'finalize.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        if rc:raise RuntimeError('Final analysis failed; see finalize.log')
        atomic_json(OPS/'coordinator.json',dict(state='complete',started_utc=started,finished_utc=utc_now(),total_calls=12544))
        return 0
    finally:
        for job in live.values():
            if job['proc'].poll() is None:job['proc'].send_signal(signal.SIGUSR1)
        for job in live.values():
            job['proc'].wait();job['log'].close()
        lock.close()


if __name__=='__main__':
    rc=1
    try:rc=main()
    except BaseException:
        traceback.print_exc()
        atomic_json(OPS/'coordinator.json',dict(state='failed',time_utc=utc_now()))
    finally:atomic_json(OPS/'supervisor_exit.json',dict(exit_code=rc,time_utc=utc_now()))
    raise SystemExit(rc)
