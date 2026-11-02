"""Read existing journals; write an ingestion cost report outside frozen inference.

Run with the project .venv in tmux on gpu-04. No model is loaded and no
checkpoint, journal, configuration, launcher or GPU process is modified.
"""
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def jsonl(path):
    """Ignore only an incomplete trailing write; never repair the live journal."""
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    result = []
    for i, line in enumerate(lines):
        try:
            result.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if i == len(lines) - 1 and not line.endswith(b'\n'):
                continue
            raise
    return result


def distribution(values):
    known = sorted(v for v in values if v is not None)
    return dict(known_count=len(known), unknown_count=len(values)-len(known),
                total=sum(known), mean=statistics.mean(known) if known else None,
                median=statistics.median(known) if known else None,
                p95=known[math.ceil(.95*len(known))-1] if known else None)


def summarize(calls):
    return dict(completed_attempts=len(calls),
                status_counts=dict(Counter(c['status'] for c in calls)),
                retry_attempts=sum(c.get('attempt', 0) > 0 for c in calls),
                input_tokens=distribution([c.get('input_tokens') for c in calls]),
                output_tokens=distribution([c.get('output_tokens') for c in calls]),
                latency_seconds=distribution([c.get('latency_seconds') for c in calls]),
                prefill_seconds=distribution([c.get('prefill_seconds') for c in calls]),
                failed_attempt_seconds=sum(c.get('latency_seconds') or 0 for c in calls if c['status'] != 'ok'),
                retry_attempt_seconds=sum(c.get('latency_seconds') or 0 for c in calls if c.get('attempt', 0) > 0))


def main():
    if not os.environ.get('TMUX'):
        raise RuntimeError('Run this read-only profile in tmux with durable logs.')
    started = datetime.now(timezone.utc)
    stamp = started.strftime('%Y%m%dT%H%M%SZ')
    destination = ROOT/'results/operations/ingestion_profile_v1'/stamp
    destination.mkdir(parents=True, exist_ok=False)
    config = read(ROOT/'configs/experiment.json')
    raw = ROOT/'results/raw_results'/config['run_id']
    ops = ROOT/'results/operations/direct_parallel_v1'
    coordinator = read(ops/'full_coordinator.json')
    operational = read(ops/'manifest.json')
    original = read(raw/'manifest.json')
    frozen = [(ROOT/p, h) for p, h in original['source_sha256'].items()]
    frozen += [(ROOT/'operations/direct_parallel_v1'/p, h)
               for p, h in operational['adapter_source_sha256'].items()]
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == h for p, h in frozen)
    calls, unmatched, turns, checkpoints = [], [], [], []
    for folder in sorted((raw/'conversations').glob('*')):
        if not folder.is_dir():
            continue
        memory = read(folder/'memory.json')
        committed = {n['id'] for n in memory.get('notes', [])}
        checkpoints.append(dict(conversation=folder.name, notes=len(committed)))
        records = jsonl(folder/'calls.jsonl')
        key = lambda c: (c['call_id'], c.get('attempt'), c.get('time_utc'), c.get('worker_pid'))
        complete = [c for c in records if c.get('event') == 'call_complete'
                    and c.get('phase') in ('note', 'evolution')]
        assert len({key(c) for c in complete}) == len(complete), 'Duplicate completed attempt'
        done = {key(c) for c in complete}
        unmatched.extend({k: c.get(k) for k in ('call_id', 'phase', 'attempt', 'time_utc', 'worker_pid')}
                         for c in records if c.get('event') == 'call_started'
                         and c.get('phase') in ('note', 'evolution') and key(c) not in done)
        calls.extend(complete)
        by_id = defaultdict(list)
        for c in complete:
            by_id[c['call_id']].append(c)
        events = {e['event_id']: e for e in jsonl(folder/'memory_events.jsonl')}
        for event in events.values():
            if event['turn_id'] not in committed:
                continue
            ids = [event['note_call_id'], event['evolution_call_id']]
            assert all(any(c['status'] == 'ok' for c in by_id[i]) for i in ids)
            attempts = [c for i in ids for c in by_id[i]]
            assert all(c.get('output_tokens') is not None and c.get('latency_seconds') is not None
                       for c in attempts), 'Unknown completed-attempt cost; report separately before aggregating'
            turns.append(dict(conversation=folder.name, turn_id=event['turn_id'],
                              note_number=event['note_number'], commit_event_utc=event['time_utc'],
                              phase='direct_parallel' if event['time_utc'] >= coordinator['started_utc'] else 'initial_sequential',
                              completed_attempts=len(attempts),
                              input_tokens=sum(c['input_tokens'] for c in attempts),
                              output_tokens=sum(c['output_tokens'] for c in attempts),
                              llm_attempt_seconds=sum(c['latency_seconds'] for c in attempts),
                              observed_build_seconds=event['latency_seconds'],
                              should_evolve=event['decision']['should_evolve'],
                              index_refreshed=event['index_refreshed'],
                              neighbors=len(event['neighbor_indices'])))
    assert len(turns) == sum(c['notes'] for c in checkpoints), 'Committed-turn evidence missing'
    scopes = {'all_run': calls,
              'initial_sequential': [c for c in calls if c['time_utc'] < coordinator['started_utc']],
              'direct_parallel': [c for c in calls if c['time_utc'] >= coordinator['started_utc']]}
    stage_stats = {scope: {stage: summarize([c for c in selected if c['phase'] == stage])
                           for stage in ('note', 'evolution')} for scope, selected in scopes.items()}
    per_turn = {}
    for scope in scopes:
        selected = turns if scope == 'all_run' else [t for t in turns if t['phase'] == scope]
        per_turn[scope] = dict(committed_turns=len(selected), **{
            metric: distribution([t[metric] for t in selected]) for metric in
            ('input_tokens', 'output_tokens', 'llm_attempt_seconds', 'observed_build_seconds', 'completed_attempts')})
    finished = datetime.now(timezone.utc)
    phase_end = datetime.fromisoformat(coordinator['finished_utc']) if coordinator.get('finished_utc') else finished
    elapsed = (phase_end-datetime.fromisoformat(coordinator['started_utc'])).total_seconds()
    direct_notes = len([t for t in turns if t['phase'] == 'direct_parallel'])
    hardware = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total,memory.used,utilization.gpu', '--format=csv'], text=True)
    report = dict(schema_version=1, run_id=config['run_id'], partial=coordinator.get('state') != 'complete',
                  collection_started_utc=started.isoformat(), collection_finished_utc=finished.isoformat(),
                  configuration=config, hostname=os.uname().nodename, gpu_snapshot=hardware,
                  cpu_info=subprocess.check_output(['lscpu'], text=True),
                  coordinator=coordinator, checkpoints=checkpoints, stages=stage_stats,
                  committed_turns=per_turn, started_attempts_without_completion=unmatched,
                  direct_phase=dict(started_utc=coordinator['started_utc'], elapsed_wall_seconds=elapsed,
                                    committed_turns=direct_notes, notes_per_wall_minute=60*direct_notes/elapsed,
                                    preserved_initial_notes=operational['preserved_notes'],
                                    live_workers_at_snapshot=len(coordinator['workers'])),
                  failures=[read(p) for p in sorted((ops/'failures').glob('*.json'))],
                  verification=dict(frozen_hashes_matched=len(frozen), committed_turns_matched=len(turns)),
                  profiler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  definitions=dict(
                      stage_means='Completed LLM attempts, including retries and invalid outputs; split by attempt start time. Unknowns are counted, not replaced with zero.',
                      turn_means='All completed note/evolution attempts referenced by each committed turn. Includes earlier attempts replayed across the execution transition; split by commit event time. Excludes work on uncommitted or terminally failed turns, which remains in stage costs.',
                      build_time='Existing memory.add timer includes LLM, embedding, retrieval and in-memory update work, but excludes checkpoint writing/model loading. Replayed calls may undercount this timer.',
                      latency='Wall duration of one worker call; summed across workers is not elapsed wall time. Includes CPU decoding/validation overhead, so it is not pure GPU execution time.',
                      throughput='Committed turns divided by elapsed direct-phase wall time. Concurrency varies and includes startup/idle/failure effects; not an ETA or a speedup comparison.',
                      percentile='Nearest-rank p95; mean and median both retained.',
                      unmatched='May be active, interrupted or abandoned. Known completed costs exclude these attempts; this is not a crash count.',
                      snapshot='Live files read without repair or locks. Checkpoints are atomic; journals are append-only. This is a short collection interval, not a globally atomic snapshot.'))
    (destination/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    with (destination/'per_turn.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(turns[0]) if turns else ['conversation'])
        writer.writeheader()
        writer.writerows(turns)
    (destination/'profile.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(dict(output_directory=str(destination), stages=stage_stats['direct_parallel'],
                          per_turn=per_turn['direct_parallel'], progress=report['direct_phase'],
                          failures=[dict(sample_id=f['sample_id'], error=f['error'], time_utc=f['time_utc'])
                                    for f in report['failures']], verification=report['verification']), indent=2))


if __name__ == '__main__':
    main()

