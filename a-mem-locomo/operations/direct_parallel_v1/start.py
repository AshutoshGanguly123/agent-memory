"""Start a durable preparation, validation or direct run in a named tmux."""
import argparse
import os
import shlex
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OPS = ROOT / 'results/operations/direct_parallel_v1'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'probe', 'full'])
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    OPS.mkdir(parents=True, exist_ok=True)
    if args.mode != 'prepare' and os.uname().nodename.split('.')[0] != 'gpu-04':
        raise RuntimeError('SSH to gpu-04 before starting GPU work')
    if args.inside:
        if not os.environ.get('TMUX'):
            raise RuntimeError('Runner requires tmux')
        os.environ['AMEM_EXECUTION_MODE'] = 'direct_ssh'
        command = [str(ROOT / '.venv/bin/python'), '-u', str(CODE / ('prepare.py' if args.mode == 'prepare' else 'supervise.py'))]
        if args.mode == 'probe':
            command.append('--probe')
        result = subprocess.run(command, cwd=ROOT)
        from amem_reproduction.storage import atomic_json, utc_now
        atomic_json(OPS / (args.mode + '_launcher_exit.json'), dict(exit_code=result.returncode, time_utc=utc_now()))
        return result.returncode
    name = 'amem_direct_' + args.mode
    command = 'source ' + shlex.quote(str(ROOT / 'scripts/runtime.sh')) + '; '
    command += shlex.join([str(ROOT / '.venv/bin/python'), str(CODE / 'start.py'), args.mode, '--inside'])
    command += ' >> ' + shlex.quote(str(OPS / (args.mode + '_launch.log'))) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-c', str(ROOT), '-s', name, 'bash', '-c', command], check=True)
    print('Started tmux ' + name + '; log ' + str(OPS / (args.mode + '_launch.log')))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
