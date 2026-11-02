"""Durable preparation and a four-conversation GPU-availability supervisor."""
import argparse
import os
import shlex
import subprocess
from pathlib import Path
from amem_reproduction.storage import atomic_json, utc_now

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OPS = ROOT / 'results/operations/recovery_gpu_nodes_v2'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'run'])
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    OPS.mkdir(parents=True, exist_ok=True)
    import re
    if not re.fullmatch(r'gpu-\d{2}', os.uname().nodename.split('.')[0]):
        raise RuntimeError('User permits only gpu-xx nodes')
    if args.inside:
        if not os.environ.get('TMUX'):
            raise RuntimeError('Dedicated tmux session required')
        os.environ['AMEM_EXECUTION_MODE'] = 'direct_ssh'
        script = CODE / ('prepare.py' if args.mode == 'prepare' else 'supervise.py')
        result = subprocess.run([str(ROOT / '.venv/bin/python'), '-u', str(script)], cwd=ROOT)
        atomic_json(OPS / (args.mode + '_launcher_exit.json'),
                    dict(exit_code=result.returncode, time_utc=utc_now()))
        return result.returncode
    name = 'amem_recovery_nodes_' + args.mode
    if subprocess.run(['tmux', 'has-session', '-t', name], capture_output=True).returncode == 0:
        print('Existing tmux session: ' + name)
        return 0
    command = 'source ' + shlex.quote(str(ROOT / 'scripts/runtime.sh')) + '; exec '
    command += shlex.join([str(ROOT / '.venv/bin/python'), '-u', str(CODE / 'start.py'), args.mode, '--inside'])
    command += ' >> ' + shlex.quote(str(OPS / (args.mode + '_launch.log'))) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-c', str(ROOT), '-s', name,
                    shlex.join(['bash', '-lc', command])], check=True)
    print('Started tmux ' + name)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
