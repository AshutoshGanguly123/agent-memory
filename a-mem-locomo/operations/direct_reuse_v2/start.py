"""Start the validated replacement supervisor in its own durable tmux session."""
import argparse
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = Path(__file__).resolve().parent
OPS = ROOT/'results/operations/direct_reuse_v2'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inside', action='store_true')
    args = parser.parse_args()
    if os.uname().nodename.split('.')[0] != 'gpu-04':
        raise RuntimeError('Restricted to gpu-04')
    if args.inside:
        if not os.environ.get('TMUX'):
            raise RuntimeError('tmux required')
        os.environ['AMEM_EXECUTION_MODE'] = 'direct_ssh'
        result = subprocess.run([str(ROOT/'.venv/bin/python'), '-u', str(CODE/'supervise.py')], cwd=ROOT)
        (OPS/'launcher_exit.json').write_text(json.dumps(dict(exit_code=result.returncode,
            time_utc=datetime.now(timezone.utc).isoformat())))
        return result.returncode
    command = 'source '+shlex.quote(str(ROOT/'scripts/runtime.sh'))+'; '
    command += shlex.join([str(ROOT/'.venv/bin/python'), '-u', str(CODE/'start.py'), '--inside'])
    command += ' >> '+shlex.quote(str(OPS/'launch.log'))+' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', 'amem_direct_full_v2', '-c', str(ROOT),
                    'bash', '-c', command], check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

