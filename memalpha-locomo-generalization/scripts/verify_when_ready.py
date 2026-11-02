"""Wait for environment setup, then complete CPU-only imports/cache/tests."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
setup = ROOT / 'results/setup'
ready = setup / 'environment.exit.json'
code, error = 1, None
try:
    if not os.environ.get('TMUX'):
        raise RuntimeError('Run in server tmux')
    while not ready.exists():
        time.sleep(10)
    if json.loads(ready.read_text())['exit_code'] != 0:
        raise RuntimeError('Environment setup failed; CPU verification was not run')
    result = subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/verify_environment.py')], cwd=ROOT)
    code = result.returncode
except BaseException as exc:
    error = repr(exc)
    raise
finally:
    with (setup / 'verification.exit.json').open('w') as output:
        json.dump({'exit_code': code, 'error': error, 'finished_utc': datetime.now(timezone.utc).isoformat()}, output, indent=2)
        output.flush()
        os.fsync(output.fileno())
