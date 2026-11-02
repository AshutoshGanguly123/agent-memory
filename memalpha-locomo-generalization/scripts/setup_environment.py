"""Build an independent inference environment; never modifies older projects."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
UV = Path.home() / '.local/bin/uv'


def run(*args):
    subprocess.run([str(x) for x in args], check=True, cwd=ROOT)


def main():
    if sys.platform != 'linux' or not os.environ.get('TMUX'):
        raise RuntimeError('Run setup in server tmux')
    run(UV, 'venv', '--python', '/usr/bin/python3', ROOT / '.venv')
    python = ROOT / '.venv/bin/python'
    run(UV, 'pip', 'install', '--python', python, '-r', ROOT / 'scripts/requirements.inference.txt')
    run(UV, 'pip', 'check', '--python', python)
    run(python, '-c', 'import torch,transformers,vllm,openai,qwen_agent; print({"torch":torch.__version__,"transformers":transformers.__version__,"vllm":vllm.__version__,"openai":openai.__version__})')
    run(python, '-m', 'nltk.downloader', '-d', ROOT / '.venv/nltk_data', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4')
    packages = subprocess.check_output([str(UV), 'pip', 'freeze', '--python', str(python)], text=True)
    (ROOT / 'results/setup/requirements.resolved.txt').write_text(packages)


if __name__ == '__main__':
    code, error = 0, None
    try:
        main()
    except BaseException as exc:
        code, error = 1, repr(exc)
        raise
    finally:
        path = ROOT / 'results/setup/environment.exit.json'
        value = {'exit_code': code, 'error': error, 'finished_utc': datetime.now(timezone.utc).isoformat()}
        with path.open('w') as output:
            json.dump(value, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
