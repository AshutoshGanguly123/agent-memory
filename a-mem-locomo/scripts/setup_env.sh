#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
test -n "${TMUX:-}"
if [ ! -d .git ]; then git init -b main; fi
mkdir -p results/setup
if [ -f results/setup/setup.exit ]; then mv results/setup/setup.exit "results/setup/setup.exit.previous.$(date +%s)"; fi
trap 'echo "$?" > results/setup/setup.exit' EXIT
UV=/home/ashutosh.ganguly/.local/bin/uv
if [ ! -x .venv/bin/python ]; then "$UV" venv --python /usr/bin/python3 .venv; fi
"$UV" pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 'torch==2.7.1+cu128'
if [ -f requirements.lock ]; then
  "$UV" pip install --python .venv/bin/python -r requirements.lock
else
  "$UV" pip install --python .venv/bin/python -r requirements.in
fi
"$UV" pip install --python .venv/bin/python --no-deps -e .
"$UV" pip freeze --python .venv/bin/python > requirements.lock
.venv/bin/python - <<'PY'
from pathlib import Path
p=Path('requirements.lock')
p.write_text('\n'.join('-e .' if line.startswith('-e ') else line for line in p.read_text().splitlines())+'\n')
PY
.venv/bin/python scripts/prepare_assets.py
.venv/bin/python scripts/setup_report.py
touch results/setup/ASSETS_READY
