#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
test -n "${TMUX:-}"
trap 'echo "$?" > results/setup/checks.exit' EXIT
python -m compileall -q src scripts tests
python -m pytest -q
python - <<'PY'
from pathlib import Path
import subprocess
for p in Path('scripts').glob('*.sh'):
    subprocess.run(['bash','-n',str(p)],check=True)
PY
