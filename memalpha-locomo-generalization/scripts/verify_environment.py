"""CPU-only final runtime and upstream import check before GPU preflight."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'upstream'))
os.environ['TIKTOKEN_CACHE_DIR'] = str(ROOT / 'models/tiktoken_cache')
os.environ['NLTK_DATA'] = str(ROOT / '.venv/nltk_data')
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tiktoken
import nltk
import torch
import transformers
import vllm
import openai
import qwen_agent
from memory import Memory
from functions import get_memory_tool_schemas
from agent import MemoryAgent
from memory_server import MemoryProcessor

tiktoken.encoding_for_model('gpt-4o-mini').encode('Synthetic tokenization cache check.')
nltk.data.path.insert(0, os.environ['NLTK_DATA'])
nltk.data.find('tokenizers/punkt_tab')
subprocess.run([sys.executable, '-m', 'pytest', '-q', str(ROOT / 'tests/test_ops.py')], check=True, cwd=ROOT)
packages = dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))
cache = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT / 'models/tiktoken_cache').iterdir() if p.is_file()}
record = {'state': 'ready', 'python': platform.python_version(), 'packages': packages, 'tiktoken_cache_sha256': cache,
          'gpu_inference_performed': False}
path = ROOT / 'results/setup/environment_verified.json'
with path.open('w') as output:
    json.dump(record, output, indent=2, sort_keys=True)
    output.flush()
    os.fsync(output.fileno())
print(json.dumps({'state': 'ready', 'packages': len(packages), 'tiktoken_cache_files': len(cache)}))
