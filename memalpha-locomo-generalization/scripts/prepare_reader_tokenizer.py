"""Make the pinned reader tokenizer ready while weight shards download."""
from pathlib import Path
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
for filename in ('tokenizer.json', 'tokenizer_config.json', 'merges.txt', 'vocab.json'):
    hf_hub_download('Qwen/Qwen3-32B', filename,
                    revision='9216db5781bf21249d130ec9da846c4624c16137',
                    local_dir=str(ROOT / 'models/reader'))
print('Pinned reader tokenizer downloaded; model weights have a separate readiness gate.', flush=True)
