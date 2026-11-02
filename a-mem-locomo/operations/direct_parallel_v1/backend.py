"""Direct-node initialization; all inference methods inherited unchanged.

The model/tokenizer/precision initialization below is copied from the frozen
CudaLlama constructor. Only launch-context checks and execution metadata differ.
Tests compare the computational initialization AST against the original.
"""
import json
import os
import time
from pathlib import Path
from amem_reproduction.cuda_backend import CudaLlama


class DirectCudaLlama(CudaLlama):
    def __init__(self, config, root, worker):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        from common import validate_context
        validate_context()
        if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 1:
            raise RuntimeError('Expected exactly one Slurm-visible CUDA GPU')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Allocated BF16 CUDA GPU required; no CPU fallback')
        if os.environ.get('PYTHONHASHSEED') != str(config['python_hash_seed']):
            raise RuntimeError('Python hash seed differs from frozen configuration')
        from amem_reproduction.device_guard import allocated_gpu,same_gpu
        occupancy=allocated_gpu()
        props=torch.cuda.get_device_properties(0)
        actual_uuid=str(props.uuid)
        if not same_gpu(actual_uuid,occupancy['uuid']):
            raise RuntimeError(f'CUDA/NVML device mapping differs: {actual_uuid} versus {occupancy["uuid"]}; refuse to load model')
        self.torch, self.config, self.root, self.worker = torch, config, Path(root), worker
        self.progress_callback = None
        self.partial_generation = {}
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        model_path = str(self.root/config['model_path'])
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.enforcer_data = build_token_enforcer_tokenizer_data(self.tokenizer)
        generation = json.loads((Path(model_path)/'generation_config.json').read_text())
        self.stop_ids = generation['eos_token_id']
        if isinstance(self.stop_ids, int): self.stop_ids = [self.stop_ids]
        started = time.perf_counter()
        print('Loading pinned Llama weights onto allocated CUDA GPU',flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation=config['attention']).eval().to('cuda:0')
        if any(p.device.type != 'cuda' for p in self.model.parameters()):
            raise RuntimeError('Found offloaded model parameter')
        if self.model.config.max_position_embeddings < config['max_context']:
            raise RuntimeError('Model context does not match protocol')
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter()-started
        props = torch.cuda.get_device_properties(0)
        self.device_info = dict(name=props.name, uuid=occupancy['uuid'], cuda_uuid=str(props.uuid), total_memory_bytes=props.total_memory,
            compute_capability=[props.major,props.minor], visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
            slurm_job_id=None, execution_mode='direct_ssh', node=os.uname().nodename, torch=torch.__version__,
            cuda=torch.version.cuda, dtype=str(next(self.model.parameters()).dtype),occupancy_before_load=occupancy)

