"""CUDA-only BF16 forwards; audited host sampling and schema policy from TT."""
import gc
import json
import os
import time
from pathlib import Path


class CudaLlama:
    def __init__(self, config, root, worker):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        if not os.environ.get('SLURM_JOB_ID') or os.environ.get('AMEM_LAUNCHED_FROM_TMUX') != '1':
            raise RuntimeError('CUDA forwards require Slurm allocation launched by the tmux coordinator')
        if not os.environ.get('CUDA_VISIBLE_DEVICES') or torch.cuda.device_count() != 1:
            raise RuntimeError('Expected exactly one Slurm-visible CUDA GPU')
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Allocated BF16 CUDA GPU required; no CPU fallback')
        if os.environ.get('PYTHONHASHSEED') != str(config['python_hash_seed']):
            raise RuntimeError('Python hash seed differs from frozen configuration')
        from .device_guard import allocated_gpu,same_gpu
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
            slurm_job_id=os.environ['SLURM_JOB_ID'], node=os.uname().nodename, torch=torch.__version__,
            cuda=torch.version.cuda, dtype=str(next(self.model.parameters()).dtype),occupancy_before_load=occupancy)

    def encode(self, prompt):
        return self.tokenizer.apply_chat_template([
            {'role':'system','content':'You must respond with a JSON object.'},
            {'role':'user','content':prompt}], add_generation_prompt=True, return_tensors='pt')

    def _prefill(self, tokens, chunk_tokens=None):
        from transformers import DynamicCache
        torch = self.torch
        cache = DynamicCache()
        size = chunk_tokens or self.config['prefill_chunk_tokens']
        n = tokens.shape[1]
        with torch.inference_mode():
            for start in range(0,n,size):
                chunk = tokens[:,start:start+size].to('cuda:0')
                output = self.model(input_ids=chunk, past_key_values=cache, use_cache=True,
                                    logits_to_keep=1, return_dict=True)
                cache = output.past_key_values
                logits = output.logits[0,-1].float().cpu()
                if self.progress_callback:
                    self.progress_callback(prefill_tokens_done=min(start+size,n), prefill_tokens_total=n)
                del output, chunk
        return logits, cache

    def generate(self, tokens, schema, seed):
        from lmformatenforcer import TokenEnforcer, JsonSchemaParser
        from jsonschema import Draft202012Validator, ValidationError
        import psutil
        torch = self.torch
        n = int(tokens.shape[1])
        if n+self.config['max_new_tokens'] > self.config['max_context']:
            raise ValueError('Overflow must be caught before generation')
        self.partial_generation = dict(raw_response='', output_token_ids=[], output_tokens=0)
        enforcer = TokenEnforcer(self.enforcer_data, JsonSchemaParser(schema))
        validator = Draft202012Validator(schema)
        rng = torch.Generator(device='cpu').manual_seed(seed)
        generated, parsed, cache = [], None, None
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            scores, cache = self._prefill(tokens)
            prefill_seconds = time.perf_counter()-started
            reason = 'output_limit'
            with torch.inference_mode():
                for step in range(self.config['max_new_tokens']):
                    if not torch.isfinite(scores).all():
                        raise RuntimeError('Non-finite CUDA logits')
                    allowed = list(dict.fromkeys(t for t in enforcer.get_allowed_tokens(generated).allowed_tokens if t < len(scores)))
                    if not allowed:
                        reason = 'no_valid_tokens'; break
                    values = scores[torch.tensor(allowed,dtype=torch.long)]
                    if self.config['temperature'] == 0:
                        token = allowed[int(values.argmax())]
                    else:
                        probabilities = torch.softmax(values/self.config['temperature'],dim=-1)
                        token = allowed[int(torch.multinomial(probabilities,1,generator=rng))]
                    generated.append(token)
                    raw = self.tokenizer.decode(generated,skip_special_tokens=True)
                    self.partial_generation = dict(raw_response=raw,output_token_ids=list(generated),
                        output_tokens=len(generated),prefill_seconds=prefill_seconds)
                    if self.progress_callback and step%64 == 0:
                        self.progress_callback(current_output_tokens=len(generated),generation_elapsed_seconds=time.perf_counter()-started)
                    if token in self.stop_ids:
                        reason = 'eos'; break
                    if raw.rstrip().endswith('}'):
                        try:
                            obj = json.loads(raw); validator.validate(obj)
                            parsed = obj; reason = 'json_complete'; break
                        except (json.JSONDecodeError,ValidationError):
                            pass
                    if step+1 == self.config['max_new_tokens']: break
                    output = self.model(input_ids=torch.tensor([[token]],device='cuda:0'),
                        past_key_values=cache,use_cache=True,logits_to_keep=1,return_dict=True)
                    cache = output.past_key_values
                    scores = output.logits[0,-1].float().cpu()
                    del output
            raw = self.tokenizer.decode(generated,skip_special_tokens=True)
            if parsed is None:
                try:
                    parsed = json.loads(raw); validator.validate(parsed)
                except (json.JSONDecodeError,ValidationError): parsed = None
            torch.cuda.synchronize()
            return dict(raw_response=raw,parsed=parsed,output_token_ids=generated,output_tokens=len(generated),
                prefill_seconds=prefill_seconds,latency_seconds=time.perf_counter()-started,stop_reason=reason,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                host_rss_bytes=psutil.Process().memory_info().rss)
        finally:
            del cache
            gc.collect()

    def context_probe(self, input_length):
        """Exercise real long prefill plus one decode; no semantic accuracy claim."""
        torch = self.torch
        ids = self.tokenizer.encode('This is a synthetic context capacity test. ',add_special_tokens=False)
        tokens = torch.tensor(([self.tokenizer.bos_token_id]+ids*((input_length//len(ids))+1))[:input_length]).reshape(1,-1)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        scores, cache = self._prefill(tokens)
        assert int(cache.get_seq_length()) == input_length
        with torch.inference_mode():
            output = self.model(input_ids=torch.tensor([[int(scores.argmax())]],device='cuda:0'),
                past_key_values=cache,use_cache=True,logits_to_keep=1)
            assert torch.isfinite(output.logits).all()
        torch.cuda.synchronize()
        result = dict(input_tokens=input_length,decode_steps=1,seconds=time.perf_counter()-started,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            cache_tokens=int(output.past_key_values.get_seq_length()))
        del output,cache,scores,tokens
        gc.collect();torch.cuda.empty_cache()
        return result

    def prefill_parity_probe(self):
        torch = self.torch
        tokens = self.encode('Remember every earlier statement. '+('Maya owns a red bicycle. Alex lives in Berlin. '*40))
        one,cache = self._prefill(tokens,chunk_tokens=int(tokens.shape[1]))
        del cache
        many,cache = self._prefill(tokens,chunk_tokens=128)
        del cache
        cosine = float(torch.nn.functional.cosine_similarity(one,many,dim=0))
        maximum = float((one-many).abs().max())
        # BF16 chunking selects different SDPA kernels and need not be numerically
        # identical. Independently verify causal/cache correctness in FP32 using
        # the same pinned weights. Keep the experimental BF16 instance untouched.
        from transformers import AutoModelForCausalLM
        bf16_model=self.model
        torch.cuda.empty_cache()
        self.model=AutoModelForCausalLM.from_pretrained(str(self.root/self.config['model_path']),
            local_files_only=True,torch_dtype=torch.float32,attn_implementation=self.config['attention']).eval().to('cuda:0')
        try:
            fp32_one,cache=self._prefill(tokens,chunk_tokens=int(tokens.shape[1]));del cache
            fp32_many,cache=self._prefill(tokens,chunk_tokens=128);del cache
            fp32_max=float((fp32_one-fp32_many).abs().max())
            torch.testing.assert_close(fp32_one,fp32_many,atol=5e-4,rtol=5e-4)
        finally:
            del self.model
            self.model=bf16_model
            gc.collect();torch.cuda.empty_cache()
        return dict(tokens=int(tokens.shape[1]),bf16_cosine=cosine,bf16_max_abs_logit_difference=maximum,
                    fp32_max_abs_logit_difference=fp32_max,fp32_tolerance='atol=5e-4, rtol=5e-4',
                    bf16_difference_is_reported_not_assumed_zero=True)

    def close(self):
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()
