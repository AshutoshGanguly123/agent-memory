"""Bounded owned-allocation I/O and import diagnostic; no model inference."""
import faulthandler,json,os,subprocess,sys,time
from pathlib import Path
faulthandler.enable()
root=Path(__file__).resolve().parents[1]
print('NODE',os.uname().nodename,'PYTHON',sys.version,'EXECUTABLE',sys.executable,flush=True)
print('ALLOCATION',{k:os.environ.get(k) for k in ['SLURM_JOB_ID','CUDA_VISIBLE_DEVICES','TMUX']},flush=True)
print(subprocess.check_output(['df','-h',str(root),'/tmp'],text=True),flush=True)
p=root/'results/operations'/('compute_io_'+os.environ['SLURM_JOB_ID']+'.bin')
block=os.urandom(1024*1024);start=time.perf_counter()
with p.open('wb') as f:
    for _ in range(32):f.write(block)
    f.flush();os.fsync(f.fileno())
with p.open('rb') as f:
    for _ in range(32):assert f.read(len(block))==block
p.unlink()
print('IO_FSYNC_READ_OK',time.perf_counter()-start,flush=True)
print('IMPORT_TORCH_START',flush=True)
import torch
print('TORCH',torch.__version__,'CUDA_AVAILABLE',torch.cuda.is_available(),'VISIBLE_COUNT',torch.cuda.device_count(),flush=True)
print('IMPORT_SENTENCE_TRANSFORMERS_START',flush=True)
from sentence_transformers import SentenceTransformer
print('IMPORT_SENTENCE_TRANSFORMERS_OK',flush=True)
c=json.loads((root/'configs/experiment.json').read_text())
m=SentenceTransformer(str(root/c['embedding_path']),device='cpu',local_files_only=True)
print('MINILM_ENCODE',m.encode(['diagnostic sentence']).shape,flush=True)
print('COMPUTE_DIAGNOSTIC_PASSED',flush=True)
