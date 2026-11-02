import ast
import json
from pathlib import Path
import pytest
import common
from common import claim, ROOT, CODE


def constructor_region(path, name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    start = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]).startswith('(self.torch,'))
    end = next(i for i, n in enumerate(fn.body) if isinstance(n, ast.Assign)
               and ast.unparse(n.targets[0]) == 'self.load_seconds')
    return ast.dump(ast.Module(body=fn.body[start:end+1], type_ignores=[]))


def test_same_model_initialization_and_inherited_inference():
    from backend import DirectCudaLlama
    from amem_reproduction.cuda_backend import CudaLlama
    assert constructor_region(CODE / 'backend.py', 'DirectCudaLlama') == constructor_region(
        ROOT / 'src/amem_reproduction/cuda_backend.py', 'CudaLlama')
    for name in ['encode', '_prefill', 'generate', 'context_probe', 'prefill_parity_probe', 'close']:
        assert getattr(DirectCudaLlama, name) is getattr(CudaLlama, name)


def test_claims_are_exclusive_and_released_for_resume(tmp_path):
    out, ops = tmp_path / 'out', tmp_path / 'ops'
    sid, one = claim(['a', 'b'], out, ops)
    assert sid == 'a'
    sid, two = claim(['a', 'b'], out, ops)
    assert sid == 'b'
    assert claim(['a', 'b'], out, ops) == (None, None)
    one.close()
    sid, resumed = claim(['a', 'b'], out, ops)
    assert sid == 'a'
    resumed.close()
    two.close()


def test_claim_skips_completed_and_failed_conversations(tmp_path):
    out, ops = tmp_path / 'out', tmp_path / 'ops'
    complete = out / 'conversations/a/complete.json'
    complete.parent.mkdir(parents=True)
    complete.write_text('{}')
    failed = ops / 'failures/b.json'
    failed.parent.mkdir(parents=True)
    failed.write_text('{}')
    sid, handle = claim(['a', 'b', 'c'], out, ops)
    assert sid == 'c'
    handle.close()


def test_direct_context_refuses_login_or_slurm_impersonation(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(common.os, 'uname', lambda: SimpleNamespace(nodename='lo-01'))
    with pytest.raises(RuntimeError, match='gpu-04'):
        common.validate_context()
    monkeypatch.setattr(common.os, 'uname', lambda: SimpleNamespace(nodename='gpu-04'))
    monkeypatch.setenv('TMUX', 'test-only')
    monkeypatch.setenv('AMEM_EXECUTION_MODE', 'direct_ssh')
    monkeypatch.setenv('SLURM_JOB_ID', '123')
    with pytest.raises(RuntimeError, match='impersonate'):
        common.validate_context()


def test_scheduling_query_is_read_only(monkeypatch):
    from types import SimpleNamespace
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout='123|another-user\n')
    monkeypatch.setattr(common.subprocess, 'run', run)
    assert common.scheduled_jobs() == ['123|another-user']
    assert len(commands) == 1 and commands[0][0] == 'squeue'
