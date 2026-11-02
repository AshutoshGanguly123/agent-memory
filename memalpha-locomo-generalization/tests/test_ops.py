import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location('memalpha_ops', Path(__file__).resolve().parents[1] / 'ops.py')
ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ops)


def test_reservations_require_physical_indices():
    assert ops.parse_reservations('NodeName=gpu-04 State=MIXED GresDrain=N/A GresUsed=gpu:ada:3(IDX:0-1,7) ', 'gpu-04') == {0, 1, 7}
    assert ops.parse_reservations('NodeName=gpu-04 State=MIXED GresDrain=N/A GresUsed=gpu:ada:0(IDX:N/A) ', 'gpu-04') == set()
    for evidence in ('gpu:ada:2(IDX:0)', 'gpu:ada:2(IDX:3-2)', 'gpu:ada:2', 'gpu:ada:2(IDX:0-1),other:2'):
        with pytest.raises(ValueError):
            ops.parse_reservations('NodeName=gpu-04 State=MIXED GresDrain=N/A GresUsed=' + evidence + ' ', 'gpu-04')
    with pytest.raises(ValueError):
        ops.parse_reservations('NodeName=gpu-03 State=MIXED GresDrain=N/A GresUsed=N/A ', 'gpu-04')
    with pytest.raises(ValueError):
        ops.parse_reservations('NodeName=gpu-04 State=DRAIN GresDrain=N/A GresUsed=N/A ', 'gpu-04')


def test_partial_multicard_lease_rolls_back(tmp_path):
    first, second = tmp_path / 'a', tmp_path / 'b'
    with ops.lease([second]):
        with pytest.raises(RuntimeError):
            with ops.lease([first, second]):
                pytest.fail('A partial multicard lease must never launch')
        with ops.lease([first]):
            pass
    with ops.lease([first, second]):
        pass


def test_card_selection_excludes_occupancy_reservations_and_insufficient_memory():
    cards = [dict(uuid=f'GPU-{i}', reserved=False, pids=[], free_mib=32*1024) for i in range(5)]
    cards[0]['reserved'] = True
    cards[1]['pids'] = [123]
    cards[2]['free_mib'] = 28*1024 - 1
    snapshot = {'host_available_bytes': 128*1024**3, 'cards': cards}
    assert ops.select_cards(snapshot, 2, 128) == ['GPU-3', 'GPU-4']
    with pytest.raises(RuntimeError):
        ops.select_cards(snapshot, 3, 64)
    with pytest.raises(RuntimeError):
        ops.select_cards(snapshot, 1, 129)


def test_numeric_cuda_mapping_requires_nvml_pci_order_parity():
    cards = [{'index': 0, 'uuid': 'GPU-a', 'pci_bus_id': '00000000:01:00.0'},
             {'index': 1, 'uuid': 'GPU-b', 'pci_bus_id': '00000000:02:00.0'}]
    assert ops.cuda_indices({'cards': cards}, ['GPU-b', 'GPU-a']) == [1, 0]
    cards[0]['pci_bus_id'] = '00000000:03:00.0'
    with pytest.raises(RuntimeError, match='ordering'):
        ops.cuda_indices({'cards': cards}, ['GPU-b'])
    cards[0]['pci_bus_id'] = 'invalid'
    with pytest.raises(RuntimeError, match='PCI'):
        ops.cuda_indices({'cards': cards}, ['GPU-b'])


def test_descendant_ownership_does_not_include_test_process():
    pytest.importorskip('psutil')
    code = 'import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); print(p.pid,flush=True); time.sleep(30)'
    child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True, start_new_session=True)
    owned = ops.OwnedProcesses(child)
    try:
        descendant_pid = int(child.stdout.readline())
        assert owned.is_owned(child.pid)
        assert owned.is_owned(descendant_pid)
        assert not owned.is_owned(os.getpid())
        owned.identities[os.getpid()] = 0.0
        assert not owned.is_owned(os.getpid())
        owned.signal(signal.SIGTERM)
        child.wait(timeout=5)
        deadline = time.monotonic() + 5
        while owned.living() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not owned.living()
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


def test_competing_pid_stops_only_owned_child_group(tmp_path, monkeypatch):
    pytest.importorskip('psutil')
    monkeypatch.setattr(ops, 'OPS', tmp_path / 'operations')
    monkeypatch.setattr(ops, 'SHARED_LOCKS', tmp_path / 'shared')
    monkeypatch.setattr(ops, 'node_name', lambda: 'gpu-04')
    monkeypatch.setenv('TMUX', 'synthetic-test-only')
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    calls = []

    def snapshot():
        calls.append(1)
        return {'node': 'gpu-04', 'host_available_bytes': 128*1024**3,
                'cards': [{'uuid': 'GPU-abc', 'index': 0, 'reserved': False, 'free_mib': 32*1024,
                           'pci_bus_id': '00000000:01:00.0',
                           'pids': [] if len(calls) == 1 else [os.getpid()]}]}

    monkeypatch.setattr(ops, 'inventory', snapshot)
    args = SimpleNamespace(name='synthetic_lease_test', gpus='GPU-abc', attempt='fake_no_gpu',
                           min_host_gib=64, command=[sys.executable, '-c', 'import time; time.sleep(30)'])
    assert ops.locked(args) == 75
    assert len(calls) >= 2
    assert os.getpid() > 0


def test_tracker_constructor_failure_happens_before_popen(tmp_path, monkeypatch):
    monkeypatch.setattr(ops, 'OPS', tmp_path / 'operations')
    monkeypatch.setattr(ops, 'SHARED_LOCKS', tmp_path / 'shared')
    monkeypatch.setattr(ops, 'node_name', lambda: 'gpu-04')
    monkeypatch.setenv('TMUX', 'synthetic-test-only')
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.setattr(ops, 'inventory', lambda: {'node': 'gpu-04', 'host_available_bytes': 128*1024**3,
        'cards': [{'uuid': 'GPU-abc', 'index': 0, 'reserved': False, 'free_mib': 32*1024,
                   'pci_bus_id': '00000000:01:00.0', 'pids': []}]})

    def fail_constructor(*args):
        raise RuntimeError('Synthetic ownership constructor failure')

    calls = []

    def forbid_launch(*args, **kwargs):
        calls.append(1)
        raise AssertionError('Popen must not run before ownership tracker construction')

    monkeypatch.setattr(ops, 'OwnedProcesses', fail_constructor)
    monkeypatch.setattr(ops.subprocess, 'Popen', forbid_launch)
    args = SimpleNamespace(name='synthetic_startup_test', gpus='GPU-abc', attempt='fake_no_gpu',
                           min_host_gib=64, command=[sys.executable, '-c', 'pass'])
    assert ops.locked(args) == 1
    assert not calls


def test_fast_exited_parent_descendants_recovered_by_new_session():
    pytest.importorskip('psutil')
    owned = ops.OwnedProcesses()
    code = 'import subprocess,sys,os; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); print(p.pid,flush=True); os._exit(0)'
    before = time.time()
    child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True, start_new_session=True)
    after = time.time()
    descendant_pid = int(child.stdout.readline())
    child.wait(timeout=5)
    try:
        owned.attach(child, before, after)
        assert descendant_pid in owned.living()
        assert not owned.is_owned(os.getpid())
        owned.signal(signal.SIGTERM)
        deadline = time.monotonic() + 5
        while owned.living() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not owned.living()
    finally:
        if owned.living():
            owned.signal(signal.SIGKILL)
