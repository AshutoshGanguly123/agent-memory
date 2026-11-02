"""Supervisor tests use mocked dispatch only: no model or GPU jobs launch."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pipeline


@pytest.fixture
def environment(tmp_path, monkeypatch):
    root = tmp_path / "experiment"
    base = root / "runs/frozen-test"
    monkeypatch.setattr(pipeline, "ROOT", root)
    monkeypatch.setattr(pipeline, "STATE", root / "results/pipeline/status.json")
    monkeypatch.setattr(pipeline, "OPS", root / "results/operations")
    monkeypatch.setattr(pipeline, "PYTHON", "/mock/python")
    monkeypatch.setattr(pipeline, "run_root", lambda: base)
    monkeypatch.setenv("TMUX", "synthetic-no-processes")
    verified = []
    monkeypatch.setattr(pipeline, "verify_freeze", lambda: verified.append(True))
    clock = {"seconds": 0}
    monkeypatch.setattr(pipeline, "time", SimpleNamespace(
        sleep=lambda seconds: clock.update(seconds=clock["seconds"] + seconds),
        monotonic=lambda: clock["seconds"]))

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def frozen_memories():
        for arm in ("memalpha", "qwen3_base"):
            for index in range(10):
                write(base / arm / "conversations" / str(index) / "memory_frozen.json", {})

    def completed_builds():
        for arm in ("memalpha", "qwen3_base"):
            write(base / arm / "build_status.json", {"status": "complete"})
        frozen_memories()

    return SimpleNamespace(root=root, base=base, verified=verified, write=write,
                           frozen_memories=frozen_memories, completed_builds=completed_builds, clock=clock)


def test_dispatch_reader_propagates_only_allowlisted_runtime_and_uses_argv(environment, monkeypatch):
    environment.write(environment.root / "results/setup/reader_runtime.json",
                      {"environment": {"NCCL_P2P_DISABLE": "1", "NCCL_IB_DISABLE": "1"}})
    calls = []

    def capture(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"state": "running", "child_pid": 123}), stderr="")

    monkeypatch.setattr(pipeline.subprocess, "run", capture)
    assert pipeline.dispatch("reader", 4, ["answer"], 7) == "full_reader_007"
    command, kwargs = calls[0]
    assert command[command.index("--gpu-count") + 1] == "4"
    assert command[command.index("--min-host-gib") + 1] == "128"
    assert command[command.index("--") + 1:] == ["env", "NCCL_IB_DISABLE=1", "NCCL_P2P_DISABLE=1",
                                               "/mock/python", str(environment.root / "run.py"), "answer"]
    assert "shell" not in kwargs
    calls.clear()
    pipeline.dispatch("memalpha", 1, ["build", "--arm", "memalpha"], 1)
    command = calls[0][0]
    assert command[command.index("--") + 1] == "/mock/python"


def test_dispatch_rejects_unreviewed_runtime_before_launch(environment, monkeypatch):
    environment.write(environment.root / "results/setup/reader_runtime.json", {"environment": {"LD_PRELOAD": "unreviewed"}})
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: pytest.fail("Must not launch"))
    with pytest.raises(RuntimeError, match="Unreviewed"):
        pipeline.dispatch("reader", 4, ["answer"], 1)


def test_dispatch_resource_wait_is_retryable_but_existing_child_is_not(environment, monkeypatch):
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=1, stdout="", stderr="no_available_node"))
    assert pipeline.dispatch("reader", 4, ["answer"], 1) is None
    assert json.loads(pipeline.STATE.read_text())["state"] == "waiting_for_resources"
    environment.write(pipeline.OPS / "full_reader_002/status.json", {"state": "running", "child_pid": 123})
    with pytest.raises(RuntimeError, match="child needs inspection"):
        pipeline.dispatch("reader", 4, ["answer"], 2)


def test_unconfirmed_startup_waits_for_status_then_completion(environment, monkeypatch):
    count = []

    def advance(seconds):
        environment.clock["seconds"] += seconds
        count.append(seconds)
        environment.write(pipeline.OPS / "full_reader_001/status.json",
                          {"state": "starting" if len(count) == 1 else "complete", "exit_code": 0})

    monkeypatch.setattr(pipeline.time, "sleep", advance)
    assert pipeline.wait_operation("full_reader_001")["state"] == "complete"
    assert len(count) == 2


def test_missing_startup_status_times_out_without_redispatch(environment):
    with pytest.raises(RuntimeError, match="inspect before retry"):
        pipeline.wait_operation("full_reader_001")
    assert environment.clock["seconds"] == pipeline.STARTUP_STATUS_TIMEOUT_SECONDS


def test_stale_starting_operation_requires_inspection(environment):
    environment.write(pipeline.OPS / "full_reader_001/status.json", {"state": "starting"})
    with pytest.raises(RuntimeError, match="never confirmed its child"):
        pipeline.wait_operation("full_reader_001")


def test_stop_before_launch_fans_out_to_both_existing_writers(environment, monkeypatch):
    for name in ("full_memalpha_001", "full_base_001"):
        environment.write(pipeline.OPS / name / "status.json", {"state": "running"})
    environment.write(environment.root / "results/pipeline/STOP", {})
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: pytest.fail("STOP must prevent launch"))
    assert pipeline.run() == 75
    assert environment.verified == [True]
    assert all((pipeline.OPS / name / "STOP").exists() for name in ("full_memalpha_001", "full_base_001"))
    assert json.loads(pipeline.STATE.read_text())["state"] == "stop_requested"


def test_stop_during_resource_wait_prevents_further_dispatch(environment, monkeypatch):
    calls = []
    def unavailable(*args, **kwargs):
        calls.append(1)
        return SimpleNamespace(returncode=1, stdout="", stderr="no_available_node")
    monkeypatch.setattr(pipeline.subprocess, "run", unavailable)
    monkeypatch.setattr(pipeline.time, "sleep", lambda seconds: environment.write(environment.root / "results/pipeline/STOP", {}))
    assert pipeline.run() == 75
    assert calls == [1]


def test_stop_during_wait_signals_all_active_operations_not_just_current(environment):
    for name in ("full_memalpha_001", "full_base_001"):
        environment.write(pipeline.OPS / name / "status.json", {"state": "running"})
    environment.write(pipeline.OPS / "full_old_001/status.json", {"state": "complete"})
    environment.write(environment.root / "results/pipeline/STOP", {})
    with pytest.raises(pipeline.StopRequested):
        pipeline.wait_operation("full_memalpha_001")
    assert (pipeline.OPS / "full_memalpha_001/STOP").exists()
    assert (pipeline.OPS / "full_base_001/STOP").exists()
    assert not (pipeline.OPS / "full_old_001/STOP").exists()


def install_fake_stages(environment, monkeypatch, completion=None, complete_metrics=True):
    events = []
    def dispatch(stage, count, args, attempt):
        assert environment.verified, "No dispatch before verified freeze"
        events.append(("dispatch", stage, count, attempt))
        return f"full_{stage}_{attempt:03d}"
    def wait(name):
        events.append(("wait", name))
        return completion(name) if completion else {"state": "complete", "exit_code": 0}
    def score(command, **kwargs):
        assert command[-1] == "score"
        events.append(("score",))
        environment.write(environment.base / "report/metrics.json", {"complete": complete_metrics})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(pipeline, "dispatch", dispatch)
    monkeypatch.setattr(pipeline, "wait_operation", wait)
    monkeypatch.setattr(pipeline.subprocess, "run", score)
    return events


def test_two_writers_finish_before_reader_and_score(environment, monkeypatch):
    environment.frozen_memories()
    events = install_fake_stages(environment, monkeypatch)
    assert pipeline.run() == 0
    assert events == [("dispatch", "memalpha", 1, 1), ("dispatch", "base", 1, 1),
                      ("wait", "full_memalpha_001"), ("wait", "full_base_001"),
                      ("dispatch", "reader", 4, 1), ("wait", "full_reader_001"), ("score",)]


def test_starting_writer_is_adopted_on_restart(environment, monkeypatch):
    environment.frozen_memories()
    environment.write(environment.base / "qwen3_base/build_status.json", {"status": "complete"})
    environment.write(environment.base / "answer_status.json", {"status": "complete"})
    environment.write(pipeline.OPS / "full_memalpha_001/status.json", {"state": "starting"})
    events = install_fake_stages(environment, monkeypatch)
    assert pipeline.run() == 0
    assert events == [("wait", "full_memalpha_001"), ("score",)]


def test_reader_cannot_launch_without_all_twenty_memories(environment, monkeypatch):
    events = install_fake_stages(environment, monkeypatch)
    with pytest.raises(RuntimeError, match="All 20"):
        pipeline.run()
    assert not any(event[:2] == ("dispatch", "reader") for event in events)
    assert ("score",) not in events


def test_fatal_writer_failure_prevents_reader(environment, monkeypatch):
    environment.frozen_memories()
    events = install_fake_stages(environment, monkeypatch, completion=lambda name: {"state": "failed", "exit_code": 1})
    with pytest.raises(RuntimeError, match="Writer operation needs inspection"):
        pipeline.run()
    assert not any(event[:2] == ("dispatch", "reader") for event in events)


def test_previous_fatal_writer_is_not_silently_restarted(environment, monkeypatch):
    environment.write(pipeline.OPS / "full_memalpha_001/status.json", {"state": "failed", "exit_code": 1, "child_pid": 123})
    events = install_fake_stages(environment, monkeypatch)
    with pytest.raises(RuntimeError, match="Previous writer operation"):
        pipeline.run()
    assert events == []


def test_prelaunch_resource_failure_can_be_retried_after_restart(environment, monkeypatch):
    environment.frozen_memories()
    environment.write(environment.base / "qwen3_base/build_status.json", {"status": "complete"})
    environment.write(environment.base / "answer_status.json", {"status": "complete"})
    environment.write(pipeline.OPS / "full_memalpha_001/status.json", {"state": "failed", "exit_code": 1})
    events = install_fake_stages(environment, monkeypatch)
    assert pipeline.run() == 0
    assert events[0] == ("dispatch", "memalpha", 1, 2)


def test_checkpointed_writer_can_resume_exit_75(environment, monkeypatch):
    environment.frozen_memories()
    environment.write(environment.base / "qwen3_base/build_status.json", {"status": "complete"})
    environment.write(environment.base / "answer_status.json", {"status": "complete"})
    def completion(name):
        return {"state": "stopped", "exit_code": 75} if name.endswith("001") else {"state": "complete", "exit_code": 0}
    events = install_fake_stages(environment, monkeypatch, completion=completion)
    assert pipeline.run() == 0
    assert events == [("dispatch", "memalpha", 1, 1), ("wait", "full_memalpha_001"),
                      ("dispatch", "memalpha", 1, 2), ("wait", "full_memalpha_002"), ("score",)]


def test_reader_error_outputs_retry_three_times_then_report_incomplete(environment, monkeypatch):
    environment.completed_builds()
    events = install_fake_stages(environment, monkeypatch,
                                 completion=lambda name: {"state": "failed", "exit_code": 2}, complete_metrics=False)
    assert pipeline.run() == 2
    assert [event for event in events if event[0] == "dispatch"] == [
        ("dispatch", "reader", 4, 1), ("dispatch", "reader", 4, 2), ("dispatch", "reader", 4, 3)]
    assert events[-1] == ("score",)
    assert json.loads(pipeline.STATE.read_text())["state"] == "complete_with_errors"


def test_invalid_freeze_prevents_all_dispatch(environment, monkeypatch):
    def invalid():
        raise RuntimeError("Frozen source changed")
    monkeypatch.setattr(pipeline, "verify_freeze", invalid)
    monkeypatch.setattr(pipeline.subprocess, "run", lambda *args, **kwargs: pytest.fail("No unverified jobs"))
    with pytest.raises(RuntimeError, match="Frozen source changed"):
        pipeline.run()
