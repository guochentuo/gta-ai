from __future__ import annotations

import fcntl
import signal
import subprocess
from pathlib import Path
from unittest.mock import Mock

from training.idle_trainer import IdleTrainer, TrainerConfig, _write_json


def config(tmp_path: Path) -> TrainerConfig:
    return TrainerConfig(
        router_state_path=tmp_path / "router.json",
        pending_manifest_path=tmp_path / "pending.json",
        deployed_manifest_path=tmp_path / "deployed.json",
        candidate_manifest_path=tmp_path / "candidate.json",
        active_adapter_path=tmp_path / "active",
        status_path=tmp_path / "status.json",
        gpu_lock_path=tmp_path / "gpu.lock",
        train_command="/test/train",
        idle_seconds=1,
        poll_seconds=0.01,
        terminate_grace_seconds=1,
        failure_backoff_seconds=60,
        training_container_name="test-trainer",
        validator_container_name="test-validator",
    )


def test_pending_revision_is_not_retrained_after_deployment(tmp_path: Path) -> None:
    trainer = IdleTrainer(config(tmp_path))
    _write_json(trainer.config.pending_manifest_path, {"revision": "r1"})
    assert trainer._has_pending_work()
    _write_json(trainer.config.deployed_manifest_path, {"revision": "r1"})
    assert not trainer._has_pending_work()


def test_preemption_terminates_whole_training_process_group(tmp_path: Path, monkeypatch) -> None:
    trainer = IdleTrainer(config(tmp_path))
    process = Mock(spec=subprocess.Popen)
    process.pid = 42001
    process.poll.return_value = None
    trainer.training_process = process
    killpg = Mock()
    cleanup = Mock()
    monkeypatch.setattr("training.idle_trainer.os.killpg", killpg)
    monkeypatch.setattr(trainer, "_cleanup_training_containers", cleanup)

    trainer._preempt_training()

    killpg.assert_called_once_with(42001, signal.SIGTERM)
    cleanup.assert_called_once_with()
    process.wait.assert_called_once_with(timeout=1)
    assert trainer.training_process is None


def test_container_cleanup_force_removes_each_container_cgroup(
    tmp_path: Path, monkeypatch
) -> None:
    trainer = IdleTrainer(config(tmp_path))
    run = Mock()
    monkeypatch.setattr("training.idle_trainer.subprocess.run", run)

    trainer._cleanup_training_containers()

    assert [call.args[0] for call in run.call_args_list] == [
        ["podman", "rm", "--force", "--time", "0", "test-trainer"],
        ["podman", "rm", "--force", "--time", "0", "test-validator"],
    ]


def test_invalid_candidate_is_never_activated(tmp_path: Path) -> None:
    trainer = IdleTrainer(config(tmp_path))
    _write_json(trainer.config.pending_manifest_path, {"revision": "r2"})
    _write_json(
        trainer.config.candidate_manifest_path,
        {"revision": "r1", "adapter_path": str(tmp_path / "adapter")},
    )
    assert not trainer._activate_candidate()
    assert not trainer.config.active_adapter_path.exists()


def test_training_cannot_take_gpu_while_inference_holds_shared_lock(tmp_path: Path) -> None:
    trainer = IdleTrainer(config(tmp_path))
    descriptor = trainer.config.gpu_lock_path.open("a+")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_SH)
        assert not trainer._try_acquire_gpu_lock()
    finally:
        descriptor.close()


def test_failed_revision_obeys_backoff_but_new_revision_does_not(tmp_path: Path) -> None:
    trainer = IdleTrainer(config(tmp_path))
    _write_json(trainer.config.pending_manifest_path, {"revision": "r1"})
    trainer.failed_revision = "r1"
    trainer.retry_not_before = float("inf")
    assert not trainer._retry_allowed()
    _write_json(trainer.config.pending_manifest_path, {"revision": "r2"})
    assert trainer._retry_allowed()
