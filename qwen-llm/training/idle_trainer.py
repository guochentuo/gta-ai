from __future__ import annotations

import fcntl
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainerConfig:
    router_state_path: Path = Path("/opt/gta-ai/qwen-llm/state/router-state.json")
    pending_manifest_path: Path = Path(
        "/opt/gta-ai/qwen-llm/data/training/queue/pending.json"
    )
    deployed_manifest_path: Path = Path(
        "/opt/gta-ai/qwen-llm/data/training/runtime/deployed.json"
    )
    candidate_manifest_path: Path = Path(
        "/opt/gta-ai/qwen-llm/data/training/runtime/candidate.json"
    )
    active_adapter_path: Path = Path(
        "/opt/gta-ai/qwen-llm/data/training/adapters/active"
    )
    status_path: Path = Path(
        "/opt/gta-ai/qwen-llm/data/training/runtime/trainer-state.json"
    )
    gpu_lock_path: Path = Path("/opt/gta-ai/qwen-llm/state/gpu.lock")
    train_command: str = "/opt/gta-ai/bin/run-identity-training"
    inference_service: str = "gta-ai-vllm.service"
    idle_seconds: float = 900.0
    poll_seconds: float = 0.25
    terminate_grace_seconds: float = 3.0
    failure_backoff_seconds: float = 3600.0
    training_container_name: str = "gta-ai-identity-trainer"
    validator_container_name: str = "gta-ai-identity-validator"

    @classmethod
    def from_environment(cls) -> TrainerConfig:
        return cls(
            router_state_path=Path(
                os.getenv("GTA_AI_TRAINER_ROUTER_STATE_PATH", str(cls.router_state_path))
            ),
            pending_manifest_path=Path(
                os.getenv("GTA_AI_TRAINER_PENDING_MANIFEST_PATH", str(cls.pending_manifest_path))
            ),
            deployed_manifest_path=Path(
                os.getenv("GTA_AI_TRAINER_DEPLOYED_MANIFEST_PATH", str(cls.deployed_manifest_path))
            ),
            candidate_manifest_path=Path(
                os.getenv(
                    "GTA_AI_TRAINER_CANDIDATE_MANIFEST_PATH", str(cls.candidate_manifest_path)
                )
            ),
            active_adapter_path=Path(
                os.getenv("GTA_AI_TRAINER_ACTIVE_ADAPTER_PATH", str(cls.active_adapter_path))
            ),
            status_path=Path(os.getenv("GTA_AI_TRAINER_STATUS_PATH", str(cls.status_path))),
            gpu_lock_path=Path(os.getenv("GTA_AI_TRAINER_GPU_LOCK_PATH", str(cls.gpu_lock_path))),
            train_command=os.getenv("GTA_AI_TRAINER_COMMAND", cls.train_command),
            inference_service=os.getenv("GTA_AI_TRAINER_INFERENCE_SERVICE", cls.inference_service),
            idle_seconds=float(os.getenv("GTA_AI_TRAINER_IDLE_SECONDS", str(cls.idle_seconds))),
            poll_seconds=float(os.getenv("GTA_AI_TRAINER_POLL_SECONDS", str(cls.poll_seconds))),
            terminate_grace_seconds=float(
                os.getenv(
                    "GTA_AI_TRAINER_TERMINATE_GRACE_SECONDS",
                    str(cls.terminate_grace_seconds),
                )
            ),
            failure_backoff_seconds=float(
                os.getenv(
                    "GTA_AI_TRAINER_FAILURE_BACKOFF_SECONDS",
                    str(cls.failure_backoff_seconds),
                )
            ),
            training_container_name=os.getenv(
                "GTA_AI_TRAINER_CONTAINER_NAME", cls.training_container_name
            ),
            validator_container_name=os.getenv(
                "GTA_AI_TRAINER_VALIDATOR_CONTAINER_NAME", cls.validator_container_name
            ),
        )


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class IdleTrainer:
    def __init__(self, config: TrainerConfig) -> None:
        self.config = config
        self.training_process: subprocess.Popen[bytes] | None = None
        self.gpu_lock_descriptor: int | None = None
        self.running = True
        previous_status = _read_json(config.status_path)
        self.failed_revision = str(previous_status.get("failed_revision", ""))
        self.retry_not_before = float(previous_status.get("retry_not_before", 0.0))

    def _systemctl(self, action: str) -> None:
        subprocess.run(
            ["systemctl", "--user", action, self.config.inference_service],
            check=True,
            timeout=1800,
        )

    def _inference_active(self) -> bool:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "--quiet", self.config.inference_service],
            check=False,
        )
        return result.returncode == 0

    def _pending_revision(self) -> str:
        return str(_read_json(self.config.pending_manifest_path).get("revision", ""))

    def _deployed_revision(self) -> str:
        return str(_read_json(self.config.deployed_manifest_path).get("revision", ""))

    def _has_pending_work(self) -> bool:
        pending = self._pending_revision()
        return bool(pending) and pending != self._deployed_revision()

    def _retry_allowed(self) -> bool:
        pending = self._pending_revision()
        return pending != self.failed_revision or time.time() >= self.retry_not_before

    def _runtime(self) -> tuple[int, float]:
        state = _read_json(self.config.router_state_path)
        active = int(state.get("active_requests", 0))
        last = float(state.get("last_workload_at", time.time()))
        return active, last

    def _set_status(self, mode: str, **details: object) -> None:
        _write_json(
            self.config.status_path,
            {
                "schema_version": 1,
                "mode": mode,
                "updated_at": time.time(),
                "pending_revision": self._pending_revision(),
                "deployed_revision": self._deployed_revision(),
                **details,
            },
        )

    def _try_acquire_gpu_lock(self) -> bool:
        if self.gpu_lock_descriptor is not None:
            return True
        self.config.gpu_lock_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.config.gpu_lock_path, os.O_CREAT | os.O_RDWR, 0o660)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return False
        self.gpu_lock_descriptor = descriptor
        return True

    def _release_gpu_lock(self) -> None:
        descriptor = self.gpu_lock_descriptor
        if descriptor is None:
            return
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        self.gpu_lock_descriptor = None

    def _ensure_inference(self) -> None:
        try:
            if not self._inference_active():
                self._systemctl("start")
        finally:
            self._release_gpu_lock()

    def _start_training(self) -> None:
        active, _ = self._runtime()
        if active > 0 or not self._try_acquire_gpu_lock():
            return
        # 请求先登记 active 再获取共享锁; 拿到独占锁后必须复查一次。
        active, _ = self._runtime()
        if active > 0:
            self._release_gpu_lock()
            return
        try:
            self._systemctl("stop")
            self.training_process = subprocess.Popen(
                [self.config.train_command],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except BaseException:
            self._ensure_inference()
            raise
        self._set_status("training", pid=self.training_process.pid)

    def _preempt_training(self) -> None:
        process = self.training_process
        if process is None or process.poll() is not None:
            self.training_process = None
            self._cleanup_training_containers()
            return
        # 容器才是真正占用 GPU 的进程; 必须先明确停止容器, 再回收包装脚本进程组。
        self._cleanup_training_containers()
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=self.config.terminate_grace_seconds)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
        self.training_process = None
        self._set_status("preempted_for_inference")

    def _cleanup_training_containers(self) -> None:
        names = [self.config.training_container_name, self.config.validator_container_name]
        # ms-swift spawns nested Python workers. Removing each container
        # forcibly reclaims its full libpod cgroup before inference resumes.
        for name in names:
            subprocess.run(
                ["podman", "rm", "--force", "--time", "0", name],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=15,
            )

    def _activate_candidate(self) -> bool:
        candidate = _read_json(self.config.candidate_manifest_path)
        adapter_text = str(candidate.get("adapter_path", ""))
        revision = str(candidate.get("revision", ""))
        adapter = Path(adapter_text) if adapter_text else None
        if not revision or revision != self._pending_revision() or adapter is None:
            self._set_status("candidate_invalid")
            return False
        if not (adapter / "adapter_config.json").is_file():
            self._set_status("candidate_missing_adapter", adapter_path=str(adapter))
            return False

        self.config.active_adapter_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.config.active_adapter_path.with_name(".active.tmp")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(adapter)
        temporary.replace(self.config.active_adapter_path)
        _write_json(
            self.config.deployed_manifest_path,
            {
                **candidate,
                "activated_at": time.time(),
            },
        )
        self._set_status("candidate_activated", adapter_path=str(adapter))
        return True

    def _finish_training_if_needed(self) -> None:
        process = self.training_process
        if process is None:
            return
        return_code = process.poll()
        if return_code is None:
            return
        self.training_process = None
        if return_code == 0:
            self.failed_revision = ""
            self.retry_not_before = 0
            self._activate_candidate()
        else:
            self.failed_revision = self._pending_revision()
            self.retry_not_before = time.time() + self.config.failure_backoff_seconds
            self._set_status(
                "training_failed",
                exit_code=return_code,
                failed_revision=self.failed_revision,
                retry_not_before=self.retry_not_before,
            )
        self._ensure_inference()

    def run(self) -> None:
        self._ensure_inference()
        self._set_status("inference")
        while self.running:
            self._finish_training_if_needed()
            active, last_workload = self._runtime()
            if active > 0:
                if self.training_process is not None:
                    self._preempt_training()
                self._ensure_inference()
                self._set_status("inference", active_requests=active)
            elif (
                self.training_process is None
                and self._has_pending_work()
                and self._retry_allowed()
                and time.time() - last_workload >= self.config.idle_seconds
            ):
                self._start_training()
            time.sleep(self.config.poll_seconds)

    def stop(self) -> None:
        self.running = False
        self._preempt_training()
        self._ensure_inference()


def main() -> None:
    trainer = IdleTrainer(TrainerConfig.from_environment())

    def stop_handler(_: int, __: object) -> None:
        trainer.stop()

    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    trainer.run()


if __name__ == "__main__":
    main()
