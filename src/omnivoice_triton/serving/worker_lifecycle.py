"""Lingua worker callback lifecycle for Vast.ai OmniVoice workers."""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from typing import Any

logger = logging.getLogger(__name__)


def _normalize_worker_type(value: str | None) -> str:
    worker_type = (value or "tts").strip().lower()
    if worker_type == "chatterbox":
        return "tts"
    return worker_type or "tts"


def _parse_positive_int(value: str | None, default: int) -> int:
    try:
        parsed = int(str(value).strip()) if value is not None else default
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _parse_positive_float(value: str | None, default: float) -> float:
    try:
        parsed = float(str(value).strip()) if value is not None else default
    except ValueError:
        return default
    return parsed if parsed > 0.0 else default


def _parse_interval_seconds(value: str | None, default: float) -> float:
    parsed = _parse_positive_float(value, default)
    return min(max(parsed, 15.0), 30.0)


def _env_first(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def _has_env_with_prefix(prefix: str) -> bool:
    return any(
        name.startswith(prefix) and value.strip() for name, value in os.environ.items()
    )


def _detect_instance_id() -> str:
    value = _env_first(
        "LINGUA_WORKER_INSTANCE_ID",
        "VAST_INSTANCE_ID",
        "INSTANCE_ID",
        "CONTAINER_ID",
        "HOSTNAME",
    )
    if value:
        return value
    return socket.gethostname()


@dataclass(frozen=True)
class WorkerPublicEndpoint:
    base_url: str | None
    public_ip: str | None
    public_port: str | None
    internal_port: int


def detect_public_endpoint(port: int) -> WorkerPublicEndpoint:
    """Detect the public endpoint Lingua should use to call this worker."""
    explicit = _env_first(
        "LINGUA_WORKER_BASE_URL",
        "LINGUA_WORKER_PUBLIC_BASE_URL",
        "OMNIVOICE_PUBLIC_BASE_URL",
        "PUBLIC_BASE_URL",
    )
    if explicit:
        return WorkerPublicEndpoint(
            base_url=explicit.rstrip("/"),
            public_ip=None,
            public_port=None,
            internal_port=port,
        )

    public_ip = _env_first(
        "LINGUA_WORKER_PUBLIC_IP",
        "VAST_PUBLIC_IP",
        "PUBLIC_IPADDR",
        "PUBLIC_IP",
    )
    public_port = _env_first(
        "LINGUA_WORKER_PUBLIC_PORT",
        "VAST_PUBLIC_PORT",
        f"VAST_TCP_PORT_{port}",
        "PUBLIC_PORT",
    )
    base_url: str | None = None
    if public_ip and public_port:
        base_url = f"http://{public_ip}:{public_port}"
    elif public_ip and not _has_env_with_prefix("VAST_TCP_PORT_"):
        base_url = f"http://{public_ip}:{port}"

    return WorkerPublicEndpoint(
        base_url=base_url,
        public_ip=public_ip,
        public_port=public_port,
        internal_port=port,
    )


def detect_public_base_url(port: int) -> str | None:
    """Detect the public URL Lingua should use to call this worker."""
    return detect_public_endpoint(port).base_url


class WorkerRuntimeState:
    """Thread-safe request and drain state shared by API and heartbeat loop."""

    def __init__(self, capacity: int = 1) -> None:
        self.capacity = max(1, int(capacity))
        self._lock = threading.Lock()
        self._active_requests = 0
        self._draining = False
        self._cuda_fault: dict[str, str] | None = None

    def begin_request(self) -> bool:
        with self._lock:
            if self._draining or self._cuda_fault is not None:
                return False
            self._active_requests += 1
            return True

    def mark_cuda_fault(self, reason: str) -> None:
        """Record that the CUDA context failed; from now on the worker reports dead."""
        with self._lock:
            if self._cuda_fault is None:
                self._cuda_fault = {
                    "reason": reason,
                    "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }

    @property
    def cuda_fault(self) -> dict[str, str] | None:
        with self._lock:
            return dict(self._cuda_fault) if self._cuda_fault is not None else None

    def end_request(self) -> None:
        with self._lock:
            self._active_requests = max(0, self._active_requests - 1)

    def request_drain(self) -> None:
        with self._lock:
            self._draining = True

    def wait_for_idle(self, timeout_s: float) -> bool:
        deadline_at = time.monotonic() + max(0.0, timeout_s)
        while True:
            with self._lock:
                if self._active_requests <= 0:
                    return True
            remaining = deadline_at - time.monotonic()
            if remaining <= 0.0:
                return False
            time.sleep(min(0.1, remaining))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active_requests = self._active_requests
            draining = self._draining
            cuda_fault = (
                dict(self._cuda_fault) if self._cuda_fault is not None else None
            )
        if cuda_fault is not None:
            status = "dead"
        else:
            status = "draining" if draining else "healthy"
        return {
            "status": status,
            "active_requests": active_requests,
            "queue_depth": active_requests,
            "capacity": self.capacity,
            "assignable": (
                not draining and cuda_fault is None and active_requests < self.capacity
            ),
            "draining": draining,
            "cuda_fault": cuda_fault,
        }


@dataclass(frozen=True)
class WorkerCallbackConfig:
    worker_type: str
    worker_id: str
    instance_id: str
    register_url: str
    heartbeat_url: str
    token: str
    capacity: int
    public_base_url: str | None
    public_ip: str | None
    public_port: str | None
    internal_port: int
    region: str | None
    gpu_type: str | None
    heartbeat_interval_s: float
    request_timeout_s: float

    @classmethod
    def from_env(cls, *, port: int) -> "WorkerCallbackConfig | None":
        token = _env_first("LINGUA_WORKER_TOKEN")
        base_url = _env_first("LINGUA_CONTROL_PLANE_BASE_URL")
        register_url = _env_first("LINGUA_WORKER_REGISTER_URL")
        heartbeat_url = _env_first("LINGUA_WORKER_HEARTBEAT_URL")
        if not token or not (base_url or (register_url and heartbeat_url)):
            return None

        if base_url:
            base_url = base_url.rstrip("/")
        register_url = register_url or f"{base_url}/api/runtime/workers/register"
        heartbeat_url = heartbeat_url or f"{base_url}/api/runtime/workers/heartbeat"
        worker_type = _normalize_worker_type(_env_first("LINGUA_WORKER_TYPE"))
        instance_id = _detect_instance_id()
        worker_id = _env_first("WORKER_ID", "LINGUA_WORKER_ID") or (
            f"{worker_type}-{instance_id}"
        )
        capacity = _parse_positive_int(
            _env_first(
                "LINGUA_WORKER_DEFAULT_CAPACITY",
                "OMNIVOICE_WORKER_DEFAULT_CAPACITY",
            ),
            1,
        )
        public_endpoint = detect_public_endpoint(port)
        return cls(
            worker_type=worker_type,
            worker_id=worker_id,
            instance_id=instance_id,
            register_url=register_url,
            heartbeat_url=heartbeat_url,
            token=token,
            capacity=capacity,
            public_base_url=public_endpoint.base_url,
            public_ip=public_endpoint.public_ip,
            public_port=public_endpoint.public_port,
            internal_port=public_endpoint.internal_port,
            region=_env_first("LINGUA_WORKER_REGION"),
            gpu_type=_env_first("LINGUA_WORKER_GPU_TYPE"),
            heartbeat_interval_s=_parse_interval_seconds(
                _env_first("LINGUA_WORKER_HEARTBEAT_INTERVAL_SECONDS"),
                20.0,
            ),
            request_timeout_s=_parse_positive_float(
                _env_first("LINGUA_WORKER_CALLBACK_TIMEOUT_SECONDS"),
                8.0,
            ),
        )


class WorkerLifecycleReporter:
    """Registers this worker with Lingua and sends periodic heartbeats."""

    def __init__(
        self,
        config: WorkerCallbackConfig,
        runtime_state: WorkerRuntimeState,
    ) -> None:
        self.config = config
        self.runtime_state = runtime_state
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="lingua-worker-callback",
            daemon=True,
        )
        self._thread.start()

    def report_now(self) -> None:
        """Send one heartbeat now instead of at the next interval (e.g. once dead)."""
        if not self.config.public_base_url:
            return
        threading.Thread(
            target=self._post,
            args=(self.config.heartbeat_url, "heartbeat"),
            name="lingua-worker-callback-now",
            daemon=True,
        ).start()

    def stop(self, timeout_s: float = 5.0, *, final_heartbeat: bool = True) -> None:
        if final_heartbeat and self.config.public_base_url:
            self.runtime_state.request_drain()
            self._post(self.config.heartbeat_url, "final heartbeat")
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout_s)
            self._thread = None

    def payload(self) -> dict[str, Any]:
        snapshot = self.runtime_state.snapshot()
        metadata = {
            "provider": "omnivoice",
            "model_id": _env_first("OMNIVOICE_MODEL_ID") or "omnivoice",
            "queue_depth": snapshot["queue_depth"],
            "active_requests": snapshot["active_requests"],
            "max_concurrent": _parse_positive_int(
                _env_first("OMNIVOICE_MAX_CONCURRENT"),
                snapshot["capacity"],
            ),
            "gpu_utilization": 0,
            "gpu_memory_pct": 0,
            "p95_latency_ms": 0,
        }
        return {
            "worker_type": self.config.worker_type,
            "worker_id": self.config.worker_id,
            "instance_id": self.config.instance_id,
            "base_url": self.config.public_base_url,
            "endpoint_url": self.config.public_base_url,
            "public_ip": self.config.public_ip,
            "public_port": self.config.public_port,
            "internal_port": self.config.internal_port,
            "status": snapshot["status"],
            "capacity": snapshot["capacity"],
            "region": self.config.region,
            "gpu_type": self.config.gpu_type,
            "metadata": metadata,
        }

    def _run(self) -> None:
        self._retry_until_public_base_url()
        if not self.config.public_base_url:
            return

        self._retry_until_registered()
        while not self._stop_event.wait(self.config.heartbeat_interval_s):
            self._post_with_backoff(
                self.config.heartbeat_url,
                "heartbeat",
                forever=False,
            )

    def _retry_until_public_base_url(self) -> None:
        backoff_s = 1.0
        while not self._stop_event.is_set() and not self.config.public_base_url:
            public_endpoint = detect_public_endpoint(self.config.internal_port)
            if public_endpoint.base_url:
                self.config = replace(
                    self.config,
                    public_base_url=public_endpoint.base_url,
                    public_ip=public_endpoint.public_ip,
                    public_port=public_endpoint.public_port,
                )
                logger.info(
                    "Detected Lingua worker public base URL worker_id=%s base_url=%s",
                    self.config.worker_id,
                    self.config.public_base_url,
                )
                return
            logger.warning(
                "Lingua worker registration waiting for public Vast URL "
                "worker_id=%s internal_port=%s. Set LINGUA_WORKER_PUBLIC_BASE_URL "
                "or expose VAST_TCP_PORT_%s.",
                self.config.worker_id,
                self.config.internal_port,
                self.config.internal_port,
            )
            self._stop_event.wait(backoff_s)
            backoff_s = min(backoff_s * 2.0, 30.0)

    def _retry_until_registered(self) -> None:
        backoff_s = 1.0
        while not self._stop_event.is_set():
            if self._post(self.config.register_url, "register"):
                return
            self._stop_event.wait(backoff_s)
            backoff_s = min(backoff_s * 2.0, 30.0)

    def _post_with_backoff(self, url: str, action: str, *, forever: bool) -> bool:
        backoff_s = 1.0
        while not self._stop_event.is_set():
            if self._post(url, action):
                return True
            if not forever:
                return False
            self._stop_event.wait(backoff_s)
            backoff_s = min(backoff_s * 2.0, 30.0)
        return False

    def _post(self, url: str, action: str) -> bool:
        body = json.dumps(self.payload()).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "X-Worker-Token": self.config.token,
            },
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.request_timeout_s,
            ) as response:
                if 200 <= int(response.status) < 300:
                    logger.info(
                        "Lingua worker %s succeeded worker_id=%s status=%s",
                        action,
                        self.config.worker_id,
                        self.runtime_state.snapshot()["status"],
                    )
                    return True
                logger.warning(
                    "Lingua worker %s failed worker_id=%s status_code=%s",
                    action,
                    self.config.worker_id,
                    response.status,
                )
        except urllib.error.HTTPError as exc:
            logger.warning(
                "Lingua worker %s failed worker_id=%s status_code=%s",
                action,
                self.config.worker_id,
                exc.code,
            )
        except Exception as exc:
            logger.warning(
                "Lingua worker %s failed worker_id=%s error=%s",
                action,
                self.config.worker_id,
                exc,
            )
        return False
