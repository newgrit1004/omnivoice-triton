"""GPU utilization and VRAM instrumentation helpers for serving."""

from __future__ import annotations

import logging
import threading
from dataclasses import asdict, dataclass
from typing import Any

import torch

try:  # pragma: no cover - availability depends on runtime environment
    import pynvml
except Exception:  # pragma: no cover - defensive fallback
    pynvml = None

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchGpuMetrics:
    """GPU metrics collected while one merged batch is executing."""

    sample_count: int = 0
    gpu_util_avg_pct: float | None = None
    gpu_util_peak_pct: float | None = None
    device_vram_used_gb_avg: float | None = None
    device_vram_used_gb_peak: float | None = None
    device_vram_util_avg_pct: float | None = None
    device_vram_util_peak_pct: float | None = None
    process_peak_vram_gb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a JSON-friendly dict."""
        return asdict(self)


@dataclass(frozen=True)
class _GpuSample:
    gpu_util_pct: float
    device_vram_used_gb: float
    device_vram_util_pct: float


class GPUMetricsMonitor:
    """Sample GPU utilization and VRAM usage during batch execution."""

    _nvml_lock = threading.Lock()
    _nvml_initialized = False

    def __init__(self, device: str, *, sample_interval_ms: float = 25.0) -> None:
        self.device = device
        self.sample_interval_ms = max(5.0, float(sample_interval_ms))
        self.device_index: int | None = None
        self.device_name: str | None = None
        self.memory_total_gb: float | None = None
        self.available = False
        self.unavailable_reason: str | None = None
        self._handle: Any | None = None
        self._last_batch_lock = threading.Lock()
        self._last_batch_metrics: BatchGpuMetrics | None = None

        self._initialize()

    def _initialize(self) -> None:
        if not self.device.startswith("cuda"):
            self.unavailable_reason = "non_cuda_device"
            return
        if not torch.cuda.is_available():
            self.unavailable_reason = "cuda_unavailable"
            return
        if pynvml is None:
            self.unavailable_reason = "pynvml_unavailable"
            return

        try:
            self.device_index = self._resolve_device_index(self.device)
            self._ensure_nvml_initialized()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            device_name = pynvml.nvmlDeviceGetName(self._handle)
            if isinstance(device_name, bytes):
                device_name = device_name.decode("utf-8", errors="replace")
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            self.device_name = str(device_name)
            self.memory_total_gb = memory_info.total / 1024**3
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on NVML runtime
            self.unavailable_reason = f"{type(exc).__name__}: {exc}"
            logger.warning("GPU metrics disabled: %s", self.unavailable_reason)

    def _resolve_device_index(self, device: str) -> int:
        if ":" in device:
            _, raw_index = device.split(":", 1)
            return int(raw_index)
        return int(torch.cuda.current_device())

    @classmethod
    def _ensure_nvml_initialized(cls) -> None:
        with cls._nvml_lock:
            if not cls._nvml_initialized:
                pynvml.nvmlInit()
                cls._nvml_initialized = True

    def _read_sample(self) -> _GpuSample | None:
        if not self.available or self._handle is None:
            return None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        except Exception as exc:  # pragma: no cover - depends on NVML runtime
            logger.debug("Failed to collect GPU sample: %s", exc)
            return None

        total_bytes = max(int(memory_info.total), 1)
        return _GpuSample(
            gpu_util_pct=float(util.gpu),
            device_vram_used_gb=float(memory_info.used) / 1024**3,
            device_vram_util_pct=float(memory_info.used) * 100.0 / float(total_bytes),
        )

    def snapshot(self) -> dict[str, Any]:
        """Return a point-in-time view of current GPU metrics."""
        payload: dict[str, Any] = {
            "available": self.available,
            "device": self.device,
            "device_index": self.device_index,
            "device_name": self.device_name,
            "sample_interval_ms": self.sample_interval_ms,
            "memory_total_gb": self.memory_total_gb,
        }
        if not self.available:
            payload["reason"] = self.unavailable_reason
            return payload

        sample = self._read_sample()
        if sample is None:
            payload["sample_error"] = "sample_unavailable"
            return payload

        payload.update(
            {
                "gpu_util_pct": sample.gpu_util_pct,
                "device_vram_used_gb": sample.device_vram_used_gb,
                "device_vram_util_pct": sample.device_vram_util_pct,
            }
        )
        return payload

    def create_batch_sampler(self) -> _BatchGpuSampler | None:
        """Create a sampler for one batch execution window."""
        if not self.available:
            return None
        return _BatchGpuSampler(self, interval_s=self.sample_interval_ms / 1000.0)

    def record_last_batch(self, metrics: BatchGpuMetrics) -> None:
        """Store the latest batch-level GPU metrics for health reporting."""
        with self._last_batch_lock:
            self._last_batch_metrics = metrics

    def get_last_batch_metrics(self) -> dict[str, Any] | None:
        """Return the most recent batch-level GPU metrics."""
        with self._last_batch_lock:
            if self._last_batch_metrics is None:
                return None
            return self._last_batch_metrics.to_dict()


class _BatchGpuSampler:
    """Background sampler for one merged batch execution."""

    def __init__(self, monitor: GPUMetricsMonitor, *, interval_s: float) -> None:
        self._monitor = monitor
        self._interval_s = max(0.005, interval_s)
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._sample_count = 0
        self._gpu_util_sum = 0.0
        self._gpu_util_peak = 0.0
        self._device_vram_used_sum = 0.0
        self._device_vram_used_peak = 0.0
        self._device_vram_util_sum = 0.0
        self._device_vram_util_peak = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._run,
            name="omnivoice-gpu-sampler",
            daemon=True,
        )
        self._thread.start()

    def _run(self) -> None:
        self._record_sample()
        while not self._stop_event.wait(self._interval_s):
            self._record_sample()

    def _record_sample(self) -> None:
        sample = self._monitor._read_sample()
        if sample is None:
            return
        with self._lock:
            self._sample_count += 1
            self._gpu_util_sum += sample.gpu_util_pct
            self._gpu_util_peak = max(self._gpu_util_peak, sample.gpu_util_pct)
            self._device_vram_used_sum += sample.device_vram_used_gb
            self._device_vram_used_peak = max(
                self._device_vram_used_peak,
                sample.device_vram_used_gb,
            )
            self._device_vram_util_sum += sample.device_vram_util_pct
            self._device_vram_util_peak = max(
                self._device_vram_util_peak,
                sample.device_vram_util_pct,
            )

    def stop(self, *, process_peak_vram_gb: float | None = None) -> BatchGpuMetrics:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

        with self._lock:
            if self._sample_count <= 0:
                metrics = BatchGpuMetrics(process_peak_vram_gb=process_peak_vram_gb)
            else:
                metrics = BatchGpuMetrics(
                    sample_count=self._sample_count,
                    gpu_util_avg_pct=self._gpu_util_sum / self._sample_count,
                    gpu_util_peak_pct=self._gpu_util_peak,
                    device_vram_used_gb_avg=(
                        self._device_vram_used_sum / self._sample_count
                    ),
                    device_vram_used_gb_peak=self._device_vram_used_peak,
                    device_vram_util_avg_pct=(
                        self._device_vram_util_sum / self._sample_count
                    ),
                    device_vram_util_peak_pct=self._device_vram_util_peak,
                    process_peak_vram_gb=process_peak_vram_gb,
                )

        self._monitor.record_last_batch(metrics)
        return metrics
