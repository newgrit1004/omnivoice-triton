"""Batching primitives for the FastAPI serving layer."""

from __future__ import annotations

import gc
import hashlib
import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, TypeVar

import torch
from omnivoice import OmniVoiceGenerationConfig

from omnivoice_triton.models.faster_runner import CudaGraphCaptureError
from omnivoice_triton.serving.gpu_metrics import BatchGpuMetrics, GPUMetricsMonitor

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CudaFaultError(RuntimeError):
    """The process's CUDA context failed; the worker refuses work until restarted."""


def _exception_chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        yield item
        for linked in (item.__cause__, item.__context__):
            if linked is not None:
                pending.append(linked)


def is_cuda_fault(exc: BaseException) -> bool:
    """Whether an error means the CUDA context can no longer be trusted.

    Device-side asserts, illegal addresses and failed CUDA Graph captures are sticky or
    leave corrupted state behind; out-of-memory errors are recoverable and do not count.
    """
    accelerator_error = getattr(torch, "AcceleratorError", None)
    for item in _exception_chain(exc):
        if isinstance(item, torch.OutOfMemoryError):
            continue
        if isinstance(item, CudaFaultError | CudaGraphCaptureError):
            return True
        if accelerator_error is not None and isinstance(item, accelerator_error):
            return True
        if isinstance(item, RuntimeError) and "CUDA error" in str(item):
            return True
    return False


def _release_failure_frames(exc: BaseException) -> None:
    """Drop the tracebacks of a failed GPU call (log them first).

    The frames hold the call's GPU tensors, and once the error is wrapped and re-raised
    through the HTTP stack they sit in reference cycles until a full GC pass: after a
    burst of out-of-memory failures the server kept ~17 GB allocated and every later
    request failed too.
    """
    for item in _exception_chain(exc):
        item.__traceback__ = None


def _describe_fault(exc: BaseException, where: str) -> str:
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{where}: {type(exc).__name__}: {message}"


@dataclass(frozen=True)
class GenerationBatchKey:
    """Compatibility key for joining requests into the same batch."""

    lane: str
    num_step: int
    guidance_scale: float
    t_shift: float
    layer_penalty_factor: float
    position_temperature: float
    class_temperature: float
    denoise: bool
    postprocess_mode: str
    audio_chunk_duration: float
    audio_chunk_threshold: float


@dataclass(frozen=True)
class BatchedGenerationResult:
    """Per-request result returned from a merged batch execution."""

    audio: Any
    queue_wait_ms: float
    batch_exec_ms: float
    batch_requests: int
    batch_target_tokens: int
    batch_conditioning_tokens: int
    batch_max_sequence_length: int
    peak_vram_gb: float
    gpu_metrics: BatchGpuMetrics | None = None
    generation_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class PendingGeneration:
    """Prepared API request waiting to be merged into a batch."""

    request_id: str
    mode: str
    text: str
    language: str | None
    instruct: str | None
    voice_clone_prompt: Any | None
    speed: float | None
    duration: float | None
    target_tokens: int
    conditioning_tokens: int
    max_sequence_length: int
    batch_key: GenerationBatchKey
    generation_config: OmniVoiceGenerationConfig
    enqueued_at: float = field(default_factory=time.perf_counter)
    future: Future[BatchedGenerationResult] = field(default_factory=Future)


@dataclass(frozen=True)
class ClonePromptCacheResult:
    """Prepared clone prompt plus cache-source metadata."""

    prompt: Any
    source: str


class ClonePromptCache:
    """Thread-safe cache for prepared voice-clone prompts."""

    def __init__(self, max_size: int = 128) -> None:
        self.max_size = max_size
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, str | None, bool], Any] = OrderedDict()
        self._inflight: dict[tuple[str, str | None, bool], threading.Event] = {}

    def get_or_create(
        self,
        *,
        ref_audio_bytes: bytes,
        ref_text: str | None,
        preprocess_prompt: bool,
        factory: Any,
    ) -> ClonePromptCacheResult:
        """Return a cached prompt or create and cache it on demand."""
        if self.max_size <= 0:
            return ClonePromptCacheResult(
                prompt=self._to_cpu_prompt(factory()),
                source="disabled",
            )

        key = (
            hashlib.sha256(ref_audio_bytes).hexdigest(),
            ref_text,
            preprocess_prompt,
        )

        while True:
            with self._lock:
                cached = self._entries.get(key)
                if cached is not None:
                    self._entries.move_to_end(key)
                    return ClonePromptCacheResult(prompt=cached, source="hit")

                event = self._inflight.get(key)
                if event is None:
                    event = threading.Event()
                    self._inflight[key] = event
                    break

            event.wait()

        try:
            prompt = self._to_cpu_prompt(factory())
        except Exception:
            with self._lock:
                self._inflight.pop(key, None)
                event.set()
            raise

        with self._lock:
            self._entries[key] = prompt
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)
            self._inflight.pop(key, None)
            event.set()

        return ClonePromptCacheResult(prompt=prompt, source="miss")

    def _to_cpu_prompt(self, prompt: Any) -> Any:
        return prompt_to_cpu(prompt)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


def prompt_to_cpu(prompt: Any) -> Any:
    """Copy a clone prompt's audio tokens to host memory (a no-op if already there)."""
    ref_audio_tokens = getattr(prompt, "ref_audio_tokens", None)
    if isinstance(ref_audio_tokens, torch.Tensor):
        return type(prompt)(
            ref_audio_tokens=ref_audio_tokens.detach().cpu(),
            ref_text=getattr(prompt, "ref_text"),
            ref_rms=float(getattr(prompt, "ref_rms")),
        )
    return prompt


@dataclass
class _GpuTask:
    """Work queued to run on the batcher thread between batches."""

    name: str
    fn: Callable[[], Any]
    future: Future[Any] = field(default_factory=Future)


class GenerationBatcher:
    """Background worker that merges compatible requests into one generate call.

    Its thread is the only one that should touch CUDA: batches run on it, and other GPU
    work (encoding a voice prompt) goes through run_on_gpu_thread so it can never
    overlap a CUDA Graph capture or pile up next to a batch.
    """

    def __init__(
        self,
        model: Any,
        *,
        collect_ms: float = 10.0,
        max_batch_requests: int = 32,
        max_batch_target_tokens: int = 24000,
        max_batch_conditioning_tokens: int = 12000,
        max_batch_padding_ratio: float = 1.5,
        batch_bucket_sizes: tuple[int, ...] | None = None,
        gpu_metrics_monitor: GPUMetricsMonitor | None = None,
        on_cuda_fault: Callable[[str], None] | None = None,
    ) -> None:
        self._model = model
        self.collect_ms = max(0.0, collect_ms)
        self.max_batch_requests = max(1, max_batch_requests)
        self.max_batch_target_tokens = max(1, max_batch_target_tokens)
        self.max_batch_conditioning_tokens = max(1, max_batch_conditioning_tokens)
        self.max_batch_padding_ratio = max(1.0, max_batch_padding_ratio)
        self.batch_bucket_sizes = tuple(
            sorted({size for size in batch_bucket_sizes or () if size > 0})
        )
        self._gpu_metrics_monitor = gpu_metrics_monitor
        self._on_cuda_fault = on_cuda_fault

        self._condition = threading.Condition()
        self._pending: deque[PendingGeneration] = deque()
        self._tasks: deque[_GpuTask] = deque()
        self._last_ran_task = False
        self._closed = False
        self._cuda_fault: str | None = None
        self._worker = threading.Thread(
            target=self._run,
            name="omnivoice-batcher",
            daemon=True,
        )
        self._worker.start()

    @property
    def cuda_fault(self) -> str | None:
        """Why the CUDA context is considered failed, or None while healthy."""
        with self._condition:
            return self._cuda_fault

    def submit(self, request: PendingGeneration) -> BatchedGenerationResult:
        """Queue a request and block until its merged batch finishes."""
        with self._condition:
            self._raise_if_unavailable_locked()
            self._pending.append(request)
            self._condition.notify()

        return request.future.result()

    def run_on_gpu_thread(self, fn: Callable[[], T], *, name: str) -> T:
        """Run fn on the batcher thread between batches and return its result."""
        if threading.current_thread() is self._worker:
            return fn()
        task = _GpuTask(name=name, fn=fn)
        with self._condition:
            self._raise_if_unavailable_locked()
            self._tasks.append(task)
            self._condition.notify()

        return task.future.result()

    def close(self) -> None:
        """Stop the worker and fail any queued requests."""
        with self._condition:
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            tasks = list(self._tasks)
            self._tasks.clear()
            self._condition.notify_all()

        self._fail_queued(pending, tasks, RuntimeError("Server is shutting down."))
        self._worker.join(timeout=5.0)

    def _raise_if_unavailable_locked(self) -> None:
        if self._closed:
            raise RuntimeError("Batcher is closed.")
        if self._cuda_fault is not None:
            raise CudaFaultError(self._cuda_fault)

    @staticmethod
    def _fail_queued(
        pending: list[PendingGeneration],
        tasks: list[_GpuTask],
        error: BaseException,
    ) -> None:
        for request in pending:
            if not request.future.done():
                request.future.set_exception(error)
        for task in tasks:
            if not task.future.done():
                task.future.set_exception(error)

    def _run(self) -> None:
        while True:
            batch: list[PendingGeneration] = []
            with self._condition:
                while not self._pending and not self._tasks and not self._closed:
                    self._condition.wait()

                if self._closed and not self._pending and not self._tasks:
                    return

                # Short GPU tasks (prompt encoding) run between batches; when both are
                # waiting they take turns, so a stream of /clone-prompts calls cannot
                # starve /generate.
                run_task = bool(self._tasks) and not (
                    self._pending and self._last_ran_task
                )
                task = self._tasks.popleft() if run_task else None
                self._last_ran_task = task is not None
                if task is None:
                    if self.collect_ms > 0:
                        self._condition.wait(timeout=self.collect_ms / 1000.0)
                    batch = self._select_batch_locked()

            if task is not None:
                self._execute_task(task)
                continue

            if not batch:
                continue

            self._execute_batch(batch)

    def _execute_task(self, task: _GpuTask) -> None:
        try:
            result = task.fn()
        except Exception as exc:
            if isinstance(exc, RuntimeError):
                logger.exception("GPU task failed name=%s", task.name)
                _release_failure_frames(exc)
                if is_cuda_fault(exc):
                    self._mark_cuda_fault(_describe_fault(exc, task.name))
                else:
                    _cleanup_cuda_after_gpu_failure(self._model, task.name)
            task.future.set_exception(exc)
        else:
            task.future.set_result(result)

    def _mark_cuda_fault(self, reason: str) -> None:
        """Refuse all further work: the context is broken until the process restarts."""
        with self._condition:
            if self._cuda_fault is not None:
                return
            self._cuda_fault = reason
            pending = list(self._pending)
            self._pending.clear()
            tasks = list(self._tasks)
            self._tasks.clear()

        logger.error(
            "CUDA context failed; refusing new work until restart reason=%s",
            reason,
        )
        self._fail_queued(pending, tasks, CudaFaultError(reason))
        if self._on_cuda_fault is not None:
            try:
                self._on_cuda_fault(reason)
            except Exception:  # pragma: no cover - reporting must not hide the fault
                logger.exception("CUDA fault callback failed.")

    def _select_batch_locked(self) -> list[PendingGeneration]:
        if not self._pending:
            return []

        anchor = self._pending[0]
        selected: list[PendingGeneration] = []
        remaining: deque[PendingGeneration] = deque()

        total_target_tokens = 0
        total_conditioning_tokens = 0
        total_sequence_length = 0
        max_sequence_length = 0

        for request in self._pending:
            if request.batch_key != anchor.batch_key:
                remaining.append(request)
                continue

            next_batch_size = len(selected) + 1
            next_target_tokens = total_target_tokens + request.target_tokens
            next_conditioning_tokens = (
                total_conditioning_tokens + request.conditioning_tokens
            )
            next_total_sequence_length = (
                total_sequence_length + request.max_sequence_length
            )
            next_max_sequence_length = max(
                max_sequence_length,
                request.max_sequence_length,
            )
            padding_ratio = (
                next_batch_size * next_max_sequence_length / next_total_sequence_length
            )

            if selected and (
                next_batch_size > self.max_batch_requests
                or next_target_tokens > self.max_batch_target_tokens
                or next_conditioning_tokens > self.max_batch_conditioning_tokens
                or padding_ratio > self.max_batch_padding_ratio
            ):
                remaining.append(request)
                continue

            selected.append(request)
            total_target_tokens = next_target_tokens
            total_conditioning_tokens = next_conditioning_tokens
            total_sequence_length = next_total_sequence_length
            max_sequence_length = next_max_sequence_length

        if selected and self.batch_bucket_sizes:
            selected, remaining = self._apply_batch_buckets(
                selected,
                remaining,
                anchor.batch_key.lane,
            )

        self._pending = remaining
        return selected

    def _apply_batch_buckets(
        self,
        selected: list[PendingGeneration],
        remaining: deque[PendingGeneration],
        lane: str,
    ) -> tuple[list[PendingGeneration], deque[PendingGeneration]]:
        batch_size = len(selected)
        if batch_size <= 1 or batch_size in self.batch_bucket_sizes:
            return selected, remaining

        target_size = max(
            (size for size in self.batch_bucket_sizes if size <= batch_size),
            default=0,
        )
        if target_size <= 0 or target_size >= batch_size:
            return selected, remaining

        trimmed = selected[target_size:]
        selected = selected[:target_size]
        remaining = deque(trimmed + list(remaining))
        logger.info(
            "Batch bucket trim lane=%s original_size=%d trimmed_size=%d",
            lane,
            batch_size,
            target_size,
        )
        return selected, remaining

    def _execute_batch(self, batch: list[PendingGeneration]) -> None:
        anchor = batch[0]
        batch_started = time.perf_counter()

        request_ids = ",".join(request.request_id for request in batch)
        logger.info(
            "Executing batch lane=%s size=%d request_ids=%s",
            anchor.batch_key.lane,
            len(batch),
            request_ids,
        )

        try:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            batch_gpu_sampler = None
            if self._gpu_metrics_monitor is not None:
                batch_gpu_sampler = self._gpu_metrics_monitor.create_batch_sampler()
                if batch_gpu_sampler is not None:
                    batch_gpu_sampler.start()

            generate_kwargs: dict[str, Any] = {
                "text": [request.text for request in batch],
                "generation_config": anchor.generation_config,
            }

            if any(request.language is not None for request in batch):
                generate_kwargs["language"] = [request.language for request in batch]

            if any(request.instruct is not None for request in batch):
                generate_kwargs["instruct"] = [request.instruct for request in batch]

            if all(request.voice_clone_prompt is not None for request in batch):
                generate_kwargs["voice_clone_prompt"] = [
                    request.voice_clone_prompt for request in batch
                ]

            if any(request.duration is not None for request in batch):
                generate_kwargs["duration"] = [request.duration for request in batch]
            if any(request.speed is not None for request in batch):
                generate_kwargs["speed"] = [request.speed for request in batch]

            outputs = self._model.generate(**generate_kwargs)
            batch_finished = time.perf_counter()
            batch_exec_ms = (batch_finished - batch_started) * 1000.0
            generation_metrics: dict[str, float] = {}
            metrics_getter = getattr(self._model, "get_runtime_stage_metrics", None)
            if callable(metrics_getter):
                raw_metrics = metrics_getter() or {}
                generation_metrics = {
                    str(name): float(value)
                    for name, value in raw_metrics.items()
                    if value is not None
                }

            peak_vram_gb = 0.0
            if torch.cuda.is_available():
                peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
            batch_gpu_metrics = None
            if batch_gpu_sampler is not None:
                batch_gpu_metrics = batch_gpu_sampler.stop(
                    process_peak_vram_gb=peak_vram_gb
                )
                batch_gpu_sampler = None

            if batch_gpu_metrics is not None:
                logger.info(
                    "Completed batch lane=%s size=%d batch_exec_ms=%.2f "
                    "gpu_util_avg_pct=%s gpu_util_peak_pct=%s "
                    "device_vram_used_gb_peak=%s device_vram_util_peak_pct=%s "
                    "process_peak_vram_gb=%.3f gpu_sample_count=%d",
                    anchor.batch_key.lane,
                    len(batch),
                    batch_exec_ms,
                    _format_optional_float(batch_gpu_metrics.gpu_util_avg_pct),
                    _format_optional_float(batch_gpu_metrics.gpu_util_peak_pct),
                    _format_optional_float(batch_gpu_metrics.device_vram_used_gb_peak),
                    _format_optional_float(batch_gpu_metrics.device_vram_util_peak_pct),
                    peak_vram_gb,
                    batch_gpu_metrics.sample_count,
                )
            else:
                logger.info(
                    "Completed batch lane=%s size=%d batch_exec_ms=%.2f "
                    "process_peak_vram_gb=%.3f gpu_metrics=unavailable",
                    anchor.batch_key.lane,
                    len(batch),
                    batch_exec_ms,
                    peak_vram_gb,
                )

            if generation_metrics:
                logger.info(
                    "Batch model timings lane=%s size=%d generate_total_ms=%s "
                    "prepare_inputs_ms=%s iterative_ms=%s chunked_ms=%s "
                    "decode_postprocess_ms=%s",
                    anchor.batch_key.lane,
                    len(batch),
                    _format_optional_float(generation_metrics.get("generate_total_ms")),
                    _format_optional_float(
                        generation_metrics.get("prepare_inference_inputs_ms")
                    ),
                    _format_optional_float(
                        generation_metrics.get("iterative_generate_ms")
                    ),
                    _format_optional_float(
                        generation_metrics.get("chunked_generate_ms")
                    ),
                    _format_optional_float(
                        generation_metrics.get("decode_postprocess_ms")
                    ),
                )

            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"Expected {len(batch)} outputs from model.generate(), got "
                    f"{len(outputs)}."
                )

            batch_target_tokens = sum(request.target_tokens for request in batch)
            batch_conditioning_tokens = sum(
                request.conditioning_tokens for request in batch
            )
            batch_max_sequence_length = max(
                request.max_sequence_length for request in batch
            )

            for request, audio in zip(batch, outputs, strict=True):
                queue_wait_ms = (batch_started - request.enqueued_at) * 1000.0
                request.future.set_result(
                    BatchedGenerationResult(
                        audio=audio,
                        queue_wait_ms=queue_wait_ms,
                        batch_exec_ms=batch_exec_ms,
                        batch_requests=len(batch),
                        batch_target_tokens=batch_target_tokens,
                        batch_conditioning_tokens=batch_conditioning_tokens,
                        batch_max_sequence_length=batch_max_sequence_length,
                        peak_vram_gb=peak_vram_gb,
                        gpu_metrics=batch_gpu_metrics,
                        generation_metrics=generation_metrics,
                    )
                )
        except Exception as exc:
            try:
                peak_vram_gb = (
                    torch.cuda.max_memory_allocated() / 1024**3
                    if torch.cuda.is_available()
                    else 0.0
                )
                if "batch_gpu_sampler" in locals() and batch_gpu_sampler is not None:
                    batch_gpu_sampler.stop(process_peak_vram_gb=peak_vram_gb)
            except Exception:  # pragma: no cover - metrics should not mask failures
                logger.debug(
                    "Failed to stop GPU batch sampler after error.",
                    exc_info=True,
                )
            logger.exception(
                "Batch execution failed lane=%s size=%d",
                anchor.batch_key.lane,
                len(batch),
            )
            _release_failure_frames(exc)
            if is_cuda_fault(exc):
                self._mark_cuda_fault(_describe_fault(exc, "batch"))
            else:
                _cleanup_cuda_after_gpu_failure(self._model, "failed batch")
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(exc)


def _cleanup_cuda_after_gpu_failure(model: Any, what: str) -> None:
    """Release failure-created CUDA state before the next request wave.

    Runs on the batcher thread only: clearing CUDA Graphs from another thread could free
    a graph while it is being replayed.
    """
    forward = getattr(model, "forward", None)
    clear = getattr(forward, "clear", None)
    cleared_graphs = False
    if callable(clear):
        try:
            clear()
            cleared_graphs = True
        except Exception:
            logger.debug(
                "Failed to clear CUDA Graph cache after %s.",
                what,
                exc_info=True,
            )

    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        logger.debug("CUDA synchronize failed after %s.", what, exc_info=True)
    try:
        # Collect first: tensors still held by reference cycles cannot be returned.
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        logger.info(
            "CUDA cleanup complete after %s cleared_graphs=%s",
            what,
            cleared_graphs,
        )
    except Exception:
        logger.debug("CUDA cleanup failed after %s.", what, exc_info=True)


def _format_optional_float(value: float | None) -> str:
    """Format optional floating point values for human-readable logs."""
    if value is None:
        return "-"
    return f"{value:.2f}"
