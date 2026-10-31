"""Tests for the batcher's GPU thread: task routing, faults and frame release."""

from __future__ import annotations

import logging
import threading
import time
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omnivoice import OmniVoiceGenerationConfig

from omnivoice_triton.models.faster_runner import CudaGraphCaptureError
from omnivoice_triton.serving import (
    CudaFaultError,
    GenerationBatcher,
    GenerationBatchKey,
    PendingGeneration,
    is_cuda_fault,
)


@pytest.fixture
def batcher_and_faults():
    faults: list[str] = []
    batcher = GenerationBatcher(SimpleNamespace(), on_cuda_fault=faults.append)
    yield batcher, faults
    batcher.close()


def test_run_on_gpu_thread_runs_on_the_batcher_thread(batcher_and_faults) -> None:
    batcher, faults = batcher_and_faults

    assert (
        batcher.run_on_gpu_thread(lambda: threading.current_thread().name, name="probe")
        == "omnivoice-batcher"
    )
    # nested calls from the batcher thread run inline instead of deadlocking
    assert (
        batcher.run_on_gpu_thread(
            lambda: batcher.run_on_gpu_thread(lambda: 7, name="inner"), name="outer"
        )
        == 7
    )
    assert faults == []


def _pending_generation(request_id: str) -> PendingGeneration:
    return PendingGeneration(
        request_id=request_id,
        mode="design",
        text="Hello.",
        language=None,
        instruct=None,
        voice_clone_prompt=None,
        speed=None,
        duration=None,
        target_tokens=10,
        conditioning_tokens=10,
        max_sequence_length=20,
        batch_key=GenerationBatchKey(
            lane="short_noref",
            num_step=16,
            guidance_scale=2.0,
            t_shift=0.1,
            layer_penalty_factor=5.0,
            position_temperature=5.0,
            class_temperature=0.0,
            denoise=True,
            postprocess_mode="off",
            audio_chunk_duration=15.0,
            audio_chunk_threshold=30.0,
        ),
        generation_config=OmniVoiceGenerationConfig(),
    )


def test_gpu_tasks_and_batches_take_turns() -> None:
    """A stream of prompt-encoding tasks must not starve queued generate requests."""
    order: list[str] = []

    def generate(**kwargs):
        order.append("batch")
        return [np.zeros(10, dtype=np.float32) for _ in kwargs["text"]]

    batcher = GenerationBatcher(SimpleNamespace(generate=generate), collect_ms=0.0)
    gate = threading.Event()

    def task(index: int):
        def run() -> None:
            order.append(f"task{index}")
            time.sleep(0.02)

        return run

    threads = [
        threading.Thread(
            target=batcher.run_on_gpu_thread,
            args=(gate.wait,),
            kwargs={"name": "blocker"},
        )
    ]
    threads[0].start()
    time.sleep(0.1)  # the blocker now holds the batcher thread
    threads += [
        threading.Thread(
            target=batcher.run_on_gpu_thread, args=(task(i),), kwargs={"name": f"t{i}"}
        )
        for i in range(4)
    ]
    threads.append(
        threading.Thread(target=batcher.submit, args=(_pending_generation("g1"),))
    )
    for thread in threads[1:]:
        thread.start()
    time.sleep(0.2)  # everything is queued behind the blocker
    gate.set()
    for thread in threads:
        thread.join(timeout=5.0)
    batcher.close()

    assert order[0] == "batch"
    assert sorted(order[1:]) == ["task0", "task1", "task2", "task3"]


def test_failed_gpu_task_releases_its_frames(batcher_and_faults, monkeypatch) -> None:
    """A failed call's locals (GPU tensors in production) must not outlive it."""
    batcher, _ = batcher_and_faults
    # pytest's log capture keeps each record's traceback; production handlers do not
    monkeypatch.setattr(
        logging.getLogger("omnivoice_triton.serving.batching"), "disabled", True
    )

    class Payload:
        pass

    refs: list[weakref.ref[Payload]] = []

    def encode() -> None:
        payload = Payload()
        refs.append(weakref.ref(payload))
        raise RuntimeError("prompt encoder ran out of memory")

    with pytest.raises(RuntimeError, match="prompt encoder") as excinfo:
        batcher.run_on_gpu_thread(encode, name="encode")

    assert excinfo.value is not None  # the caller still holds the error...
    assert refs[0]() is None  # ...but not the failed call's frames


def test_cuda_fault_refuses_all_later_work(batcher_and_faults) -> None:
    batcher, faults = batcher_and_faults

    def broken() -> None:
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

    with pytest.raises(RuntimeError, match="illegal memory access"):
        batcher.run_on_gpu_thread(broken, name="broken_task")

    assert batcher.cuda_fault is not None
    assert batcher.cuda_fault.startswith("broken_task: RuntimeError: CUDA error")
    assert faults == [batcher.cuda_fault]
    with pytest.raises(CudaFaultError):
        batcher.run_on_gpu_thread(lambda: None, name="after")


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (RuntimeError("CUDA error: device-side assert triggered"), True),
        (RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR"), False),
        (
            torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB."),
            False,
        ),
        (
            CudaGraphCaptureError("CUDA Graph capture failed for shape (2, 8, 491)"),
            True,
        ),
        (CudaFaultError("batch: RuntimeError: CUDA error"), True),
        (ValueError("bad input"), False),
    ],
)
def test_is_cuda_fault(error: BaseException, expected: bool) -> None:
    assert is_cuda_fault(error) is expected


def test_is_cuda_fault_follows_the_cause_chain() -> None:
    try:
        try:
            raise RuntimeError(
                "CUDA error: operation failed due to a previous error during capture"
            )
        except RuntimeError as inner:
            raise RuntimeError("Batch execution failed") from inner
    except RuntimeError as outer:
        assert is_cuda_fault(outer)
