"""FastAPI server for omnivoice-triton runners."""

from __future__ import annotations

import argparse
import gc
import io
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse, Response
from omnivoice import OmniVoiceGenerationConfig
from omnivoice.utils.lang_map import LANG_NAME_TO_ID, lang_display_name

from omnivoice_triton import ALL_RUNNER_NAMES, __version__, create_runner
from omnivoice_triton.models.base_runner import require_cuda_available
from omnivoice_triton.serving import (
    ClonePromptCache,
    GenerationBatcher,
    GenerationBatchKey,
    GPUMetricsMonitor,
    PendingGeneration,
    QueueFullError,
    WorkerCallbackConfig,
    WorkerLifecycleReporter,
    WorkerRuntimeState,
    is_cuda_fault,
    prompt_to_cpu,
)
from omnivoice_triton.serving.sentence_pauses import PAUSE_SECONDS, split_sentences

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "k2-fsa/OmniVoice"
DEFAULT_PORT = 8002
DEFAULT_RUNNER = "hybrid"
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_BATCH_COLLECT_MS = 10.0
DEFAULT_MAX_BATCH_REQUESTS = 32
DEFAULT_MAX_BATCH_TARGET_TOKENS = 24000
DEFAULT_MAX_BATCH_CONDITIONING_TOKENS = 12000
DEFAULT_MAX_BATCH_PADDING_RATIO = 1.5
DEFAULT_CLONE_PROMPT_CACHE_SIZE = 128
DEFAULT_BATCH_BUCKET_SIZES = (1, 2, 4, 8, 16, 32)
DEFAULT_REGISTERED_CLONE_PROMPT_STORE_SIZE = 256
DEFAULT_TOKEN_ESTIMATE_CACHE_SIZE = 4096
DEFAULT_DECODE_POSTPROCESS_WORKERS = max(1, min(4, os.cpu_count() or 1))
DEFAULT_MAX_CUDA_GRAPHS = 8
DEFAULT_PREWARM_CLONE_BATCH_SIZES: tuple[int, ...] = ()
DEFAULT_PREWARM_CLONE_SEQUENCE_LENGTHS: tuple[int, ...] = ()
_SLUG_RE = re.compile(r"[^a-z0-9]+")


class GenerateMode(str, Enum):
    """Supported request modes."""

    auto = "auto"
    design = "design"
    clone = "clone"


class TokenEstimateCache:
    """Thread-safe LRU cache for tokenizer-based batch estimates."""

    def __init__(self, max_size: int = 4096) -> None:
        self.max_size = max(0, max_size)
        self._lock = threading.Lock()
        self._entries: OrderedDict[tuple[str, bool], int] = OrderedDict()

    def get_or_create(
        self,
        *,
        text: str,
        add_special_tokens: bool,
        factory: Callable[[], int],
    ) -> int:
        if self.max_size <= 0:
            return int(factory())

        key = (text, add_special_tokens)
        with self._lock:
            cached = self._entries.get(key)
            if cached is not None:
                self._entries.move_to_end(key)
                return cached

        value = int(factory())
        with self._lock:
            self._entries[key] = value
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)
        return value

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)


class RegisteredClonePromptStore:
    """Thread-safe LRU registry for reusable clone prompts."""

    def __init__(self, max_size: int = 256, storage_device: str = "cpu") -> None:
        self.max_size = max(0, max_size)
        self.storage_device = storage_device
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, Any] = OrderedDict()

    def register(self, prompt: Any) -> str:
        if self.max_size <= 0:
            raise RuntimeError("Registered clone prompt store is disabled.")

        prompt_id = uuid4().hex[:12]
        stored = self._prepare_prompt(prompt)
        with self._lock:
            self._entries[prompt_id] = stored
            self._entries.move_to_end(prompt_id)
            while len(self._entries) > self.max_size:
                self._entries.popitem(last=False)
        return prompt_id

    def get(self, prompt_id: str) -> Any | None:
        with self._lock:
            prompt = self._entries.get(prompt_id)
            if prompt is None:
                return None
            self._entries.move_to_end(prompt_id)
            return prompt

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _prepare_prompt(self, prompt: Any) -> Any:
        ref_audio_tokens = getattr(prompt, "ref_audio_tokens", None)
        if not isinstance(ref_audio_tokens, torch.Tensor):
            return prompt

        tokens = ref_audio_tokens.detach()
        if self.storage_device.startswith("cuda") and torch.cuda.is_available():
            try:
                tokens = tokens.to(self.storage_device, non_blocking=True)
            except Exception as exc:
                logger.warning(
                    "Failed to store registered clone prompt on %s; falling back "
                    "to CPU storage: %s",
                    self.storage_device,
                    exc,
                )
                _cleanup_cuda_after_failure(
                    "registered_clone_prompt_gpu_store_error",
                )
                tokens = ref_audio_tokens.detach().cpu()
                try:
                    tokens = tokens.pin_memory()
                except RuntimeError:
                    pass
        else:
            tokens = tokens.cpu()
            if torch.cuda.is_available():
                try:
                    tokens = tokens.pin_memory()
                except RuntimeError:
                    pass

        prompt_fields = dict(vars(prompt))
        prompt_fields["ref_audio_tokens"] = tokens
        if "ref_rms" in prompt_fields:
            prompt_fields["ref_rms"] = float(prompt_fields["ref_rms"])
        return type(prompt)(**prompt_fields)


def get_best_device() -> str:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        prog="omnivoice-api",
        description="Launch a FastAPI server for omnivoice-triton.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Model checkpoint path or HuggingFace repo id.",
    )
    parser.add_argument(
        "--runner",
        default=DEFAULT_RUNNER,
        choices=ALL_RUNNER_NAMES,
        help="Runner backend to use.",
    )
    parser.add_argument(
        "--sage-attention",
        action="store_true",
        default=False,
        help="Enable SageAttention for Triton/Hybrid runners.",
    )
    parser.add_argument(
        "--full-triton-patch",
        action="store_true",
        default=False,
        help="Patch all decoder layers for Triton/Hybrid runners.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use. Auto-detected if not specified.",
    )
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["bf16", "fp16", "fp32"],
        help="Runner dtype.",
    )
    parser.add_argument("--ip", default="0.0.0.0", help="Server IP to bind to.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port.")
    parser.add_argument(
        "--root-path",
        default=None,
        help="Root path for reverse proxy deployments.",
    )
    parser.add_argument(
        "--no-asr",
        action="store_true",
        default=False,
        help="Skip loading Whisper ASR at startup and require ref_text for clone "
        "requests unless --allow-lazy-asr is also set.",
    )
    parser.add_argument(
        "--allow-lazy-asr",
        action="store_true",
        default=False,
        help="Allow clone requests without ref_text to load Whisper ASR on demand.",
    )
    parser.add_argument(
        "--save-dir",
        default=None,
        help="Optional directory to persist a copy of each generated WAV.",
    )
    parser.add_argument(
        "--batch-collect-ms",
        type=float,
        default=DEFAULT_BATCH_COLLECT_MS,
        help=(
            "How long to wait for nearby compatible requests before launching a batch."
        ),
    )
    parser.add_argument(
        "--max-batch-requests",
        type=int,
        default=DEFAULT_MAX_BATCH_REQUESTS,
        help="Maximum number of requests to merge into one model.generate() call.",
    )
    parser.add_argument(
        "--max-pending-requests",
        type=int,
        default=64,
        help="Maximum prepared requests waiting for a GPU batch (HTTP 429 when full).",
    )
    parser.add_argument(
        "--max-pending-gpu-tasks",
        type=int,
        default=16,
        help="Maximum reference/cache tasks waiting for the GPU (HTTP 429 when full).",
    )
    parser.add_argument(
        "--max-batch-target-tokens",
        type=int,
        default=DEFAULT_MAX_BATCH_TARGET_TOKENS,
        help="Maximum sum of estimated target audio tokens in one batch.",
    )
    parser.add_argument(
        "--max-batch-conditioning-tokens",
        type=int,
        default=DEFAULT_MAX_BATCH_CONDITIONING_TOKENS,
        help="Maximum sum of estimated conditioning tokens in one batch.",
    )
    parser.add_argument(
        "--max-batch-padding-ratio",
        type=float,
        default=DEFAULT_MAX_BATCH_PADDING_RATIO,
        help="Maximum padded-sequence expansion ratio allowed when merging requests.",
    )
    parser.add_argument(
        "--clone-prompt-cache-size",
        type=int,
        default=DEFAULT_CLONE_PROMPT_CACHE_SIZE,
        help="Number of prepared clone prompts to keep in the in-memory cache.",
    )
    parser.add_argument(
        "--registered-clone-prompt-store-size",
        type=int,
        default=DEFAULT_REGISTERED_CLONE_PROMPT_STORE_SIZE,
        help="Number of registered clone prompts to keep for prompt_id reuse.",
    )
    parser.add_argument(
        "--token-estimate-cache-size",
        type=int,
        default=DEFAULT_TOKEN_ESTIMATE_CACHE_SIZE,
        help="LRU cache size for tokenizer-based batching estimates.",
    )
    parser.add_argument(
        "--decode-postprocess-workers",
        type=int,
        default=DEFAULT_DECODE_POSTPROCESS_WORKERS,
        help="Worker threads for CPU-side audio postprocessing after decode.",
    )
    parser.add_argument(
        "--max-cuda-graphs",
        type=int,
        default=DEFAULT_MAX_CUDA_GRAPHS,
        help=(
            "Maximum CUDA Graph shapes to retain for faster/hybrid runners "
            "(default: 8). "
            "Use 0 to disable graph caching."
        ),
    )
    parser.add_argument(
        "--batch-bucket-sizes",
        default="1,2,4,8,16,32",
        help="Preferred merged request-count buckets, comma-separated.",
    )
    parser.add_argument(
        "--prewarm-clone-batch-sizes",
        default=None,
        help=(
            "Optional comma-separated clone request batch sizes to prewarm for "
            "CUDA Graph capture, such as 2,4,8,16,32."
        ),
    )
    parser.add_argument(
        "--prewarm-clone-sequence-lengths",
        default=None,
        help=(
            "Optional comma-separated clone max sequence lengths to prewarm for "
            "CUDA Graph capture, such as 163."
        ),
    )
    return parser


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso_utc(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _format_optional_metric(value: float | None) -> str:
    """Format optional numeric metrics for structured logs."""
    if value is None:
        return "-"
    return f"{value:.2f}"


def _normalize_text(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _normalize_language(language: str | None) -> str | None:
    value = _normalize_text(language)
    if value is None:
        return None
    if value.lower() in {"auto", "none"}:
        return None
    return value


def _parse_positive_int(name: str, raw: str | None, default: int) -> int:
    value = _normalize_text(raw)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise HTTPException(400, detail=f"{name} must be an integer.") from exc
    if parsed <= 0:
        raise HTTPException(400, detail=f"{name} must be > 0.")
    return parsed


def _require_worker_token(
    expected_token: str | None,
    supplied_token: str | None,
) -> None:
    if expected_token is None:
        return
    if supplied_token != expected_token:
        raise HTTPException(401, detail="Invalid worker token.")


def _worker_draining_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={"ok": False, "error": "worker_draining"},
    )


def _worker_unavailable_response(runtime: WorkerRuntimeState) -> JSONResponse:
    """503 for work the worker will not take: draining, or its CUDA context failed."""
    cuda_fault = runtime.cuda_fault
    if cuda_fault is not None:
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "error": "cuda_fault",
                "detail": cuda_fault["reason"],
            },
        )
    return _worker_draining_response()


def _runtime_error_status(exc: RuntimeError) -> int:
    """Distinguish overload, a failed CUDA context, and other runtime failures."""
    if isinstance(exc, QueueFullError):
        return 429
    return 503 if is_cuda_fault(exc) else 500


def _parse_env_nonnegative_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        parsed = float(raw.strip())
    except ValueError:
        logger.warning("Ignoring invalid %s=%r; using %.2f", name, raw, default)
        return default
    if parsed < 0.0:
        logger.warning("Ignoring negative %s=%r; using %.2f", name, raw, default)
        return default
    return parsed


def _parse_nonnegative_float(name: str, raw: str | None, default: float) -> float:
    value = _normalize_text(raw)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as exc:
        raise HTTPException(400, detail=f"{name} must be a number.") from exc
    if parsed < 0.0:
        raise HTTPException(400, detail=f"{name} must be >= 0.")
    return parsed


def _parse_optional_positive_float(name: str, raw: str | None) -> float | None:
    value = _normalize_text(raw)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise HTTPException(400, detail=f"{name} must be a number.") from exc
    if parsed <= 0.0:
        raise HTTPException(400, detail=f"{name} must be > 0.")
    return parsed


def _parse_optional_duration(raw: str | None) -> float | None:
    """Parse duration the same way as the Gradio demo.

    The upstream demo treats empty or non-positive duration as "unset" and
    falls back to speed / duration estimation instead of raising an error.
    """
    value = _normalize_text(raw)
    if value is None:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise HTTPException(400, detail="duration must be a number.") from exc
    if parsed <= 0.0:
        return None
    return parsed


def _parse_postprocess_mode(
    raw_output: str | None,
    raw_mode: str | None = None,
) -> tuple[bool, str]:
    """Parse legacy bool-style and new named postprocess modes."""
    value = _normalize_text(raw_mode) or _normalize_text(raw_output)
    if value is None:
        return True, "full"

    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on", "full"}:
        return True, "full"
    if normalized in {"light", "fast"}:
        return False, "light"
    if normalized in {"0", "false", "no", "off", "none"}:
        return False, "off"
    raise HTTPException(
        400,
        detail="postprocess_output/postprocess_mode must be one of "
        "true, false, full, light, or off.",
    )


def _parse_positive_int_list(
    raw: str | None,
    *,
    option_name: str,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    value = _normalize_text(raw)
    if value is None:
        return default

    sizes: set[int] = set()
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        try:
            parsed = int(item)
        except ValueError as exc:
            raise ValueError(
                f"{option_name} must be a comma-separated list of integers."
            ) from exc
        if parsed <= 0:
            raise ValueError(f"{option_name} values must be > 0.")
        sizes.add(parsed)
    return tuple(sorted(sizes))


def _parse_batch_bucket_sizes(raw: str | None) -> tuple[int, ...]:
    return _parse_positive_int_list(
        raw,
        option_name="--batch-bucket-sizes",
        default=DEFAULT_BATCH_BUCKET_SIZES,
    )


def _to_numpy(audio: Any) -> np.ndarray:
    if isinstance(audio, list):
        audio = audio[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze().cpu().float().numpy()
    waveform = np.asarray(audio, dtype=np.float32).squeeze()
    if waveform.ndim != 1:
        waveform = waveform.reshape(-1)
    return np.clip(waveform, -1.0, 1.0)


def _to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _cleanup_cuda_after_failure(
    reason: str,
    *,
    model: Any | None = None,
    clear_cuda_graphs: bool = False,
) -> None:
    """Best-effort cleanup for failed requests that may have touched CUDA.

    Call only on the batcher thread (inside GenerationBatcher.run_on_gpu_thread): from a
    request thread it would race CUDA Graph capture and replay.
    """
    cleared_graphs = False
    if clear_cuda_graphs and model is not None:
        forward = getattr(model, "forward", None)
        clear = getattr(forward, "clear", None)
        if callable(clear):
            try:
                clear()
                cleared_graphs = True
            except Exception:
                logger.debug(
                    "Failed to clear CUDA Graph cache after %s.",
                    reason,
                    exc_info=True,
                )

    if not torch.cuda.is_available():
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        logger.debug(
            "CUDA synchronize failed during cleanup after %s.",
            reason,
            exc_info=True,
        )
    try:
        gc.collect()
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        logger.info(
            "CUDA cleanup complete after %s cleared_graphs=%s",
            reason,
            cleared_graphs,
        )
    except Exception:
        logger.debug(
            "CUDA cache cleanup failed after %s.",
            reason,
            exc_info=True,
        )


def _slugify(text: str, max_len: int = 48) -> str:
    slug = _SLUG_RE.sub("-", text.lower()).strip("-")
    if not slug:
        slug = "tts"
    return slug[:max_len].strip("-") or "tts"


def _decode_ref_audio_upload(ref_audio_bytes: bytes) -> tuple[torch.Tensor, int]:
    try:
        waveform, sample_rate = sf.read(
            io.BytesIO(ref_audio_bytes),
            dtype="float32",
            always_2d=True,
        )
    except RuntimeError as exc:
        raise HTTPException(400, detail=f"Invalid ref_audio: {exc}") from exc

    if waveform.size == 0:
        raise HTTPException(400, detail="ref_audio is empty.")

    channels_first = np.ascontiguousarray(waveform.T)
    return torch.from_numpy(channels_first), int(sample_rate)


def _estimate_target_tokens(
    model: Any,
    *,
    text: str,
    voice_clone_prompt: Any | None,
    speed: float | None,
    duration: float | None,
) -> int:
    frame_rate = int(model.audio_tokenizer.config.frame_rate)
    if duration is not None:
        return max(1, int(duration * frame_rate))

    ref_text = getattr(voice_clone_prompt, "ref_text", None)
    ref_audio_tokens = getattr(voice_clone_prompt, "ref_audio_tokens", None)
    ref_audio_len = (
        int(ref_audio_tokens.size(-1))
        if isinstance(ref_audio_tokens, torch.Tensor)
        else None
    )
    return int(
        model._estimate_target_tokens(
            text,
            ref_text,
            ref_audio_len,
            speed=speed if speed is not None else 1.0,
        )
    )


def _estimate_conditioning_tokens(
    model: Any,
    *,
    text: str,
    language: str | None,
    instruct: str | None,
    voice_clone_prompt: Any | None,
    denoise: bool,
    token_cache: TokenEstimateCache | None = None,
) -> int:
    tokenizer = getattr(model, "text_tokenizer", None)
    if tokenizer is None:
        ref_audio_tokens = getattr(voice_clone_prompt, "ref_audio_tokens", None)
        ref_audio_len = (
            int(ref_audio_tokens.size(-1))
            if isinstance(ref_audio_tokens, torch.Tensor)
            else 0
        )
        return len(text) + ref_audio_len

    style_text = ""
    if denoise and voice_clone_prompt is not None:
        style_text += "<|denoise|>"
    style_text += f"<|lang_start|>{language or 'None'}<|lang_end|>"
    style_text += f"<|instruct_start|>{instruct or 'None'}<|instruct_end|>"
    style_tokens = (
        token_cache.get_or_create(
            text=style_text,
            add_special_tokens=True,
            factory=lambda: int(
                tokenizer(style_text, return_tensors="pt").input_ids.shape[-1]
            ),
        )
        if token_cache is not None
        else int(tokenizer(style_text, return_tensors="pt").input_ids.shape[-1])
    )

    combined_text = text.strip()
    ref_text = getattr(voice_clone_prompt, "ref_text", None)
    if ref_text:
        combined_text = f"{ref_text.strip()} {combined_text}"
    combined_text = re.sub(r"[\r\n]+", "", combined_text)
    combined_text = re.sub(r"[ \t]+", " ", combined_text).strip()
    wrapped_text = f"<|text_start|>{combined_text}<|text_end|>"
    text_tokens = (
        token_cache.get_or_create(
            text=wrapped_text,
            add_special_tokens=False,
            factory=lambda: int(
                tokenizer(
                    wrapped_text,
                    add_special_tokens=False,
                    return_tensors="pt",
                ).input_ids.shape[-1]
            ),
        )
        if token_cache is not None
        else int(
            tokenizer(
                wrapped_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.shape[-1]
        )
    )

    ref_audio_tokens = getattr(voice_clone_prompt, "ref_audio_tokens", None)
    ref_audio_len = (
        int(ref_audio_tokens.size(-1))
        if isinstance(ref_audio_tokens, torch.Tensor)
        else 0
    )
    return style_tokens + text_tokens + ref_audio_len


def _build_batch_key(
    model: Any,
    *,
    voice_clone_prompt: Any | None,
    target_tokens: int,
    generation_config: OmniVoiceGenerationConfig,
) -> GenerationBatchKey:
    frame_rate = int(model.audio_tokenizer.config.frame_rate)
    chunk_threshold_tokens = int(generation_config.audio_chunk_threshold * frame_rate)
    is_long = target_tokens > chunk_threshold_tokens
    has_prompt = voice_clone_prompt is not None
    lane = f"{'long' if is_long else 'short'}_{'ref' if has_prompt else 'noref'}"

    return GenerationBatchKey(
        lane=lane,
        num_step=generation_config.num_step,
        guidance_scale=generation_config.guidance_scale,
        t_shift=generation_config.t_shift,
        layer_penalty_factor=generation_config.layer_penalty_factor,
        position_temperature=generation_config.position_temperature,
        class_temperature=generation_config.class_temperature,
        denoise=generation_config.denoise,
        postprocess_mode=str(
            getattr(
                generation_config,
                "postprocess_mode",
                "full" if generation_config.postprocess_output else "off",
            )
        ),
        audio_chunk_duration=generation_config.audio_chunk_duration,
        audio_chunk_threshold=generation_config.audio_chunk_threshold,
    )


@torch.inference_mode()
def _create_voice_clone_prompt(
    model: Any,
    *,
    ref_audio_bytes: bytes,
    ref_text: str | None,
    preprocess_prompt: bool,
    asr_load_lock: threading.Lock,
    allow_lazy_asr: bool,
) -> Any:
    if (
        ref_text is None
        and hasattr(model, "load_asr_model")
        and getattr(model, "_asr_pipe", None) is None
    ):
        if not allow_lazy_asr:
            raise HTTPException(
                400,
                detail=(
                    "ref_text is required for clone requests when lazy ASR is "
                    "disabled. Start with --allow-lazy-asr to permit on-demand "
                    "Whisper loading."
                ),
            )
        with asr_load_lock:
            if getattr(model, "_asr_pipe", None) is None:
                model.load_asr_model()

    return model.create_voice_clone_prompt(
        ref_audio=_decode_ref_audio_upload(ref_audio_bytes),
        ref_text=ref_text,
        preprocess_prompt=preprocess_prompt,
    )


def _describe_clone_prompt(prompt: Any, fallback_device: str) -> dict[str, Any]:
    ref_audio_tokens = getattr(prompt, "ref_audio_tokens", None)
    prompt_audio_tokens = (
        int(ref_audio_tokens.size(-1))
        if isinstance(ref_audio_tokens, torch.Tensor)
        else None
    )
    stored_device = (
        str(ref_audio_tokens.device)
        if isinstance(ref_audio_tokens, torch.Tensor)
        else fallback_device
    )
    return {
        "prompt_audio_tokens": prompt_audio_tokens,
        "ref_text": getattr(prompt, "ref_text", None),
        "stored_device": stored_device,
    }


def _save_output(
    wav_bytes: bytes,
    save_dir: Path,
    mode: GenerateMode,
    text: str,
    request_id: str,
) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    stamp = _utc_now().strftime("%Y%m%dT%H%M%SZ")
    output_path = save_dir / f"{stamp}_{mode.value}_{_slugify(text)}_{request_id}.wav"
    output_path.write_bytes(wav_bytes)
    return output_path


def _build_runner(
    runner_name: str,
    *,
    model_checkpoint: str,
    device: str,
    dtype: str,
    enable_sage_attention: bool,
    full_triton_patch: bool,
    decode_postprocess_workers: int,
    max_cuda_graphs: int,
    runner_factory: Callable[..., Any],
) -> Any:
    kwargs: dict[str, Any] = {
        "device": device,
        "model_id": model_checkpoint,
        "dtype": dtype,
        "decode_postprocess_workers": decode_postprocess_workers,
    }
    if runner_name in {"faster", "hybrid"}:
        kwargs["max_cuda_graphs"] = max(0, int(max_cuda_graphs))
    if full_triton_patch:
        if runner_name not in {"triton", "hybrid"}:
            raise ValueError(
                "--full-triton-patch is only supported for the triton and "
                "hybrid runners."
            )
        kwargs["patch_range"] = None
    if enable_sage_attention:
        if runner_name not in {"triton", "hybrid"}:
            raise ValueError(
                "--sage-attention is only supported for the triton and hybrid runners."
            )
        kwargs["enable_sage_attention"] = True
    return runner_factory(runner_name, **kwargs)


def _build_clone_prewarm_shapes(
    model: Any,
    *,
    request_batch_sizes: tuple[int, ...],
    sequence_lengths: tuple[int, ...],
) -> tuple[tuple[int, int, int], ...]:
    if not request_batch_sizes or not sequence_lengths:
        return ()

    num_audio_codebook = int(
        getattr(getattr(model, "config", None), "num_audio_codebook", 8)
    )
    shapes = {
        (int(batch_size) * 2, num_audio_codebook, int(seq_len))
        for batch_size in request_batch_sizes
        for seq_len in sequence_lengths
        if batch_size > 0 and seq_len > 0
    }
    return tuple(sorted(shapes))


def create_app(
    model_checkpoint: str = DEFAULT_MODEL_ID,
    runner_name: str = DEFAULT_RUNNER,
    device: str | None = None,
    dtype: str = "fp16",
    load_asr: bool = True,
    allow_lazy_asr: bool = False,
    enable_sage_attention: bool = False,
    full_triton_patch: bool = False,
    decode_postprocess_workers: int = DEFAULT_DECODE_POSTPROCESS_WORKERS,
    max_cuda_graphs: int = DEFAULT_MAX_CUDA_GRAPHS,
    save_dir: str | None = None,
    batch_collect_ms: float = DEFAULT_BATCH_COLLECT_MS,
    max_batch_requests: int = DEFAULT_MAX_BATCH_REQUESTS,
    max_pending_requests: int = 64,
    max_pending_gpu_tasks: int = 16,
    max_batch_target_tokens: int = DEFAULT_MAX_BATCH_TARGET_TOKENS,
    max_batch_conditioning_tokens: int = DEFAULT_MAX_BATCH_CONDITIONING_TOKENS,
    max_batch_padding_ratio: float = DEFAULT_MAX_BATCH_PADDING_RATIO,
    clone_prompt_cache_size: int = DEFAULT_CLONE_PROMPT_CACHE_SIZE,
    registered_clone_prompt_store_size: int = (
        DEFAULT_REGISTERED_CLONE_PROMPT_STORE_SIZE
    ),
    token_estimate_cache_size: int = DEFAULT_TOKEN_ESTIMATE_CACHE_SIZE,
    batch_bucket_sizes: tuple[int, ...] = DEFAULT_BATCH_BUCKET_SIZES,
    prewarm_clone_batch_sizes: tuple[int, ...] = DEFAULT_PREWARM_CLONE_BATCH_SIZES,
    prewarm_clone_sequence_lengths: tuple[
        int, ...
    ] = DEFAULT_PREWARM_CLONE_SEQUENCE_LENGTHS,
    server_port: int = DEFAULT_PORT,
    start_worker_callback: bool = True,
    runner_factory: Callable[..., Any] | None = None,
) -> FastAPI:
    """Create the FastAPI app."""
    resolved_device = device or get_best_device()
    if resolved_device.startswith("cuda"):
        require_cuda_available()
    elif runner_name != "base":
        raise ValueError(
            f"Runner '{runner_name}' requires CUDA. Use --runner base or a CUDA device."
        )

    factory = runner_factory or create_runner
    worker_callback_config = WorkerCallbackConfig.from_env(port=server_port)
    worker_capacity = (
        worker_callback_config.capacity if worker_callback_config is not None else 1
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        runner = _build_runner(
            runner_name,
            model_checkpoint=model_checkpoint,
            device=resolved_device,
            dtype=dtype,
            enable_sage_attention=enable_sage_attention,
            full_triton_patch=full_triton_patch,
            decode_postprocess_workers=decode_postprocess_workers,
            max_cuda_graphs=max_cuda_graphs,
            runner_factory=factory,
        )
        runner.load_model()
        model = runner.model
        if load_asr and hasattr(model, "load_asr_model"):
            model.load_asr_model()
        app.state.runner = runner
        app.state.sample_rate = int(
            getattr(model, "sampling_rate", DEFAULT_SAMPLE_RATE) or DEFAULT_SAMPLE_RATE
        )
        app.state.clone_prompt_cache = ClonePromptCache(
            max_size=clone_prompt_cache_size
        )
        app.state.registered_clone_prompts = RegisteredClonePromptStore(
            max_size=registered_clone_prompt_store_size,
            storage_device=resolved_device,
        )
        app.state.token_estimate_cache = TokenEstimateCache(
            max_size=token_estimate_cache_size
        )
        app.state.gpu_metrics_monitor = GPUMetricsMonitor(resolved_device)

        def _handle_cuda_fault(reason: str) -> None:
            # /health and /healthz turn 503, and Lingua hears "dead" right away.
            app.state.worker_runtime.mark_cuda_fault(reason)
            reporter = app.state.worker_lifecycle_reporter
            if reporter is not None:
                reporter.report_now()

        app.state.batcher = GenerationBatcher(
            model,
            collect_ms=batch_collect_ms,
            max_batch_requests=max_batch_requests,
            max_pending_requests=max_pending_requests,
            max_pending_gpu_tasks=max_pending_gpu_tasks,
            max_batch_target_tokens=max_batch_target_tokens,
            max_batch_conditioning_tokens=max_batch_conditioning_tokens,
            max_batch_padding_ratio=max_batch_padding_ratio,
            batch_bucket_sizes=batch_bucket_sizes,
            gpu_metrics_monitor=app.state.gpu_metrics_monitor,
            on_cuda_fault=_handle_cuda_fault,
        )
        requested_prewarm_batch_sizes = tuple(
            size for size in prewarm_clone_batch_sizes if size <= max_batch_requests
        )
        requested_prewarm_shapes = _build_clone_prewarm_shapes(
            model,
            request_batch_sizes=requested_prewarm_batch_sizes,
            sequence_lengths=prewarm_clone_sequence_lengths,
        )
        prewarm_started = time.perf_counter()
        prewarm_summary: dict[str, Any] = {
            "requested_batch_sizes": list(requested_prewarm_batch_sizes),
            "requested_sequence_lengths": list(prewarm_clone_sequence_lengths),
            "requested_shapes": [list(shape) for shape in requested_prewarm_shapes],
            "warmed_shapes": [],
            "supported": hasattr(runner, "prewarm_cuda_graph_shapes"),
        }
        if requested_prewarm_shapes and hasattr(runner, "prewarm_cuda_graph_shapes"):
            prewarm_summary.update(
                runner.prewarm_cuda_graph_shapes(requested_prewarm_shapes)
            )
        prewarm_summary["duration_ms"] = round(
            (time.perf_counter() - prewarm_started) * 1000.0,
            2,
        )
        app.state.cuda_graph_prewarm_summary = prewarm_summary
        if worker_callback_config is not None and start_worker_callback:
            worker_reporter = WorkerLifecycleReporter(
                worker_callback_config,
                app.state.worker_runtime,
            )
            app.state.worker_lifecycle_reporter = worker_reporter
            worker_reporter.start()
        logger.info(
            "API server ready runner=%s model=%s device=%s dtype=%s asr_loaded=%s "
            "batch_collect_ms=%.1f max_batch_requests=%d batch_bucket_sizes=%s "
            "full_triton_patch=%s decode_postprocess_workers=%d "
            "max_cuda_graphs=%d allow_lazy_asr=%s "
            "prewarm_clone_batch_sizes=%s "
            "prewarm_clone_sequence_lengths=%s gpu_metrics_available=%s "
            "gpu_device=%s",
            runner_name,
            model_checkpoint,
            resolved_device,
            dtype,
            bool(getattr(model, "_asr_pipe", None)),
            batch_collect_ms,
            max_batch_requests,
            list(batch_bucket_sizes),
            full_triton_patch,
            decode_postprocess_workers,
            max_cuda_graphs,
            allow_lazy_asr,
            list(requested_prewarm_batch_sizes),
            list(prewarm_clone_sequence_lengths),
            app.state.gpu_metrics_monitor.available,
            app.state.gpu_metrics_monitor.device_name
            or app.state.gpu_metrics_monitor.unavailable_reason
            or "-",
        )
        try:
            yield
        finally:
            worker_reporter = app.state.worker_lifecycle_reporter
            app.state.worker_runtime.request_drain()
            idle = app.state.worker_runtime.wait_for_idle(
                _parse_env_nonnegative_float(
                    "LINGUA_WORKER_DRAIN_TIMEOUT_SECONDS", 30.0
                )
            )
            if not idle:
                logger.warning(
                    "Worker shutdown continuing with active_requests=%s",
                    app.state.worker_runtime.snapshot()["active_requests"],
                )
            if worker_reporter is not None:
                worker_reporter.stop()
                app.state.worker_lifecycle_reporter = None
            if app.state.batcher is not None:
                app.state.batcher.close()
                app.state.batcher = None
            app.state.gpu_metrics_monitor = None
            if app.state.runner is not None:
                app.state.runner.unload_model()
                app.state.runner = None

    app = FastAPI(
        title="omnivoice-triton API",
        version=__version__,
        description="HTTP API for OmniVoice text-to-speech with Triton runners.",
        lifespan=lifespan,
    )
    app.state.model_checkpoint = model_checkpoint
    app.state.runner_name = runner_name
    app.state.device = resolved_device
    app.state.dtype = dtype
    app.state.load_asr = load_asr
    app.state.allow_lazy_asr = allow_lazy_asr
    app.state.enable_sage_attention = enable_sage_attention
    app.state.full_triton_patch = full_triton_patch
    app.state.decode_postprocess_workers = decode_postprocess_workers
    app.state.max_cuda_graphs = max(0, int(max_cuda_graphs))
    app.state.save_dir = Path(save_dir) if save_dir else None
    app.state.runner = None
    app.state.sample_rate = DEFAULT_SAMPLE_RATE
    app.state.batcher = None
    app.state.clone_prompt_cache = None
    app.state.registered_clone_prompts = None
    app.state.token_estimate_cache = None
    app.state.gpu_metrics_monitor = None
    app.state.asr_load_lock = threading.Lock()
    app.state.batch_collect_ms = batch_collect_ms
    app.state.max_batch_requests = max_batch_requests
    app.state.max_batch_target_tokens = max_batch_target_tokens
    app.state.max_batch_conditioning_tokens = max_batch_conditioning_tokens
    app.state.max_batch_padding_ratio = max_batch_padding_ratio
    app.state.clone_prompt_cache_size = clone_prompt_cache_size
    app.state.registered_clone_prompt_store_size = registered_clone_prompt_store_size
    app.state.token_estimate_cache_size = token_estimate_cache_size
    app.state.batch_bucket_sizes = list(batch_bucket_sizes)
    app.state.prewarm_clone_batch_sizes = list(prewarm_clone_batch_sizes)
    app.state.prewarm_clone_sequence_lengths = list(prewarm_clone_sequence_lengths)
    app.state.cuda_graph_prewarm_summary = None
    app.state.worker_runtime = WorkerRuntimeState(capacity=worker_capacity)
    app.state.worker_callback_config = worker_callback_config
    app.state.worker_lifecycle_reporter = None

    def _on_gpu_thread(name: str, fn: Callable[[], Any]) -> Any:
        """Run CUDA work for a request on the batcher thread (see GenerationBatcher)."""
        batcher = app.state.batcher
        if batcher is None:
            raise HTTPException(503, detail="Batcher is not ready yet.")
        return batcher.run_on_gpu_thread(fn, name=name)

    @app.get("/health")
    def health() -> Any:
        runner = app.state.runner
        model = getattr(runner, "model", None)
        graph_metrics: dict[str, Any] = {}
        generation_metrics: dict[str, Any] = {}
        gpu_metrics: dict[str, Any] = {"available": False, "reason": "not_initialized"}
        last_batch_gpu_metrics: dict[str, Any] | None = None
        if runner is not None and hasattr(runner, "get_cuda_graph_metrics"):
            try:
                graph_metrics = runner.get_cuda_graph_metrics()
            except Exception:  # pragma: no cover - health should stay resilient
                logger.exception("Failed to collect CUDA graph metrics.")
        if runner is not None and hasattr(runner, "get_generation_metrics"):
            try:
                generation_metrics = runner.get_generation_metrics()
            except Exception:  # pragma: no cover - health should stay resilient
                logger.exception("Failed to collect generation metrics.")
        gpu_monitor = app.state.gpu_metrics_monitor
        if gpu_monitor is not None:
            try:
                gpu_metrics = gpu_monitor.snapshot()
                last_batch_gpu_metrics = gpu_monitor.get_last_batch_metrics()
            except Exception:  # pragma: no cover - health should stay resilient
                logger.exception("Failed to collect GPU metrics.")
        worker_config = app.state.worker_callback_config
        worker_runtime = app.state.worker_runtime.snapshot()
        cuda_fault = worker_runtime["cuda_fault"]
        body = {
            "status": "ok" if cuda_fault is None else "error",
            "cuda_fault": cuda_fault,
            "model": app.state.model_checkpoint,
            "runner": app.state.runner_name,
            "device": app.state.device,
            "dtype": app.state.dtype,
            "sample_rate": app.state.sample_rate,
            "load_asr": app.state.load_asr,
            "allow_lazy_asr": app.state.allow_lazy_asr,
            "asr_loaded": bool(getattr(model, "_asr_pipe", None)),
            "save_dir": str(app.state.save_dir) if app.state.save_dir else None,
            "sage_attention": bool(app.state.enable_sage_attention),
            "full_triton_patch": bool(app.state.full_triton_patch),
            "decode_postprocess_workers": int(app.state.decode_postprocess_workers),
            "max_cuda_graphs": int(app.state.max_cuda_graphs),
            "batch_collect_ms": app.state.batch_collect_ms,
            "max_batch_requests": app.state.max_batch_requests,
            "batcher_queue": app.state.batcher.queue_snapshot(),
            "max_batch_target_tokens": app.state.max_batch_target_tokens,
            "max_batch_conditioning_tokens": app.state.max_batch_conditioning_tokens,
            "max_batch_padding_ratio": app.state.max_batch_padding_ratio,
            "clone_prompt_cache_size": app.state.clone_prompt_cache_size,
            "registered_clone_prompt_store_size": (
                app.state.registered_clone_prompt_store_size
            ),
            "registered_clone_prompt_count": len(
                app.state.registered_clone_prompts or ()
            ),
            "token_estimate_cache_size": app.state.token_estimate_cache_size,
            "token_estimate_cache_entries": len(app.state.token_estimate_cache or ()),
            "batch_bucket_sizes": app.state.batch_bucket_sizes,
            "prewarm_clone_batch_sizes": app.state.prewarm_clone_batch_sizes,
            "prewarm_clone_sequence_lengths": app.state.prewarm_clone_sequence_lengths,
            "cuda_graph_capture_count": int(graph_metrics.get("capture_count", 0)),
            "cuda_graph_max_graphs": int(
                graph_metrics.get("max_graphs", app.state.max_cuda_graphs)
            ),
            "cuda_graph_shapes": graph_metrics.get("shapes", []),
            "cuda_graph_prewarm": app.state.cuda_graph_prewarm_summary,
            "last_generation_metrics": generation_metrics,
            "gpu_metrics": gpu_metrics,
            "last_batch_gpu_metrics": last_batch_gpu_metrics,
            "worker": {
                "worker_type": (
                    worker_config.worker_type if worker_config is not None else "tts"
                ),
                "worker_id": (
                    worker_config.worker_id if worker_config is not None else None
                ),
                "instance_id": (
                    worker_config.instance_id if worker_config is not None else None
                ),
                "base_url": (
                    worker_config.public_base_url if worker_config is not None else None
                ),
                "endpoint_url": (
                    worker_config.public_base_url if worker_config is not None else None
                ),
                "public_ip": (
                    worker_config.public_ip if worker_config is not None else None
                ),
                "public_port": (
                    worker_config.public_port if worker_config is not None else None
                ),
                "internal_port": (
                    worker_config.internal_port if worker_config is not None else None
                ),
                "callback_enabled": worker_config is not None,
                **worker_runtime,
            },
        }
        if cuda_fault is not None:
            # A failed CUDA context fails every request; tell health checks so.
            return JSONResponse(status_code=503, content=jsonable_encoder(body))
        return body

    @app.get("/healthz")
    def healthz() -> Any:
        runner = app.state.runner
        if runner is None or app.state.batcher is None:
            raise HTTPException(503, detail="Model is not ready yet.")
        cuda_fault = app.state.worker_runtime.cuda_fault
        if cuda_fault is not None:
            return JSONResponse(
                status_code=503,
                content={
                    "ok": False,
                    "service": "omnivoice",
                    "worker_type": "tts",
                    "status": "dead",
                    "error": "cuda_fault",
                    "detail": cuda_fault["reason"],
                },
            )
        return {
            "ok": True,
            "service": "omnivoice",
            "worker_type": "tts",
            "status": app.state.worker_runtime.snapshot()["status"],
        }

    @app.get("/languages")
    def languages() -> dict[str, Any]:
        items = [
            {
                "id": LANG_NAME_TO_ID[name],
                "name": name,
                "display_name": lang_display_name(name),
            }
            for name in sorted(LANG_NAME_TO_ID, key=lang_display_name)
        ]
        return {"count": len(items), "languages": items}

    @app.get("/worker/state")
    @app.get("/worker")
    def worker_status() -> dict[str, Any]:
        worker_config = app.state.worker_callback_config
        runtime = app.state.worker_runtime.snapshot()
        return {
            "worker_type": worker_config.worker_type if worker_config else "tts",
            "worker_id": worker_config.worker_id if worker_config else None,
            "instance_id": worker_config.instance_id if worker_config else None,
            "base_url": worker_config.public_base_url if worker_config else None,
            "endpoint_url": worker_config.public_base_url if worker_config else None,
            "public_ip": worker_config.public_ip if worker_config else None,
            "public_port": worker_config.public_port if worker_config else None,
            "internal_port": worker_config.internal_port if worker_config else None,
            "callback_enabled": worker_config is not None,
            **runtime,
        }

    @app.post("/drain")
    def drain_worker(
        x_worker_token: str | None = Header(default=None, alias="X-Worker-Token"),
    ) -> dict[str, Any]:
        expected_token = (
            app.state.worker_callback_config.token
            if app.state.worker_callback_config is not None
            else os.environ.get("LINGUA_WORKER_TOKEN")
        )
        _require_worker_token(expected_token, x_worker_token)
        app.state.worker_runtime.request_drain()
        return worker_status()

    @app.post("/clone-prompts")
    def register_clone_prompt(
        ref_audio: UploadFile = File(...),
        ref_text: str | None = Form(None),
        preprocess_prompt: bool = Form(True),
    ) -> Any:
        runner = app.state.runner
        if runner is None:
            raise HTTPException(503, detail="Model is not ready yet.")

        prompt_store = app.state.registered_clone_prompts
        if prompt_store is None or prompt_store.max_size <= 0:
            raise HTTPException(
                503,
                detail="Registered clone prompt store is disabled.",
            )
        request_counted = app.state.worker_runtime.begin_request()
        if not request_counted:
            return _worker_unavailable_response(app.state.worker_runtime)

        request_id = uuid4().hex[:12]
        started_at = _utc_now()
        started_perf = time.perf_counter()
        ref_text_value = _normalize_text(ref_text)

        try:
            ref_audio_bytes = ref_audio.file.read()
            if not ref_audio_bytes:
                raise HTTPException(400, detail="ref_audio is empty.")

            prompt_id = _on_gpu_thread(
                "clone_prompt_register",
                lambda: prompt_store.register(
                    _create_voice_clone_prompt(
                        runner.model,
                        ref_audio_bytes=ref_audio_bytes,
                        ref_text=ref_text_value,
                        preprocess_prompt=preprocess_prompt,
                        asr_load_lock=app.state.asr_load_lock,
                        allow_lazy_asr=app.state.allow_lazy_asr,
                    )
                ),
            )
            finished_at = _utc_now()
            duration_ms = (time.perf_counter() - started_perf) * 1000.0
            stored_prompt = prompt_store.get(prompt_id)
            ref_audio_tokens = getattr(stored_prompt, "ref_audio_tokens", None)
            prompt_audio_tokens = (
                int(ref_audio_tokens.size(-1))
                if isinstance(ref_audio_tokens, torch.Tensor)
                else None
            )
            stored_device = (
                str(ref_audio_tokens.device)
                if isinstance(ref_audio_tokens, torch.Tensor)
                else app.state.device
            )

            logger.info(
                "clone_prompt_id=%s request_id=%s status=registered started_at=%s "
                "finished_at=%s duration_ms=%.2f prompt_audio_tokens=%s "
                "stored_device=%s has_ref_text=%s",
                prompt_id,
                request_id,
                _iso_utc(started_at),
                _iso_utc(finished_at),
                duration_ms,
                prompt_audio_tokens if prompt_audio_tokens is not None else "-",
                stored_device,
                bool(getattr(stored_prompt, "ref_text", None)),
            )

            return {
                "status": "ok",
                "prompt_id": prompt_id,
                "request_id": request_id,
                "created_at": _iso_utc(finished_at),
                "duration_ms": round(duration_ms, 2),
                "prompt_audio_tokens": prompt_audio_tokens,
                "ref_text": getattr(stored_prompt, "ref_text", None),
                "stored_device": stored_device,
            }
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("request_id=%s status=runtime_error", request_id)
            raise HTTPException(
                status_code=_runtime_error_status(exc),
                detail=str(exc),
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive server guard
            logger.exception("request_id=%s status=unexpected_error", request_id)
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        finally:
            app.state.worker_runtime.end_request()
            ref_audio.file.close()

    def _warm_clone_prompt_cache(
        *,
        runner: Any,
        ref_audio_bytes: bytes,
        ref_text_value: str | None,
        preprocess_prompt: bool,
    ) -> dict[str, Any]:
        prompt_cache = app.state.clone_prompt_cache
        if prompt_cache is None:
            raise HTTPException(503, detail="Clone prompt cache is not ready yet.")

        def _create_prompt() -> Any:
            return _on_gpu_thread(
                "clone_prompt_cache_warm",
                lambda: prompt_to_cpu(
                    _create_voice_clone_prompt(
                        runner.model,
                        ref_audio_bytes=ref_audio_bytes,
                        ref_text=ref_text_value,
                        preprocess_prompt=preprocess_prompt,
                        asr_load_lock=app.state.asr_load_lock,
                        allow_lazy_asr=app.state.allow_lazy_asr,
                    )
                ),
            )

        prompt_result = prompt_cache.get_or_create(
            ref_audio_bytes=ref_audio_bytes,
            ref_text=ref_text_value,
            preprocess_prompt=preprocess_prompt,
            factory=_create_prompt,
        )
        return {
            "cache": "clone_prompt",
            "source": prompt_result.source,
            "warmed": prompt_result.source != "disabled",
            **_describe_clone_prompt(prompt_result.prompt, app.state.device),
        }

    def _run_cache_warm_background(
        *,
        request_id: str,
        runner: Any,
        ref_audio_bytes: bytes,
        ref_text_value: str | None,
        preprocess_prompt: bool,
        started_at: datetime,
    ) -> None:
        started_perf = time.perf_counter()
        try:
            result = _warm_clone_prompt_cache(
                runner=runner,
                ref_audio_bytes=ref_audio_bytes,
                ref_text_value=ref_text_value,
                preprocess_prompt=preprocess_prompt,
            )
            logger.info(
                "cache_warm request_id=%s status=success wait=false started_at=%s "
                "finished_at=%s duration_ms=%.2f source=%s "
                "prompt_audio_tokens=%s stored_device=%s has_ref_text=%s",
                request_id,
                _iso_utc(started_at),
                _iso_utc(_utc_now()),
                (time.perf_counter() - started_perf) * 1000.0,
                result["source"],
                result["prompt_audio_tokens"]
                if result["prompt_audio_tokens"] is not None
                else "-",
                result["stored_device"],
                bool(result["ref_text"]),
            )
        except Exception:
            logger.exception(
                "cache_warm request_id=%s status=background_error",
                request_id,
            )
        finally:
            app.state.worker_runtime.end_request()

    @app.post("/cache/warm")
    def warm_cache(
        wait: bool = True,
        mode: GenerateMode = Form(GenerateMode.clone),
        language: str | None = Form(None),
        ref_text: str | None = Form(None),
        num_step: str | None = Form(None),
        denoise: bool = Form(True),
        preprocess_prompt: bool = Form(True),
        postprocess_mode: str | None = Form(None),
        ref_audio: UploadFile | None = File(None),
    ) -> Any:
        runner = app.state.runner
        if runner is None:
            raise HTTPException(503, detail="Model is not ready yet.")

        request_id = uuid4().hex[:12]
        started_at = _utc_now()
        started_perf = time.perf_counter()
        language_value = _normalize_language(language)
        num_step_value = _parse_positive_int("num_step", num_step, 32)
        _, postprocess_mode_value = _parse_postprocess_mode(None, postprocess_mode)
        ref_text_value = _normalize_text(ref_text)
        request_counted = app.state.worker_runtime.begin_request()
        if not request_counted:
            return _worker_unavailable_response(app.state.worker_runtime)
        release_request = True

        try:
            if mode != GenerateMode.clone:
                return {
                    "status": "ok",
                    "request_id": request_id,
                    "mode": mode.value,
                    "wait": wait,
                    "warmed": False,
                    "cache": "none",
                    "reason": "mode_has_no_reference_cache",
                    "language": language_value,
                    "num_step": num_step_value,
                    "denoise": denoise,
                    "postprocess_mode": postprocess_mode_value,
                }

            if ref_audio is None:
                raise HTTPException(400, detail="ref_audio is required for cache warm.")

            ref_audio_bytes = ref_audio.file.read()
            if not ref_audio_bytes:
                raise HTTPException(400, detail="ref_audio is empty.")

            if (
                ref_text_value is None
                and hasattr(runner.model, "load_asr_model")
                and getattr(runner.model, "_asr_pipe", None) is None
                and not app.state.allow_lazy_asr
            ):
                raise HTTPException(
                    400,
                    detail=(
                        "ref_text is required for clone requests when lazy ASR is "
                        "disabled. Start with --allow-lazy-asr to permit on-demand "
                        "Whisper loading."
                    ),
                )

            if not wait:
                thread = threading.Thread(
                    target=_run_cache_warm_background,
                    name=f"omnivoice-cache-warm-{request_id}",
                    daemon=True,
                    kwargs={
                        "request_id": request_id,
                        "runner": runner,
                        "ref_audio_bytes": ref_audio_bytes,
                        "ref_text_value": ref_text_value,
                        "preprocess_prompt": preprocess_prompt,
                        "started_at": started_at,
                    },
                )
                try:
                    thread.start()
                except Exception:
                    raise
                release_request = False
                return {
                    "status": "accepted",
                    "request_id": request_id,
                    "mode": mode.value,
                    "wait": False,
                    "warmed": False,
                    "cache": "clone_prompt",
                    "language": language_value,
                    "num_step": num_step_value,
                    "denoise": denoise,
                    "postprocess_mode": postprocess_mode_value,
                }

            result = _warm_clone_prompt_cache(
                runner=runner,
                ref_audio_bytes=ref_audio_bytes,
                ref_text_value=ref_text_value,
                preprocess_prompt=preprocess_prompt,
            )

            finished_at = _utc_now()
            duration_ms = (time.perf_counter() - started_perf) * 1000.0
            logger.info(
                "cache_warm request_id=%s status=success wait=true started_at=%s "
                "finished_at=%s duration_ms=%.2f source=%s "
                "prompt_audio_tokens=%s stored_device=%s has_ref_text=%s",
                request_id,
                _iso_utc(started_at),
                _iso_utc(finished_at),
                duration_ms,
                result["source"],
                result["prompt_audio_tokens"]
                if result["prompt_audio_tokens"] is not None
                else "-",
                result["stored_device"],
                bool(result["ref_text"]),
            )
            return {
                "status": "ok",
                "request_id": request_id,
                "mode": mode.value,
                "wait": True,
                "language": language_value,
                "num_step": num_step_value,
                "denoise": denoise,
                "postprocess_mode": postprocess_mode_value,
                "duration_ms": round(duration_ms, 2),
                **result,
            }
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception(
                "cache_warm request_id=%s status=runtime_error",
                request_id,
            )
            raise HTTPException(
                status_code=_runtime_error_status(exc),
                detail=str(exc),
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive server guard
            logger.exception(
                "cache_warm request_id=%s status=unexpected_error",
                request_id,
            )
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        finally:
            if release_request:
                app.state.worker_runtime.end_request()
            if ref_audio is not None:
                ref_audio.file.close()

    @app.get("/cache/status")
    def cache_status(cache_key: str | None = None) -> dict[str, Any]:
        prompt_store = app.state.registered_clone_prompts
        clone_prompt_cache = app.state.clone_prompt_cache
        cached = False
        if cache_key and prompt_store is not None:
            cached = prompt_store.get(cache_key) is not None
        return {
            "status": "ready" if cached else "unknown",
            "cached": cached,
            "cache_key": cache_key,
            "prompt_id": cache_key if cached else None,
            "registered_clone_prompt_count": len(prompt_store or ()),
            "clone_prompt_cache_entries": len(clone_prompt_cache or ()),
        }

    @app.post(
        "/generate",
        response_class=Response,
        responses={200: {"content": {"audio/wav": {}}}},
    )
    def generate(
        mode: GenerateMode = Form(...),
        text: str = Form(...),
        language: str | None = Form(None),
        instruct: str | None = Form(None),
        prompt_id: str | None = Form(None),
        ref_text: str | None = Form(None),
        num_step: str | None = Form(None),
        guidance_scale: str | None = Form(None),
        t_shift: str | None = Form(None),
        layer_penalty_factor: str | None = Form(None),
        position_temperature: str | None = Form(None),
        class_temperature: str | None = Form(None),
        speed: str | None = Form(None),
        duration: str | None = Form(None),
        denoise: bool = Form(True),
        preprocess_prompt: bool = Form(True),
        insert_pauses: bool = Form(False),
        postprocess_output: str | None = Form("true"),
        postprocess_mode: str | None = Form(None),
        ref_audio: UploadFile | None = File(None),
    ) -> Response:
        runner = app.state.runner
        if runner is None:
            raise HTTPException(503, detail="Model is not ready yet.")
        batcher = app.state.batcher
        if batcher is None:
            raise HTTPException(503, detail="Batcher is not ready yet.")

        text_value = _normalize_text(text)
        if text_value is None:
            raise HTTPException(400, detail="text is required.")

        instruct_value = _normalize_text(instruct)
        prompt_id_value = _normalize_text(prompt_id)
        ref_text_value = _normalize_text(ref_text)
        language_value = _normalize_language(language)

        if mode == GenerateMode.auto:
            if instruct_value is not None:
                raise HTTPException(
                    400, detail="instruct is not allowed for mode=auto."
                )
            if (
                prompt_id_value is not None
                or ref_audio is not None
                or ref_text_value is not None
            ):
                raise HTTPException(
                    400,
                    detail=(
                        "prompt_id/ref_audio/ref_text are not allowed for mode=auto."
                    ),
                )
        elif mode == GenerateMode.design:
            if (
                prompt_id_value is not None
                or ref_audio is not None
                or ref_text_value is not None
            ):
                raise HTTPException(
                    400,
                    detail=(
                        "prompt_id/ref_audio/ref_text are not allowed for mode=design."
                    ),
                )
        else:
            if prompt_id_value is not None and ref_audio is not None:
                raise HTTPException(
                    400,
                    detail=(
                        "Provide either prompt_id or ref_audio for mode=clone, "
                        "not both."
                    ),
                )
            if prompt_id_value is None and ref_audio is None:
                raise HTTPException(
                    400,
                    detail="prompt_id or ref_audio is required for mode=clone.",
                )
            if prompt_id_value is not None and ref_text_value is not None:
                raise HTTPException(
                    400,
                    detail="ref_text cannot be combined with prompt_id.",
                )

        num_step_value = _parse_positive_int("num_step", num_step, 32)
        guidance_scale_value = _parse_nonnegative_float(
            "guidance_scale", guidance_scale, 2.0
        )
        t_shift_value = _parse_nonnegative_float("t_shift", t_shift, 0.1)
        layer_penalty_factor_value = _parse_nonnegative_float(
            "layer_penalty_factor", layer_penalty_factor, 5.0
        )
        position_temperature_value = _parse_nonnegative_float(
            "position_temperature", position_temperature, 5.0
        )
        class_temperature_value = _parse_nonnegative_float(
            "class_temperature", class_temperature, 0.0
        )
        speed_value = _parse_optional_positive_float("speed", speed)
        duration_value = _parse_optional_duration(duration)
        postprocess_enabled, postprocess_mode_value = _parse_postprocess_mode(
            postprocess_output,
            postprocess_mode,
        )

        sentences = split_sentences(text_value) if insert_pauses else [text_value]
        if len(sentences) > 1 and duration_value is not None:
            raise HTTPException(
                400,
                detail="duration cannot be combined with insert_pauses for multiple "
                "sentences; use speed instead.",
            )

        request_id = uuid4().hex[:12]
        started_at = _utc_now()
        request_started_perf = time.perf_counter()
        saved_path: Path | None = None
        prompt_source = "none"
        prompt_prepare_ms = 0.0
        batch_estimate_ms = 0.0
        response_encode_ms = 0.0
        request_counted = app.state.worker_runtime.begin_request()
        if not request_counted:
            return _worker_unavailable_response(app.state.worker_runtime)

        try:
            model = runner.model
            gen_config = OmniVoiceGenerationConfig(
                num_step=num_step_value,
                guidance_scale=guidance_scale_value,
                t_shift=t_shift_value,
                layer_penalty_factor=layer_penalty_factor_value,
                position_temperature=position_temperature_value,
                class_temperature=class_temperature_value,
                denoise=denoise,
                preprocess_prompt=preprocess_prompt,
                postprocess_output=postprocess_enabled,
            )
            setattr(gen_config, "postprocess_mode", postprocess_mode_value)

            voice_clone_prompt = None
            if mode == GenerateMode.clone:
                prompt_started_perf = time.perf_counter()
                if prompt_id_value is not None:
                    prompt_store = app.state.registered_clone_prompts
                    if prompt_store is None or prompt_store.max_size <= 0:
                        raise HTTPException(
                            503,
                            detail="Registered clone prompt store is disabled.",
                        )
                    voice_clone_prompt = prompt_store.get(prompt_id_value)
                    if voice_clone_prompt is None:
                        raise HTTPException(404, detail="Unknown prompt_id.")
                    prompt_source = "prompt_id"
                else:
                    assert ref_audio is not None  # for type checkers
                    ref_audio_bytes = ref_audio.file.read()
                    if not ref_audio_bytes:
                        raise HTTPException(400, detail="ref_audio is empty.")

                    prompt_cache = app.state.clone_prompt_cache

                    def _create_prompt() -> Any:
                        return _on_gpu_thread(
                            "clone_prompt_upload",
                            lambda: prompt_to_cpu(
                                _create_voice_clone_prompt(
                                    model,
                                    ref_audio_bytes=ref_audio_bytes,
                                    ref_text=ref_text_value,
                                    preprocess_prompt=preprocess_prompt,
                                    asr_load_lock=app.state.asr_load_lock,
                                    allow_lazy_asr=app.state.allow_lazy_asr,
                                )
                            ),
                        )

                    if prompt_cache is None:
                        voice_clone_prompt = _create_prompt()
                        prompt_source = "cache_disabled"
                    else:
                        prompt_result = prompt_cache.get_or_create(
                            ref_audio_bytes=ref_audio_bytes,
                            ref_text=ref_text_value,
                            preprocess_prompt=preprocess_prompt,
                            factory=_create_prompt,
                        )
                        voice_clone_prompt = prompt_result.prompt
                        prompt_source = f"upload_{prompt_result.source}"
                prompt_prepare_ms = (time.perf_counter() - prompt_started_perf) * 1000.0

            batched_instruct = None
            if mode in {GenerateMode.design, GenerateMode.clone} and instruct_value:
                batched_instruct = instruct_value

            results = []
            for sentence in sentences:
                batch_estimate_started_perf = time.perf_counter()
                target_tokens = _estimate_target_tokens(
                    model,
                    text=sentence,
                    voice_clone_prompt=voice_clone_prompt,
                    speed=speed_value,
                    duration=duration_value,
                )
                conditioning_tokens = _estimate_conditioning_tokens(
                    model,
                    text=sentence,
                    language=language_value,
                    instruct=instruct_value,
                    voice_clone_prompt=voice_clone_prompt,
                    denoise=gen_config.denoise,
                    token_cache=app.state.token_estimate_cache,
                )
                batch_estimate_ms += (
                    time.perf_counter() - batch_estimate_started_perf
                ) * 1000.0
                batch_key = _build_batch_key(
                    model,
                    voice_clone_prompt=voice_clone_prompt,
                    target_tokens=target_tokens,
                    generation_config=gen_config,
                )
                pending = PendingGeneration(
                    request_id=request_id,
                    mode=mode.value,
                    text=sentence,
                    language=language_value,
                    instruct=batched_instruct,
                    voice_clone_prompt=voice_clone_prompt,
                    speed=speed_value,
                    duration=duration_value,
                    target_tokens=target_tokens,
                    conditioning_tokens=conditioning_tokens,
                    max_sequence_length=target_tokens + conditioning_tokens,
                    batch_key=batch_key,
                    generation_config=gen_config,
                )
                if not results:
                    pre_batch_ms = (time.perf_counter() - request_started_perf) * 1000.0
                results.append(batcher.submit(pending))

            batch_result = results[0]
            if len(results) > 1:
                # Timings and token totals span all sentence batches. GPU samples
                # cannot be combined accurately, so omit those for split requests.
                batch_result = replace(
                    batch_result,
                    queue_wait_ms=sum(r.queue_wait_ms for r in results),
                    batch_exec_ms=sum(r.batch_exec_ms for r in results),
                    batch_requests=sum(r.batch_requests for r in results),
                    batch_target_tokens=sum(r.batch_target_tokens for r in results),
                    batch_conditioning_tokens=sum(
                        r.batch_conditioning_tokens for r in results
                    ),
                    batch_max_sequence_length=max(
                        r.batch_max_sequence_length for r in results
                    ),
                    peak_vram_gb=max(r.peak_vram_gb for r in results),
                    gpu_metrics=None,
                    generation_metrics={
                        key: sum(r.generation_metrics.get(key, 0.0) for r in results)
                        for key in {k for r in results for k in r.generation_metrics}
                    },
                )

            response_encode_started_perf = time.perf_counter()
            audio = _to_numpy(batch_result.audio)
            sample_rate = int(
                getattr(model, "sampling_rate", app.state.sample_rate)
                or app.state.sample_rate
            )
            if len(results) > 1:
                gap = np.zeros(round(sample_rate * PAUSE_SECONDS), dtype=audio.dtype)
                parts = [audio]
                for result in results[1:]:
                    parts.extend((gap, _to_numpy(result.audio)))
                audio = np.concatenate(parts)
            wav_bytes = _to_wav_bytes(audio, sample_rate)
            if app.state.save_dir is not None:
                saved_path = _save_output(
                    wav_bytes=wav_bytes,
                    save_dir=app.state.save_dir,
                    mode=mode,
                    text=text_value,
                    request_id=request_id,
                )
            response_encode_ms = (
                time.perf_counter() - response_encode_started_perf
            ) * 1000.0

            finished_at = _utc_now()
            latency_ms = (finished_at - started_at).total_seconds() * 1000.0
            audio_duration_s = float(audio.shape[0]) / float(sample_rate)
            rtf = (
                latency_ms / 1000.0 / audio_duration_s if audio_duration_s > 0 else None
            )
            peak_vram_gb = batch_result.peak_vram_gb
            gpu_metrics = batch_result.gpu_metrics
            generation_metrics = batch_result.generation_metrics

            headers = {
                "Content-Disposition": 'inline; filename="omnivoice.wav"',
                "X-OmniVoice-Request-Id": request_id,
                "X-OmniVoice-Started-At": _iso_utc(started_at),
                "X-OmniVoice-Finished-At": _iso_utc(finished_at),
                "X-OmniVoice-Latency-Ms": f"{latency_ms:.2f}",
                "X-OmniVoice-Audio-Duration-S": f"{audio_duration_s:.3f}",
                "X-OmniVoice-Runner": app.state.runner_name,
                "X-OmniVoice-Device": app.state.device,
                "X-OmniVoice-Peak-Vram-Gb": f"{peak_vram_gb:.3f}",
                "X-OmniVoice-Pre-Batch-Ms": f"{pre_batch_ms:.2f}",
                "X-OmniVoice-Prompt-Prepare-Ms": f"{prompt_prepare_ms:.2f}",
                "X-OmniVoice-Batch-Estimate-Ms": f"{batch_estimate_ms:.2f}",
                "X-OmniVoice-Queue-Wait-Ms": f"{batch_result.queue_wait_ms:.2f}",
                "X-OmniVoice-Batch-Exec-Ms": f"{batch_result.batch_exec_ms:.2f}",
                "X-OmniVoice-Response-Encode-Ms": f"{response_encode_ms:.2f}",
                "X-OmniVoice-Batch-Requests": str(batch_result.batch_requests),
                "X-OmniVoice-Batch-Target-Tokens": str(
                    batch_result.batch_target_tokens
                ),
                "X-OmniVoice-Batch-Conditioning-Tokens": str(
                    batch_result.batch_conditioning_tokens
                ),
                "X-OmniVoice-Batch-Max-Sequence-Length": str(
                    batch_result.batch_max_sequence_length
                ),
                "X-OmniVoice-Batch-Lane": batch_key.lane,
                "X-OmniVoice-Insert-Pauses": str(insert_pauses).lower(),
                "X-OmniVoice-Sentence-Count": str(len(sentences)),
                "X-OmniVoice-Postprocess-Mode": postprocess_mode_value,
                "X-OmniVoice-Prompt-Source": prompt_source,
            }
            if generation_metrics:
                if generation_metrics.get("generate_total_ms") is not None:
                    headers["X-OmniVoice-Generate-Total-Ms"] = (
                        f"{generation_metrics['generate_total_ms']:.2f}"
                    )
                if generation_metrics.get("prepare_inference_inputs_ms") is not None:
                    headers["X-OmniVoice-Prepare-Inference-Inputs-Ms"] = (
                        f"{generation_metrics['prepare_inference_inputs_ms']:.2f}"
                    )
                if generation_metrics.get("iterative_generate_ms") is not None:
                    headers["X-OmniVoice-Iterative-Generate-Ms"] = (
                        f"{generation_metrics['iterative_generate_ms']:.2f}"
                    )
                if generation_metrics.get("chunked_generate_ms") is not None:
                    headers["X-OmniVoice-Chunked-Generate-Ms"] = (
                        f"{generation_metrics['chunked_generate_ms']:.2f}"
                    )
                if generation_metrics.get("decode_postprocess_ms") is not None:
                    headers["X-OmniVoice-Decode-Postprocess-Ms"] = (
                        f"{generation_metrics['decode_postprocess_ms']:.2f}"
                    )
                if generation_metrics.get("audio_decode_ms") is not None:
                    headers["X-OmniVoice-Audio-Decode-Ms"] = (
                        f"{generation_metrics['audio_decode_ms']:.2f}"
                    )
                if generation_metrics.get("post_process_audio_ms") is not None:
                    headers["X-OmniVoice-Post-Process-Audio-Ms"] = (
                        f"{generation_metrics['post_process_audio_ms']:.2f}"
                    )
            if gpu_metrics is not None:
                headers["X-OmniVoice-Gpu-Sample-Count"] = str(gpu_metrics.sample_count)
                if gpu_metrics.gpu_util_avg_pct is not None:
                    headers["X-OmniVoice-Gpu-Util-Avg-Pct"] = (
                        f"{gpu_metrics.gpu_util_avg_pct:.2f}"
                    )
                if gpu_metrics.gpu_util_peak_pct is not None:
                    headers["X-OmniVoice-Gpu-Util-Peak-Pct"] = (
                        f"{gpu_metrics.gpu_util_peak_pct:.2f}"
                    )
                if gpu_metrics.device_vram_used_gb_avg is not None:
                    headers["X-OmniVoice-Device-Vram-Used-Gb-Avg"] = (
                        f"{gpu_metrics.device_vram_used_gb_avg:.3f}"
                    )
                if gpu_metrics.device_vram_used_gb_peak is not None:
                    headers["X-OmniVoice-Device-Vram-Used-Gb-Peak"] = (
                        f"{gpu_metrics.device_vram_used_gb_peak:.3f}"
                    )
                if gpu_metrics.device_vram_util_avg_pct is not None:
                    headers["X-OmniVoice-Device-Vram-Util-Avg-Pct"] = (
                        f"{gpu_metrics.device_vram_util_avg_pct:.2f}"
                    )
                if gpu_metrics.device_vram_util_peak_pct is not None:
                    headers["X-OmniVoice-Device-Vram-Util-Peak-Pct"] = (
                        f"{gpu_metrics.device_vram_util_peak_pct:.2f}"
                    )
            if prompt_id_value is not None:
                headers["X-OmniVoice-Prompt-Id"] = prompt_id_value
            if rtf is not None:
                headers["X-OmniVoice-RTF"] = f"{rtf:.4f}"
            if saved_path is not None:
                headers["X-OmniVoice-Saved-Path"] = str(saved_path)

            logger.info(
                "request_id=%s status=success mode=%s lane=%s runner=%s started_at=%s "
                "finished_at=%s latency_ms=%.2f pre_batch_ms=%.2f "
                "prompt_prepare_ms=%.2f batch_estimate_ms=%.2f "
                "queue_wait_ms=%.2f batch_exec_ms=%.2f response_encode_ms=%.2f "
                "generate_total_ms=%s prepare_inputs_ms=%s iterative_ms=%s "
                "chunked_ms=%s decode_postprocess_ms=%s audio_decode_ms=%s "
                "post_process_audio_ms=%s "
                "batch_requests=%d audio_s=%.3f rtf=%s peak_vram_gb=%.3f "
                "gpu_util_avg_pct=%s gpu_util_peak_pct=%s "
                "device_vram_used_gb_peak=%s device_vram_util_peak_pct=%s "
                "text_chars=%d "
                "prompt_source=%s postprocess_mode=%s language=%s device=%s "
                "saved_path=%s",
                request_id,
                mode.value,
                batch_key.lane,
                app.state.runner_name,
                _iso_utc(started_at),
                _iso_utc(finished_at),
                latency_ms,
                pre_batch_ms,
                prompt_prepare_ms,
                batch_estimate_ms,
                batch_result.queue_wait_ms,
                batch_result.batch_exec_ms,
                response_encode_ms,
                _format_optional_metric(generation_metrics.get("generate_total_ms")),
                _format_optional_metric(
                    generation_metrics.get("prepare_inference_inputs_ms")
                ),
                _format_optional_metric(
                    generation_metrics.get("iterative_generate_ms")
                ),
                _format_optional_metric(generation_metrics.get("chunked_generate_ms")),
                _format_optional_metric(
                    generation_metrics.get("decode_postprocess_ms")
                ),
                _format_optional_metric(generation_metrics.get("audio_decode_ms")),
                _format_optional_metric(
                    generation_metrics.get("post_process_audio_ms")
                ),
                batch_result.batch_requests,
                audio_duration_s,
                f"{rtf:.4f}" if rtf is not None else "-",
                peak_vram_gb,
                _format_optional_metric(getattr(gpu_metrics, "gpu_util_avg_pct", None)),
                _format_optional_metric(
                    getattr(gpu_metrics, "gpu_util_peak_pct", None)
                ),
                _format_optional_metric(
                    getattr(gpu_metrics, "device_vram_used_gb_peak", None)
                ),
                _format_optional_metric(
                    getattr(gpu_metrics, "device_vram_util_peak_pct", None)
                ),
                len(text_value),
                prompt_source,
                postprocess_mode_value,
                language_value or "auto",
                app.state.device,
                str(saved_path) if saved_path is not None else "-",
            )

            return Response(
                content=wav_bytes,
                media_type="audio/wav",
                headers=headers,
            )
        # A failed batch was cleaned up on the batcher thread; no CUDA calls here.
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            logger.exception("request_id=%s status=runtime_error", request_id)
            raise HTTPException(
                status_code=_runtime_error_status(exc),
                detail=str(exc),
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive server guard
            logger.exception("request_id=%s status=unexpected_error", request_id)
            raise HTTPException(
                status_code=500,
                detail=f"{type(exc).__name__}: {exc}",
            ) from exc
        finally:
            app.state.worker_runtime.end_request()
            if ref_audio is not None:
                ref_audio.file.close()

    return app


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint."""
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        batch_bucket_sizes = _parse_batch_bucket_sizes(args.batch_bucket_sizes)
        prewarm_clone_batch_sizes = _parse_positive_int_list(
            args.prewarm_clone_batch_sizes,
            option_name="--prewarm-clone-batch-sizes",
            default=DEFAULT_PREWARM_CLONE_BATCH_SIZES,
        )
        prewarm_clone_sequence_lengths = _parse_positive_int_list(
            args.prewarm_clone_sequence_lengths,
            option_name="--prewarm-clone-sequence-lengths",
            default=DEFAULT_PREWARM_CLONE_SEQUENCE_LENGTHS,
        )
    except ValueError as exc:
        parser.error(str(exc))

    logging.basicConfig(
        level=getattr(
            logging,
            os.environ.get("OMNIVOICE_LOG_LEVEL", "INFO").upper(),
            logging.INFO,
        ),
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    app = create_app(
        model_checkpoint=args.model,
        runner_name=args.runner,
        device=args.device,
        dtype=args.dtype,
        load_asr=not args.no_asr,
        allow_lazy_asr=args.allow_lazy_asr,
        enable_sage_attention=args.sage_attention,
        full_triton_patch=args.full_triton_patch,
        decode_postprocess_workers=args.decode_postprocess_workers,
        max_cuda_graphs=args.max_cuda_graphs,
        save_dir=args.save_dir,
        batch_collect_ms=args.batch_collect_ms,
        max_batch_requests=args.max_batch_requests,
        max_pending_requests=args.max_pending_requests,
        max_pending_gpu_tasks=args.max_pending_gpu_tasks,
        max_batch_target_tokens=args.max_batch_target_tokens,
        max_batch_conditioning_tokens=args.max_batch_conditioning_tokens,
        max_batch_padding_ratio=args.max_batch_padding_ratio,
        clone_prompt_cache_size=args.clone_prompt_cache_size,
        registered_clone_prompt_store_size=args.registered_clone_prompt_store_size,
        token_estimate_cache_size=args.token_estimate_cache_size,
        batch_bucket_sizes=batch_bucket_sizes,
        prewarm_clone_batch_sizes=prewarm_clone_batch_sizes,
        prewarm_clone_sequence_lengths=prewarm_clone_sequence_lengths,
        server_port=args.port,
    )

    uvicorn.run(
        app,
        host=args.ip,
        port=args.port,
        root_path=args.root_path or "",
    )


if __name__ == "__main__":
    main()
