"""Base OmniVoice runner using HuggingFace transformers."""

import logging
import threading
import time
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "k2-fsa/OmniVoice"
DEFAULT_SAMPLE_RATE = 24000

# dtype string to torch dtype mapping
_DTYPE_MAP: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def cuda_preflight_error() -> str | None:
    """Return a diagnostic message if CUDA cannot be used by PyTorch."""
    try:
        if torch.cuda.is_available():
            return None
        device_count = torch.cuda.device_count()
    except Exception as exc:
        return f"PyTorch CUDA initialization failed: {exc}"

    torch_version = getattr(torch, "__version__", "unknown")
    cuda_version = getattr(torch.version, "cuda", None) or "not built with CUDA"
    if device_count > 0:
        return (
            "PyTorch can see an NVIDIA device but CUDA is not usable. "
            f"Installed torch={torch_version} was built for CUDA {cuda_version}. "
            "Install a PyTorch wheel that matches the host driver, for example "
            "the cu128 wheel on Vast.ai CUDA 12.8 images."
        )
    return "PyTorch CUDA is unavailable and no CUDA devices were detected."


def require_cuda_available() -> None:
    """Raise a clear error when a CUDA runner is requested without CUDA."""
    error = cuda_preflight_error()
    if error is not None:
        raise RuntimeError(error)


def _reset_peak_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _max_memory_allocated_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024**3


def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Convert dtype string to torch.dtype."""
    if isinstance(dtype, torch.dtype):
        return dtype
    key = dtype.lower().replace("bfloat16", "bf16").replace("float16", "fp16")
    key = key.replace("float32", "fp32")
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{dtype}'. Use: bf16, fp16, fp32")
    return _DTYPE_MAP[key]


def _to_numpy(audio: Any) -> np.ndarray:
    """Convert audio output to numpy array."""
    if isinstance(audio, list):
        audio = audio[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.squeeze().cpu().float().numpy()
    return audio


def _ensure_audio_batch(audio: np.ndarray | Any) -> np.ndarray:
    """Normalize decoded audio to a float32 array with shape (C, T)."""
    waveform = np.asarray(audio, dtype=np.float32)
    if waveform.ndim == 0:
        return waveform.reshape(1, 1)
    if waveform.ndim == 1:
        return waveform[np.newaxis, :]
    if waveform.ndim > 2:
        return waveform.reshape(waveform.shape[0], -1)
    return waveform


def _decode_audio_token_batch(
    results: list[Any],
    *,
    audio_tokenizer: Any,
    sampling_rate: int,
    cross_fade_chunks: Callable[..., Any],
    max_batch_size: int = 1,
) -> list[np.ndarray]:
    """Decode equal-shaped token sequences together without padding any audio.

    The default preserves the original one-at-a-time decoder path. Opt-in batching
    changes only the codec batch dimension: generated token IDs, sequence lengths,
    chunk crossfades, output order and postprocessing remain unchanged. Different
    batch shapes can select different GPU convolution algorithms, so callers must
    qualify numerical and speech quality rather than assume bitwise equivalence.
    """
    tokenizer_device = audio_tokenizer.device

    def decode_one(tokens: Any) -> np.ndarray:
        if isinstance(tokens, list):
            chunk_audios = [
                audio_tokenizer.decode(t.to(tokenizer_device).unsqueeze(0))
                .audio_values[0]
                .detach()
                .cpu()
                .numpy()
                for t in tokens
            ]
            return _ensure_audio_batch(cross_fade_chunks(chunk_audios, sampling_rate))
        return _ensure_audio_batch(
            audio_tokenizer.decode(tokens.to(tokenizer_device).unsqueeze(0))
            .audio_values[0]
            .detach()
            .cpu()
            .numpy()
        )

    if max_batch_size <= 1:
        return [decode_one(tokens) for tokens in results]

    decoded: list[np.ndarray | None] = [None] * len(results)
    groups: dict[tuple[Any, ...], list[tuple[int, torch.Tensor]]] = defaultdict(list)
    for index, tokens in enumerate(results):
        if isinstance(tokens, list):
            # Long-form chunk boundaries and crossfade order stay exactly as before.
            decoded[index] = decode_one(tokens)
            continue
        key = (tuple(tokens.shape), tokens.dtype, tokens.device)
        groups[key].append((index, tokens))

    for entries in groups.values():
        for start in range(0, len(entries), max_batch_size):
            batch = entries[start : start + max_batch_size]
            if len(batch) == 1:
                index, tokens = batch[0]
                decoded[index] = decode_one(tokens)
                continue
            # Exact lengths only: zero padding can alter codec boundary samples.
            stacked = torch.stack([tokens for _, tokens in batch]).to(tokenizer_device)
            audio_values = audio_tokenizer.decode(stacked).audio_values
            if audio_values.shape[0] != len(batch):
                raise RuntimeError(
                    "Audio tokenizer returned an invalid batch dimension"
                )
            audio_arrays = audio_values.detach().cpu().numpy()
            for offset, (index, _) in enumerate(batch):
                decoded[index] = _ensure_audio_batch(audio_arrays[offset])

    if any(audio is None for audio in decoded):
        raise RuntimeError("Audio tokenizer did not return every requested waveform")
    return [audio for audio in decoded if audio is not None]


def _resolve_postprocess_mode(value: Any) -> str:
    """Normalize postprocess mode names across bool and string inputs."""
    if isinstance(value, bool):
        return "full" if value else "off"
    text = str(value).strip().lower()
    if text in {"", "1", "true", "yes", "on", "full"}:
        return "full"
    if text in {"light", "fast"}:
        return "light"
    if text in {"0", "false", "no", "off", "none"}:
        return "off"
    return "full"


def _resolve_postprocess_mode_from_config(gen_config: Any) -> str:
    """Read postprocess mode from a generation config with backward compatibility."""
    if hasattr(gen_config, "postprocess_mode"):
        return _resolve_postprocess_mode(getattr(gen_config, "postprocess_mode"))
    return _resolve_postprocess_mode(getattr(gen_config, "postprocess_output", True))


def _trim_silence_edges_numpy(
    audio: np.ndarray,
    sample_rate: int,
    *,
    lead_sil_ms: int = 100,
    trail_sil_ms: int = 100,
    silence_threshold_db: float = -50.0,
) -> np.ndarray:
    """Trim leading and trailing silence using a lightweight numpy scan."""
    waveform = _ensure_audio_batch(audio)
    if waveform.shape[-1] == 0:
        return waveform

    threshold = float(10.0 ** (silence_threshold_db / 20.0))
    amplitude = np.max(np.abs(waveform), axis=0)
    non_silent = np.flatnonzero(amplitude > threshold)
    if non_silent.size == 0:
        return waveform[:, :0]

    lead_keep = int(sample_rate * lead_sil_ms / 1000.0)
    trail_keep = int(sample_rate * trail_sil_ms / 1000.0)
    start = max(0, int(non_silent[0]) - lead_keep)
    end = min(waveform.shape[-1], int(non_silent[-1]) + 1 + trail_keep)
    return waveform[:, start:end]


class BaseRunner:
    """Load and run OmniVoice from HuggingFace transformers.

    Args:
        device: Target device (default: "cuda").
        model_id: HuggingFace model ID or local path.
        dtype: Model dtype ("bf16", "fp16", "fp32").
    """

    def __init__(
        self,
        device: str = "cuda",
        model_id: str = DEFAULT_MODEL_ID,
        dtype: str | torch.dtype = "fp16",
        decode_postprocess_workers: int = 0,
        audio_decode_batch_size: int = 1,
    ) -> None:
        self.device = device
        self.model_id = model_id
        self.dtype = _resolve_dtype(dtype)
        self.decode_postprocess_workers = max(0, int(decode_postprocess_workers))
        self.audio_decode_batch_size = max(1, int(audio_decode_batch_size))
        self._model: Any = None
        self._decode_postprocess_executor: ThreadPoolExecutor | None = None
        self._generation_metrics_lock = threading.Lock()
        self._last_generation_metrics: dict[str, float] = {}

    def load_model(self) -> None:
        """Download and load model onto device."""
        from omnivoice import OmniVoice

        logger.info("Loading %s ...", self.model_id)
        if self.device.startswith("cuda"):
            require_cuda_available()
        _reset_peak_memory_stats()

        self._model = OmniVoice.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=self.dtype,
        )

        if self.decode_postprocess_workers > 1:
            self._decode_postprocess_executor = ThreadPoolExecutor(
                max_workers=self.decode_postprocess_workers,
                thread_name_prefix="omnivoice-postprocess",
            )
        self._install_generation_metrics(self._model)
        vram_gb = _max_memory_allocated_gb()
        logger.info("Model loaded. VRAM: %.2f GB", vram_gb)

    @property
    def model(self) -> Any:
        """Internal model (for patching)."""
        return self._model

    def _check_loaded(self) -> None:
        """Raise if model not loaded."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

    def generate(
        self,
        text: str,
        language: str | None = None,
        *,
        num_step: int = 32,
        guidance_scale: float = 2.0,
        class_temperature: float = 0.0,
    ) -> dict:
        """Generate speech from text.

        Args:
            text: Input text to synthesize.
            language: Language hint (auto-detected if None).
            num_step: Number of iterative decoding steps.
            guidance_scale: Classifier-free guidance strength.
            class_temperature: Token sampling temperature (0=greedy).

        Returns:
            Dict with audio, sample_rate, time_s, peak_vram_gb.
        """
        self._check_loaded()
        _reset_peak_memory_stats()

        from omnivoice import OmniVoiceGenerationConfig

        gen_config = OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=guidance_scale,
            class_temperature=class_temperature,
        )

        kwargs: dict[str, Any] = {
            "text": text,
            "generation_config": gen_config,
        }
        if language is not None:
            kwargs["language"] = language

        start = time.perf_counter()
        audio_list = self._model.generate(**kwargs)
        elapsed = time.perf_counter() - start

        return {
            "audio": _to_numpy(audio_list),
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "time_s": elapsed,
            "peak_vram_gb": _max_memory_allocated_gb(),
        }

    def generate_voice_clone(
        self,
        text: str,
        ref_audio: str,
        ref_text: str = "",
        language: str | None = None,
        *,
        num_step: int = 32,
        guidance_scale: float = 2.0,
        class_temperature: float = 0.0,
    ) -> dict:
        """Generate speech by cloning a reference voice.

        Args:
            text: Input text to synthesize.
            ref_audio: Path to reference audio file.
            ref_text: Transcription of the reference audio.
            language: Language hint (auto-detected if None).
            num_step: Number of iterative decoding steps.
            guidance_scale: Classifier-free guidance strength.
            class_temperature: Token sampling temperature.

        Returns:
            Dict with audio, sample_rate, time_s, peak_vram_gb.
        """
        self._check_loaded()
        _reset_peak_memory_stats()

        from omnivoice import OmniVoiceGenerationConfig

        gen_config = OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=guidance_scale,
            class_temperature=class_temperature,
        )

        kwargs: dict[str, Any] = {
            "text": text,
            "ref_audio": ref_audio,
            "generation_config": gen_config,
        }
        if ref_text:
            kwargs["ref_text"] = ref_text
        if language is not None:
            kwargs["language"] = language

        start = time.perf_counter()
        audio_list = self._model.generate(**kwargs)
        elapsed = time.perf_counter() - start

        return {
            "audio": _to_numpy(audio_list),
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "time_s": elapsed,
            "peak_vram_gb": _max_memory_allocated_gb(),
        }

    def generate_voice_design(
        self,
        text: str,
        instruct: str,
        language: str | None = None,
        *,
        num_step: int = 32,
        guidance_scale: float = 2.0,
        class_temperature: float = 0.0,
    ) -> dict:
        """Generate speech with a designed voice from instructions.

        Args:
            text: Input text to synthesize.
            instruct: Speaker attribute instructions
                (e.g. "female, young adult, high pitch").
            language: Language hint (auto-detected if None).
            num_step: Number of iterative decoding steps.
            guidance_scale: Classifier-free guidance strength.
            class_temperature: Token sampling temperature.

        Returns:
            Dict with audio, sample_rate, time_s, peak_vram_gb.
        """
        self._check_loaded()
        _reset_peak_memory_stats()

        from omnivoice import OmniVoiceGenerationConfig

        gen_config = OmniVoiceGenerationConfig(
            num_step=num_step,
            guidance_scale=guidance_scale,
            class_temperature=class_temperature,
        )

        kwargs: dict[str, Any] = {
            "text": text,
            "instruct": instruct,
            "generation_config": gen_config,
        }
        if language is not None:
            kwargs["language"] = language

        start = time.perf_counter()
        audio_list = self._model.generate(**kwargs)
        elapsed = time.perf_counter() - start

        return {
            "audio": _to_numpy(audio_list),
            "sample_rate": DEFAULT_SAMPLE_RATE,
            "time_s": elapsed,
            "peak_vram_gb": _max_memory_allocated_gb(),
        }

    def unload_model(self) -> None:
        """Free model from GPU memory."""
        if self._decode_postprocess_executor is not None:
            self._decode_postprocess_executor.shutdown(wait=True)
            self._decode_postprocess_executor = None
        del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model unloaded.")

    def get_generation_metrics(self) -> dict[str, float]:
        """Return the most recent model-internal generation timing breakdown."""
        with self._generation_metrics_lock:
            return dict(self._last_generation_metrics)

    def _install_generation_metrics(self, model: Any) -> None:
        """Attach lightweight runtime stage timers to the OmniVoice model."""
        from omnivoice import OmniVoiceGenerationConfig
        from omnivoice.utils.audio import (
            cross_fade_chunks,
            fade_and_pad_audio,
            remove_silence,
        )

        if getattr(model, "_omnivoice_generation_metrics_installed", False):
            return

        original_prepare = getattr(model, "_prepare_inference_inputs", None)
        original_iterative = getattr(model, "_generate_iterative", None)
        original_chunked = getattr(model, "_generate_chunked", None)
        original_decode = getattr(model, "_decode_and_post_process", None)

        def _current_metrics() -> dict[str, float] | None:
            return getattr(model, "_omnivoice_current_generation_metrics", None)

        def _record_metric(name: str, elapsed_ms: float) -> None:
            metrics = _current_metrics()
            if metrics is None:
                return
            metrics[name] = metrics.get(name, 0.0) + elapsed_ms

        def _snapshot_metrics() -> dict[str, float]:
            metrics = getattr(model, "_omnivoice_last_generation_metrics", None) or {}
            return {key: float(value) for key, value in metrics.items()}

        def _post_process_audio_array(
            generated_audio: np.ndarray,
            ref_rms: float | None,
            gen_config: Any,
        ) -> np.ndarray:
            processed = _ensure_audio_batch(generated_audio)
            postprocess_mode = _resolve_postprocess_mode_from_config(gen_config)

            if postprocess_mode == "full":
                processed = remove_silence(
                    processed,
                    model.sampling_rate,
                    mid_sil=500,
                    lead_sil=100,
                    trail_sil=100,
                )
            elif postprocess_mode == "light":
                processed = _trim_silence_edges_numpy(
                    processed,
                    model.sampling_rate,
                    lead_sil_ms=100,
                    trail_sil_ms=100,
                    silence_threshold_db=-50.0,
                )

            if ref_rms is not None and ref_rms < 0.1:
                processed = processed * ref_rms / 0.1
            elif ref_rms is None:
                peak = float(np.abs(processed).max()) if processed.size else 0.0
                if peak > 1e-6:
                    processed = processed / peak * 0.5

            processed = fade_and_pad_audio(
                processed,
                sample_rate=model.sampling_rate,
            )
            return processed.squeeze(0)

        def enhanced_generate(
            text: str | list[str],
            language: str | list[str] | None = None,
            ref_text: str | list[str] | None = None,
            ref_audio: Any = None,
            voice_clone_prompt: Any = None,
            instruct: str | list[str] | None = None,
            duration: float | list[float | None] | None = None,
            speed: float | list[float | None] | None = None,
            generation_config: Any | None = None,
            **kwargs: Any,
        ) -> list[np.ndarray]:
            if model.audio_tokenizer is None or model.text_tokenizer is None:
                raise RuntimeError(
                    "Model is not loaded with audio/text tokenizers. Make sure you "
                    "loaded the model with OmniVoice.from_pretrained()."
                )

            gen_config = (
                generation_config
                if generation_config is not None
                else OmniVoiceGenerationConfig.from_dict(kwargs)
            )
            postprocess_mode = _resolve_postprocess_mode_from_config(gen_config)
            setattr(gen_config, "postprocess_mode", postprocess_mode)
            gen_config.postprocess_output = postprocess_mode == "full"

            model.eval()

            full_task = model._preprocess_all(
                text=text,
                language=language,
                ref_text=ref_text,
                ref_audio=ref_audio,
                voice_clone_prompt=voice_clone_prompt,
                instruct=instruct,
                preprocess_prompt=gen_config.preprocess_prompt,
                speed=speed,
                duration=duration,
            )

            short_idx, long_idx = full_task.get_indices(
                gen_config,
                model.audio_tokenizer.config.frame_rate,
            )

            results = [None] * full_task.batch_size

            if short_idx:
                short_task = full_task.slice_task(short_idx)
                short_results = model._generate_iterative(short_task, gen_config)
                for idx, res in zip(short_idx, short_results):
                    results[idx] = res

            if long_idx:
                long_task = full_task.slice_task(long_idx)
                long_results = model._generate_chunked(long_task, gen_config)
                for idx, res in zip(long_idx, long_results):
                    results[idx] = res

            decode_started = time.perf_counter()
            for index in range(full_task.batch_size):
                assert results[index] is not None, f"Result {index} was not generated"
            decoded_audios = _decode_audio_token_batch(
                results,
                audio_tokenizer=model.audio_tokenizer,
                sampling_rate=model.sampling_rate,
                cross_fade_chunks=cross_fade_chunks,
                max_batch_size=self.audio_decode_batch_size,
            )
            audio_decode_ms = (time.perf_counter() - decode_started) * 1000.0
            _record_metric("audio_decode_ms", audio_decode_ms)

            postprocess_started = time.perf_counter()
            decode_inputs = list(zip(decoded_audios, full_task.ref_rms, strict=True))
            if self._decode_postprocess_executor is not None and len(decode_inputs) > 1:
                generated_audios = list(
                    self._decode_postprocess_executor.map(
                        lambda item: _post_process_audio_array(
                            item[0],
                            item[1],
                            gen_config,
                        ),
                        decode_inputs,
                    )
                )
            else:
                generated_audios = [
                    _post_process_audio_array(audio, rms, gen_config)
                    for audio, rms in decode_inputs
                ]
            post_process_audio_ms = (
                time.perf_counter() - postprocess_started
            ) * 1000.0
            _record_metric("post_process_audio_ms", post_process_audio_ms)
            _record_metric(
                "decode_postprocess_ms",
                audio_decode_ms + post_process_audio_ms,
            )
            return generated_audios

        def wrapped_generate(*args: Any, **kwargs: Any) -> Any:
            metrics: dict[str, float] = {}
            setattr(model, "_omnivoice_current_generation_metrics", metrics)
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    return enhanced_generate(*args, **kwargs)
            finally:
                metrics["generate_total_ms"] = (time.perf_counter() - started) * 1000.0
                setattr(model, "_omnivoice_last_generation_metrics", dict(metrics))
                setattr(model, "_omnivoice_current_generation_metrics", None)
                with self._generation_metrics_lock:
                    self._last_generation_metrics = dict(metrics)

        if original_prepare is not None:
            def wrapped_prepare(*args: Any, **kwargs: Any) -> Any:
                started = time.perf_counter()
                try:
                    return original_prepare(*args, **kwargs)
                finally:
                    _record_metric(
                        "prepare_inference_inputs_ms",
                        (time.perf_counter() - started) * 1000.0,
                    )

            model._prepare_inference_inputs = wrapped_prepare

        if original_iterative is not None:
            def wrapped_iterative(*args: Any, **kwargs: Any) -> Any:
                started = time.perf_counter()
                try:
                    return original_iterative(*args, **kwargs)
                finally:
                    _record_metric(
                        "iterative_generate_ms",
                        (time.perf_counter() - started) * 1000.0,
                    )

            model._generate_iterative = wrapped_iterative

        if original_chunked is not None:
            def wrapped_chunked(*args: Any, **kwargs: Any) -> Any:
                started = time.perf_counter()
                try:
                    return original_chunked(*args, **kwargs)
                finally:
                    _record_metric(
                        "chunked_generate_ms",
                        (time.perf_counter() - started) * 1000.0,
                    )

            model._generate_chunked = wrapped_chunked

        if original_decode is not None:
            def wrapped_decode(*args: Any, **kwargs: Any) -> Any:
                started = time.perf_counter()
                try:
                    return original_decode(*args, **kwargs)
                finally:
                    _record_metric(
                        "decode_postprocess_ms",
                        (time.perf_counter() - started) * 1000.0,
                    )

            model._decode_and_post_process = wrapped_decode

        model.generate = wrapped_generate
        model.get_runtime_stage_metrics = _snapshot_metrics
        model._omnivoice_generation_metrics_installed = True
