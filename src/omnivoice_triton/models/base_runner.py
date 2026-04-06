"""Base OmniVoice runner using HuggingFace transformers."""

import logging
import time
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
    ) -> None:
        self.device = device
        self.model_id = model_id
        self.dtype = _resolve_dtype(dtype)
        self._model: Any = None

    def load_model(self) -> None:
        """Download and load model onto device."""
        from omnivoice import OmniVoice

        logger.info("Loading %s ...", self.model_id)
        torch.cuda.reset_peak_memory_stats()

        self._model = OmniVoice.from_pretrained(
            self.model_id,
            device_map=self.device,
            dtype=self.dtype,
        )

        vram_gb = torch.cuda.max_memory_allocated() / 1024**3
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
        torch.cuda.reset_peak_memory_stats()

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
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
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
        torch.cuda.reset_peak_memory_stats()

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
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
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
        torch.cuda.reset_peak_memory_stats()

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
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }

    def unload_model(self) -> None:
        """Free model from GPU memory."""
        del self._model
        self._model = None
        torch.cuda.empty_cache()
        logger.info("Model unloaded.")
