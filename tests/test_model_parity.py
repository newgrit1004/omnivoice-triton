"""Tier 2: Model-level parity test — all runners produce valid audio.

OmniVoice is a stochastic NAR model (Gumbel-Softmax sampling), so
each generation produces different audio even for the same text.
Signal-level comparison (cosine/SNR) is NOT meaningful.

Instead, we verify:
  - Each runner generates non-empty, non-silent audio
  - Audio duration is within a reasonable range
  - Audio amplitude is within normal bounds

Run:
    uv run pytest tests/test_model_parity.py -v --no-header
"""

import logging

import numpy as np
import pytest
import torch

logger = logging.getLogger(__name__)

TEXT = "Hello, this is a model parity test for OmniVoice."
SAMPLE_RATE = 24000
MIN_DURATION_S = 0.5
MAX_DURATION_S = 30.0


@pytest.mark.gpu
@pytest.mark.slow
class TestModelParity:
    """Verify all runners produce valid audio output."""

    def _generate_with_runner(self, name: str) -> dict:
        from omnivoice_triton.models import create_runner

        runner = create_runner(name)
        runner.load_model()
        result = runner.generate(text=TEXT)
        runner.unload_model()
        torch.cuda.empty_cache()
        return result

    def _assert_valid_audio(self, result: dict, runner_name: str) -> None:
        audio = result["audio"]
        duration = len(audio) / result["sample_rate"]
        max_abs = float(np.max(np.abs(audio)))

        logger.info(
            "[%s] duration=%.2fs, max_abs=%.4f, time=%.3fs, vram=%.2fGB",
            runner_name,
            duration,
            max_abs,
            result["time_s"],
            result["peak_vram_gb"],
        )

        assert len(audio) > 0, f"[{runner_name}] Empty audio"
        assert max_abs > 0.001, f"[{runner_name}] Silent audio (max_abs={max_abs})"
        assert (
            duration > MIN_DURATION_S
        ), f"[{runner_name}] Too short: {duration:.2f}s < {MIN_DURATION_S}s"
        assert (
            duration < MAX_DURATION_S
        ), f"[{runner_name}] Too long: {duration:.2f}s > {MAX_DURATION_S}s"

    def test_base(self) -> None:
        result = self._generate_with_runner("base")
        self._assert_valid_audio(result, "base")

    def test_triton(self) -> None:
        result = self._generate_with_runner("triton")
        self._assert_valid_audio(result, "triton")

    def test_faster(self) -> None:
        result = self._generate_with_runner("faster")
        self._assert_valid_audio(result, "faster")

    def test_hybrid(self) -> None:
        result = self._generate_with_runner("hybrid")
        self._assert_valid_audio(result, "hybrid")
