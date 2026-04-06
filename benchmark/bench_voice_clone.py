"""Voice cloning benchmarks using LJSpeech reference audio.

Compares all 4 runners on voice cloning latency and RTF.

Usage:
    python -m benchmark.bench_voice_clone
"""

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"

# LJSpeech dataset location (HuggingFace cache)
LJSPEECH_CACHE = Path.home() / ".cache/huggingface/hub/datasets--keithito--lj_speech"

CLONE_TEXT = (
    "This is a voice cloning test using a reference audio sample from LJSpeech."
)


def _find_ljspeech_samples(max_samples: int = 3) -> list[dict[str, str]]:
    """Find WAV files and transcripts from cached LJSpeech dataset."""
    samples = []

    # Search for WAV files in the HF cache
    if not LJSPEECH_CACHE.exists():
        logger.warning("LJSpeech cache not found at %s", LJSPEECH_CACHE)
        return samples

    wav_files = sorted(LJSPEECH_CACHE.rglob("*.wav"))[:max_samples]
    for wav in wav_files:
        samples.append(
            {
                "ref_audio": str(wav),
                "ref_text": "",  # Transcription not strictly required for OmniVoice
            }
        )

    logger.info("Found %d LJSpeech samples", len(samples))
    return samples


def bench_voice_clone(
    runner_name: str,
    samples: list[dict[str, str]],
    warmup: int = 2,
    repeat: int = 5,
) -> list[dict[str, Any]]:
    """Benchmark voice cloning for a single runner."""
    from omnivoice_triton.models import create_runner

    results = []
    runner = create_runner(runner_name)

    try:
        runner.load_model()

        for sample in samples:
            # Warmup
            for _ in range(warmup):
                runner.generate_voice_clone(
                    text=CLONE_TEXT,
                    ref_audio=sample["ref_audio"],
                    ref_text=sample["ref_text"],
                )

            # Measured
            timings = []
            for _ in range(repeat):
                torch.cuda.synchronize()
                start = time.perf_counter()
                result = runner.generate_voice_clone(
                    text=CLONE_TEXT,
                    ref_audio=sample["ref_audio"],
                    ref_text=sample["ref_text"],
                )
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)

            audio_dur = len(result["audio"]) / result["sample_rate"]
            mean_t = np.mean(timings)

            results.append(
                {
                    "runner": runner_name,
                    "ref_audio": Path(sample["ref_audio"]).name,
                    "mean_s": round(float(mean_t), 3),
                    "std_s": round(float(np.std(timings)), 3),
                    "rtf": round(audio_dur / mean_t, 2),
                    "audio_duration_s": round(audio_dur, 2),
                    "peak_vram_gb": round(
                        torch.cuda.max_memory_allocated() / 1024**3, 2
                    ),
                }
            )
    finally:
        runner.unload_model()

    return results


def run_voice_clone_benchmarks(
    warmup: int = 2,
    repeat: int = 5,
) -> list[dict[str, Any]]:
    """Run voice cloning benchmarks for all runners."""
    from omnivoice_triton.models import ALL_RUNNER_NAMES

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    samples = _find_ljspeech_samples(max_samples=2)

    if not samples:
        logger.error("No LJSpeech samples found. Skipping voice clone benchmark.")
        return []

    all_results: list[dict[str, Any]] = []

    for name in ALL_RUNNER_NAMES:
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Voice clone benchmark: %s", name)
        try:
            results = bench_voice_clone(name, samples, warmup, repeat)
            all_results.extend(results)
            for r in results:
                logger.info(
                    "  [%s] %s: %.3fs (RTF %.2f)",
                    name,
                    r["ref_audio"],
                    r["mean_s"],
                    r["rtf"],
                )
        except Exception:
            logger.exception("Failed voice clone benchmark for %s", name)

    out_path = RESULTS_DIR / "voice_clone_benchmarks.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved to %s", out_path)
    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    run_voice_clone_benchmarks()
