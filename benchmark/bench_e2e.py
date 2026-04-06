"""End-to-end benchmarks for OmniVoice inference modes.

Compares 6 runner configurations (Base, Triton, Triton+Sage, Faster,
Hybrid, Hybrid+Sage) on RTF, total time, and peak VRAM with proper
CUDA event timing, warmup, and statistical reporting.

Usage:
    python -m benchmark.bench_e2e --warmup 3 --repeat 5 --output results.json
"""

import argparse
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

SAMPLE_TEXTS = [
    {"text": "안녕하세요, 오늘 날씨가 정말 좋네요.", "language": "ko"},
    {
        "text": "Hello, welcome to the OmniVoice text-to-speech system.",
        "language": "en",
    },
    {"text": "你好，今天天气真好。", "language": "zh"},
]


def _reset_gpu() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _compute_stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def _calculate_rtf(
    audio_samples: int, sample_rate: int, generation_time: float
) -> float:
    if generation_time <= 0:
        return 0.0
    audio_duration = audio_samples / sample_rate
    return audio_duration / generation_time


def _get_runner_configs() -> list[dict[str, Any]]:
    """Return all runner configurations to benchmark."""
    from omnivoice_triton.models.base_runner import BaseRunner
    from omnivoice_triton.models.faster_runner import FasterRunner
    from omnivoice_triton.models.triton_faster_runner import TritonFasterRunner
    from omnivoice_triton.models.triton_runner import TritonRunner

    return [
        {"name": "Base", "factory": lambda: BaseRunner()},
        {"name": "Triton", "factory": lambda: TritonRunner()},
        {
            "name": "Triton+Sage",
            "factory": lambda: TritonRunner(enable_sage_attention=True),
        },
        {"name": "Faster", "factory": lambda: FasterRunner()},
        {"name": "Hybrid", "factory": lambda: TritonFasterRunner()},
        {
            "name": "Hybrid+Sage",
            "factory": lambda: TritonFasterRunner(enable_sage_attention=True),
        },
    ]


def bench_runner(
    config: dict[str, Any],
    texts: list[dict[str, str]],
    warmup: int = 3,
    repeat: int = 5,
) -> list[dict[str, Any]]:
    """Benchmark a single runner configuration."""
    results = []
    runner_name = config["name"]
    runner = config["factory"]()

    try:
        _reset_gpu()
        t_load = time.perf_counter()
        runner.load_model()
        torch.cuda.synchronize()
        load_time = round(time.perf_counter() - t_load, 3)
        logger.info("[%s] Model loaded in %.3fs", runner_name, load_time)

        for sample in texts:
            # Warmup
            for i in range(warmup):
                runner.generate(text=sample["text"])
            torch.cuda.synchronize()

            # Measured runs
            baseline_vram = torch.cuda.memory_allocated() / (1024**3)
            torch.cuda.reset_peak_memory_stats()

            timings_ms = []
            rtf_values = []

            for run_idx in range(repeat):
                torch.cuda.empty_cache()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                output = runner.generate(text=sample["text"])
                end.record()
                torch.cuda.synchronize()

                elapsed_ms = start.elapsed_time(end)
                timings_ms.append(elapsed_ms)

                audio = output.get("audio")
                sr = output.get("sample_rate", 24000)
                audio_len = len(audio) if audio is not None else 0
                rtf = _calculate_rtf(audio_len, sr, elapsed_ms / 1000.0)
                rtf_values.append(rtf)

            peak_delta = torch.cuda.max_memory_allocated() / (1024**3) - baseline_vram

            entry: dict[str, Any] = {
                "runner": runner_name,
                "text": sample["text"][:40],
                "language": sample.get("language", "auto"),
                "warmup": warmup,
                "repeat": repeat,
                "time_ms": _compute_stats(timings_ms),
                "rtf": _compute_stats(rtf_values),
                "peak_vram_gb": round(baseline_vram + peak_delta, 3),
                "model_load_time_s": load_time,
            }
            results.append(entry)
    finally:
        runner.unload_model()
        _reset_gpu()

    return results


def _format_table(results: list[dict[str, Any]]) -> str:
    header = (
        f"{'Runner':<15} {'Lang':<5} {'Mean(ms)':>9} {'Std':>7} "
        f"{'P50':>8} {'P95':>8} {'RTF':>6} {'VRAM':>6}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for r in results:
        t = r["time_ms"]
        lines.append(
            f"{r['runner']:<15} {r['language']:<5} "
            f"{t['mean']:>9.1f} {t['std']:>7.1f} "
            f"{t['p50']:>8.1f} {t['p95']:>8.1f} "
            f"{r['rtf']['mean']:>6.2f} {r['peak_vram_gb']:>6.2f}"
        )
    lines.append(sep)
    return "\n".join(lines)


def run_e2e_benchmarks(
    texts: list[dict[str, str]] | None = None,
    warmup: int = 3,
    repeat: int = 5,
    output: str | None = None,
) -> list[dict[str, Any]]:
    """Run end-to-end benchmarks for all runner configurations."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    texts = texts or SAMPLE_TEXTS

    configs = _get_runner_configs()
    all_results: list[dict[str, Any]] = []

    for config in configs:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("Benchmarking %s ...", config["name"])
        try:
            results = bench_runner(config, texts, warmup, repeat)
            all_results.extend(results)
        except Exception:
            logger.exception("Failed to benchmark %s", config["name"])

    if all_results:
        table = _format_table(all_results)
        for line in table.split("\n"):
            logger.info(line)

    out_path = output or str(RESULTS_DIR / "e2e_benchmarks.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(all_results, indent=2))
    logger.info("Results saved to %s", out_path)

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    parser = argparse.ArgumentParser(description="OmniVoice E2E benchmark")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run_e2e_benchmarks(warmup=args.warmup, repeat=args.repeat, output=args.output)
