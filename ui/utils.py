"""UI helper functions for VRAM measurement, RTF calculation, and JSON loading.

Provides utilities shared across the Streamlit dashboard for measuring
GPU memory usage, computing audio quality metrics (RTF), formatting
display values, and loading benchmark result files.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def get_vram_usage_gb() -> float:
    """Get current GPU VRAM usage in GB."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024**3)
    except ImportError:
        return 0.0


def get_peak_vram_gb() -> float:
    """Get peak GPU VRAM usage in GB."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024**3)
    except ImportError:
        return 0.0


def reset_vram_stats() -> None:
    """Reset VRAM statistics and clear cache."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def calculate_rtf(
    audio_samples: int,
    sample_rate: int,
    generation_time_s: float,
) -> float:
    """Calculate Real-Time Factor.

    RTF = audio_duration / generation_time.
    RTF > 1 means faster than real-time.

    Args:
        audio_samples: Number of audio samples generated.
        sample_rate: Audio sample rate in Hz.
        generation_time_s: Wall-clock generation time in seconds.

    Returns:
        Real-Time Factor (higher is better).
    """
    if generation_time_s <= 0 or sample_rate <= 0:
        return 0.0
    audio_duration = audio_samples / sample_rate
    return audio_duration / generation_time_s


def format_speedup(baseline_time: float, optimized_time: float) -> str:
    """Format speedup ratio as string.

    Args:
        baseline_time: Baseline execution time.
        optimized_time: Optimized execution time.

    Returns:
        Formatted speedup string like "2.5x".
    """
    if optimized_time <= 0:
        return "N/A"
    ratio = baseline_time / optimized_time
    return f"{ratio:.2f}x"


def load_json_list(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON list from a file, returning [] on error.

    Args:
        path: Path to the JSON file.

    Returns:
        List of dicts, empty list on any error.
    """
    p = Path(path)
    if not p.exists():
        logger.warning("File not found: %s", p)
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to parse JSON: %s", p)
        return []


def load_json_dict(path: str | Path) -> dict[str, Any] | None:
    """Load a JSON dict from a file, returning None on error.

    Args:
        path: Path to the JSON file.

    Returns:
        Dict on success, None on any error.
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        logger.exception("Failed to parse JSON: %s", p)
        return None
