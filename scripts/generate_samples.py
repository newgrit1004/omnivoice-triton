"""Generate audio samples for all runner modes × sample texts.

Produces WAV files at 24kHz and a metadata.json summary for the
assets/audio_samples/ directory. Useful for demonstrating runner
quality differences in the open-source release.

Usage:
    uv run python scripts/generate_samples.py
    uv run python scripts/generate_samples.py --modes base triton
    uv run python scripts/generate_samples.py --output-dir /tmp/samples
"""

import argparse
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

SAMPLE_TEXTS: dict[str, list[str]] = {
    "ko": [
        "안녕하세요, 반갑습니다.",
        "오늘 날씨가 정말 좋네요.",
        "옴니보이스 음성 합성 시스템입니다.",
    ],
    "en": [
        "Hello, nice to meet you.",
        "The weather is really nice today.",
        "Welcome to the OmniVoice text-to-speech system.",
    ],
    "zh": [
        "你好，很高兴认识你。",
        "今天天气真好。",
        "欢迎使用OmniVoice语音合成系统。",
    ],
}

VOICE_DESIGN_SAMPLES: list[dict[str, str]] = [
    {
        "text": "Hello, I am a young female voice with a cheerful tone.",
        "instruct": "female, young, cheerful",
        "language": "en",
        "label": "female_young_cheerful",
    },
    {
        "text": "안녕하세요, 저는 차분한 남성 목소리입니다.",
        "instruct": "male, adult, calm",
        "language": "ko",
        "label": "male_adult_calm",
    },
]

ALL_MODES = ["base", "triton", "triton_sage", "faster", "hybrid", "hybrid_sage"]

# Modes that enable SageAttention
_SAGE_MODES: dict[str, str] = {
    "triton_sage": "triton",
    "hybrid_sage": "hybrid",
}


DEFAULT_SEED = 42


def _set_seed(seed: int = DEFAULT_SEED) -> None:
    """Fix all random seeds for reproducible generation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_runner(mode: str) -> Any:
    """Instantiate a runner by mode name."""
    from omnivoice_triton import create_runner

    if mode in _SAGE_MODES:
        return create_runner(_SAGE_MODES[mode], enable_sage_attention=True)
    return create_runner(mode)


def _get_hardware_info() -> dict[str, Any]:
    """Collect GPU and system hardware info."""
    info: dict[str, Any] = {"gpu": None, "cuda_version": None}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
    except Exception as exc:
        logger.warning("Could not collect hardware info: %s", exc)
    return info


def _save_audio(audio: Any, path: Path, sample_rate: int = 24000) -> None:
    """Save audio tensor to WAV file using soundfile."""
    import numpy as np
    import soundfile as sf

    # Accepts torch tensor or numpy array; ensure 1-D float32 numpy
    try:
        data = audio.cpu().numpy()
    except AttributeError:
        data = np.asarray(audio)

    if data.ndim > 1:
        data = data.squeeze()

    data = data.astype(np.float32)
    sf.write(str(path), data, sample_rate)


def generate_mode_samples(
    mode: str,
    output_dir: Path,
    sample_texts: dict[str, list[str]],
    voice_design_samples: list[dict[str, str]],
) -> list[dict[str, Any]]:
    """Generate all samples for one runner mode.

    Args:
        mode: Runner mode name (base/triton/faster/hybrid).
        output_dir: Root output directory; a subdirectory per mode is created.
        sample_texts: Dict of language -> list of texts to synthesize.
        voice_design_samples: List of voice design sample configs.

    Returns:
        List of sample metadata dicts.
    """
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []

    try:
        runner = _get_runner(mode)
    except Exception as exc:
        logger.error("[%s] Failed to create runner: %s", mode, exc)
        return samples

    try:
        logger.info("[%s] Loading model...", mode)
        runner.load_model()
        logger.info("[%s] Model loaded.", mode)

        # Basic generation samples
        for lang, texts in sample_texts.items():
            for idx, text in enumerate(texts):
                filename = f"{lang}_{idx:02d}.wav"
                out_path = mode_dir / filename
                try:
                    _set_seed(DEFAULT_SEED)
                    t0 = time.perf_counter()
                    output = runner.generate(text=text)
                    elapsed = time.perf_counter() - t0

                    audio = output.get("audio") if isinstance(output, dict) else output
                    sr = (
                        output.get("sample_rate", 24000)
                        if isinstance(output, dict)
                        else 24000
                    )
                    _save_audio(audio, out_path, sr)
                    logger.info("[%s] Saved %s (%.2fs)", mode, out_path.name, elapsed)
                    samples.append(
                        {
                            "mode": mode,
                            "type": "generate",
                            "language": lang,
                            "text": text,
                            "file": f"{mode}/{filename}",
                            "sample_rate": sr,
                            "generation_time_s": round(elapsed, 3),
                        }
                    )
                except Exception as exc:
                    logger.warning(
                        "[%s] Failed to generate %s/%d: %s", mode, lang, idx, exc
                    )

        # Voice design samples
        for vd in voice_design_samples:
            filename = f"voice_design_{vd['label']}.wav"
            out_path = mode_dir / filename
            try:
                _set_seed(DEFAULT_SEED)
                t0 = time.perf_counter()
                output = runner.generate(text=vd["text"], instruct=vd["instruct"])
                elapsed = time.perf_counter() - t0

                audio = output.get("audio") if isinstance(output, dict) else output
                sr = (
                    output.get("sample_rate", 24000)
                    if isinstance(output, dict)
                    else 24000
                )
                _save_audio(audio, out_path, sr)
                logger.info("[%s] Saved %s (%.2fs)", mode, out_path.name, elapsed)
                samples.append(
                    {
                        "mode": mode,
                        "type": "voice_design",
                        "language": vd["language"],
                        "text": vd["text"],
                        "instruct": vd["instruct"],
                        "file": f"{mode}/{filename}",
                        "sample_rate": sr,
                        "generation_time_s": round(elapsed, 3),
                    }
                )
            except Exception as exc:
                logger.warning(
                    "[%s] Failed voice design %s: %s", mode, vd["label"], exc
                )

    finally:
        try:
            runner.unload_model()
        except Exception:
            pass

    return samples


def main() -> None:
    """CLI entry point for generating audio samples."""
    parser = argparse.ArgumentParser(
        description="Generate OmniVoice audio samples for all runner modes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=ALL_MODES,
        choices=ALL_MODES,
        help="Runner modes to generate samples for.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("assets/audio_samples"),
        help="Root output directory for generated WAV files.",
    )
    parser.add_argument(
        "--no-voice-design",
        action="store_true",
        help="Skip voice design samples.",
    )
    args = parser.parse_args()

    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA not available. Samples may not generate correctly.")
    except ImportError:
        logger.warning("torch not installed; skipping GPU check.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    voice_design = [] if args.no_voice_design else VOICE_DESIGN_SAMPLES
    all_samples: list[dict[str, Any]] = []

    for mode in args.modes:
        logger.info("=" * 50)
        logger.info("Generating samples for mode: %s", mode)
        logger.info("=" * 50)
        mode_samples = generate_mode_samples(
            mode, args.output_dir, SAMPLE_TEXTS, voice_design
        )
        all_samples.extend(mode_samples)

    # Write metadata.json
    metadata: dict[str, Any] = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "hardware": _get_hardware_info(),
        "modes": args.modes,
        "samples": all_samples,
    }
    meta_path = args.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
    logger.info("Metadata saved to %s", meta_path)
    logger.info("Done. Generated %d samples total.", len(all_samples))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    main()
