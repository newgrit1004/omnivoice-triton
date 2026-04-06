"""Tier 3 E2E quality evaluation: independent distribution comparison.

Evaluates Base vs optimized runners by generating audio independently,
computing per-sample metrics (CER, UTMOS, Speaker Similarity), then
comparing distributions.

OmniVoice uses non-autoregressive iterative unmasking (32 fixed steps),
which is near-deterministic at temperature=0. This means pair-level
comparison IS valid, unlike stochastic AR models.

Usage:
    # Fast evaluation (~5 min, whisper-small, 1 run/sentence)
    uv run python -m benchmark.eval_quality --mode fast

    # Full evaluation (~30 min, whisper-large-v3, 3 runs/sentence)
    uv run python -m benchmark.eval_quality --mode full
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

from benchmark.eval_config import EVAL_SENTENCES, THRESHOLDS

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
OUTPUTS_DIR = Path(__file__).parent / "output" / "eval"

# Cache for expensive model loads
_whisper_cache: dict[str, Any] = {}
_utmos_cache: list[Any] = []
_voice_encoder_cache: list[Any] = []

# Eval mode configs
EVAL_CONFIGS: dict[str, dict[str, Any]] = {
    "fast": {
        "runs_per_sentence": 1,
        "asr_model": "small",
        "warmup_runs": 1,
        "sentences_per_lang": 5,
    },
    "full": {
        "runs_per_sentence": 3,
        "asr_model": "large-v3",
        "warmup_runs": 2,
        "sentences_per_lang": None,  # all
    },
}


# ────────────────────────────────────────────────────────────
# Model caching helpers
# ────────────────────────────────────────────────────────────


def _get_whisper(model_size: str) -> Any:
    if model_size not in _whisper_cache:
        import whisper

        _whisper_cache[model_size] = whisper.load_model(model_size)
    return _whisper_cache[model_size]


def _get_utmos() -> Any:
    if not _utmos_cache:
        predictor = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0",
            "utmos22_strong",
            trust_repo=True,
        )
        _utmos_cache.append(predictor)
    return _utmos_cache[0]


def _get_voice_encoder() -> Any:
    if not _voice_encoder_cache:
        from resemblyzer import VoiceEncoder

        _voice_encoder_cache.append(VoiceEncoder())
    return _voice_encoder_cache[0]


# ────────────────────────────────────────────────────────────
# Per-sample metric computation
# ────────────────────────────────────────────────────────────


def compute_cer(
    wav_path: str | Path, ground_truth: str, asr_model: str = "small"
) -> dict[str, Any]:
    """Compute CER for a single WAV against ground truth text."""
    from jiwer import cer

    model = _get_whisper(asr_model)
    result = model.transcribe(str(wav_path), language=None)
    transcript = result["text"].strip()
    cer_value = cer(ground_truth, transcript)
    return {"cer": float(cer_value), "transcript": transcript}


def compute_utmos(wav_path: str | Path) -> float:
    """Predict MOS score using UTMOS."""
    import librosa

    predictor = _get_utmos()
    wave, sr = librosa.load(str(wav_path), sr=None, mono=True)
    tensor = torch.from_numpy(wave).unsqueeze(0)
    score = predictor(tensor, sr)
    return float(score.mean().item())


def compute_speaker_similarity(wav_a: str | Path, wav_b: str | Path) -> float:
    """Compute speaker embedding cosine similarity between two WAVs.

    Falls back to librosa MFCC cosine similarity if resemblyzer is unavailable
    (webrtcvad requires pkg_resources which may be missing in uv envs).
    """
    try:
        from resemblyzer import preprocess_wav

        encoder = _get_voice_encoder()
        audio_a = preprocess_wav(Path(wav_a))
        audio_b = preprocess_wav(Path(wav_b))
        embed_a = encoder.embed_utterance(audio_a)
        embed_b = encoder.embed_utterance(audio_b)
    except (ImportError, ModuleNotFoundError):
        import librosa

        audio_a, sr_a = librosa.load(str(wav_a), sr=16000, mono=True)
        audio_b, sr_b = librosa.load(str(wav_b), sr=16000, mono=True)
        embed_a = np.mean(librosa.feature.mfcc(y=audio_a, sr=sr_a, n_mfcc=40), axis=1)
        embed_b = np.mean(librosa.feature.mfcc(y=audio_b, sr=sr_b, n_mfcc=40), axis=1)

    similarity = float(
        np.dot(embed_a, embed_b) / (np.linalg.norm(embed_a) * np.linalg.norm(embed_b))
    )
    return similarity


# ────────────────────────────────────────────────────────────
# Generation + evaluation
# ────────────────────────────────────────────────────────────


def generate_and_evaluate(
    runner: Any,
    sentence: dict[str, str],
    output_dir: Path,
    runner_name: str,
    idx: int,
    run: int,
    asr_model: str = "small",
) -> dict[str, Any]:
    """Generate audio with one runner and evaluate independently."""
    output = runner.generate(text=sentence["text"])
    audio = output.get("audio")
    sr = output.get("sample_rate", 24000)

    wav_path = output_dir / f"{runner_name}_{idx:03d}_r{run}.wav"
    sf.write(str(wav_path), audio, sr)

    cer_result = compute_cer(wav_path, sentence["text"], asr_model)
    utmos_score = compute_utmos(wav_path)

    return {
        "wav_path": str(wav_path),
        "sentence_idx": idx,
        "run": run,
        "text": sentence["text"][:60],
        "language": sentence["language"],
        "cer": cer_result["cer"],
        "transcript": cer_result["transcript"],
        "utmos": utmos_score,
    }


def _create_runner_by_name(name: str) -> Any:
    """Create runner, handling sage variants."""
    from omnivoice_triton import create_runner

    sage_map = {
        "triton_sage": ("triton", True),
        "hybrid_sage": ("hybrid", True),
    }
    if name in sage_map:
        base_name, sage = sage_map[name]
        return create_runner(base_name, enable_sage_attention=sage)
    return create_runner(name)


def _run_model_evaluation(
    runner_name: str,
    sentences: list[dict[str, str]],
    runs_per_sentence: int,
    output_dir: Path,
    asr_model: str,
    warmup_runs: int,
) -> list[dict[str, Any]]:
    """Generate and evaluate all sentences with one runner."""
    runner = _create_runner_by_name(runner_name)
    results: list[dict[str, Any]] = []

    try:
        runner.load_model()

        # Warmup
        for i in range(warmup_runs):
            logger.info("[%s] Warmup %d/%d", runner_name, i + 1, warmup_runs)
            runner.generate(text=sentences[0]["text"])

        # Generate and evaluate
        total = len(sentences) * runs_per_sentence
        count = 0
        for run in range(runs_per_sentence):
            for idx, sent in enumerate(sentences):
                count += 1
                logger.info(
                    "[%s] %d/%d (sent %d, run %d)",
                    runner_name,
                    count,
                    total,
                    idx,
                    run,
                )
                result = generate_and_evaluate(
                    runner,
                    sent,
                    output_dir,
                    runner_name,
                    idx,
                    run,
                    asr_model,
                )
                results.append(result)
    finally:
        runner.unload_model()

    return results


# ────────────────────────────────────────────────────────────
# Distribution comparison
# ────────────────────────────────────────────────────────────


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.array(values)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def _compute_verdict(
    ref_results: list[dict[str, Any]],
    opt_results: list[dict[str, Any]],
    ref_name: str,
    opt_name: str,
    mode: str,
) -> dict[str, Any]:
    """Compare distributions and produce PASS/FAIL verdict."""
    ref_utmos = [r["utmos"] for r in ref_results]
    opt_utmos = [r["utmos"] for r in opt_results]
    ref_cers = [r["cer"] for r in ref_results]
    opt_cers = [r["cer"] for r in opt_results]

    utmos_delta = abs(float(np.mean(ref_utmos)) - float(np.mean(opt_utmos)))
    cer_delta = abs(float(np.mean(ref_cers)) - float(np.mean(opt_cers)))

    # Speaker similarity (matched pairs, run 0)
    sims = _compute_speaker_similarities(ref_results, opt_results)
    sim_mean = round(float(np.mean(sims)), 4) if sims else 0.0

    # Mann-Whitney U test (full mode only)
    mann_whitney = None
    if mode == "full" and len(ref_utmos) >= 3:
        from scipy.stats import mannwhitneyu

        stat, p = mannwhitneyu(ref_utmos, opt_utmos, alternative="two-sided")
        mann_whitney = {
            "statistic": float(stat),
            "p_value": float(p),
            "equivalent": float(p) > 0.05,
        }

    # Check thresholds
    failures: list[str] = []
    if utmos_delta > THRESHOLDS.get("utmos_delta_max", 0.3):
        failures.append(
            f"UTMOS delta {utmos_delta:.4f} > {THRESHOLDS['utmos_delta_max']}"
        )
    if float(np.mean(opt_utmos)) < THRESHOLDS.get("utmos_min", 3.0):
        failures.append(
            f"{opt_name} UTMOS {np.mean(opt_utmos):.3f} < {THRESHOLDS['utmos_min']}"
        )
    if cer_delta > THRESHOLDS.get("cer_delta_max", 0.05):
        failures.append(f"CER delta {cer_delta:.4f} > {THRESHOLDS['cer_delta_max']}")
    if sims and sim_mean < THRESHOLDS.get("speaker_sim_min", 0.7):
        failures.append(f"Speaker sim {sim_mean:.4f} < {THRESHOLDS['speaker_sim_min']}")
    if mode == "full" and mann_whitney and not mann_whitney["equivalent"]:
        failures.append(f"Mann-Whitney p={mann_whitney['p_value']:.4f} < 0.05")

    return {
        "ref": ref_name,
        "opt": opt_name,
        "status": "FAIL" if failures else "PASS",
        "failures": failures,
        "ref_metrics": {"utmos": _stats(ref_utmos), "cer": _stats(ref_cers)},
        "opt_metrics": {"utmos": _stats(opt_utmos), "cer": _stats(opt_cers)},
        "utmos_delta": round(utmos_delta, 4),
        "cer_delta": round(cer_delta, 4),
        "speaker_sim_mean": sim_mean,
        "mann_whitney": mann_whitney,
    }


def _compute_speaker_similarities(
    base_results: list[dict[str, Any]],
    opt_results: list[dict[str, Any]],
) -> list[float]:
    """Compute speaker similarity for matched sentence pairs (run 0 only)."""
    base_by_idx = {
        r["sentence_idx"]: r["wav_path"] for r in base_results if r["run"] == 0
    }
    opt_by_idx = {
        r["sentence_idx"]: r["wav_path"] for r in opt_results if r["run"] == 0
    }

    sims: list[float] = []
    for idx in sorted(base_by_idx.keys()):
        if idx not in opt_by_idx:
            continue
        sim = compute_speaker_similarity(base_by_idx[idx], opt_by_idx[idx])
        sims.append(sim)
        logger.info("Speaker sim sentence %d: %.4f", idx, sim)
    return sims


# ────────────────────────────────────────────────────────────
# Main pipeline
# ────────────────────────────────────────────────────────────


def run_tier3(
    mode: str = "fast",
    ref_runner: str = "base",
    opt_runners: list[str] | None = None,
) -> dict[str, Any]:
    """Run Tier 3 quality evaluation."""
    if opt_runners is None:
        opt_runners = ["triton", "triton_sage", "faster", "hybrid", "hybrid_sage"]

    cfg = EVAL_CONFIGS[mode]
    sentences = _select_sentences(mode)
    output_dir = OUTPUTS_DIR / f"tier3_{mode}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Tier 3 [%s]: %s vs %s | %d sentences | %d runs/sent | ASR=%s",
        mode,
        ref_runner,
        opt_runners,
        len(sentences),
        cfg["runs_per_sentence"],
        cfg["asr_model"],
    )

    t_start = time.perf_counter()

    # Generate reference
    logger.info("Generating reference (%s)...", ref_runner)
    ref_results = _run_model_evaluation(
        ref_runner,
        sentences,
        cfg["runs_per_sentence"],
        output_dir,
        cfg["asr_model"],
        cfg["warmup_runs"],
    )

    # Generate optimized runners
    opt_all: dict[str, list[dict[str, Any]]] = {}
    for name in opt_runners:
        logger.info("Generating %s...", name)
        opt_all[name] = _run_model_evaluation(
            name,
            sentences,
            cfg["runs_per_sentence"],
            output_dir,
            cfg["asr_model"],
            cfg["warmup_runs"],
        )

    # Compare distributions
    comparisons = []
    for name in opt_runners:
        verdict = _compute_verdict(ref_results, opt_all[name], ref_runner, name, mode)
        comparisons.append(verdict)

    eval_time = round(time.perf_counter() - t_start, 2)
    overall = "PASS" if all(c["status"] == "PASS" for c in comparisons) else "FAIL"

    # Strip wav_path from samples (keep JSON clean)
    for s in ref_results:
        s.pop("wav_path", None)
    for samples in opt_all.values():
        for s in samples:
            s.pop("wav_path", None)

    result = {
        "status": overall,
        "mode": mode,
        "ref_runner": ref_runner,
        "opt_runners": opt_runners,
        "num_sentences": len(sentences),
        "runs_per_sentence": cfg["runs_per_sentence"],
        "asr_model": cfg["asr_model"],
        "eval_time_s": eval_time,
        "comparisons": comparisons,
        "ref_samples": ref_results,
        "opt_samples": {k: v for k, v in opt_all.items()},
    }

    _print_summary(result)
    return result


def _select_sentences(mode: str) -> list[dict[str, str]]:
    """Select evaluation sentences based on mode."""
    cfg = EVAL_CONFIGS[mode]
    limit = cfg["sentences_per_lang"]
    sentences: list[dict[str, str]] = []
    for lang, texts in EVAL_SENTENCES.items():
        subset = texts[:limit] if limit else texts
        for text in subset:
            sentences.append({"text": text, "language": lang})
    return sentences


def _print_summary(result: dict[str, Any]) -> None:
    """Log formatted summary table."""
    sep = "=" * 70
    logger.info("")
    logger.info("TIER 3 QUALITY EVALUATION (%s mode)", result["mode"])
    logger.info(sep)
    logger.info(
        "Sentences: %d | Runs/sent: %d | ASR: %s | Time: %.1fs",
        result["num_sentences"],
        result["runs_per_sentence"],
        result["asr_model"],
        result["eval_time_s"],
    )
    logger.info(sep)

    header = f"{'Runner':<12}{'UTMOS':<18}{'CER':<18}{'Speaker Sim':<14}{'Status'}"
    logger.info(header)
    logger.info("-" * 70)

    for c in result["comparisons"]:
        # Reference row (first time only)
        opt_m = c["opt_metrics"]

        opt_utmos = f"{opt_m['utmos']['mean']:.2f}\u00b1{opt_m['utmos']['std']:.2f}"
        opt_cer = f"{opt_m['cer']['mean']:.2f}\u00b1{opt_m['cer']['std']:.2f}"
        sim = f"{c['speaker_sim_mean']:.2f}"
        logger.info(f"{c['opt']:<12}{opt_utmos:<18}{opt_cer:<18}{sim:<14}{c['status']}")

    logger.info(sep)
    logger.info("Overall: %s", result["status"])

    for c in result["comparisons"]:
        if c["failures"]:
            logger.info("  %s failures:", c["opt"])
            for f in c["failures"]:
                logger.info("    - %s", f)
    logger.info(sep)


# ────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniVoice Tier 3 quality evaluation")
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default="fast",
        help="fast (~5min, whisper-small) or full (~30min, whisper-large-v3)",
    )
    parser.add_argument(
        "--runners",
        nargs="*",
        default=None,
        help="Optimized runners to compare (default: triton faster hybrid)",
    )
    parser.add_argument(
        "--ref", default="base", help="Reference runner (default: base)"
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = run_tier3(args.mode, ref_runner=args.ref, opt_runners=args.runners)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.output or str(RESULTS_DIR / f"tier3_{args.mode}_multi.json")

    Path(out_path).write_text(json.dumps(result, indent=2, default=str))
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)8s] %(message)s",
    )
    main()
