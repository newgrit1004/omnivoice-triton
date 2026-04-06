"""Configuration for OmniVoice quality evaluation (Tier 3).

Defines evaluation sentences, pass/fail thresholds, and runner
settings used by benchmark/eval_quality.py.

Metrics:
    UTMOS   — MOS prediction (higher is better; range ~1–5)
    CER     — Character Error Rate (lower is better; range 0–1)
    Speaker sim — cosine similarity to reference (higher is better; range 0–1)
"""

from typing import Any

# ─────────────────────────────────────────────
# Evaluation sentences
# ─────────────────────────────────────────────

EVAL_SENTENCES: dict[str, list[str]] = {
    "ko": [
        "인공지능 음성 합성 기술이 빠르게 발전하고 있습니다.",
        "오늘 서울의 날씨는 맑고 기온은 이십삼도입니다.",
        "이 프로젝트는 트리톤 커널 퓨전을 통해 추론 속도를 높입니다.",
        "대한민국은 반도체와 디스플레이 산업에서 세계적인 경쟁력을 가지고 있습니다.",
        "깊은 학습 모델의 최적화는 실시간 서비스 배포에 매우 중요합니다.",
    ],
    "en": [
        "Artificial intelligence voice synthesis technology is advancing rapidly these days.",  # noqa: E501
        "This project accelerates inference through Triton kernel fusion and CUDA graph optimization.",  # noqa: E501
        "The non-autoregressive architecture generates all tokens in parallel using iterative unmasking.",  # noqa: E501
        "Modern text to speech systems can produce natural sounding audio from written text.",  # noqa: E501
        "Deep learning model optimization is crucial for deploying real-time services at scale.",  # noqa: E501
    ],
    "zh": [
        "人工智能语音合成技术正在快速发展，应用场景越来越广泛。",
        "这个项目通过自定义的三叉戟内核融合来加速推理过程。",
        "非自回归架构通过迭代去掩码的方式并行生成所有音频标记。",
        "现代文本转语音系统能够从书面文字生成自然流畅的语音输出。",
        "深度学习模型的优化对于部署实时服务至关重要。",
    ],
}

# ─────────────────────────────────────────────
# Pass/fail thresholds
# ─────────────────────────────────────────────

THRESHOLDS: dict[str, float] = {
    # UTMOS score must be at least this value (MOS-like, 1–5 scale)
    "utmos_min": 3.0,
    # Character Error Rate must be at most this value (0 = perfect)
    "cer_max": 0.3,
    # Speaker similarity (cosine) for voice-clone mode
    "speaker_sim_min": 0.7,
    # Maximum allowed UTMOS degradation vs. base runner
    "utmos_delta_max": 0.3,
    # Maximum allowed CER increase vs. base runner
    "cer_delta_max": 0.05,
}

# ─────────────────────────────────────────────
# Runner pairs to compare in quality eval
# ─────────────────────────────────────────────

RUNNER_COMPARISONS: list[dict[str, str]] = [
    {"ref": "base", "opt": "triton"},
    {"ref": "base", "opt": "faster"},
    {"ref": "base", "opt": "hybrid"},
]

# ─────────────────────────────────────────────
# Eval modes
# ─────────────────────────────────────────────

EVAL_MODES: list[str] = ["generate", "voice_design"]

# Voice design instruction used during quality eval
VOICE_DESIGN_INSTRUCT: str = "female, young adult, natural"

# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────


def get_all_sentences() -> list[dict[str, str]]:
    """Return a flat list of {text, language} dicts for all languages.

    Returns:
        List of dicts with 'text' and 'language' keys.
    """
    items: list[dict[str, str]] = []
    for lang, sentences in EVAL_SENTENCES.items():
        for text in sentences:
            items.append({"text": text, "language": lang})
    return items


def check_thresholds(metrics: dict[str, Any]) -> dict[str, bool]:
    """Check whether metrics satisfy the configured thresholds.

    Args:
        metrics: Dict with optional keys utmos, cer, speaker_sim,
                 utmos_delta, cer_delta.

    Returns:
        Dict mapping threshold name to bool (True = passed).
    """
    results: dict[str, bool] = {}
    if "utmos" in metrics:
        results["utmos_min"] = metrics["utmos"] >= THRESHOLDS["utmos_min"]
    if "cer" in metrics:
        results["cer_max"] = metrics["cer"] <= THRESHOLDS["cer_max"]
    if "speaker_sim" in metrics:
        results["speaker_sim_min"] = (
            metrics["speaker_sim"] >= THRESHOLDS["speaker_sim_min"]
        )
    if "utmos_delta" in metrics:
        results["utmos_delta_max"] = (
            abs(metrics["utmos_delta"]) <= THRESHOLDS["utmos_delta_max"]
        )
    if "cer_delta" in metrics:
        results["cer_delta_max"] = metrics["cer_delta"] <= THRESHOLDS["cer_delta_max"]
    return results
