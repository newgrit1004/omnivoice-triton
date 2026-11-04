"""Generate OmniVoice default/design voice sweeps for Lingua voice review.

The sweep is intentionally metadata-heavy: every generated WAV is paired with
the exact instruction tags, dtype, language, text case, and timing data needed
to compare OmniVoice design voices against Gemini TTS clone references.

Examples:
    .venv/bin/python scripts/generate_default_voice_sweep.py --dry-run
    .venv/bin/python scripts/generate_default_voice_sweep.py --languages en,es,fr
    .venv/bin/python scripts/generate_default_voice_sweep.py --profile-set full
    .venv/bin/python scripts/generate_default_voice_sweep.py --include-gemini-clones
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import random
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ID = "k2-fsa/OmniVoice"
DEFAULT_OUTPUT_ROOT = Path("tts/omnivoice_default_voice_sweep")
DEFAULT_GEMINI_MANIFEST = Path("tts/gemini_tts_clone_samples/manifest.json")
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_SEED = 42
LANGUAGE_HINT_OVERRIDES: dict[str, str | None] = {
    # The installed OmniVoice 0.1.4 language map warns on "ar"; Arabic text still
    # generates in language-agnostic mode, so omit the hint for cleaner sweeps.
    "ar": None,
}


@dataclass(frozen=True)
class TextCase:
    label: str
    language: str
    text: str
    notes: str


@dataclass(frozen=True)
class VoiceProfile:
    label: str
    instruct: str | None
    source: str
    gender: str | None = None
    age: str | None = None
    pitch: str | None = None
    accent: str | None = None
    dialect: str | None = None
    style: str | None = None
    persona: str | None = None
    notes: str = ""


TEXT_CASES: dict[str, list[TextCase]] = {
    "en": [
        TextCase(
            "baseline",
            "en",
            "Hello, I am your language tutor. Listen carefully and repeat after me.",
            "English tutor baseline.",
        ),
        TextCase(
            "pronunciation",
            "en",
            "Please practice clearly: three thin thinkers threw thirty-three things.",
            "English fricatives and pacing.",
        ),
        TextCase(
            "mixed_tutor",
            "en",
            "In Spanish, usted is formal and tu is casual. Try saying both with me.",
            "Tutor-style mixed language explanation.",
        ),
    ],
    "es": [
        TextCase(
            "baseline",
            "es",
            "Hola, soy tu tutora de idiomas. Escucha con atencion y repite conmigo.",
            "Spanish tutor baseline.",
        ),
        TextCase(
            "pronunciation",
            "es",
            "Practiquemos la erre: perro, carro, rapido, alrededor.",
            "Spanish rolled-r practice.",
        ),
        TextCase(
            "mixed_tutor",
            "es",
            "Usted es formal, pero tu es casual. Repite las dos formas conmigo.",
            "Spanish contrast useful in Lingua lessons.",
        ),
    ],
    "fr": [
        TextCase(
            "baseline",
            "fr",
            "Bonjour, je suis votre professeur de langue. Ecoutez puis repetez apres moi.",
            "French tutor baseline.",
        ),
        TextCase(
            "pronunciation",
            "fr",
            "Repetez doucement: rue, rouge, regarder, tres heureux.",
            "French r and vowel practice.",
        ),
        TextCase(
            "mixed_tutor",
            "fr",
            "En francais, tu est familier et vous est plus poli. Repetez les deux.",
            "French politeness contrast.",
        ),
    ],
    "de": [
        TextCase(
            "baseline",
            "de",
            "Hallo, ich bin dein Sprachlehrer. Hoer gut zu und sprich mir nach.",
            "German tutor baseline.",
        ),
        TextCase(
            "pronunciation",
            "de",
            "Bitte uebe deutlich: ich, ach, richtig, freundlich.",
            "German ch sounds.",
        ),
        TextCase(
            "mixed_tutor",
            "de",
            "Du ist informell, aber Sie ist hoeflich. Wiederhole beide Formen.",
            "German formality contrast.",
        ),
    ],
    "ar": [
        TextCase(
            "baseline",
            "ar",
            "مرحبا، أنا معلم اللغة الخاص بك. استمع جيدا ثم كرر بعدي.",
            "Arabic tutor baseline.",
        ),
        TextCase(
            "pronunciation",
            "ar",
            "هيا نتدرب بوضوح: خ، ح، ع، غ، ق.",
            "Arabic pronunciation-sensitive letters.",
        ),
        TextCase(
            "mixed_tutor",
            "ar",
            "كلمة مرحبا تعني hello. كررها ببطء معي: مرحبا.",
            "Arabic mixed tutor prompt.",
        ),
    ],
    "ur": [
        TextCase(
            "baseline",
            "ur",
            "السلام علیکم، میں آپ کا زبان کا استاد ہوں۔ غور سے سنیں اور میرے بعد دہرائیں۔",
            "Urdu tutor baseline.",
        ),
        TextCase(
            "pronunciation",
            "ur",
            "آئیے صاف بولنے کی مشق کریں: خ، غ، ق، ٹ، ڑ۔",
            "Urdu pronunciation-sensitive letters.",
        ),
        TextCase(
            "mixed_tutor",
            "ur",
            "آپ رسمی ہے، اور تم غیر رسمی ہے۔ دونوں الفاظ میرے ساتھ دہرائیں۔",
            "Urdu formality contrast.",
        ),
    ],
    "zh": [
        TextCase(
            "baseline",
            "zh",
            "你好，我是你的语言老师。请仔细听，然后跟我重复。",
            "Chinese tutor baseline.",
        ),
        TextCase(
            "pronunciation",
            "zh",
            "请练习声调：妈，麻，马，骂。",
            "Mandarin tone contrast.",
        ),
        TextCase(
            "mixed_tutor",
            "zh",
            "你好 means hello. 请慢慢跟我说：你好。",
            "Chinese mixed tutor prompt.",
        ),
    ],
}


OFFICIAL_PROFILES: list[VoiceProfile] = [
    VoiceProfile("auto_default", None, "default", notes="No instruct; OmniVoice default."),
    VoiceProfile(
        "female_young_high",
        "female, young adult, high pitch",
        "official",
        gender="female",
        age="young adult",
        pitch="high pitch",
    ),
    VoiceProfile(
        "female_young_moderate",
        "female, young adult, moderate pitch",
        "official",
        gender="female",
        age="young adult",
        pitch="moderate pitch",
    ),
    VoiceProfile(
        "female_middle_low",
        "female, middle-aged, low pitch",
        "official",
        gender="female",
        age="middle-aged",
        pitch="low pitch",
        notes="Lower/deeper female voice candidate.",
    ),
    VoiceProfile(
        "male_young_moderate",
        "male, young adult, moderate pitch",
        "official",
        gender="male",
        age="young adult",
        pitch="moderate pitch",
    ),
    VoiceProfile(
        "male_middle_low",
        "male, middle-aged, low pitch",
        "official",
        gender="male",
        age="middle-aged",
        pitch="low pitch",
        notes="Deep voice candidate.",
    ),
    VoiceProfile(
        "male_middle_very_low",
        "male, middle-aged, very low pitch",
        "official",
        gender="male",
        age="middle-aged",
        pitch="very low pitch",
        notes="Deepest official pitch tag.",
    ),
    VoiceProfile(
        "female_whisper",
        "female, young adult, moderate pitch, whisper",
        "official",
        gender="female",
        age="young adult",
        pitch="moderate pitch",
        style="whisper",
    ),
    VoiceProfile(
        "male_whisper",
        "male, middle-aged, low pitch, whisper",
        "official",
        gender="male",
        age="middle-aged",
        pitch="low pitch",
        style="whisper",
    ),
    VoiceProfile(
        "female_elderly_low",
        "female, elderly, low pitch",
        "official",
        gender="female",
        age="elderly",
        pitch="low pitch",
    ),
    VoiceProfile(
        "male_elderly_low",
        "male, elderly, low pitch",
        "official",
        gender="male",
        age="elderly",
        pitch="low pitch",
    ),
    VoiceProfile(
        "child_high",
        "child, high pitch",
        "official",
        age="child",
        pitch="high pitch",
    ),
    VoiceProfile(
        "teenager_high",
        "teenager, high pitch",
        "official",
        age="teenager",
        pitch="high pitch",
    ),
]


ENGLISH_ACCENT_PROFILES: list[VoiceProfile] = [
    VoiceProfile(
        f"female_{accent.replace(' ', '_')}",
        f"female, young adult, moderate pitch, {accent}",
        "official_accent",
        gender="female",
        age="young adult",
        pitch="moderate pitch",
        accent=accent,
    )
    for accent in [
        "american accent",
        "british accent",
        "australian accent",
        "canadian accent",
        "indian accent",
        "chinese accent",
        "korean accent",
        "japanese accent",
        "russian accent",
        "portuguese accent",
    ]
]


CHINESE_DIALECT_PROFILES: list[VoiceProfile] = [
    VoiceProfile(
        f"female_zh_{label}",
        f"女，青年，中音调，{dialect}",
        "official_dialect",
        gender="female",
        age="young adult",
        pitch="moderate pitch",
        dialect=dialect,
    )
    for label, dialect in [
        ("henan", "河南话"),
        ("shaanxi", "陕西话"),
        ("sichuan", "四川话"),
        ("guizhou", "贵州话"),
        ("yunnan", "云南话"),
        ("guilin", "桂林话"),
        ("jinan", "济南话"),
        ("shijiazhuang", "石家庄话"),
        ("gansu", "甘肃话"),
        ("ningxia", "宁夏话"),
        ("qingdao", "青岛话"),
        ("northeast", "东北话"),
    ]
]


MATRIX_GENDERS = ["female", "male"]
MATRIX_AGES = ["child", "teenager", "young adult", "middle-aged", "elderly"]
MATRIX_PITCHES = [
    "very low pitch",
    "low pitch",
    "moderate pitch",
    "high pitch",
    "very high pitch",
]


AGE_PITCH_MATRIX_PROFILES: list[VoiceProfile] = [
    VoiceProfile(
        f"{gender}_{age.replace('-', '').replace(' ', '_')}_{pitch.replace(' ', '_')}",
        f"{gender}, {age}, {pitch}",
        "age_pitch_matrix",
        gender=gender,
        age=age,
        pitch=pitch,
        notes="Full gender x age x pitch matrix profile.",
    )
    for gender in MATRIX_GENDERS
    for age in MATRIX_AGES
    for pitch in MATRIX_PITCHES
]


WHISPER_MATRIX_PROFILES: list[VoiceProfile] = [
    VoiceProfile(
        f"{gender}_{age.replace('-', '').replace(' ', '_')}_moderate_pitch_whisper",
        f"{gender}, {age}, moderate pitch, whisper",
        "whisper_matrix",
        gender=gender,
        age=age,
        pitch="moderate pitch",
        style="whisper",
        notes="Whisper style matrix profile at moderate pitch.",
    )
    for gender in MATRIX_GENDERS
    for age in MATRIX_AGES
]


EXPLORATORY_PERSONAS: list[VoiceProfile] = [
    VoiceProfile(
        "warm_trusting_tutor",
        "female, young adult, moderate pitch",
        "exploratory",
        gender="female",
        age="young adult",
        pitch="moderate pitch",
        persona="warm and trusting",
        notes="Persona label inspired by Gemini Despina; OmniVoice instruct only accepts official tags.",
    ),
    VoiceProfile(
        "thoughtful_calm_tutor",
        "female, middle-aged, low pitch",
        "exploratory",
        gender="female",
        age="middle-aged",
        pitch="low pitch",
        persona="thoughtful and calm",
        notes="Persona label inspired by Gemini Aoede; OmniVoice instruct only accepts official tags.",
    ),
    VoiceProfile(
        "bright_playful_tutor",
        "female, young adult, high pitch",
        "exploratory",
        gender="female",
        age="young adult",
        pitch="high pitch",
        persona="bright and playful",
        notes="Persona label inspired by Gemini Zephyr; OmniVoice instruct only accepts official tags.",
    ),
    VoiceProfile(
        "confident_youthful_tutor",
        "female, young adult, moderate pitch",
        "exploratory",
        gender="female",
        age="young adult",
        pitch="moderate pitch",
        persona="confident and youthful",
        notes="Persona label inspired by Gemini Kore; OmniVoice instruct only accepts official tags.",
    ),
    VoiceProfile(
        "calm_reassuring_deep",
        "male, middle-aged, low pitch",
        "exploratory",
        gender="male",
        age="middle-aged",
        pitch="low pitch",
        persona="calm and reassuring",
        notes="Gemini Charon-like persona label plus deep voice tag.",
    ),
    VoiceProfile(
        "friendly_casual_deep",
        "male, young adult, low pitch",
        "exploratory",
        gender="male",
        age="young adult",
        pitch="low pitch",
        persona="friendly and casual",
        notes="Persona label inspired by Gemini Iapetus; OmniVoice instruct only accepts official tags.",
    ),
    VoiceProfile(
        "clear_helpful_deep",
        "male, middle-aged, very low pitch",
        "exploratory",
        gender="male",
        age="middle-aged",
        pitch="very low pitch",
        persona="clear and helpful",
        notes="Persona label inspired by Gemini Fenrir; OmniVoice instruct only accepts official tags.",
    ),
    VoiceProfile(
        "gentle_slow_tutor",
        "female, middle-aged, moderate pitch",
        "exploratory",
        gender="female",
        age="middle-aged",
        pitch="moderate pitch",
        persona="gentle slow patient",
        notes="Tutor persona label; use --speed below 1.0 for actual slower delivery.",
    ),
    VoiceProfile(
        "neutral_clear_tutor",
        "middle-aged, moderate pitch",
        "exploratory",
        age="middle-aged",
        pitch="moderate pitch",
        persona="neutral clear",
        notes="Tutor persona label; OmniVoice instruct only accepts official tags.",
    ),
]


PROFILE_SETS: dict[str, list[VoiceProfile]] = {
    "pilot": [
        OFFICIAL_PROFILES[0],
        OFFICIAL_PROFILES[1],
        OFFICIAL_PROFILES[2],
        OFFICIAL_PROFILES[5],
        OFFICIAL_PROFILES[6],
        OFFICIAL_PROFILES[7],
        EXPLORATORY_PERSONAS[0],
        EXPLORATORY_PERSONAS[1],
        EXPLORATORY_PERSONAS[4],
        EXPLORATORY_PERSONAS[6],
    ],
    "official": OFFICIAL_PROFILES + ENGLISH_ACCENT_PROFILES + CHINESE_DIALECT_PROFILES,
    "exploratory": OFFICIAL_PROFILES + EXPLORATORY_PERSONAS,
    "matrix": AGE_PITCH_MATRIX_PROFILES + WHISPER_MATRIX_PROFILES,
    "full": (
        OFFICIAL_PROFILES
        + ENGLISH_ACCENT_PROFILES
        + CHINESE_DIALECT_PROFILES
        + EXPLORATORY_PERSONAS
        + AGE_PITCH_MATRIX_PROFILES
        + WHISPER_MATRIX_PROFILES
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fp16 OmniVoice default/design voice samples.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runner", default="hybrid", help="Runner backend.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="Model id/path.")
    parser.add_argument(
        "--dtype",
        default="fp16",
        choices=["fp16", "bf16", "fp32"],
        help="Model dtype. Use fp16 for the Lingua comparison.",
    )
    parser.add_argument("--device", default="cuda", help="Device for generation.")
    parser.add_argument(
        "--max-cuda-graphs",
        type=int,
        default=32,
        help="Maximum retained CUDA Graph shapes for faster/hybrid runners. Use 0 to disable.",
    )
    parser.add_argument(
        "--languages",
        default="en,es,fr,de,ar,ur",
        help="Comma-separated language codes to generate.",
    )
    parser.add_argument(
        "--profile-set",
        default="pilot",
        choices=sorted(PROFILE_SETS),
        help="Which instruct/profile set to sweep.",
    )
    parser.add_argument(
        "--text-case",
        action="append",
        default=None,
        choices=["baseline", "pronunciation", "mixed_tutor"],
        help="Restrict text cases. Repeatable. Defaults to all cases.",
    )
    parser.add_argument("--num-step", type=int, default=16, help="Generation steps.")
    parser.add_argument(
        "--also-num-step",
        action="append",
        type=int,
        default=[],
        help="Additional num_step values to generate, e.g. --also-num-step 32.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=2.0,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--class-temperature",
        type=float,
        default=0.0,
        help="Class token sampling temperature.",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=None,
        help="Optional speed override. 1.0 is omitted to match default behavior.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional fixed duration override in seconds.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory for timestamped sweep output.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional output run id. Defaults to UTC timestamp.",
    )
    parser.add_argument(
        "--gemini-manifest",
        type=Path,
        default=DEFAULT_GEMINI_MANIFEST,
        help="Manifest for Gemini clone reference metadata.",
    )
    parser.add_argument(
        "--include-gemini-clones",
        action="store_true",
        help="Also generate clone-mode outputs from Gemini reference WAVs.",
    )
    parser.add_argument(
        "--skip-design",
        action="store_true",
        help="Skip OmniVoice default/design generation. Useful for clone-only reruns.",
    )
    parser.add_argument(
        "--gemini-voices",
        default="Aoede,Despina,Charon,Fenrir",
        help="Comma-separated Gemini voices to clone when --include-gemini-clones.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after this many generated WAVs; useful for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write catalogs/manifests showing planned samples, but do not load model.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate files that already exist in the run directory.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser


def _utc_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _slug(value: str, max_len: int = 96) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", value.lower()).strip("_")
    return (normalized[:max_len].strip("_") or "sample")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_numpy(audio: Any) -> np.ndarray:
    if isinstance(audio, list):
        audio = audio[0]
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().squeeze().cpu().float().numpy()
    data = np.asarray(audio, dtype=np.float32).squeeze()
    if data.ndim != 1:
        data = data.reshape(-1)
    return np.clip(data, -1.0, 1.0)


def _save_audio(audio: Any, path: Path, sample_rate: int) -> dict[str, Any]:
    data = _to_numpy(audio)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sample_rate)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    return {
        "sample_rate": sample_rate,
        "samples": int(data.size),
        "duration_s": round(float(data.size) / float(sample_rate), 3)
        if sample_rate
        else 0.0,
        "peak_abs": round(peak, 6),
        "clipping_risk": bool(peak >= 0.999),
        "silent_risk": bool(peak < 0.001),
    }


def _build_text_cases(languages: list[str], wanted: set[str] | None) -> list[TextCase]:
    cases: list[TextCase] = []
    for language in languages:
        lang_cases = TEXT_CASES.get(language)
        if not lang_cases:
            logger.warning("No built-in text cases for language=%s; skipping.", language)
            continue
        for case in lang_cases:
            if wanted is None or case.label in wanted:
                cases.append(case)
    return cases


def _load_gemini_entries(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("Gemini manifest not found: %s", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [row for row in data.get("results", []) if row.get("status") == "success"]


def _gemini_reference_path(manifest_path: Path, row: dict[str, Any]) -> Path | None:
    local_path = row.get("local_path")
    if local_path:
        candidate = Path(local_path)
        if candidate.exists():
            return candidate
    lang = row.get("lang_code")
    voice = str(row.get("voice_name", "")).lower()
    language_name = str(row.get("language_name", "")).lower()
    if lang and voice and language_name:
        candidate = manifest_path.parent / lang / f"{lang}_{language_name}_{voice}.wav"
        if candidate.exists():
            return candidate
    return None


def _matching_gemini_rows(
    manifest_path: Path,
    languages: set[str],
    voices: set[str],
) -> list[dict[str, Any]]:
    rows = []
    for row in _load_gemini_entries(manifest_path):
        if row.get("lang_code") not in languages:
            continue
        if str(row.get("voice_name", "")).lower() not in voices:
            continue
        ref_path = _gemini_reference_path(manifest_path, row)
        if ref_path is None:
            continue
        copied = dict(row)
        copied["reference_audio"] = str(ref_path)
        rows.append(copied)
    return rows


def _create_runner(args: argparse.Namespace) -> Any:
    from omnivoice_triton import create_runner

    runner_kwargs: dict[str, Any] = {
        "device": args.device,
        "model_id": args.model,
        "dtype": args.dtype,
    }
    if args.runner in {"faster", "hybrid"}:
        runner_kwargs["max_cuda_graphs"] = args.max_cuda_graphs
    runner = create_runner(args.runner, **runner_kwargs)
    runner.load_model()
    return runner


def _cleanup_cuda_after_exception(runner: Any, exc: BaseException) -> None:
    is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(
        exc
    ).lower()
    if not is_oom or not torch.cuda.is_available():
        return

    graph_forward = getattr(runner, "_graph_forward", None)
    clear = getattr(graph_forward, "clear", None)
    if callable(clear):
        try:
            clear()
            logger.warning("Cleared retained CUDA Graphs after OOM.")
        except Exception:
            logger.debug("Failed to clear CUDA Graphs after OOM.", exc_info=True)
    try:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
    except Exception:
        logger.debug("CUDA cache cleanup failed after OOM.", exc_info=True)


def _generation_config(args: argparse.Namespace, num_step: int) -> Any:
    from omnivoice import OmniVoiceGenerationConfig

    return OmniVoiceGenerationConfig(
        num_step=num_step,
        guidance_scale=args.guidance_scale,
        class_temperature=args.class_temperature,
    )


def _generate_design(
    runner: Any,
    args: argparse.Namespace,
    *,
    text_case: TextCase,
    profile: VoiceProfile,
    num_step: int,
) -> Any:
    language_hint = LANGUAGE_HINT_OVERRIDES.get(text_case.language, text_case.language)
    kwargs: dict[str, Any] = {
        "text": text_case.text,
        "generation_config": _generation_config(args, num_step),
    }
    if language_hint is not None:
        kwargs["language"] = language_hint
    if profile.instruct:
        kwargs["instruct"] = profile.instruct
    if args.speed is not None and args.speed != 1.0:
        kwargs["speed"] = args.speed
    if args.duration is not None and args.duration > 0:
        kwargs["duration"] = args.duration
    return runner.model.generate(**kwargs)


def _generate_clone(
    runner: Any,
    args: argparse.Namespace,
    *,
    text_case: TextCase,
    gemini_row: dict[str, Any],
    num_step: int,
) -> Any:
    prompt = runner.model.create_voice_clone_prompt(
        ref_audio=gemini_row["reference_audio"],
        ref_text=gemini_row.get("sentence") or None,
    )
    language_hint = LANGUAGE_HINT_OVERRIDES.get(text_case.language, text_case.language)
    kwargs: dict[str, Any] = {
        "text": text_case.text,
        "voice_clone_prompt": prompt,
        "generation_config": _generation_config(args, num_step),
    }
    if language_hint is not None:
        kwargs["language"] = language_hint
    if args.speed is not None and args.speed != 1.0:
        kwargs["speed"] = args.speed
    if args.duration is not None and args.duration > 0:
        kwargs["duration"] = args.duration
    return runner.model.generate(**kwargs)


def _planned_design_rows(
    text_cases: list[TextCase],
    profiles: list[VoiceProfile],
    num_steps: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for text_case in text_cases:
        for profile in profiles:
            if profile.source == "official_accent" and text_case.language != "en":
                continue
            if profile.source == "official_dialect" and text_case.language != "zh":
                continue
            for num_step in num_steps:
                rows.append(
                    {
                        "mode": "design",
                        "language": text_case.language,
                        "text_case": text_case.label,
                        "profile": profile.label,
                        "num_step": num_step,
                        "instruct": profile.instruct,
                    }
                )
    return rows


def _write_instruct_catalog(path: Path, profiles: list[VoiceProfile]) -> None:
    official_tags = {
        "gender": ["male", "female"],
        "age": ["child", "teenager", "young adult", "middle-aged", "elderly"],
        "pitch": [
            "very low pitch",
            "low pitch",
            "moderate pitch",
            "high pitch",
            "very high pitch",
        ],
        "style": ["whisper"],
        "english_accent": [
            "american accent",
            "british accent",
            "australian accent",
            "chinese accent",
            "canadian accent",
            "indian accent",
            "korean accent",
            "portuguese accent",
            "russian accent",
            "japanese accent",
        ],
        "chinese_dialect": [
            "河南话",
            "陕西话",
            "四川话",
            "贵州话",
            "云南话",
            "桂林话",
            "济南话",
            "石家庄话",
            "甘肃话",
            "宁夏话",
            "青岛话",
            "东北话",
        ],
    }
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_notes": [
            "Official tags were copied from the installed OmniVoice Gradio demo and voice_design utility.",
            "OmniVoice 0.1.4 validates instruct as controlled tags, not open-ended prose.",
            "Exploratory personas keep tutor/persona wording in metadata only; generated instruct strings stay valid.",
            "Deep voice experiments should start with low pitch and very low pitch.",
        ],
        "official_tags": official_tags,
        "profiles": [asdict(profile) for profile in profiles],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _audio_rel(path: str, base_dir: Path) -> str:
    try:
        return Path(path).resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def _write_review_html(path: Path, manifest: dict[str, Any]) -> None:
    base_dir = path.parent
    rows = manifest.get("samples", [])
    options = "\n".join(
        f"<option value='{html.escape(str(v))}'>{html.escape(str(v))}</option>"
        for v in ["", 1, 2, 3, 4, 5]
    )
    body_rows = []
    for row in rows:
        file_path = row.get("file")
        audio = ""
        if file_path and Path(file_path).exists():
            src = html.escape(_audio_rel(file_path, base_dir))
            audio = f"<audio controls preload='none' src='{src}'></audio>"
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(row.get('mode', ''))}</td>"
            f"<td>{html.escape(row.get('language', ''))}</td>"
            f"<td>{html.escape(row.get('text_case', ''))}</td>"
            f"<td>{html.escape(row.get('profile', row.get('voice_name', '')))}</td>"
            f"<td>{html.escape(str(row.get('num_step', '')))}</td>"
            f"<td>{html.escape(row.get('instruct') or row.get('voice_persona') or '')}</td>"
            f"<td>{audio}</td>"
            f"<td><select>{options}</select></td>"
            f"<td><select>{options}</select></td>"
            f"<td><select>{options}</select></td>"
            "<td><input type='checkbox'></td>"
            "<td><input type='text' placeholder='notes'></td>"
            "</tr>"
        )
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>OmniVoice Default Voice Sweep Review</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f5f5f5; z-index: 1; }}
    audio {{ width: 260px; }}
    input[type="text"] {{ width: 220px; }}
    .meta {{ color: #555; margin-bottom: 18px; }}
  </style>
</head>
<body>
  <h1>OmniVoice Default Voice Sweep Review</h1>
  <div class="meta">
    runner={html.escape(manifest['config']['runner'])},
    dtype={html.escape(manifest['config']['dtype'])},
    profile_set={html.escape(manifest['config']['profile_set'])},
    samples={len(rows)}
  </div>
  <table>
    <thead>
      <tr>
        <th>Mode</th><th>Lang</th><th>Text Case</th><th>Profile/Voice</th>
        <th>Step</th><th>Instruct/Persona</th><th>Audio</th>
        <th>Natural</th><th>Accent</th><th>Tutor Fit</th><th>Ship?</th><th>Notes</th>
      </tr>
    </thead>
    <tbody>
      {''.join(body_rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(content, encoding="utf-8")


def _write_readme(path: Path, manifest: dict[str, Any]) -> None:
    cfg = manifest["config"]
    lines = [
        "# OmniVoice Default Voice Sweep",
        "",
        "This run compares OmniVoice default/design voices against optional Gemini clone baselines.",
        "",
        "## Config",
        "",
        f"- runner: `{cfg['runner']}`",
        f"- dtype: `{cfg['dtype']}`",
        f"- profile_set: `{cfg['profile_set']}`",
        f"- num_steps: `{cfg['num_steps']}`",
        f"- languages: `{cfg['languages']}`",
        f"- include_gemini_clones: `{cfg['include_gemini_clones']}`",
        "",
        "## Files",
        "",
        "- `manifest.json`: all generated sample metadata",
        "- `instruct_catalog.json`: official and exploratory voice-design instructions",
        "- `review.html`: audio-player review sheet with manual rating columns",
        "",
        "## Notes",
        "",
        "- Official OmniVoice voice-design tags include gender, age, pitch, whisper, English accent, and Chinese dialect.",
        "- Deep voice experiments use `low pitch` and `very low pitch`.",
        "- Exploratory profiles keep tutor/persona words in metadata, because OmniVoice 0.1.4 rejects unsupported prose inside `instruct`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.dtype != "fp16":
        logger.warning("Lingua comparison requested fp16; current dtype=%s", args.dtype)

    run_id = args.run_id or _utc_stamp()
    run_dir = args.output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    languages = _split_csv(args.languages)
    text_cases = _build_text_cases(languages, set(args.text_case) if args.text_case else None)
    profiles = PROFILE_SETS[args.profile_set]
    num_steps = sorted({args.num_step, *args.also_num_step})
    planned_design = (
        [] if args.skip_design else _planned_design_rows(text_cases, profiles, num_steps)
    )

    gemini_rows: list[dict[str, Any]] = []
    if args.include_gemini_clones:
        gemini_rows = _matching_gemini_rows(
            args.gemini_manifest,
            set(languages),
            {voice.lower() for voice in _split_csv(args.gemini_voices)},
        )

    _write_instruct_catalog(run_dir / "instruct_catalog.json", profiles)

    manifest: dict[str, Any] = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "dry_run": args.dry_run,
        "config": {
            "runner": args.runner,
            "model": args.model,
            "dtype": args.dtype,
            "device": args.device,
            "languages": languages,
            "profile_set": args.profile_set,
            "num_steps": num_steps,
            "guidance_scale": args.guidance_scale,
            "class_temperature": args.class_temperature,
            "speed": args.speed,
            "duration": args.duration,
            "include_gemini_clones": args.include_gemini_clones,
            "skip_design": args.skip_design,
            "gemini_voices": _split_csv(args.gemini_voices),
            "seed": args.seed,
            "max_cuda_graphs": args.max_cuda_graphs,
        },
        "planned": {
            "design_samples": len(planned_design),
            "gemini_clone_references": len(gemini_rows),
        },
        "samples": [],
        "errors": [],
    }

    if args.dry_run:
        manifest["planned"]["design_preview"] = planned_design[:50]
        manifest["planned"]["gemini_clone_preview"] = [
            {
                "lang_code": row.get("lang_code"),
                "voice_name": row.get("voice_name"),
                "reference_audio": row.get("reference_audio"),
            }
            for row in gemini_rows[:50]
        ]
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _write_review_html(run_dir / "review.html", manifest)
        _write_readme(run_dir / "README.md", manifest)
        logger.info("Dry run written to %s", run_dir)
        return

    runner = _create_runner(args)
    generated = 0

    try:
        if not args.skip_design:
            for text_case in text_cases:
                for profile in profiles:
                    if (
                        profile.source == "official_accent"
                        and text_case.language != "en"
                    ):
                        continue
                    if (
                        profile.source == "official_dialect"
                        and text_case.language != "zh"
                    ):
                        continue
                    for num_step in num_steps:
                        if args.max_samples is not None and generated >= args.max_samples:
                            break
                        filename = (
                            f"{text_case.language}/design/"
                            f"{text_case.label}_{profile.label}_step{num_step}.wav"
                        )
                        out_path = run_dir / filename
                        if out_path.exists() and not args.overwrite:
                            logger.info("Skipping existing %s", out_path)
                            continue
                        try:
                            _set_seed(args.seed)
                            started = time.perf_counter()
                            audio = _generate_design(
                                runner,
                                args,
                                text_case=text_case,
                                profile=profile,
                                num_step=num_step,
                            )
                            elapsed = time.perf_counter() - started
                            audio_info = _save_audio(audio, out_path, DEFAULT_SAMPLE_RATE)
                            row = {
                                "mode": "design",
                                "language": text_case.language,
                                "text_case": text_case.label,
                                "text": text_case.text,
                                "profile": profile.label,
                                "instruct": profile.instruct,
                                "profile_source": profile.source,
                                "num_step": num_step,
                                "dtype": args.dtype,
                                "file": str(out_path),
                                "generation_time_s": round(elapsed, 3),
                                **audio_info,
                                **{
                                    key: value
                                    for key, value in asdict(profile).items()
                                    if key
                                    not in {
                                        "label",
                                        "instruct",
                                        "source",
                                    }
                                },
                            }
                            manifest["samples"].append(row)
                            generated += 1
                            logger.info("Generated %s", out_path)
                        except Exception as exc:
                            logger.exception("Failed design sample %s", out_path)
                            _cleanup_cuda_after_exception(runner, exc)
                            manifest["errors"].append(
                                {
                                    "mode": "design",
                                    "language": text_case.language,
                                    "text_case": text_case.label,
                                    "profile": profile.label,
                                    "num_step": num_step,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                    if args.max_samples is not None and generated >= args.max_samples:
                        break
                if args.max_samples is not None and generated >= args.max_samples:
                    break

        if args.include_gemini_clones and (
            args.max_samples is None or generated < args.max_samples
        ):
            rows_by_lang: dict[str, list[dict[str, Any]]] = {}
            for row in gemini_rows:
                rows_by_lang.setdefault(str(row.get("lang_code")), []).append(row)

            for text_case in text_cases:
                for gemini_row in rows_by_lang.get(text_case.language, []):
                    for num_step in num_steps:
                        if args.max_samples is not None and generated >= args.max_samples:
                            break
                        voice_name = str(gemini_row.get("voice_name", "voice"))
                        filename = (
                            f"{text_case.language}/clone/"
                            f"{text_case.label}_{_slug(voice_name)}_step{num_step}.wav"
                        )
                        out_path = run_dir / filename
                        if out_path.exists() and not args.overwrite:
                            logger.info("Skipping existing %s", out_path)
                            continue
                        try:
                            _set_seed(args.seed)
                            started = time.perf_counter()
                            audio = _generate_clone(
                                runner,
                                args,
                                text_case=text_case,
                                gemini_row=gemini_row,
                                num_step=num_step,
                            )
                            elapsed = time.perf_counter() - started
                            audio_info = _save_audio(audio, out_path, DEFAULT_SAMPLE_RATE)
                            manifest["samples"].append(
                                {
                                    "mode": "clone",
                                    "language": text_case.language,
                                    "text_case": text_case.label,
                                    "text": text_case.text,
                                    "voice_name": voice_name,
                                    "voice_gender": gemini_row.get("voice_gender"),
                                    "voice_persona": gemini_row.get("voice_persona"),
                                    "reference_audio": gemini_row.get("reference_audio"),
                                    "reference_text": gemini_row.get("sentence"),
                                    "num_step": num_step,
                                    "dtype": args.dtype,
                                    "file": str(out_path),
                                    "generation_time_s": round(elapsed, 3),
                                    **audio_info,
                                }
                            )
                            generated += 1
                            logger.info("Generated %s", out_path)
                        except Exception as exc:
                            logger.exception("Failed clone sample %s", out_path)
                            _cleanup_cuda_after_exception(runner, exc)
                            manifest["errors"].append(
                                {
                                    "mode": "clone",
                                    "language": text_case.language,
                                    "text_case": text_case.label,
                                    "voice_name": voice_name,
                                    "num_step": num_step,
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                            )
                    if args.max_samples is not None and generated >= args.max_samples:
                        break
                if args.max_samples is not None and generated >= args.max_samples:
                    break
    finally:
        unload = getattr(runner, "unload_model", None)
        if callable(unload):
            unload()

    manifest["completed_at"] = datetime.now(UTC).isoformat()
    manifest["generated_samples"] = generated
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_review_html(run_dir / "review.html", manifest)
    _write_readme(run_dir / "README.md", manifest)
    logger.info(
        "Wrote %s samples, %s errors to %s",
        generated,
        len(manifest["errors"]),
        run_dir,
    )


if __name__ == "__main__":
    main()
