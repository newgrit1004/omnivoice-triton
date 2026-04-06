"""Audio Samples tab for the OmniVoice Triton Streamlit dashboard.

Loads pre-generated WAV files from assets/audio_samples/ with optional
metadata and displays them in a filterable, side-by-side layout.
"""

import logging
from pathlib import Path
from typing import Any

import streamlit as st

from ui.i18n import t
from ui.utils import load_json_dict

logger = logging.getLogger(__name__)

_SAMPLES_DIR = Path(__file__).resolve().parent.parent / "assets" / "audio_samples"
_METADATA_PATH = _SAMPLES_DIR / "metadata.json"

_MODES = ["base", "triton", "faster", "hybrid"]
_MODE_LABELS: dict[str, str] = {
    "base": "Base (PyTorch)",
    "triton": "Triton",
    "faster": "Faster (CUDA Graph)",
    "hybrid": "Hybrid (Triton+CUDA Graph)",
}

_GEN_TYPE_LABELS: dict[str, str] = {
    "basic": "samples.type_basic",
    "clone": "samples.type_clone",
    "design": "samples.type_design",
}


def render_samples_tab() -> None:
    """Render the Audio Samples tab."""
    metadata = load_json_dict(_METADATA_PATH)

    if metadata is None:
        # Try to scan directory directly if no metadata
        _render_directory_fallback()
        return

    samples = metadata.get("samples", [])
    if not samples:
        st.info(t("samples.no_data"))
        return

    # Filters
    col_lang, col_type = st.columns(2)
    with col_lang:
        lang_options = ["All"] + sorted(
            {s.get("language_name", s.get("language", "")) for s in samples}
        )
        lang_filter = st.selectbox(t("samples.filter_language"), lang_options)

    with col_type:
        type_options = ["All"] + sorted({s.get("type", "basic") for s in samples})
        type_filter = st.selectbox(
            t("samples.filter_type"),
            type_options,
            format_func=lambda x: t(_GEN_TYPE_LABELS.get(x, x))
            if x != "All"
            else "All",
        )

    groups = _group_by_utterance(samples, lang_filter, type_filter)

    if not groups:
        st.warning(t("samples.no_matches"))
        return

    st.caption(t("samples.total_count", count=len(groups)))

    for idx, group in enumerate(groups):
        _render_sample_group(idx, group)


def _render_directory_fallback() -> None:
    """Scan audio_samples/ subdirectories and render any WAV files found."""
    if not _SAMPLES_DIR.exists():
        st.info(t("samples.no_data"))
        return

    audio_files: dict[str, list[Path]] = {}
    for mode in _MODES:
        mode_dir = _SAMPLES_DIR / mode
        if mode_dir.exists():
            files = sorted(mode_dir.glob("*.wav")) + sorted(mode_dir.glob("*.mp3"))
            if files:
                audio_files[mode] = files

    if not audio_files:
        st.info(t("samples.no_data"))
        return

    st.info(
        "No metadata.json found. Showing raw audio files from assets/audio_samples/."
    )

    # Group by filename across modes
    all_names: set[str] = set()
    for files in audio_files.values():
        for f in files:
            all_names.add(f.name)

    for name in sorted(all_names):
        with st.expander(name, expanded=False):
            cols = st.columns(len(_MODES))
            for col, mode in zip(cols, _MODES):
                with col:
                    st.markdown(f"**{_MODE_LABELS[mode]}**")
                    path = _SAMPLES_DIR / mode / name
                    if path.exists():
                        st.audio(str(path))
                    else:
                        st.caption(t("common.file_missing"))


def _render_sample_group(idx: int, group: dict[str, Any]) -> None:
    """Render one utterance with all available modes side by side."""
    text = group["text"]
    lang = group.get("language_name", group.get("language", ""))
    gen_type = group.get("type", "basic")
    type_label = t(_GEN_TYPE_LABELS.get(gen_type, "samples.type_basic"))

    header = f"**{idx + 1}. [{lang}] {type_label}**"
    st.markdown(header)

    with st.expander(f'"{text}"', expanded=idx == 0):
        modes_present = [m for m in _MODES if m in group.get("modes", {})]
        if not modes_present:
            st.caption(t("common.no_data"))
            return

        cols = st.columns(len(modes_present))
        for col, mode in zip(cols, modes_present):
            sample = group["modes"][mode]
            with col:
                st.markdown(f"**{_MODE_LABELS.get(mode, mode)}**")
                audio_path = _SAMPLES_DIR / sample["file"]
                if audio_path.exists():
                    st.audio(str(audio_path))
                else:
                    st.warning(t("common.file_missing"))

                dur = sample.get("duration_s", 0)
                gen_time = sample.get("generation_time_s", 0)
                rtf = dur / gen_time if gen_time > 0 else 0
                if gen_time > 0:
                    st.caption(f"{gen_time:.1f}s | RTF {rtf:.1f}x")


def _group_by_utterance(
    samples: list[dict[str, Any]],
    lang_filter: str,
    type_filter: str,
) -> list[dict[str, Any]]:
    """Group samples by utterance so the same text appears once with all modes."""
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for s in samples:
        lang_name = s.get("language_name", s.get("language", ""))
        gen_type = s.get("type", "basic")

        if lang_filter != "All" and lang_name != lang_filter:
            continue
        if type_filter != "All" and gen_type != type_filter:
            continue

        key = (s["text"], gen_type, s.get("language", ""))
        if key not in groups:
            groups[key] = {
                "text": s["text"],
                "type": gen_type,
                "language": s.get("language", ""),
                "language_name": lang_name,
                "modes": {},
            }
        mode = s.get("mode", "base")
        groups[key]["modes"][mode] = s

    return list(groups.values())
