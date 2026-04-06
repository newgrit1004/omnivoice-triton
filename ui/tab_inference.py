"""Inference tab for the OmniVoice Triton Streamlit dashboard.

Supports 3 generation modes:
  - Basic: standard text-to-speech
  - Clone: voice cloning from reference audio
  - Design: voice design via instruct text
"""

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

import streamlit as st

from ui.i18n import t
from ui.utils import calculate_rtf, reset_vram_stats

logger = logging.getLogger(__name__)

_DISPLAY_TO_KEY: dict[str, str] = {
    "Base": "base",
    "Triton": "triton",
    "Faster": "faster",
    "Hybrid": "hybrid",
}

ALL_RUNNER_NAMES = ["Base", "Triton", "Faster", "Hybrid"]

GEN_MODE_BASIC = "basic"
GEN_MODE_CLONE = "clone"
GEN_MODE_DESIGN = "design"

SAMPLE_TEXTS = {
    "en": "Hello, this is an OmniVoice Triton demo. Enjoy seamless TTS synthesis.",
    "ko": "안녕하세요, OmniVoice Triton 데모입니다. 자연스러운 음성 합성을 즐겨보세요.",
    "zh": "你好，这是 OmniVoice Triton 的演示。体验流畅的文本转语音合成。",
    "ja": "こんにちは、OmniVoice Triton のデモです。",
}


def render_inference_tab() -> None:
    """Render the Inference tab with generation mode selection and audio playback."""
    # Controls
    gen_mode, text, instruct, ref_audio_bytes, ref_text, selected_runner = (
        _render_controls()
    )

    # Generate button
    if st.button(t("inference.run"), type="primary", use_container_width=True):
        if not text.strip():
            st.warning("Please enter some text to synthesize.")
            return
        if gen_mode == GEN_MODE_CLONE and ref_audio_bytes is None:
            st.warning(t("inference.upload_ref_audio"))
            return
        st.toast(t("inference.toast_started"))
        _run_inference(
            selected_runner, text, gen_mode, instruct, ref_audio_bytes, ref_text
        )

    st.markdown("---")

    result = st.session_state.get("inference_result")
    if result:
        _display_result(result, selected_runner, gen_mode)
    else:
        st.info(t("inference.prompt"))


def _render_controls() -> tuple[str, str, str, bytes | None, str, str]:
    """Render inference controls and return user inputs.

    Returns:
        Tuple of (gen_mode, text, instruct, ref_audio_bytes, ref_text, runner_name).
    """
    col_left, col_right = st.columns([3, 1])

    with col_right:
        runner_display_names = ALL_RUNNER_NAMES
        selected_runner = st.selectbox(
            t("inference.runner_label"),
            runner_display_names,
            index=3,  # default to Hybrid
            key="runner_select",
        )

        gen_mode_labels = {
            GEN_MODE_BASIC: t("inference.gen_mode_basic"),
            GEN_MODE_CLONE: t("inference.gen_mode_clone"),
            GEN_MODE_DESIGN: t("inference.gen_mode_design"),
        }
        gen_mode = st.radio(
            t("inference.gen_mode_label"),
            list(gen_mode_labels.keys()),
            format_func=lambda x: gen_mode_labels[x],
            key="gen_mode_radio",
        )

    with col_left:
        if "inference_text" not in st.session_state:
            st.session_state["inference_text"] = SAMPLE_TEXTS["en"]

        text = st.text_area(
            t("inference.text_label"),
            height=120,
            placeholder=t("inference.text_placeholder"),
            key="inference_text",
        )

        instruct = ""
        ref_audio_bytes: bytes | None = None
        ref_text = ""

        if gen_mode == GEN_MODE_DESIGN:
            instruct = st.text_area(
                t("inference.instruct_label"),
                height=80,
                placeholder=t("inference.instruct_placeholder"),
                key="instruct_text",
            )

        elif gen_mode == GEN_MODE_CLONE:
            uploaded = st.file_uploader(
                t("inference.ref_audio_label"),
                type=["wav", "mp3", "flac", "ogg"],
                key="ref_audio_uploader",
            )
            if uploaded is not None:
                ref_audio_bytes = uploaded.read()
                st.audio(ref_audio_bytes)

            ref_text = st.text_input(
                t("inference.ref_text_label"),
                placeholder=t("inference.ref_text_placeholder"),
                key="ref_text_input",
            )

    return gen_mode, text, instruct, ref_audio_bytes, ref_text, selected_runner


def _run_inference(
    runner_name: str,
    text: str,
    gen_mode: str,
    instruct: str,
    ref_audio_bytes: bytes | None,
    ref_text: str,
) -> None:
    """Execute inference with the selected runner and generation mode."""
    key = _DISPLAY_TO_KEY.get(runner_name, runner_name.lower())

    with st.status(
        t("inference.status_running", name=runner_name),
        expanded=True,
    ) as status:
        runner = _get_runner(key)
        if runner is None:
            st.error(t("inference.unavailable", name=runner_name))
            status.update(
                label=t("inference.status_error", name=runner_name),
                state="error",
            )
            return

        try:
            if key in ("faster", "hybrid"):
                st.write(t("inference.loading_warmup", name=runner_name))
            else:
                st.write(t("inference.loading_model", name=runner_name))

            reset_vram_stats()
            t_load = time.perf_counter()
            runner.load_model()
            load_s = time.perf_counter() - t_load

            st.write(t("inference.generating"))
            result = _generate(
                runner, text, gen_mode, instruct, ref_audio_bytes, ref_text
            )
            result["load_s"] = round(load_s, 2)
            result["runner"] = runner_name
            result["gen_mode"] = gen_mode

            elapsed = result.get("total_s", 0)
            rtf = result.get("rtf", 0)
            st.write(
                t(
                    "inference.status_done_detail",
                    time=f"{elapsed:.1f}",
                    rtf=f"{rtf:.2f}",
                )
            )
            status.update(
                label=t(
                    "inference.status_done", name=runner_name, time=f"{elapsed:.1f}"
                ),
                state="complete",
                expanded=False,
            )
            st.toast(t("inference.toast_done", name=runner_name, time=f"{elapsed:.1f}"))
            st.session_state["inference_result"] = result

        except Exception as exc:
            logger.exception("Inference failed for runner %s", runner_name)
            st.write(str(exc))
            status.update(
                label=t("inference.status_error", name=runner_name),
                state="error",
                expanded=True,
            )
            st.session_state["inference_result"] = {"error": str(exc)}
        finally:
            try:
                runner.unload_model()
            except Exception:
                pass
            reset_vram_stats()


def _display_result(result: dict[str, Any], runner_name: str, gen_mode: str) -> None:
    """Display the inference result with audio player and metrics."""
    st.subheader(f"{runner_name} — {gen_mode}")

    if result.get("error"):
        st.error(result["error"])
        return

    audio = result.get("audio")
    sr = result.get("sample_rate", 24000)
    if audio is not None:
        st.audio(audio, sample_rate=sr)
    else:
        st.warning("No audio generated.")
        return

    col1, col2, col3, col4 = st.columns(4)
    total_s = result.get("total_s", 0)
    load_s = result.get("load_s", 0)
    rtf = result.get("rtf", 0)
    peak_vram = result.get("peak_vram_gb", 0)

    col1.metric("Total Time", f"{total_s:.2f}s")
    col2.metric("Model Load", f"{load_s:.2f}s")
    col3.metric("RTF", f"{rtf:.2f}x", help="Real-Time Factor: higher is better")
    col4.metric("Peak VRAM", f"{peak_vram:.2f} GB")


def _get_runner(key: str) -> Any:
    """Import and instantiate a runner by key name."""
    try:
        from omnivoice_triton import create_runner

        return create_runner(key)
    except (ImportError, Exception) as exc:
        logger.warning("Failed to create runner '%s': %s", key, exc)
        return None


def _generate(
    runner: Any,
    text: str,
    gen_mode: str,
    instruct: str,
    ref_audio_bytes: bytes | None,
    ref_text: str,
) -> dict[str, Any]:
    """Run generation for the given mode (model must already be loaded)."""
    t_start = time.perf_counter()

    if gen_mode == GEN_MODE_CLONE and ref_audio_bytes is not None:
        output = _generate_voice_clone(runner, text, ref_audio_bytes, ref_text)
    elif gen_mode == GEN_MODE_DESIGN:
        output = runner.generate_voice_design(text=text, instruct=instruct)
    else:
        output = runner.generate(text=text)

    t_total = time.perf_counter() - t_start

    audio = output.get("audio")
    sr = output.get("sample_rate", 24000)
    peak_vram = output.get("peak_vram_gb", 0.0)
    audio_len = len(audio) if audio is not None else 0
    rtf = calculate_rtf(audio_len, sr, t_total)

    return {
        "audio": audio,
        "sample_rate": sr,
        "rtf": rtf,
        "total_s": round(t_total, 3),
        "peak_vram_gb": peak_vram,
    }


def _generate_voice_clone(
    runner: Any,
    text: str,
    ref_audio_bytes: bytes,
    ref_text: str,
) -> dict[str, Any]:
    """Save ref audio bytes to a temp file and call generate_voice_clone."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(ref_audio_bytes)
        tmp_path = Path(tmp.name)

    try:
        kwargs: dict[str, Any] = {"text": text, "ref_audio": tmp_path}
        if ref_text:
            kwargs["ref_text"] = ref_text
        return runner.generate_voice_clone(**kwargs)
    finally:
        tmp_path.unlink(missing_ok=True)
