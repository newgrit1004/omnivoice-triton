"""Sidebar controls for the OmniVoice Triton Streamlit dashboard.

Renders project info, GPU info toggle, and UI language selector.
"""

import streamlit as st

from ui.gpu_info import get_gpu_info
from ui.i18n import SUPPORTED_UI_LANGS, get_i18n, t


def render_sidebar() -> None:
    """Render sidebar with project info, GPU toggle, and language selector."""
    st.sidebar.title(t("sidebar.title"))
    st.sidebar.caption(t("sidebar.subtitle"))
    st.sidebar.markdown("---")

    # Language selector
    ui_lang = st.sidebar.selectbox(
        "\U0001f310 UI Language",
        list(SUPPORTED_UI_LANGS.keys()),
        format_func=lambda x: SUPPORTED_UI_LANGS[x],
        key="ui_lang_select",
    )
    get_i18n().set_language(ui_lang)

    st.sidebar.markdown("---")

    # GPU info toggle
    show_gpu = st.sidebar.toggle(t("sidebar.show_gpu_info"), value=False)
    if show_gpu:
        _render_gpu_quick_info()

    st.sidebar.markdown("---")

    # Project links
    st.sidebar.markdown(t("sidebar.links_header"))
    st.sidebar.markdown(
        "- [GitHub](https://github.com/newgrit1004/omnivoice-triton)\n"
        "- [OmniVoice](https://github.com/AIG-AudioLab/OmniVoice)"
    )


def _render_gpu_quick_info() -> None:
    """Render compact GPU info in the sidebar."""
    info = get_gpu_info()
    st.sidebar.markdown(f"**GPU:** {info['name']}")
    total = info["total_vram_gb"]
    used = info["used_vram_gb"]
    if total > 0:
        st.sidebar.progress(
            used / total,
            text=f"VRAM: {used:.1f}/{total:.1f} GB",
        )
    else:
        st.sidebar.caption(t("sidebar.no_gpu"))
    if info["temperature_c"] is not None:
        st.sidebar.caption(f"Temp: {info['temperature_c']}°C")
    if info["utilization_pct"] is not None:
        st.sidebar.caption(f"Util: {info['utilization_pct']}%")
