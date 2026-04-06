"""Streamlit dashboard for OmniVoice Triton optimization modes.

Run with: uv run streamlit run ui/app.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path so `from ui.xxx` imports work
# regardless of how Streamlit is launched.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import streamlit as st  # noqa: E402

from ui.i18n import t  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402
from ui.tab_benchmarks import render_benchmarks_tab  # noqa: E402
from ui.tab_inference import render_inference_tab  # noqa: E402
from ui.tab_overview import render_overview_tab  # noqa: E402
from ui.tab_samples import render_samples_tab  # noqa: E402
from ui.tab_verification import render_verification_tab  # noqa: E402


def main() -> None:
    """Streamlit app entry point."""
    st.set_page_config(
        page_title=t("app.page_title"),
        page_icon=":sound:",
        layout="wide",
    )
    st.title(t("app.title"))

    render_sidebar()

    (
        tab_overview,
        tab_inference,
        tab_bench,
        tab_samples,
        tab_verify,
    ) = st.tabs(
        [
            t("app.tab_overview"),
            t("app.tab_inference"),
            t("app.tab_benchmarks"),
            t("app.tab_samples"),
            t("app.tab_verification"),
        ]
    )

    with tab_overview:
        render_overview_tab()

    with tab_inference:
        render_inference_tab()

    with tab_bench:
        render_benchmarks_tab()

    with tab_samples:
        render_samples_tab()

    with tab_verify:
        render_verification_tab()


if __name__ == "__main__":
    main()
