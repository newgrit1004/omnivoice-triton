"""Benchmarks tab for the OmniVoice Triton Streamlit dashboard.

Two sections:
  A) Pre-computed E2E benchmarks (e2e_benchmarks.json)
  B) Kernel micro-benchmarks (kernel_benchmarks.json)
"""

import logging
from pathlib import Path
from typing import Any

import streamlit as st

from ui.charts import (
    render_e2e_chart,
    render_kernel_speedup_chart,
    render_kernel_speedup_ratio_chart,
    render_rtf_chart,
)
from ui.i18n import t
from ui.utils import load_json_list

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "results"


def render_benchmarks_tab() -> None:
    """Render the Benchmarks tab with E2E and kernel sections."""
    _render_e2e_benchmarks()
    st.markdown("---")
    _render_kernel_benchmarks()


def _render_e2e_benchmarks() -> None:
    """Render pre-computed E2E benchmarks from e2e_benchmarks.json."""
    st.subheader(t("benchmarks.e2e_title"))

    source_path = RESULTS_DIR / "e2e_benchmarks.json"
    data = load_json_list(source_path)

    if not data:
        st.info(t("benchmarks.no_e2e"))
        return

    st.caption("`benchmark/results/e2e_benchmarks.json`")

    # Build table rows
    rows: list[dict[str, Any]] = []
    for entry in data:
        tm = entry["time_ms"]
        rt = entry["rtf"]
        load_time = entry.get("model_load_time_s")
        load_str = f"{load_time:.1f}" if load_time is not None else "-"
        rows.append(
            {
                t("table.runner"): entry["runner"],
                t("table.language"): entry["language"],
                t("table.model_load_time"): load_str,
                t("table.mean_ms"): f"{tm['mean']:.0f}",
                t("table.std_ms"): f"{tm['std']:.0f}",
                t("table.p50_ms"): f"{tm['p50']:.0f}",
                t("table.p95_ms"): f"{tm['p95']:.0f}",
                t("table.rtf_mean"): f"{rt['mean']:.2f}",
                t("table.peak_vram"): f"{entry['peak_vram_gb']:.3f}",
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Charts
    tab_latency, tab_rtf = st.tabs(
        [t("benchmarks.e2e_title"), t("benchmarks.rtf_title")]
    )
    with tab_latency:
        render_e2e_chart(data)
    with tab_rtf:
        render_rtf_chart(data)


def _render_kernel_benchmarks() -> None:
    """Render kernel micro-benchmark results from kernel_benchmarks.json."""
    st.subheader(t("benchmarks.kernel_title"))

    source_path = RESULTS_DIR / "kernel_benchmarks.json"
    results = load_json_list(source_path)

    if not results:
        st.info(t("benchmarks.no_kernels"))
        return

    st.caption("`benchmark/results/kernel_benchmarks.json`")

    # Summary table
    rows: list[dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                t("table.kernel"): r["kernel"],
                t("table.pytorch_ms"): f"{r.get('pytorch_ms', 0) * 1000:.3f}",
                t("table.triton_ms"): f"{r.get('triton_ms', 0) * 1000:.3f}",
                t("table.speedup"): f"{r.get('speedup', 0):.2f}x",
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)

    # Charts
    tab_latency, tab_speedup = st.tabs(
        [t("benchmarks.kernel_title"), t("benchmarks.kernel_speedup_title")]
    )
    with tab_latency:
        render_kernel_speedup_chart(results)
    with tab_speedup:
        render_kernel_speedup_ratio_chart(results)
