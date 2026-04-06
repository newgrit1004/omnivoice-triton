"""Overview tab for the OmniVoice Triton Streamlit dashboard.

Displays project description, runner info cards, kernel info,
GPU info, verification summary badges, and quick benchmark summary.
"""

import logging
from pathlib import Path
from typing import Any

import streamlit as st

from ui.gpu_info import get_gpu_info
from ui.i18n import t
from ui.utils import load_json_dict, load_json_list

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "results"

_RUNNER_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "Base": ("Base", "overview.runner_base"),
    "Triton": ("Triton", "overview.runner_triton"),
    "Faster": ("Faster", "overview.runner_faster"),
    "Hybrid": ("Hybrid (Triton+CUDA Graph)", "overview.runner_hybrid"),
}

_KERNEL_DESCRIPTIONS: list[tuple[str, str]] = [
    ("RMSNorm", "overview.kernel_rmsnorm"),
    ("SwiGLU", "overview.kernel_swiglu"),
    ("FusedAddRMSNorm", "overview.kernel_fusedaddrmsnorm"),
]


def render_overview_tab() -> None:
    """Render the Overview tab."""
    _render_project_description()
    st.markdown("---")
    _render_runners_and_kernels()
    st.markdown("---")
    _render_gpu_details()
    st.markdown("---")
    _render_verification_summary()
    st.markdown("---")
    _render_quick_benchmarks()


def _render_project_description() -> None:
    """Render project title and description."""
    st.subheader(t("overview.title"))
    st.markdown(t("overview.description"))

    col1, col2, col3 = st.columns(3)
    col1.metric("Runners", "4", help="Base, Triton, Faster, Hybrid")
    col2.metric("Triton Kernels", "3", help="RMSNorm, SwiGLU, FusedAddRMSNorm")
    col3.metric("Max Speedup", "~2.8x", help="Hybrid runner vs Base")

    st.markdown(
        t("overview.architecture_title") + ": " + t("overview.architecture_desc")
    )

    st.markdown("**" + t("overview.gen_modes_title") + "**")
    st.markdown(t("overview.gen_mode_basic"))
    st.markdown(t("overview.gen_mode_clone"))
    st.markdown(t("overview.gen_mode_design"))


def _render_runners_and_kernels() -> None:
    """Render runner and kernel info cards side by side."""
    col_runners, col_kernels = st.columns(2)

    with col_runners:
        st.subheader(t("overview.runners_title"))
        for display_name, (label, desc_key) in _RUNNER_DESCRIPTIONS.items():
            with st.container(border=True):
                st.markdown(f"**{label}**")
                st.caption(t(desc_key))

    with col_kernels:
        st.subheader(t("overview.kernels_title"))
        for kernel_name, desc_key in _KERNEL_DESCRIPTIONS:
            with st.container(border=True):
                st.markdown(f"**{kernel_name}**")
                st.caption(t(desc_key))


def _render_gpu_details() -> None:
    """Render GPU information card."""
    st.subheader(t("overview.gpu_info"))
    info = get_gpu_info()

    col1, col2, col3 = st.columns(3)
    col1.metric(t("overview.gpu"), info["name"])
    col2.metric(t("overview.driver"), info["driver_version"])

    total = info["total_vram_gb"]
    used = info["used_vram_gb"]
    free = info["free_vram_gb"]
    col3.metric(
        t("overview.vram"),
        f"{used:.1f} / {total:.1f} GB",
        delta=t("overview.vram_free", free=f"{free:.1f}"),
    )

    if total > 0:
        st.progress(
            used / total,
            text=t("overview.vram_usage", used=f"{used:.1f}", total=f"{total:.1f}"),
        )
    else:
        st.info(t("overview.no_gpu"))

    extras: list[str] = []
    if info["temperature_c"] is not None:
        extras.append(t("overview.temperature", temp=info["temperature_c"]))
    if info["utilization_pct"] is not None:
        extras.append(t("overview.utilization", pct=info["utilization_pct"]))
    if extras:
        st.caption(" | ".join(extras))


def _render_verification_summary() -> None:
    """Render compact 3-Tier verification badges."""
    st.subheader(t("overview.verification"))

    report = load_json_dict(RESULTS_DIR / "verification_report.json")
    if not report:
        st.info(t("overview.no_verification"))
        return

    timestamp = report.get("timestamp", "")
    if timestamp:
        st.caption(t("overview.last_run", timestamp=timestamp))

    col1, col2, col3 = st.columns(3)

    tier1 = report.get("tier1")
    if tier1:
        passed = tier1.get("passed", 0)
        total = tier1.get("total", 0)
        badge = _badge(tier1.get("status"))
        col1.metric(t("overview.tier1"), f"{badge} {passed}/{total}")
    else:
        col1.metric(t("overview.tier1"), t("overview.not_run"))

    tier2 = report.get("tier2")
    if tier2:
        layers = tier2.get("layers", {})
        if layers:
            min_cos = min(v["cosine_sim"] for v in layers.values())
            badge = _badge(tier2.get("status"))
            col2.metric(t("overview.tier2"), f"{badge} min={min_cos:.4f}")
        else:
            pairs = tier2.get("pairs", {})
            badge = _badge(tier2.get("status"))
            col2.metric(t("overview.tier2"), f"{badge} {len(pairs)} pairs")
    else:
        col2.metric(t("overview.tier2"), t("overview.not_run"))

    tier3 = report.get("tier3")
    if tier3:
        badge = _badge(tier3.get("overall_verdict", tier3.get("status")))
        col3.metric(t("overview.tier3"), badge)
    else:
        col3.metric(t("overview.tier3"), t("overview.not_run"))


def _render_quick_benchmarks() -> None:
    """Render quick benchmark summary from e2e_benchmarks.json."""
    st.subheader(t("overview.bench_summary"))

    data = load_json_list(RESULTS_DIR / "e2e_benchmarks.json")
    if not data:
        st.info(t("overview.no_e2e"))
        return

    # Find fastest runner by mean latency
    base_mean: float | None = None
    fastest_name: str | None = None
    fastest_mean = float("inf")

    for entry in data:
        name = entry["runner"]
        mean_ms = entry["time_ms"]["mean"]
        if name == "Base" and base_mean is None:
            base_mean = mean_ms
        if mean_ms < fastest_mean:
            fastest_mean = mean_ms
            fastest_name = name

    if fastest_name and base_mean and fastest_mean > 0:
        speedup = base_mean / fastest_mean
        st.success(t("overview.fastest", name=fastest_name, speedup=f"{speedup:.1f}"))

    rows: list[dict[str, Any]] = []
    for entry in data:
        load_time = entry.get("model_load_time_s")
        load_str = f"{load_time:.1f}" if load_time is not None else "-"
        rows.append(
            {
                t("table.runner"): entry["runner"],
                t("table.language"): entry["language"],
                t("table.model_load_time"): load_str,
                t("table.mean_ms"): f"{entry['time_ms']['mean']:.0f}",
                t("table.rtf_mean"): f"{entry['rtf']['mean']:.2f}",
                t("table.peak_vram"): f"{entry['peak_vram_gb']:.3f}",
            }
        )

    st.dataframe(rows, use_container_width=True, hide_index=True)


def _badge(status: str | None) -> str:
    """Convert status to a text badge."""
    if status == "PASS":
        return "PASS"
    if status == "FAIL":
        return "FAIL"
    return "PENDING"
