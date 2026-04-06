"""Plotly chart rendering functions for the Streamlit dashboard.

All chart functions handle plotly import failure gracefully with
st.warning fallbacks.
"""

import logging
from typing import Any

import streamlit as st

logger = logging.getLogger(__name__)


def render_kernel_speedup_chart(results: list[dict[str, Any]]) -> None:
    """Render kernel speedup bar chart (PyTorch vs Triton latency).

    Args:
        results: List of dicts with 'kernel', 'pytorch_ms', 'triton_ms', 'speedup'.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install plotly for charts: `uv add plotly`")
        return

    kernels = [r["kernel"] for r in results if "pytorch_ms" in r]
    pt_times = [r["pytorch_ms"] * 1000 for r in results if "pytorch_ms" in r]
    tr_times = [r["triton_ms"] * 1000 for r in results if "triton_ms" in r]

    if not kernels:
        return

    fig = go.Figure(
        data=[
            go.Bar(name="PyTorch", x=kernels, y=pt_times, marker_color="#636EFA"),
            go.Bar(name="Triton", x=kernels, y=tr_times, marker_color="#00CC96"),
        ]
    )
    fig.update_layout(
        barmode="group",
        title="Kernel Latency: PyTorch vs Triton (us)",
        yaxis_title="Latency (us)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_kernel_speedup_ratio_chart(results: list[dict[str, Any]]) -> None:
    """Render speedup ratio bar chart for kernel benchmarks.

    Args:
        results: List of dicts with 'kernel' and 'speedup'.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install plotly for charts: `uv add plotly`")
        return

    kernels = [r["kernel"] for r in results if "speedup" in r]
    speedups = [r["speedup"] for r in results if "speedup" in r]

    if not kernels:
        return

    colors = ["#2ecc71" if s >= 2.0 else "#f39c12" for s in speedups]

    fig = go.Figure(data=[go.Bar(x=kernels, y=speedups, marker_color=colors)])
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        annotation_text="1x (baseline)",
    )
    fig.update_layout(
        title="Triton Kernel Speedup vs PyTorch",
        yaxis_title="Speedup (x)",
        height=350,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_e2e_chart(data: list[dict[str, Any]]) -> None:
    """Render grouped bar chart for E2E benchmark data by runner.

    Args:
        data: List of benchmark entries with 'runner', 'language', 'time_ms'.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install plotly for charts: `uv add plotly`")
        return

    if not data:
        return

    runners_seen: list[str] = []
    seen_set: set[str] = set()
    for d in data:
        name = d["runner"]
        if name not in seen_set:
            runners_seen.append(name)
            seen_set.add(name)

    languages = sorted({d["language"] for d in data})

    fig = go.Figure()
    for lang in languages:
        means = []
        for runner in runners_seen:
            entry = next(
                (d for d in data if d["runner"] == runner and d["language"] == lang),
                None,
            )
            means.append(entry["time_ms"]["mean"] if entry else 0)
        fig.add_trace(go.Bar(name=lang, x=runners_seen, y=means))

    fig.update_layout(
        barmode="group",
        title="E2E Mean Latency by Runner & Language (ms)",
        yaxis_title="Latency (ms)",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_rtf_chart(data: list[dict[str, Any]]) -> None:
    """Render RTF comparison chart across runners.

    Args:
        data: List of benchmark entries with 'runner', 'language', 'rtf'.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Install plotly for charts: `uv add plotly`")
        return

    if not data:
        return

    runners_seen: list[str] = []
    seen_set: set[str] = set()
    for d in data:
        name = d["runner"]
        if name not in seen_set:
            runners_seen.append(name)
            seen_set.add(name)

    languages = sorted({d["language"] for d in data})

    fig = go.Figure()
    for lang in languages:
        rtfs = []
        for runner in runners_seen:
            entry = next(
                (d for d in data if d["runner"] == runner and d["language"] == lang),
                None,
            )
            rtfs.append(entry["rtf"]["mean"] if entry else 0)
        fig.add_trace(go.Bar(name=lang, x=runners_seen, y=rtfs))

    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="red",
        annotation_text="Real-time (1x)",
    )
    fig.update_layout(
        barmode="group",
        title="Real-Time Factor (RTF) by Runner — higher is better",
        yaxis_title="RTF",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_verification_layer_chart(
    layers: dict[str, Any], *, key: str | None = None
) -> None:
    """Render cosine similarity bar chart for model layers.

    Args:
        layers: Mapping of layer index string to dict with 'cosine_sim'.
        key: Unique Streamlit element key.
    """
    try:
        import plotly.graph_objects as go

        layer_ids = [f"L{k}" for k in layers]
        cos_vals = [v["cosine_sim"] for v in layers.values()]
        colors = ["#2ecc71" if v > 0.95 else "#e74c3c" for v in cos_vals]

        fig = go.Figure(data=[go.Bar(x=layer_ids, y=cos_vals, marker_color=colors)])
        fig.add_hline(
            y=0.95,
            line_dash="dash",
            line_color="red",
            annotation_text="threshold=0.95",
        )
        fig.update_layout(
            title="Layer Cosine Similarity",
            yaxis_title="Cosine Similarity",
            yaxis_range=[0.94, 1.001],
            height=350,
            margin={"t": 40, "b": 30},
        )
        st.plotly_chart(fig, use_container_width=True, key=key)
    except ImportError:
        for k, v in layers.items():
            st.text(f"  L{k}: cos={v['cosine_sim']:.6f}")
