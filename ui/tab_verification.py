"""Verification tab for the OmniVoice Triton Streamlit dashboard.

Displays 3-Tier verification results with colored scorecards:
  Tier 1: Kernel unit tests (pytest)
  Tier 2: Model parity (cosine similarity)
  Tier 3: E2E quality (CER, UTMOS, speaker sim)
"""

import logging
from pathlib import Path
from typing import Any

import streamlit as st

from ui.charts import render_verification_layer_chart
from ui.i18n import t
from ui.utils import load_json_dict

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "benchmark" / "results"

_CSS = """
<style>
.tier-card {
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 8px;
    text-align: center;
}
.tier-pass {
    background-color: rgba(46, 204, 113, 0.12);
    border: 2px solid rgba(46, 204, 113, 0.5);
}
.tier-fail {
    background-color: rgba(231, 76, 60, 0.12);
    border: 2px solid rgba(231, 76, 60, 0.5);
}
.tier-pending {
    background-color: rgba(149, 165, 166, 0.12);
    border: 2px solid rgba(149, 165, 166, 0.5);
}
.badge-pass {
    background-color: #2ecc71;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: bold;
    font-size: 0.85em;
}
.badge-fail {
    background-color: #e74c3c;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: bold;
    font-size: 0.85em;
}
.badge-pending {
    background-color: #95a5a6;
    color: white;
    padding: 2px 10px;
    border-radius: 12px;
    font-weight: bold;
    font-size: 0.85em;
}
.metric-bar-container {
    background-color: rgba(149, 165, 166, 0.2);
    border-radius: 6px;
    height: 24px;
    position: relative;
    margin: 4px 0 12px 0;
}
.metric-bar-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.3s ease;
}
.metric-bar-pass { background-color: rgba(46, 204, 113, 0.7); }
.metric-bar-fail { background-color: rgba(231, 76, 60, 0.7); }
.metric-bar-label {
    position: absolute;
    right: 8px;
    top: 2px;
    font-size: 0.8em;
    font-weight: bold;
}
</style>
"""


def render_verification_tab() -> None:
    """Render the full 3-Tier verification deep-dive."""
    st.markdown(_CSS, unsafe_allow_html=True)

    report = load_json_dict(RESULTS_DIR / "verification_report.json")
    if not report:
        st.info(t("verification.no_report"))
        _render_how_to_run()
        return

    timestamp = report.get("timestamp", "")
    if timestamp:
        st.caption(t("verification.last_run", timestamp=timestamp))

    # Top scorecard
    _render_scorecard(report)

    st.markdown("---")

    # Tier 1
    st.subheader(t("verification.tier1"))
    tier1 = report.get("tier1")
    if tier1:
        _render_tier1(tier1)
    else:
        st.warning(t("verification.tier1_not_run"))

    st.markdown("---")

    # Tier 2
    st.subheader(t("verification.tier2"))
    tier2 = report.get("tier2")
    if tier2:
        _render_tier2(tier2)
    else:
        st.warning(t("verification.tier2_not_run"))

    st.markdown("---")

    # Tier 3
    st.subheader(t("verification.tier3"))
    tier3 = report.get("tier3")
    if tier3:
        _render_tier3(tier3)
    else:
        st.warning(t("verification.tier3_not_run"))


def _render_how_to_run() -> None:
    """Render instructions for generating a verification report."""
    with st.expander("How to generate a verification report", expanded=True):
        st.markdown(
            """
**Tier 1 — Kernel unit tests:**
```bash
make test
```

**Tier 2 — Model parity tests (GPU required):**
```bash
make test-parity
```

**Tier 3 — E2E quality evaluation:**
```bash
uv sync --extra eval
make eval-quality
```
"""
        )


def _render_scorecard(report: dict[str, Any]) -> None:
    """Render 3-tier summary scorecard with colored cards."""
    col1, col2, col3 = st.columns(3)
    tier1 = report.get("tier1")
    tier2 = report.get("tier2")
    tier3 = report.get("tier3")

    with col1:
        if tier1:
            status = tier1.get("status", "PENDING")
            passed = tier1.get("passed", 0)
            total = tier1.get("total", 0)
            duration = tier1.get("duration_s", 0)
            _scorecard_card(
                status,
                t("verification.tier1"),
                f"{passed}/{total} ({duration:.1f}s)",
            )
        else:
            _scorecard_card(
                "PENDING", t("verification.tier1"), t("verification.tier1_not_run")
            )

    with col2:
        if tier2:
            status = tier2.get("status", "PENDING")
            pairs = tier2.get("pairs", {})
            if pairs:
                pass_count = sum(1 for p in pairs.values() if p.get("status") == "PASS")
                pair_count = len(pairs)
                detail = f"{pass_count}/{pair_count} pairs"
            else:
                layers = tier2.get("layers", {})
                if layers:
                    min_cos = min(v["cosine_sim"] for v in layers.values())
                    detail = f"min cosine={min_cos:.4f}"
                else:
                    detail = ""
            _scorecard_card(status, t("verification.tier2"), detail)
        else:
            _scorecard_card(
                "PENDING", t("verification.tier2"), t("verification.tier2_not_run")
            )

    with col3:
        if tier3:
            status = tier3.get("overall_verdict", tier3.get("status", "PENDING"))
            comparisons = tier3.get("comparisons", [])
            pass_comp = sum(1 for c in comparisons if c.get("status") == "PASS")
            _scorecard_card(
                status,
                t("verification.tier3"),
                f"{pass_comp}/{len(comparisons)} pass",
            )
        else:
            _scorecard_card(
                "PENDING", t("verification.tier3"), t("verification.tier3_not_run")
            )


def _scorecard_card(status: str, title: str, detail: str) -> None:
    """Render a single scorecard card with colored background."""
    css_class = _status_css_class(status)
    badge = _html_badge(status)
    st.markdown(
        f"""<div class="tier-card {css_class}">
            <div>{badge}</div>
            <div style="font-size:1.1em;font-weight:bold;margin:8px 0">{title}</div>
            <div style="font-size: 0.9em; color: #666;">{detail}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _render_tier1(data: dict[str, Any]) -> None:
    """Render Tier 1 as pass/fail banner with expandable test details."""
    passed = data.get("passed", 0)
    total = data.get("total", 0)
    failed = data.get("failed", 0)
    duration = data.get("duration_s", 0)
    status = data.get("status", "PENDING")

    if status == "PASS":
        st.success(f"{passed}/{total} tests passed in {duration:.1f}s")
    else:
        st.error(f"{passed}/{total} passed, {failed} failed in {duration:.1f}s")

    tests = data.get("tests", [])
    if tests:
        groups = _group_tests(tests)
        with st.expander(
            t("verification.test_details", count=len(tests)), expanded=False
        ):
            for group_name, group_tests in sorted(groups.items()):
                _render_test_group(group_name, group_tests)


def _group_tests(tests: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group tests by file/module name."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for test in tests:
        fullname = test.get("fullname", "")
        parts = fullname.split("::")
        file_part = parts[0].split("/")[-1] if parts else "other.py"
        group = file_part.replace("test_", "").replace(".py", "")
        groups.setdefault(group, []).append(test)
    return groups


def _render_test_group(group_name: str, group_tests: list[dict[str, Any]]) -> None:
    """Render a single test group with per-test pass/fail icons."""
    passed_in_group = sum(1 for tt in group_tests if tt.get("status") == "PASSED")
    total_in_group = len(group_tests)
    badge = _html_badge("PASS" if passed_in_group == total_in_group else "FAIL")
    st.markdown(
        f"**{group_name}** {badge} ({passed_in_group}/{total_in_group})",
        unsafe_allow_html=True,
    )
    for tt in group_tests:
        icon = "\u2705" if tt.get("status") == "PASSED" else "\u274c"
        st.text(f"  {icon} {tt['name']}")


def _render_tier2(data: dict[str, Any]) -> None:
    """Render Tier 2 with per-pair cards and layer cosine similarity charts."""
    if "pairs" in data:
        pair_items = list(data["pairs"].items())
        if len(pair_items) >= 2:
            cols = st.columns(len(pair_items))
            for col, (pair_name, pair_data) in zip(cols, pair_items):
                with col:
                    _render_pair_card(pair_name, pair_data)
        else:
            for pair_name, pair_data in pair_items:
                _render_pair_card(pair_name, pair_data)
        return

    # Legacy single-pair format
    layers = data.get("layers", {})
    if layers:
        render_verification_layer_chart(layers, key="layer_chart_legacy")


def _render_pair_card(pair_name: str, pair_data: dict[str, Any]) -> None:
    """Render a single Tier 2 pair with status and layer chart."""
    status = pair_data.get("status", "PENDING")
    label = pair_name.replace("_", " ").title()
    badge = _html_badge(status)

    st.markdown(f"#### {label} {badge}", unsafe_allow_html=True)

    layers = pair_data.get("layers", {})
    if layers:
        min_cos = min(v["cosine_sim"] for v in layers.values())
        st.metric(t("verification.min_cosine"), f"{min_cos:.6f}")
        render_verification_layer_chart(layers, key=f"layer_{pair_name}")

    logits = pair_data.get("logits", {})
    if logits.get("output_cosine_sim") is not None:
        with st.expander(t("verification.additional"), expanded=False):
            st.text(
                t(
                    "verification.output_cosine",
                    val=f"{logits['output_cosine_sim']:.6f}",
                )
            )
            if logits.get("output_max_abs_diff") is not None:
                st.text(f"  Max abs diff: {logits['output_max_abs_diff']:.4f}")


def _render_tier3(data: dict[str, Any]) -> None:
    """Render Tier 3 with metric progress bars."""
    st.info(t("verification.tier3_deps_note"))

    comparisons = data.get("comparisons", [])
    if not comparisons:
        comp = data.get("comparison", {})
        if comp:
            comparisons = [comp]

    if not comparisons:
        return

    thresholds = _load_tier3_thresholds()

    for comp in comparisons:
        ref = comp.get("ref", "base")
        opt = comp.get("opt", "?")
        comp_status = comp.get("status", "PENDING")
        badge = _html_badge(comp_status)

        st.markdown(f"#### {ref} vs {opt} {badge}", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            _render_metric_bar(
                t("verification.cer_delta"),
                comp.get("cer_delta"),
                thresholds.get("cer_delta_max", 0.05),
                lower_better=True,
            )
        with col2:
            _render_metric_bar(
                t("verification.utmos_delta"),
                comp.get("utmos_delta"),
                thresholds.get("utmos_delta_max", 0.3),
                lower_better=True,
            )
        with col3:
            _render_metric_bar(
                t("verification.speaker_sim"),
                comp.get("speaker_sim_mean"),
                thresholds.get("speaker_sim_min", 0.75),
                lower_better=False,
            )


def _render_metric_bar(
    label: str,
    value: float | None,
    threshold: float,
    *,
    lower_better: bool,
) -> None:
    """Render a metric with a visual progress bar and pass/fail indicator."""
    if value is None:
        st.markdown(f"**{label}**")
        st.caption(t("common.n_a"))
        return

    if lower_better:
        ok = value <= threshold
        pct = min(value / threshold, 1.5) / 1.5 * 100 if threshold > 0 else 0
    else:
        ok = value >= threshold
        pct = min(value / (threshold * 1.5), 1.0) * 100 if threshold > 0 else 0

    display = f"{value:.4f} / {threshold}"
    bar_class = "metric-bar-pass" if ok else "metric-bar-fail"
    badge = _html_badge("PASS" if ok else "FAIL")

    st.markdown(f"**{label}** {badge}", unsafe_allow_html=True)
    st.markdown(
        f"""<div class="metric-bar-container">
            <div class="metric-bar-fill {bar_class}" style="width: {pct:.0f}%;"></div>
            <div class="metric-bar-label">{display}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def _load_tier3_thresholds() -> dict[str, float]:
    """Load Tier 3 thresholds from eval config with fallback defaults."""
    try:
        from benchmark.eval_config import EVAL_CONFIG  # type: ignore[import]

        return EVAL_CONFIG.get("tier3", {})
    except ImportError:
        return {
            "cer_delta_max": 0.05,
            "utmos_delta_max": 0.3,
            "speaker_sim_min": 0.75,
        }


def _status_css_class(status: str | None) -> str:
    """Return CSS class for a status."""
    if status == "PASS":
        return "tier-pass"
    if status == "FAIL":
        return "tier-fail"
    return "tier-pending"


def _html_badge(status: str | None) -> str:
    """Return an HTML badge span for a status."""
    if status == "PASS":
        return '<span class="badge-pass">PASS</span>'
    if status == "FAIL":
        return '<span class="badge-fail">FAIL</span>'
    return '<span class="badge-pending">PENDING</span>'
