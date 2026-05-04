"""
SimRewilding — Trend Change Detection using AMOC
Interactive analysis page for rewild_trend_change_amoc results.
"""

import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from tracepy.simulation.trend import ci_sim, ci_sim_ar
from tracepy.stats.metrics import trend_stats, trend_stats_ar
from scipy.stats import linregress

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trend Change Detection (AMOC)",
    page_icon="📈",
    layout="wide",
)

# ── Simulation constants (mirror rewild_trend_change_amoc.py) ─────────────────
NPRE = 24
NPOST_YEARS = 10
NPOST_MONTHS = 12 * NPOST_YEARS
NPOST_VEC = np.arange(NPRE, NPOST_MONTHS + 1, 12)  # [24, 36, ..., 120] — annual steps, matching R
LEVEL = 10.0
TREND_CONTROL = 0.005
SIGMA = 0.05
PHI_DEFAULT = 0.5
ALPHA = 0.95

TREND_INCREASE = np.round(
    LEVEL * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / NPOST_MONTHS, 4
)
EFFECT_SIZES_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]   # % of mean / 10 yr

RESULTS_PATH = Path(__file__).parent.parent / "results" / "trend_amoc" / "sim_results.pkl"


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading pre-computed simulation results…")
def load_results(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Regenerating simulation run…")
def _regenerate_iid(seed: int, delay: int, trend_inc: float):
    npre_delay  = NPRE + delay
    npost_delay = NPOST_MONTHS - delay
    return ci_sim(
        seed=seed, npre=npre_delay, npost=npost_delay,
        level=LEVEL, trend=[TREND_CONTROL, TREND_CONTROL + trend_inc], sigma=SIGMA,
    )


@st.cache_data(show_spinner="Regenerating AR(1) simulation run…")
def _regenerate_ar(seed: int, delay: int, trend_inc: float):
    npre_delay  = NPRE + delay
    npost_delay = NPOST_MONTHS - delay
    return ci_sim_ar(
        seed=seed, npre=npre_delay, npost=npost_delay,
        level=LEVEL, trend=[TREND_CONTROL, TREND_CONTROL + trend_inc],
        phi=PHI_DEFAULT, sigma=SIGMA,
    )


def trend_label(trend_val, pct):
    return f"{pct}% ({trend_val:.4f}/mo)"


# ── Colour palette (one per effect size) ──────────────────────────────────────
PALETTE = px.colors.sample_colorscale("Viridis", [i / 10 for i in range(11)])


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("📈 Trend Change Detection (AMOC)")

st.markdown("""
After a rewilding intervention (e.g. reintroducing a species, restoring a habitat), we
expect some ecological indicator to change trajectory.

**At Most One Change (AMOC)** is an *offline* changepoint detection method.
At the end of a monitoring window we have the full time series and test all possible
changepoint locations simultaneously.

Each simulated dataset contains two parallel time series:
- **Control series** — same trend throughout, no intervention effect.
- **Intervention series** — same trend as control up to the changepoint, then a
  different (steeper) trend after intervention.

Types of noise used:
- **i.i.d. noise → BACI** (Before-After Control-Intervention): the test statistic is
  computed on the **difference** series (`intervention − control`), which removes shared
  environmental variation.
- **i.i.d. noise → BA** (Before-After): same i.i.d. noise but only the intervention
  series is used.
- **AR(1) noise → BA** (Before-After): the test statistic is computed on the **intervention
  series only** with AR(1)-autocorrelated noise, without subtracting the control.
""")


# ══════════════════════════════════════════════════════════════════════════════
# PARAMETER REFERENCE (SIDEBAR)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    **On this page**
    - 📊 [Results explorer](#results-explorer)
    - 🔍 [Individual run explorer](#individual-run-explorer)
    - 🔬 [Run your own simulation](#run-your-own-simulation)
    """)

    st.divider()
    st.header("Parameter reference")

    st.markdown(f"""
    | Parameter | Value | Meaning |
    |-----------|-------|---------|
    | **npre** | {NPRE} months | Monitoring before intervention |
    | **npost_max** | {NPOST_MONTHS} months | Max post-intervention window |
    | **level** | {LEVEL} | Baseline indicator value |
    | **trend_control** | {TREND_CONTROL}/mo | Pre-intervention monthly slope |
    | **φ (phi)** | {PHI_DEFAULT} | AR(1) autocorrelation |
    | **Nsim** | 1,000 | Null simulations for critical value |
    | **simN** | 1,000 | Main simulations per effect size |
    """)


# ── Load data ─────────────────────────────────────────────────────────────────
data = load_results(RESULTS_PATH)

if data is None:
    st.error(
        f"Pre-computed results not found at `{RESULTS_PATH}`. "
        "Run `python rewild_trend_change_amoc.py` first, "
        "or use the Mini-Simulation section below to run a small analysis on the fly."
    )
    results_available = False
else:
    cv         = data.get("critical_values")
    res_iid    = data.get("detection_results", {})
    res_ar     = data.get("detection_results_ar", {})
    res_iid_ba = data.get("detection_results_iid_ba")
    if cv is None or not res_iid:
        st.warning(
            "Saved results are incomplete (critical values or detection results missing). "
            "Re-run `python rewild_trend_change_amoc.py` to regenerate."
        )
        results_available = False
    else:
        trends = sorted(res_iid.keys())
        results_available = True


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS EXPLORER  (only shown when data is available)
# ══════════════════════════════════════════════════════════════════════════════
if results_available:
    st.header("📊 Results explorer")
    st.markdown(
        "Charts below are drawn from pre-computed simulations (1,000 runs per effect size)."
    )

    tab_power, tab_ttd, tab_null, tab_err, tab_delay, tab_bias = st.tabs([
        "Power curves",
        "Time to detection",
        "Null distributions",
        "Changepoint error",
        "Detection by delay",
        "Changepoint bias",
    ])

    # ── Tab 1: Power Curves ────────────────────────────────────────────────────
    with tab_power:
        st.subheader("Power curves: detection rate vs monitoring window")
        st.markdown("""
        Each line shows how the **probability of detecting the trend change** increases as the
        post-intervention monitoring window grows from 24 to 120 months (2–10 years).
        Brighter/yellower lines represent larger effect sizes.

        The dashed lines mark the 80% and 95% power thresholds — standard benchmarks for
        "adequate" and "high" statistical power. A study designer would read off the x-axis
        to find the minimum monitoring duration needed for a given effect size.

        **Key insight:** small effect sizes (dark lines, 5–20%) require many years of monitoring
        to reach reliable detection. Larger effects are detectable within 3–5 years.
        """)

        _power_opts = ["i.i.d. BACI", "AR(1)", "i.i.d. BA", "Side-by-side comparison"]
        if res_iid_ba is None:
            _power_opts = [o for o in _power_opts if o != "i.i.d. BA"]
        noise_choice = st.radio(
            "Noise model", _power_opts,
            horizontal=True, key="power_noise"
        )

        def power_curve_fig(results, title):
            fig = go.Figure()
            for i, (trend, pct) in enumerate(zip(trends, EFFECT_SIZES_PCT)):
                entry = results.get(trend)
                if entry is None:
                    continue
                rates = entry["detection_rates"]
                fig.add_trace(go.Scatter(
                    x=NPOST_VEC, y=rates,
                    mode="lines",
                    name=f"{pct}%",
                    line=dict(color=PALETTE[i], width=2),
                    hovertemplate="npost: %{x} mo<br>Detection rate: %{y:.1%}<extra></extra>",
                ))
            fig.add_hline(y=0.80, line_dash="dash", line_color="grey",
                          annotation_text="80% power", annotation_position="right")
            fig.add_hline(y=0.95, line_dash="dot", line_color="grey",
                          annotation_text="95% power", annotation_position="right")
            fig.update_layout(
                title=title,
                xaxis_title="Post-intervention monitoring window (months)",
                yaxis_title="Detection rate",
                yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                legend_title="Effect size<br>(% of mean / 10 yr)",
                height=480,
            )
            return fig

        if noise_choice == "i.i.d. BACI":
            st.plotly_chart(power_curve_fig(res_iid, "Power curves — i.i.d. noise (BACI)"),
                            width="stretch")
        elif noise_choice == "AR(1)":
            st.plotly_chart(power_curve_fig(res_ar, "Power curves — AR(1) noise (φ=0.5)"),
                            width="stretch")
        elif noise_choice == "i.i.d. BA":
            st.plotly_chart(power_curve_fig(res_iid_ba, "Power curves — i.i.d. noise (BA)"),
                            width="stretch")
        else:
            cols = st.columns(2 if res_iid_ba is None else 3)
            with cols[0]:
                st.plotly_chart(power_curve_fig(res_iid, "i.i.d. BACI"), width="stretch")
            with cols[1]:
                st.plotly_chart(power_curve_fig(res_ar, "AR(1) BA (φ=0.5)"), width="stretch")
            if res_iid_ba is not None:
                with cols[2]:
                    st.plotly_chart(power_curve_fig(res_iid_ba, "i.i.d. BA"), width="stretch")
            st.info(
                "AR(1) autocorrelation generally **reduces power**: the correlated errors "
                "widen the null distribution, raising the detection threshold. The effect is "
                "strongest for small effect sizes and short windows.",
                icon="ℹ️",
            )

    # ── Tab 2: Time to Detection ───────────────────────────────────────────────
    with tab_ttd:
        st.subheader("Time to detection: minimum monitoring window for target power")
        st.markdown("""
        Each bar shows the **minimum number of post-intervention months** needed before the
        detection rate first reaches your chosen power threshold.

        - **Coloured bars** — the threshold was reached; bar height = first npost where
          detection ≥ target. Hover for the exact value in months and years.
        - **Grey bars labelled "Not reached"** — the power curve never hits the threshold
          within the 10-year (120-month) maximum window. No amount of monitoring within
          this study design is sufficient for this effect size.
        - **Bars at 24 months** — detection was already at or above the target at the
          earliest post-intervention measurement point (the study is powered from the start).
        """)

        ctrl_ttd1, ctrl_ttd2 = st.columns(2)
        with ctrl_ttd1:
            power_thresh = st.select_slider(
                "Target power threshold",
                options=[0.70, 0.80, 0.90, 0.95], value=0.80,
                format_func=lambda x: f"{x:.0%}",
            )
        with ctrl_ttd2:
            _ttd_opts = ["i.i.d. BACI", "AR(1)", "Both"]
            if res_iid_ba is not None:
                _ttd_opts = ["i.i.d. BACI", "AR(1)", "i.i.d. BA", "All"]
            noise_ttd = st.radio(
                "Noise model", _ttd_opts,
                horizontal=True, key="ttd_noise",
            )

        def time_to_thresh(results, thresh):
            """Return list of (months | None, max_rate_at_120) per effect size."""
            out = []
            for t in trends:
                entry = results.get(t)
                if entry is None:
                    out.append((None, 0.0))
                    continue
                rates = entry["detection_rates"]
                idx = int(np.argmax(rates >= thresh))
                if rates[idx] >= thresh:
                    out.append((int(NPOST_VEC[idx]), float(rates[-1])))
                else:
                    out.append((None, float(rates[-1])))
            return out

        ttd_iid    = time_to_thresh(res_iid, power_thresh)
        ttd_ar     = time_to_thresh(res_ar,  power_thresh)
        ttd_iid_ba = time_to_thresh(res_iid_ba, power_thresh) if res_iid_ba is not None else None
        labels  = [f"{p}%" for p in EFFECT_SIZES_PCT]

        COLOR_IID      = "rgba(33,150,243,0.85)"
        COLOR_AR       = "rgba(255,152,0,0.85)"
        COLOR_IID_BA   = "rgba(76,175,80,0.85)"
        COLOR_NONE     = "rgba(180,180,180,0.55)"

        # Y-axis: show years at every 12-month tick
        tick_vals  = list(range(0, NPOST_MONTHS + 1, 12))
        tick_text  = [f"{m // 12} yr" if m > 0 else "0" for m in tick_vals]

        def make_bar(ttd, reached_color, model_label):
            y_vals, bar_colors, bar_text, hover = [], [], [], []
            for i, (m, max_rate) in enumerate(ttd):
                pct = EFFECT_SIZES_PCT[i]
                if m is None:
                    y_vals.append(NPOST_MONTHS)
                    bar_colors.append(COLOR_NONE)
                    bar_text.append("Not<br>reached")
                    hover.append(
                        f"<b>{pct}% effect — not reached</b><br>"
                        f"Power never exceeds {power_thresh:.0%} within 10 yr<br>"
                        f"Best detection at 10 yr: {max_rate:.1%}"
                        f"<extra>{model_label}</extra>"
                    )
                else:
                    y_vals.append(m)
                    bar_colors.append(reached_color)
                    prefix = "Already at " if m == int(NPOST_VEC[0]) else ""
                    bar_text.append(f"{prefix}{m} mo<br>({m/12:.1f} yr)")
                    hover.append(
                        f"<b>{pct}% effect</b><br>"
                        f"Reaches {power_thresh:.0%} after <b>{m} months ({m/12:.1f} yr)</b><br>"
                        f"Detection at 10 yr: {max_rate:.1%}"
                        f"<extra>{model_label}</extra>"
                    )
            return go.Bar(
                x=labels, y=y_vals,
                name=model_label,
                marker_color=bar_colors,
                text=bar_text,
                textposition="outside",
                textfont=dict(size=9),
                hovertemplate=hover,
            )

        fig_ttd = go.Figure()
        if noise_ttd in ("i.i.d. BACI", "Both", "All"):
            fig_ttd.add_trace(make_bar(ttd_iid, COLOR_IID, "i.i.d. BACI"))
        if noise_ttd in ("AR(1)", "Both", "All"):
            fig_ttd.add_trace(make_bar(ttd_ar, COLOR_AR, "AR(1)"))
        if ttd_iid_ba is not None and noise_ttd in ("i.i.d. BA", "All"):
            fig_ttd.add_trace(make_bar(ttd_iid_ba, COLOR_IID_BA, "i.i.d. BA"))

        fig_ttd.update_layout(
            barmode="group",
            xaxis=dict(
                title="Effect size (% of mean / 10 yr)",
                tickmode="array",
                tickvals=labels,
                ticktext=labels,
            ),
            yaxis=dict(
                title="Post-intervention monitoring needed",
                range=[0, NPOST_MONTHS + 26],   # headroom for "Not reached" labels
                tickvals=tick_vals,
                ticktext=tick_text,
            ),
            legend_title="Noise model",
            legend=dict(
                itemsizing="constant",
                traceorder="normal",
            ),
            height=480,
            title=f"Monitoring time needed to reach {power_thresh:.0%} detection probability",
        )
        st.plotly_chart(fig_ttd, width="stretch")

        # ── Plain-English summary ──────────────────────────────────────────────
        def summarise(ttd, label):
            not_reached = [EFFECT_SIZES_PCT[i] for i, (m, _) in enumerate(ttd) if m is None]
            return not_reached

        msgs = []
        if noise_ttd in ("i.i.d. BACI", "Both", "All"):
            nr = summarise(ttd_iid, "i.i.d. BACI")
            if nr:
                pct_str = ", ".join(f"{p}%" for p in nr)
                msgs.append(f"**i.i.d. BACI:** effect size(s) {pct_str} never reach {power_thresh:.0%} within 10 yr.")
            else:
                msgs.append(f"**i.i.d. BACI:** all effect sizes reach {power_thresh:.0%} within 10 yr.")
        if noise_ttd in ("AR(1)", "Both", "All"):
            nr = summarise(ttd_ar, "AR(1)")
            if nr:
                pct_str = ", ".join(f"{p}%" for p in nr)
                msgs.append(f"**AR(1):** effect size(s) {pct_str} never reach {power_thresh:.0%} within 10 yr.")
            else:
                msgs.append(f"**AR(1):** all effect sizes reach {power_thresh:.0%} within 10 yr.")
        if ttd_iid_ba is not None and noise_ttd in ("i.i.d. BA", "All"):
            nr = summarise(ttd_iid_ba, "i.i.d. BA")
            if nr:
                pct_str = ", ".join(f"{p}%" for p in nr)
                msgs.append(f"**i.i.d. BA:** effect size(s) {pct_str} never reach {power_thresh:.0%} within 10 yr.")
            else:
                msgs.append(f"**i.i.d. BA:** all effect sizes reach {power_thresh:.0%} within 10 yr.")

        for msg in msgs:
            st.caption(msg)

    # ── Tab 4: Null Distributions ──────────────────────────────────────────────
    with tab_null:
        st.subheader("Null distributions and critical values")
        st.markdown("""
        Before testing real data, we calibrate the test by simulating **what T_max looks like
        when there is no trend change** (the null hypothesis). These histograms show the
        distribution of the maximum test statistic under the null, separately for i.i.d. and
        AR(1) noise, and for two monitoring window lengths (48 and 96 months).

        The vertical dashed line marks the **95th percentile** — the critical value. Any
        observed T_max to the right of this line leads us to declare a changepoint detected.

        **Key observation:** the AR(1) critical values are higher than the i.i.d. ones.
        Autocorrelation inflates T_max even under the null, so we need a stricter threshold to
        keep the false-positive rate at 5%.
        """)

        _has_iid_ba_cv = "iid_ba_48" in cv

        if _has_iid_ba_cv:
            fig_null = make_subplots(
                rows=3, cols=2,
                subplot_titles=[
                    "i.i.d. BACI, npost = 48 mo (~4 yr)",
                    "i.i.d. BACI, npost = 96 mo (~8 yr)",
                    "i.i.d. BA, npost = 48 mo (~4 yr)",
                    "i.i.d. BA, npost = 96 mo (~8 yr)",
                    "AR(1), npost = 48 mo (~4 yr)",
                    "AR(1), npost = 96 mo (~8 yr)",
                ],
            )
            null_keys = [
                ("iid_48", 1, 1), ("iid_96", 1, 2),
                ("iid_ba_48", 2, 1), ("iid_ba_96", 2, 2),
                ("ar1_48", 3, 1), ("ar1_96", 3, 2),
            ]
            null_height = 780
        else:
            fig_null = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    "i.i.d. BACI, npost = 48 mo (~4 yr)",
                    "i.i.d. BACI, npost = 96 mo (~8 yr)",
                    "AR(1), npost = 48 mo (~4 yr)",
                    "AR(1), npost = 96 mo (~8 yr)",
                ],
            )
            null_keys = [("iid_48", 1, 1), ("iid_96", 1, 2), ("ar1_48", 2, 1), ("ar1_96", 2, 2)]
            null_height = 550

        for key, row, col in null_keys:
            if key not in cv:
                continue
            entry = cv[key]
            dist = entry["null_dist"]
            crit = entry["critical_value"]
            colour = "#4CAF50" if key.startswith("iid_ba") else ("#2196F3" if key.startswith("iid") else "#FF9800")
            fig_null.add_trace(
                go.Histogram(
                    x=dist, nbinsx=50,
                    marker_color=colour, opacity=0.75,
                    name=key,
                    showlegend=False,
                    hovertemplate="T_max: %{x:.2f}<br>Count: %{y}<extra></extra>",
                ),
                row=row, col=col,
            )
            fig_null.add_vline(
                x=crit, line_dash="dash", line_color="crimson",
                annotation_text=f"cv = {crit:.2f}",
                annotation_position="top right",
                row=row, col=col,
            )

        fig_null.update_layout(height=null_height, title="Null distributions of T_max (1,000 simulations each)")
        fig_null.update_xaxes(title_text="T_max")
        fig_null.update_yaxes(title_text="Count")
        st.plotly_chart(fig_null, width="stretch")

        st.markdown("**Critical values:**")
        cv_scenarios = [
            ("i.i.d. BACI, npost=48 mo", "iid_48"),
            ("i.i.d. BACI, npost=96 mo", "iid_96"),
            ("AR(1), npost=48 mo",        "ar1_48"),
            ("AR(1), npost=96 mo",        "ar1_96"),
        ]
        if _has_iid_ba_cv:
            cv_scenarios += [
                ("i.i.d. BA, npost=48 mo", "iid_ba_48"),
                ("i.i.d. BA, npost=96 mo", "iid_ba_96"),
            ]
        cv_table = {
            "Scenario": [s for s, _ in cv_scenarios],
            "Critical value (95th pct)": [
                f"{cv[k]['critical_value']:.3f}" if k in cv else "—"
                for _, k in cv_scenarios
            ],
        }
        st.table(cv_table)

    # ── Tab 5: Changepoint Error ───────────────────────────────────────────────
    with tab_err:
        st.subheader("Changepoint localisation error")
        st.markdown("""
        Detecting *that* a change occurred is only part of the story — we also want to know
        *when* it happened. This tab shows how accurately the AMOC method localises the
        changepoint in time.

        The **mean absolute error** is the average number of months between the detected
        changepoint and the true changepoint (the moment of intervention). Smaller = better.

        **Key patterns:**
        - Error decreases as monitoring continues (more data → more precision).
        - Larger effect sizes produce smaller errors (a stronger signal pins down the timing
          more easily).
        - AR(1) noise generally increases localisation error versus i.i.d. noise.
        """)

        _err_opts = ["i.i.d. BACI", "AR(1)", "Comparison"]
        if res_iid_ba is not None:
            _err_opts = ["i.i.d. BACI", "AR(1)", "i.i.d. BA", "Comparison"]
        err_noise = st.radio("Noise model", _err_opts,
                             horizontal=True, key="err_noise")

        def error_fig(results, title):
            fig = go.Figure()
            for i, (trend, pct) in enumerate(zip(trends, EFFECT_SIZES_PCT)):
                entry = results.get(trend)
                if entry is None:
                    continue
                errs = entry["mean_errors"]
                fig.add_trace(go.Scatter(
                    x=NPOST_VEC, y=errs,
                    mode="lines",
                    name=f"{pct}%",
                    line=dict(color=PALETTE[i], width=2),
                    hovertemplate="npost: %{x} mo<br>Mean |error|: %{y:.1f} mo<extra></extra>",
                ))
            fig.update_layout(
                title=title,
                xaxis_title="Post-intervention monitoring window (months)",
                yaxis_title="Mean |τ̂ − τ_true| (months)",
                legend_title="Effect size",
                height=460,
            )
            return fig

        if err_noise == "i.i.d. BACI":
            st.plotly_chart(error_fig(res_iid, "Changepoint error — i.i.d. noise (BACI)"),
                            width="stretch")
        elif err_noise == "AR(1)":
            st.plotly_chart(error_fig(res_ar, "Changepoint error — AR(1) noise"),
                            width="stretch")
        elif err_noise == "i.i.d. BA":
            st.plotly_chart(error_fig(res_iid_ba, "Changepoint error — i.i.d. noise (BA)"),
                            width="stretch")
        else:
            cols = st.columns(2 if res_iid_ba is None else 3)
            with cols[0]:
                st.plotly_chart(error_fig(res_iid, "i.i.d. BACI"), width="stretch")
            with cols[1]:
                st.plotly_chart(error_fig(res_ar, "AR(1)"), width="stretch")
            if res_iid_ba is not None:
                with cols[2]:
                    st.plotly_chart(error_fig(res_iid_ba, "i.i.d. BA"), width="stretch")

    # ── Tab 6: Detection by Delay ──────────────────────────────────────────────
    with tab_delay:
        st.subheader("Effect of intervention delay on detection")
        st.markdown("""
        The ecological response to a rewilding intervention rarely begins on day one.
        There is often a **lag** before the new trend is visible in the data — vegetation may
        take a growing season to respond, or a reintroduced predator population needs time to
        establish. This tab shows how that delay erodes detection probability.

        Choose an effect size and noise model, then explore two complementary views:
        - **Line chart** — detection rate vs delay at a chosen monitoring window,
          with ±1 SE bands to show uncertainty.
        - **Heatmap** — the full 2-D picture: how delay and monitoring window length
          interact to determine detection probability.
        """)

        # ── Controls ──────────────────────────────────────────────────────────
        ctrl1, ctrl2, ctrl3 = st.columns(3)
        with ctrl1:
            delay_effect_pct = st.select_slider(
                "Effect size",
                options=EFFECT_SIZES_PCT,
                value=20,
                format_func=lambda x: f"{x}%",
                key="delay_effect",
            )
        with ctrl2:
            _delay_opts = ["i.i.d. BACI", "AR(1)", "Both"]
            if res_iid_ba is not None:
                _delay_opts = ["i.i.d. BACI", "AR(1)", "i.i.d. BA", "All"]
            delay_noise = st.radio(
                "Noise model", _delay_opts,
                horizontal=True, key="delay_noise",
            )
        with ctrl3:
            delay_view = st.radio(
                "View", ["Line chart", "Heatmap"],
                horizontal=True, key="delay_view",
            )

        delay_idx = EFFECT_SIZES_PCT.index(delay_effect_pct)
        delay_trend = trends[delay_idx]
        unique_delays = np.arange(1, 21)
        _n_per_bin = int((np.array(res_iid[delay_trend]["delays"]) == 1).sum())

        def delay_series(results, trend, npost_i):
            """Return (rates, lower_se, upper_se) arrays across delays 1-20."""
            entry = results.get(trend)
            if entry is None:
                nan = np.full(len(unique_delays), np.nan)
                return nan, nan, nan
            delays_arr = np.array(entry["delays"])
            detected_arr = entry["detected_matrix"][:, npost_i]
            rates, lo, hi = [], [], []
            for d in unique_delays:
                mask = delays_arr == d
                n = mask.sum()
                r = detected_arr[mask].mean() if n > 0 else np.nan
                se = np.sqrt(r * (1 - r) / n) if n > 1 and not np.isnan(r) else 0.0
                rates.append(r)
                lo.append(max(0.0, r - se))
                hi.append(min(1.0, r + se))
            return np.array(rates), np.array(lo), np.array(hi)

        # ── Line chart view ────────────────────────────────────────────────────
        if delay_view == "Line chart":
            npost_line = st.select_slider(
                "Monitoring window (months)",
                options=[int(v) for v in NPOST_VEC],
                value=int(NPOST_VEC[0]),
                key="delay_npost_line",
            )
            npost_i_line = int(np.searchsorted(NPOST_VEC, npost_line, side="left"))
            npost_i_line = min(npost_i_line, len(NPOST_VEC) - 1)

            noise_configs = []
            if delay_noise in ("i.i.d. BACI", "Both", "All"):
                noise_configs.append(("i.i.d. BACI", res_iid, "rgba(33,150,243,1)", "rgba(33,150,243,0.15)"))
            if delay_noise in ("AR(1)", "Both", "All"):
                noise_configs.append(("AR(1)", res_ar, "rgba(255,152,0,1)", "rgba(255,152,0,0.15)"))
            if res_iid_ba is not None and delay_noise in ("i.i.d. BA", "All"):
                noise_configs.append(("i.i.d. BA", res_iid_ba, "rgba(76,175,80,1)", "rgba(76,175,80,0.15)"))

            fig_line = go.Figure()
            for label, results, colour, colour_fill in noise_configs:
                rates, lo, hi = delay_series(results, delay_trend, npost_i_line)

                # SE band
                fig_line.add_trace(go.Scatter(
                    x=np.concatenate([unique_delays, unique_delays[::-1]]),
                    y=np.concatenate([hi, lo[::-1]]),
                    fill="toself",
                    fillcolor=colour_fill,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                # Main line
                fig_line.add_trace(go.Scatter(
                    x=unique_delays, y=rates,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=colour, width=2.5),
                    marker=dict(size=7),
                    hovertemplate=(
                        "Delay: %{x} mo<br>"
                        "Detection: %{y:.1%}<br>"
                        "<extra>" + label + "</extra>"
                    ),
                ))

            fig_line.add_hline(y=0.80, line_dash="dash", line_color="grey",
                               annotation_text="80% power", annotation_position="right")
            fig_line.update_layout(
                title=(
                    f"Detection rate vs intervention delay — {delay_effect_pct}% effect, "
                    f"npost = {npost_line} mo<br>"
                    f"<sup>Shaded band = ±1 SE (≈{_n_per_bin} simulations per delay bin)</sup>"
                ),
                xaxis=dict(title="Intervention delay (months)", dtick=2),
                yaxis=dict(
                    title="Detection rate",
                    range=[0, 1.05],
                    tickformat=".0%",
                ),
                legend=dict(title="Noise model"),
                height=460,
            )
            st.plotly_chart(fig_line, width="stretch")

            # Contextual note below chart
            # Compute slope to give a plain-English summary
            _single_noise = delay_noise not in ("Both", "All")
            if _single_noise:
                _noise_res = {"i.i.d. BACI": res_iid, "AR(1)": res_ar, "i.i.d. BA": res_iid_ba}.get(delay_noise, res_iid)
                r_arr = delay_series(_noise_res, delay_trend, npost_i_line)[0] if _noise_res else None
            if _single_noise and r_arr is not None:
                drop = float(np.nanmax(r_arr) - np.nanmin(r_arr))
                st.caption(
                    f"Across delays 1–20 months, detection rate spans "
                    f"**{float(np.nanmin(r_arr)):.0%} – {float(np.nanmax(r_arr)):.0%}** "
                    f"(a drop of {drop:.0%} from best to worst delay) "
                    f"for the {delay_effect_pct}% effect at npost = {npost_line} mo. "
                    f"Each delay bin contains approximately "
                    f"{int((np.array(_noise_res[delay_trend]['delays']) == 1).sum()):,} simulations."
                )

        # ── Heatmap view ───────────────────────────────────────────────────────
        else:
            _heat_map = {"i.i.d. BACI": res_iid, "AR(1)": res_ar, "i.i.d. BA": res_iid_ba}
            if delay_noise in ("Both", "All"):
                _heat_single_opts = (
                    ["i.i.d. BACI", "AR(1)", "i.i.d. BA"] if res_iid_ba is not None
                    else ["i.i.d. BACI", "AR(1)"]
                )
                heat_label = st.radio(
                    "Noise model for heatmap",
                    _heat_single_opts,
                    horizontal=True,
                    key="delay_heat_single",
                )
                heat_results = _heat_map[heat_label]
            else:
                heat_results = _heat_map.get(delay_noise, res_iid)
                heat_label = delay_noise

            delays_arr = np.array(heat_results[delay_trend]["delays"])
            det_mat = heat_results[delay_trend]["detected_matrix"]

            z = np.full((len(unique_delays), len(NPOST_VEC)), np.nan)
            for di, d in enumerate(unique_delays):
                mask = delays_arr == d
                if mask.any():
                    z[di, :] = det_mat[mask, :].mean(axis=0)

            fig_heat2 = go.Figure()
            fig_heat2.add_trace(go.Heatmap(
                x=NPOST_VEC,
                y=unique_delays,
                z=z,
                colorscale="Viridis",
                zmin=0, zmax=1,
                colorbar=dict(title="Detection rate", tickformat=".0%"),
                hovertemplate=(
                    "npost: %{x} mo<br>"
                    "Delay: %{y} mo<br>"
                    "Detection: %{z:.1%}"
                    "<extra></extra>"
                ),
            ))
            fig_heat2.add_trace(go.Contour(
                x=NPOST_VEC,
                y=unique_delays,
                z=z,
                contours=dict(
                    coloring="none",
                    showlabels=True,
                    start=0.80, end=0.95, size=0.15,
                    labelfont=dict(size=11, color="white"),
                ),
                line=dict(color="white", dash="dash"),
                showscale=False,
                hoverinfo="skip",
            ))
            fig_heat2.update_layout(
                title=(
                    f"Detection rate: intervention delay × monitoring window — "
                    f"{delay_effect_pct}% effect, {heat_label} noise<br>"
                    f"<sup>Contour lines at 80% and 95% power · "
                    f"Read off the monitoring window needed for your expected delay</sup>"
                ),
                xaxis_title="Post-intervention monitoring window (months)",
                yaxis=dict(title="Intervention delay (months)", dtick=2),
                height=500,
            )
            st.plotly_chart(fig_heat2, width="stretch")
            st.caption(
                "**How to read this:** find your expected delay on the y-axis, then read "
                "across to where the colour reaches 80% (first white contour line) — "
                "that x-value is the minimum monitoring window needed."
            )

    # ── Tab 6: Changepoint Bias ────────────────────────────────────────────────
    with tab_bias:
        st.subheader("Changepoint timing accuracy")
        st.markdown("""
        When a changepoint *is* detected, how close is the estimated timing to when the change
        actually happened?

        The chart shows **detection timing error** — the gap in months between the declared
        changepoint (τ̂) and the true moment of change (τ_true = month 24):

        - **Zero** — perfect timing
        - **Positive** — declared *later* than the true change (late detection)
        - **Negative** — declared *earlier* than the true change (early detection)

        Each box spans the middle 50% of detected simulations (IQR); whiskers reach the 5th
        and 95th percentiles. Only simulations that successfully detected a change are shown —
        the number included in each box is in the table below.

        The **dotted zero line** marks perfect timing — the changepoint declared exactly when
        the ecological response began. The **green band (±3 months)** is a practical
        "good enough" zone: a timing error within ±3 months is unlikely to matter for
        real monitoring decisions. Errors outside this band mean the method is attributing
        the change to the wrong point in time by a meaningful margin.
        """)

        ctrl_b1, ctrl_b2 = st.columns(2)
        with ctrl_b1:
            _bias_opts = ["i.i.d. BACI", "AR(1)", "Both"]
            if res_iid_ba is not None:
                _bias_opts = ["i.i.d. BACI", "AR(1)", "i.i.d. BA", "All"]
            bias_noise = st.radio(
                "Noise model", _bias_opts,
                horizontal=True, key="bias_noise",
            )
        with ctrl_b2:
            npost_idx = st.slider(
                "Monitoring window (months)",
                min_value=int(NPOST_VEC[0]),
                max_value=int(NPOST_VEC[-1]),
                value=int(NPOST_VEC[0]),
                step=12,
                key="bias_npost",
            )

        npost_i = int(np.searchsorted(NPOST_VEC, npost_idx, side="left"))
        npost_i = min(npost_i, len(NPOST_VEC) - 1)

        bias_configs = []
        if bias_noise in ("i.i.d. BACI", "Both", "All"):
            bias_configs.append(("i.i.d. BACI", res_iid, "rgba(33,150,243,0.85)"))
        if bias_noise in ("AR(1)", "Both", "All"):
            bias_configs.append(("AR(1)", res_ar, "rgba(255,152,0,0.85)"))
        if res_iid_ba is not None and bias_noise in ("i.i.d. BA", "All"):
            bias_configs.append(("i.i.d. BA", res_iid_ba, "rgba(76,175,80,0.85)"))

        x_labels = [f"{p}%" for p in EFFECT_SIZES_PCT]
        n_table_rows = {}

        fig_bias = go.Figure()

        # Shaded "near-zero" zone ±3 months — roughly within one data collection round
        fig_bias.add_hrect(
            y0=-3, y1=3,
            fillcolor="rgba(0,180,0,0.07)",
            line_width=0,
            annotation_text="± 3 months",
            annotation_position="top right",
            annotation_font=dict(color="green", size=10),
        )

        for label, results, colour in bias_configs:
            q1s, meds, q3s, p5s, p95s = [], [], [], [], []
            ns = []

            for trend in trends:
                entry = results.get(trend)
                if entry is None:
                    q1s.append(None); meds.append(None); q3s.append(None)
                    p5s.append(None); p95s.append(None)
                    ns.append(0)
                    continue
                det = entry["detected_matrix"][:, npost_i].astype(bool)
                err = entry["cpt_matrix"][:, npost_i][det] - NPRE
                n = len(err)
                ns.append(n)
                if n >= 5:
                    q1s.append(float(np.percentile(err, 25)))
                    meds.append(float(np.median(err)))
                    q3s.append(float(np.percentile(err, 75)))
                    p5s.append(float(np.percentile(err, 5)))
                    p95s.append(float(np.percentile(err, 95)))
                else:
                    q1s.append(None); meds.append(None); q3s.append(None)
                    p5s.append(None); p95s.append(None)

            n_table_rows[label] = ns

            fig_bias.add_trace(go.Box(
                x=x_labels,
                q1=q1s, median=meds, q3=q3s,
                lowerfence=p5s, upperfence=p95s,
                name=label,
                marker_color=colour,
                line_color=colour,
                boxpoints=False,
                hovertemplate=(
                    "<b>%{x} effect — " + label + "</b><br>"
                    "Median: %{median} mo<br>"
                    "IQR: %{q1} – %{q3} mo<br>"
                    "5th–95th pct: %{lowerfence} – %{upperfence} mo"
                    "<extra></extra>"
                ),
            ))

        fig_bias.add_hline(
            y=0, line_color="black", line_width=1.5, line_dash="dot",
            annotation_text="True changepoint timing",
            annotation_position="top left",
            annotation_font=dict(size=10),
        )

        # Arrow annotations on y-axis to clarify direction
        fig_bias.add_annotation(
            x=-0.07, y=0.75, xref="paper", yref="paper",
            text="▲ Late detection", showarrow=False,
            font=dict(size=10, color="crimson"), textangle=-90,
        )
        fig_bias.add_annotation(
            x=-0.07, y=0.25, xref="paper", yref="paper",
            text="▼ Early detection", showarrow=False,
            font=dict(size=10, color="steelblue"), textangle=-90,
        )

        fig_bias.update_layout(
            boxmode="group",
            xaxis=dict(
                title="Effect size (% of mean / 10 yr)",
                tickmode="array",
                tickvals=x_labels,
                ticktext=x_labels,
            ),
            yaxis=dict(
                title="Detection timing error (months)",
                zeroline=False,
            ),
            height=480,
            title=(
                f"Changepoint timing error at npost = {npost_idx} months — {bias_noise} noise<br>"
                f"<sup>Box = IQR · whiskers = 5th–95th percentile · "
                f"detected simulations only</sup>"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
            margin=dict(l=70),
        )
        st.plotly_chart(fig_bias, width="stretch")

        # N detected table
        n_df_cols = {"Effect size": x_labels}
        for label, ns in n_table_rows.items():
            n_df_cols[f"Detected / 1,000 runs ({label})"] = ns
        st.caption(
            "How many of the 1,000 simulations successfully detected the changepoint at "
            f"npost = {npost_idx} months. Each box above is drawn from only those runs — "
            "a low count (e.g. 5% effect) means the box represents a small, potentially "
            "unrepresentative sample and its whiskers should be read with caution."
        )
        st.dataframe(n_df_cols, hide_index=True, width="stretch")

        # Key finding callout
        st.info(
            "**Key finding:** AMOC is systematically late — it typically declares the "
            "changepoint **~10 months after it actually occurred**, across all effect sizes ≥ 20% "
            "and regardless of monitoring window length. This is a structural property of the "
            "method: evidence must accumulate before the test statistic crosses the threshold. "
            "Only very small effects (5–10%) show different behaviour, with wider and more "
            "variable errors due to low detection rates.",
            icon="💡",
        )

    # ── Individual run explorer ────────────────────────────────────────────────
    st.divider()
    st.header("🔍 Individual run explorer")
    st.markdown("""
    Browse any of the 1,000 pre-computed simulations.
    """)

    exp_c1, exp_c2 = st.columns(2)
    with exp_c1:
        exp_effect_pct = st.select_slider(
            "Effect size", options=EFFECT_SIZES_PCT, value=30,
            format_func=lambda x: f"{x}%", key="exp_effect",
        )
    with exp_c2:
        exp_npost = st.select_slider(
            "Monitoring window (months)",
            options=[int(v) for v in NPOST_VEC],
            value=int(NPOST_VEC[0]),
            key="exp_npost",
        )

    # Run navigator — shared slider + ◀ ▶ buttons
    if "exp_nav_idx" not in st.session_state:
        st.session_state["exp_nav_idx"] = 0
    if "exp_nav_slider" not in st.session_state:
        st.session_state["exp_nav_slider"] = 1

    def _exp_prev():
        new = max(0, st.session_state["exp_nav_idx"] - 1)
        st.session_state["exp_nav_idx"]    = new
        st.session_state["exp_nav_slider"] = new + 1

    def _exp_next():
        new = min(999, st.session_state["exp_nav_idx"] + 1)
        st.session_state["exp_nav_idx"]    = new
        st.session_state["exp_nav_slider"] = new + 1

    def _exp_on_slider():
        st.session_state["exp_nav_idx"] = st.session_state["exp_nav_slider"] - 1

    en1, en2, en3 = st.columns([1, 10, 1])
    with en1:
        st.button("◀", on_click=_exp_prev, key="exp_nav_prev", width="stretch")
    with en2:
        st.slider("Run", 1, 1000, key="exp_nav_slider", on_change=_exp_on_slider)
    with en3:
        st.button("▶", on_click=_exp_next, key="exp_nav_next", width="stretch")

    exp_trend   = trends[EFFECT_SIZES_PCT.index(exp_effect_pct)]
    run_i       = st.session_state.get("exp_nav_idx", 0)

    npost_i_exp = int(np.searchsorted(NPOST_VEC, exp_npost, side="left"))
    npost_i_exp = min(npost_i_exp, len(NPOST_VEC) - 1)

    # Build per-model config: (label, results, critical_value_key, use_ar)
    _exp_models = [
        ("i.i.d. BACI", res_iid,    "iid_48",    False),
        ("AR(1)",        res_ar,     "ar1_48",    True),
    ]
    if res_iid_ba is not None:
        _exp_models.append(("i.i.d. BA", res_iid_ba, "iid_ba_48", False))

    _n_runs_exp = min(len(res[exp_trend]["delays"]) for _, res, _, _ in _exp_models)
    if run_i >= _n_runs_exp:
        st.warning(
            "It looks like you have run fewer simulations than expected. "
            "Please re-run the full simulation."
        )
        st.stop()

    # ── Metrics row — one column per model ────────────────────────────────────
    m_cols = st.columns(len(_exp_models))
    for col, (label, res, crit_key, _use_ar) in zip(m_cols, _exp_models):
        _delay    = int(res[exp_trend]["delays"][run_i])
        _true_cpt = NPRE + _delay
        _tmax     = float(res[exp_trend]["tmax_matrix"][run_i, npost_i_exp])
        _cpt      = int(res[exp_trend]["cpt_matrix"][run_i, npost_i_exp])
        _detected = bool(res[exp_trend]["detected_matrix"][run_i, npost_i_exp])
        _crit_val = cv[crit_key]["critical_value"] if crit_key in cv else None
        with col:
            st.markdown(f"**{label}**")
            _thresh_str = f"threshold {_crit_val:.3f}" if _crit_val is not None else "threshold —"
            st.metric("T_max", f"{_tmax:.3f}", delta=_thresh_str, delta_color="off")
            st.metric("Detected τ̂", f"month {_cpt}" if _detected else "✗ not detected")
            if _detected:
                _err = _cpt - _true_cpt
                _dir = "late" if _err > 0 else ("early" if _err < 0 else "exact")
                st.metric("Timing error", f"{_err:+d} mo ({_dir})",
                          help=f"True τ = month {_true_cpt} (pre={NPRE} + delay={_delay})")
            else:
                st.metric("Timing error", "—",
                          help=f"True τ = month {_true_cpt} (pre={NPRE} + delay={_delay})")

    # ── Time series tabs — one per model ──────────────────────────────────────
    def _build_exp_fig(res, trend, run_idx, npost_i, npost_mo, label, use_ar=False):
        _delay    = int(res[trend]["delays"][run_idx])
        _seed     = int(res[trend]["seeds"][run_idx])
        _true_cpt = NPRE + _delay
        _tmax     = float(res[trend]["tmax_matrix"][run_idx, npost_i])
        _cpt      = int(res[trend]["cpt_matrix"][run_idx, npost_i])
        _detected = bool(res[trend]["detected_matrix"][run_idx, npost_i])
        _sim_data = (_regenerate_ar(_seed, _delay, float(trend))
                     if use_ar else
                     _regenerate_iid(_seed, _delay, float(trend)))

        nt     = NPRE + npost_mo
        t      = np.arange(1, nt + 1)
        y_ctr  = _sim_data["y_ctr"][:nt]
        y_itv  = _sim_data["y_itv"][:nt]
        diff   = y_itv - y_ctr

        t_pre  = t[:_true_cpt]
        t_post = t[_true_cpt:]
        fit_pre = fit_post = None
        if len(t_pre) >= 2:
            sl, ic, *_ = linregress(t_pre, y_itv[:_true_cpt])
            fit_pre = sl * t_pre + ic
        if len(t_post) >= 2:
            sl, ic, *_ = linregress(t_post, y_itv[_true_cpt:])
            fit_post = sl * t_post + ic

        fig = make_subplots(
            rows=2, cols=1, row_heights=[0.62, 0.38],
            subplot_titles=[
                "Control vs Intervention — raw series + fitted slopes",
                "Difference (intervention − control)",
            ],
            shared_xaxes=True, vertical_spacing=0.10,
        )
        for row in [1, 2]:
            fig.add_vrect(x0=1, x1=NPRE, fillcolor="rgba(100,149,237,0.07)",
                          line_width=0, row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=y_ctr, mode="lines", name="Control",
            line=dict(color="rgba(96,125,139,0.7)", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=y_itv, mode="lines", name="Intervention",
            line=dict(color="rgba(76,175,80,0.7)", width=1.5)), row=1, col=1)
        if fit_pre is not None:
            fig.add_trace(go.Scatter(x=t_pre, y=fit_pre, mode="lines", name="Fitted (pre)",
                line=dict(color="#1565C0", width=2, dash="dash")), row=1, col=1)
        if fit_post is not None:
            fig.add_trace(go.Scatter(x=t_post, y=fit_post, mode="lines", name="Fitted (post)",
                line=dict(color="#2E7D32", width=2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=diff, mode="lines", name="Difference",
            line=dict(color="rgba(255,152,0,0.8)", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,152,0,0.10)"), row=2, col=1)
        fig.add_hline(y=0, line_color="rgba(0,0,0,0.3)", line_width=1, line_dash="dot", row=2, col=1)

        if _delay > 0:
            for row in [1, 2]:
                fig.add_vline(x=NPRE, line_dash="dot", line_color="steelblue", line_width=1.5,
                    annotation_text="Intervention" if row == 1 else "",
                    annotation_position="top right",
                    annotation_font=dict(color="steelblue", size=10), row=row, col=1)

        true_ann_side = "top right" if _true_cpt < nt * 0.75 else "top left"
        for row in [1, 2]:
            fig.add_vline(x=_true_cpt, line_dash="dash", line_color="#1A237E", line_width=2,
                annotation_text=f"True τ={_true_cpt}" if row == 1 else "",
                annotation_position=true_ann_side,
                annotation_font=dict(color="#1A237E", size=10), row=row, col=1)

        if _detected:
            close = abs(_cpt - _true_cpt) < 6
            det_ann_side = ("top left" if true_ann_side == "top right" else "top right") if close else true_ann_side
            for row in [1, 2]:
                fig.add_vline(x=_cpt, line_dash="solid", line_color="crimson", line_width=2,
                    annotation_text=f"τ̂={_cpt}" if row == 1 else "",
                    annotation_position=det_ann_side,
                    annotation_font=dict(color="crimson", size=10), row=row, col=1)

        fig.update_layout(
            height=580,
            title=(
                f"Run {run_idx + 1} · {label} · {exp_effect_pct}% effect · "
                f"npost={npost_mo} mo · delay={_delay} mo · "
                f"T_max={_tmax:.2f} ({'✓ detected' if _detected else '✗ not detected'})"
            ),
            xaxis2_title="Month", yaxis_title="Indicator value", yaxis2_title="Difference",
            legend=dict(orientation="h", yanchor="bottom", y=-0.22, font=dict(size=11)),
            margin=dict(b=80),
        )
        return fig

    _exp_cols = st.columns(len(_exp_models))
    for col, (label, res, _, use_ar) in zip(_exp_cols, _exp_models):
        with col:
            st.plotly_chart(
                _build_exp_fig(res, exp_trend, run_i, npost_i_exp, exp_npost, label, use_ar),
                width="stretch",
            )


# ══════════════════════════════════════════════════════════════════════════════
# MINI-SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("🔬 Run your own simulation")
st.markdown("""
Run a simulation directly in the browser to explore how parameter choices affect
detection.

All three noise models — **i.i.d. BACI**, **AR(1)**, and **i.i.d. BA** — run simultaneously
so you can compare their detection rates and time series side by side.
""")

if "mini_pending_restore" in st.session_state:
    _pr = st.session_state.pop("mini_pending_restore")
    st.session_state["mini_p_n_sim"]      = _pr["n_sim"]
    st.session_state["mini_p_base_seed"]  = _pr["base_seed"]
    st.session_state["mini_p_effect_pct"] = _pr["effect_pct"]
    st.session_state["mini_p_phi"]        = _pr["phi_input"]
    st.session_state["mini_p_npre"]       = _pr["npre_sim"]
    st.session_state["mini_p_npost"]      = _pr["npost_sim"]
    st.session_state["mini_p_delay"]      = _pr["delay_sim"]

for _k, _v in [
    ("mini_p_n_sim", 30), ("mini_p_base_seed", 42), ("mini_p_effect_pct", 30),
    ("mini_p_phi", PHI_DEFAULT), ("mini_p_npre", NPRE), ("mini_p_npost", 60), ("mini_p_delay", 10),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

with st.expander("⚙️ Simulation parameters", expanded=True):
    st.caption(
        "Defaults match the benchmark simulations exactly — change any value to explore "
        "how it affects detection. The comparison chart will flag differences from the benchmark."
    )
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        n_sim = st.slider(
            "N simulations",
            min_value=10, max_value=200, step=10,
            key="mini_p_n_sim",
            help=(
                "Each simulation generates a fresh (control, intervention) pair with different "
                "random noise. Detection rate = fraction of runs where T_max exceeds the critical "
                "value."
            ),
        )
        base_seed = st.number_input(
            "Random seed",
            min_value=0, max_value=99999, step=1,
            key="mini_p_base_seed",
            help=(
                "Simulation i uses seed = base_seed + i. "
                "Change the seed to get a different draw of noise realisations."
            ),
        )

    with col_b:
        effect_pct = st.select_slider(
            "Effect size (% of mean / 10 yr)",
            options=EFFECT_SIZES_PCT,
            key="mini_p_effect_pct",
            help=(
                "Additional trend added by the intervention, expressed as a percentage of the "
                "baseline level accumulated over 10 years."
            ),
        )
        phi_input = st.slider(
            "φ (AR(1) coefficient)",
            min_value=0.0, max_value=0.95, step=0.05,
            key="mini_p_phi",
            help=(
                "Autocorrelation strength for the AR(1) model only "
                "(i.i.d. BACI and i.i.d. BA always use independent errors regardless of this setting). "
                "φ = 0.5 is a moderate default. 0 = no autocorrelation; 0.9 = strong memory."
            ),
        )
        npre_sim = st.slider(
            "Pre-intervention period (months)",
            min_value=6, max_value=60, step=6,
            key="mini_p_npre",
            help=(
                f"How long the site was monitored before the intervention. "
                f"Sets where the true changepoint sits in the time series. "
                f"The benchmark used {NPRE} months ({NPRE//12} years)."
            ),
        )

    with col_c:
        npost_sim = st.slider(
            "Monitoring window (months)",
            min_value=24, max_value=NPOST_MONTHS, step=6,
            key="mini_p_npost",
            help=(
                "Number of post-intervention months included in the test. "
                "Longer windows accumulate more evidence and generally increase power."
            ),
        )
        delay_sim = st.slider(
            "Intervention delay (months)",
            min_value=0, max_value=20, step=1,
            key="mini_p_delay",
            help=(
                "Months between the formal intervention date and when the ecological response begins."
            ),
        )

_cur_params = {
    "n_sim": n_sim, "base_seed": int(base_seed), "effect_pct": effect_pct,
    "phi_input": phi_input, "npre_sim": npre_sim, "npost_sim": npost_sim, "delay_sim": delay_sim,
}
if "mini_runs_last_params" in st.session_state:
    if st.session_state["mini_runs_last_params"] != _cur_params:
        for _k in ("mini_runs", "mini_nav_idx", "mnav_slider"):
            st.session_state.pop(_k, None)
        st.session_state["mini_hist_sel_gen"] = st.session_state.get("mini_hist_sel_gen", 0) + 1

run_btn = st.button("▶ Run simulation", type="primary")

_mini_history = st.session_state.get("mini_runs_history", [])
if _mini_history:
    _sel = st.selectbox(
        "Previous runs",
        options=range(len(_mini_history)),
        format_func=lambda i: _mini_history[i]["label"],
        index=None,
        placeholder="Select a previous run to restore…",
        key=f"mini_hist_sel_{st.session_state.get('mini_hist_sel_gen', 0)}",
    )
    if _sel is not None:
        _h  = _mini_history[_sel]
        _hp = _h["params"]
        st.session_state["mini_pending_restore"]  = _hp
        st.session_state["mini_runs"]             = _h["data"]
        st.session_state["mini_nav_idx"]          = _h["nav_idx"]
        st.session_state["mnav_slider"]           = _h["nav_idx"] + 1
        st.session_state["mini_runs_last_params"] = _hp
        st.session_state["mini_hist_sel_gen"] = st.session_state.get("mini_hist_sel_gen", 0) + 1

if run_btn:
    trend_delta  = LEVEL * (effect_pct / 100) / NPOST_MONTHS
    trend_interv = TREND_CONTROL + trend_delta

    if results_available:
        _cv_ciba = cv["iid_48"]["critical_value"]
        _cv_ar   = cv["ar1_48"]["critical_value"]
        _cv_ba   = cv.get("iid_ba_48", cv["iid_48"])["critical_value"]
    else:
        _cv_ciba = _cv_ar = _cv_ba = 2.5

    effective_npost = npost_sim - delay_sim
    if effective_npost < 2:
        st.error(
            f"Intervention delay ({delay_sim} mo) must be at least 2 months shorter than the "
            f"monitoring window ({npost_sim} mo). Reduce the delay or increase the window."
        )
        st.stop()

    true_cpt = npre_sim + delay_sim
    nt_total  = npre_sim + npost_sim

    _mini_models = {
        "i.i.d. BACI": {"all_data": [], "all_stats": [], "detected_flags": [], "detected_cpts": [], "crit_val": _cv_ciba},
        "AR(1)":        {"all_data": [], "all_stats": [], "detected_flags": [], "detected_cpts": [], "crit_val": _cv_ar},
        "i.i.d. BA":    {"all_data": [], "all_stats": [], "detected_flags": [], "detected_cpts": [], "crit_val": _cv_ba},
    }

    progress_bar = st.progress(0, text="Starting…")

    for i in range(n_sim):
        # i.i.d. data — shared between BACI (uses control+intervention) and BA (intervention only)
        sim_iid    = ci_sim(seed=int(base_seed) + i, npre=true_cpt, npost=effective_npost,
                            level=LEVEL, trend=(TREND_CONTROL, trend_interv), sigma=SIGMA)
        stats_ciba = trend_stats(y_ctr=sim_iid["y_ctr"], y_itv=sim_iid["y_itv"], nt=nt_total)
        stats_ba   = trend_stats(y_itv=sim_iid["y_itv"], nt=nt_total)

        # AR(1) data — independent realisation
        sim_ar   = ci_sim_ar(seed=int(base_seed) + i, npre=true_cpt, npost=effective_npost,
                             level=LEVEL, trend=(TREND_CONTROL, trend_interv), phi=phi_input, sigma=SIGMA)
        stats_ar = trend_stats_ar(y_itv=sim_ar["y_itv"], nt=nt_total)

        for label, sim_data, stats in [
            ("i.i.d. BACI", sim_iid, stats_ciba),
            ("AR(1)",        sim_ar,  stats_ar),
            ("i.i.d. BA",   sim_iid, stats_ba),
        ]:
            crit     = _mini_models[label]["crit_val"]
            detected = stats["Tmax"] > crit
            _mini_models[label]["all_data"].append(sim_data)
            _mini_models[label]["all_stats"].append(stats)
            _mini_models[label]["detected_flags"].append(detected)
            _mini_models[label]["detected_cpts"].append(stats["cpt"] if detected else np.nan)

        progress_bar.progress((i + 1) / n_sim, text=f"Simulation {i+1}/{n_sim}…")

    progress_bar.empty()

    for mdata in _mini_models.values():
        flags = mdata["detected_flags"]
        cpts  = mdata["detected_cpts"]
        mdata["det_rate"]   = float(np.mean(flags))
        mdata["n_detected"] = int(sum(flags))
        mdata["valid_cpts"] = [c for c in cpts if not np.isnan(c)]

    first_det = next(
        (i for i, d in enumerate(_mini_models["i.i.d. BACI"]["detected_flags"]) if d), 0
    )
    st.session_state["mini_runs"] = {
        "n_sim":      n_sim,
        "base_seed":  int(base_seed),
        "effect_pct": effect_pct,
        "npre_sim":   npre_sim,
        "npost_sim":  npost_sim,
        "delay_sim":  delay_sim,
        "true_cpt":   true_cpt,
        "models":     _mini_models,
    }
    st.session_state["mini_nav_idx"] = first_det
    st.session_state["mnav_slider"]  = first_det + 1

    _run_n     = len(st.session_state.get("mini_runs_history", [])) + 1
    _baci_rate = _mini_models["i.i.d. BACI"]["det_rate"]
    _ts        = datetime.now().strftime("%H:%M")
    _hist_lbl  = (
        f"#{_run_n} · {_ts} · N={n_sim} seed={int(base_seed)} eff={effect_pct}% φ={phi_input:.2f} "
        f"pre={npre_sim}mo post={npost_sim}mo delay={delay_sim}mo · det={_baci_rate:.0%}"
    )
    if "mini_runs_history" not in st.session_state:
        st.session_state["mini_runs_history"] = []
    st.session_state["mini_runs_history"].insert(0, {
        "label":   _hist_lbl,
        "data":    st.session_state["mini_runs"],
        "nav_idx": first_det,
        "params":  _cur_params,
    })
    st.session_state["mini_runs_last_params"] = _cur_params


# ── Results — persists across rerenders via session state ─────────────────────
if "mini_runs" in st.session_state and "models" in st.session_state["mini_runs"]:
    mr            = st.session_state["mini_runs"]
    _model_labels = list(mr["models"].keys())

    det_rates_str = ", ".join(
        f"**{k}: {mr['models'][k]['det_rate']:.1%}**" for k in _model_labels
    )
    st.success(f"Simulation complete!")

    # ── Summary metrics — one column per model ─────────────────────────────────
    sm_cols = st.columns(len(_model_labels))
    for col, label in zip(sm_cols, _model_labels):
        mdata       = mr["models"][label]
        valid_cpts  = mdata["valid_cpts"]
        true_cpt_mr = mr["true_cpt"]
        with col:
            st.markdown(f"**{label}**")
            st.metric("Detection rate", f"{mdata['det_rate']:.1%}",
                      help=f"{mdata['n_detected']} of {mr['n_sim']} simulations")
            st.metric("Critical value", f"{mdata['crit_val']:.3f}")
            if valid_cpts:
                mean_err   = np.mean(np.abs(np.array(valid_cpts) - true_cpt_mr))
                median_lag = float(np.median(np.array(valid_cpts) - true_cpt_mr))
                lag_dir    = "late" if median_lag > 0 else ("early" if median_lag < 0 else "exact")
                st.metric("Mean |timing error|", f"{mean_err:.1f} mo",
                          help="Average |τ̂ − true τ| across detected runs.")
                st.metric("Median detection lag", f"{median_lag:+.1f} mo ({lag_dir})",
                          help="Median (τ̂ − true τ). Positive = declared later than true change.")
            else:
                st.metric("Mean |timing error|", "n/a")
                st.metric("Median detection lag", "n/a")

    # ── Run navigator ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Browse simulation runs")

    n_sim_mr    = mr["n_sim"]
    det_indices = [i for i, d in enumerate(mr["models"]["i.i.d. BACI"]["detected_flags"]) if d]

    if "mini_nav_idx" not in st.session_state:
        st.session_state["mini_nav_idx"] = 0
    if "mnav_slider" not in st.session_state:
        st.session_state["mnav_slider"] = 1

    def _nav_prev():
        new = max(0, st.session_state["mini_nav_idx"] - 1)
        st.session_state["mini_nav_idx"] = new
        st.session_state["mnav_slider"]  = new + 1

    def _nav_next():
        new = min(n_sim_mr - 1, st.session_state["mini_nav_idx"] + 1)
        st.session_state["mini_nav_idx"] = new
        st.session_state["mnav_slider"]  = new + 1

    def _on_slider():
        st.session_state["mini_nav_idx"] = st.session_state["mnav_slider"] - 1

    nc1, nc2, nc3 = st.columns([1, 10, 1])
    with nc1:
        st.button("◀", on_click=_nav_prev, key="mnav_prev", width="stretch")
    with nc2:
        st.slider("Run", 1, n_sim_mr, key="mnav_slider", on_change=_on_slider)
    with nc3:
        st.button("▶", on_click=_nav_next, key="mnav_next", width="stretch")

    det_pct = len(det_indices) / n_sim_mr if n_sim_mr > 0 else 0
    st.caption(
        f"i.i.d. BACI detected a change in {len(det_indices)} of {n_sim_mr} runs ({det_pct:.0%})."
        if det_indices else
        "No runs detected a change (i.i.d. BACI) — try a larger effect size or longer window."
    )

    # ── Current run ───────────────────────────────────────────────────────────
    show_idx   = st.session_state.get("mini_nav_idx", 0)
    run_num    = show_idx + 1
    show_seed  = mr["base_seed"] + show_idx
    true_cpt   = mr["true_cpt"]
    npre_sim   = mr["npre_sim"]
    npost_sim  = mr["npost_sim"]
    delay_sim  = mr["delay_sim"]
    effect_pct = mr["effect_pct"]

    st.subheader(
        f"Run #{run_num} of {n_sim_mr} · base seed {mr['base_seed']} · run seed {show_seed}"
    )
    st.markdown(f"""
        - **Colour code:** `grey = control site`; `green = intervention site`.
        - **Dashed fitted lines** show trends before and after τ on the intervention series.
        - **Bottom panel:** difference (intervention − control) isolates the signal.
        """
    )
    if delay_sim > 0:
        st.caption(
            f"Blue dotted = month {npre_sim} (formal intervention). "
            f"Navy dashed = month {true_cpt} (ecological response begins, {delay_sim} month(s) later). "
            "Red solid = detected τ̂ (where present)."
        )

    # ── Per-model metrics ──────────────────────────────────────────────────────
    pm_cols = st.columns(len(_model_labels))
    for col, label in zip(pm_cols, _model_labels):
        mdata        = mr["models"][label]
        show_stats   = mdata["all_stats"][show_idx]
        was_detected = mdata["detected_flags"][show_idx]
        crit_val     = mdata["crit_val"]
        det_cpt      = show_stats["cpt"] if was_detected else None
        err          = (det_cpt - true_cpt) if det_cpt is not None else None
        err_dir      = "late" if (err and err > 0) else ("early" if (err and err < 0) else "exact")
        with col:
            st.markdown(f"**{label}**")
            st.metric("T_max", f"{show_stats['Tmax']:.3f}",
                      delta=f"threshold {crit_val:.3f}", delta_color="off")
            st.metric("Detected τ̂", f"month {det_cpt}" if was_detected else "✗ not detected")
            if was_detected:
                st.metric("Timing error", f"{err:+d} mo ({err_dir})",
                          help=f"True τ = month {true_cpt} (pre={npre_sim} + delay={delay_sim})")
            else:
                st.metric("Timing error", "—",
                          help=f"True τ = month {true_cpt} (pre={npre_sim} + delay={delay_sim})")

    # ── Time series charts — one per model, side by side ──────────────────────
    def _build_mini_fig(sim_data, stats, was_detected, label, npre, npost, delay, true_cpt_val, crit_val):
        nt     = npre + npost
        t      = np.arange(1, nt + 1)
        y_ctr  = sim_data["y_ctr"]
        y_itv  = sim_data["y_itv"]
        diff   = y_itv - y_ctr
        t_pre  = t[:true_cpt_val]
        t_post = t[true_cpt_val:]

        sl_pre,  ic_pre,  *_ = linregress(t_pre,  y_itv[:true_cpt_val])
        sl_post, ic_post, *_ = linregress(t_post, y_itv[true_cpt_val:])
        fit_pre  = sl_pre  * t_pre  + ic_pre
        fit_post = sl_post * t_post + ic_post
        fit_diff_post = None
        if len(t_post) >= 2:
            sl_d, ic_d, *_ = linregress(t_post, diff[true_cpt_val:])
            fit_diff_post = sl_d * t_post + ic_d

        det_badge = "✓ detected" if was_detected else "✗ not detected"
        fig = make_subplots(
            rows=2, cols=1, row_heights=[0.62, 0.38],
            subplot_titles=[
                "Control vs Intervention",
                "Difference (intervention − control)",
            ],
            shared_xaxes=True, vertical_spacing=0.10,
        )
        for row in [1, 2]:
            fig.add_vrect(x0=1, x1=npre, fillcolor="rgba(100,149,237,0.07)",
                          line_width=0, row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=y_ctr, mode="lines", name="Control",
            line=dict(color="rgba(96,125,139,0.7)", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=y_itv, mode="lines", name="Intervention",
            line=dict(color="rgba(76,175,80,0.7)", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_pre, y=fit_pre, mode="lines", name="Fitted (pre)",
            line=dict(color="#1565C0", width=2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=t_post, y=fit_post, mode="lines", name="Fitted (post)",
            line=dict(color="#2E7D32", width=2, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=diff, mode="lines", name="Difference",
            line=dict(color="rgba(255,152,0,0.8)", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,152,0,0.10)"), row=2, col=1)
        if fit_diff_post is not None:
            fig.add_trace(go.Scatter(x=t_post, y=fit_diff_post, mode="lines", name="Divergence trend",
                line=dict(color="#E65100", width=2, dash="dash")), row=2, col=1)
        fig.add_hline(y=0, line_color="rgba(0,0,0,0.3)", line_width=1, line_dash="dot", row=2, col=1)

        if delay > 0:
            for row in [1, 2]:
                fig.add_vline(x=npre, line_dash="dot", line_color="steelblue", line_width=1.5,
                    annotation_text="Intervention" if row == 1 else "",
                    annotation_position="top right",
                    annotation_font=dict(color="steelblue", size=10), row=row, col=1)

        true_ann_side = "top right" if true_cpt_val < nt * 0.75 else "top left"
        for row in [1, 2]:
            fig.add_vline(x=true_cpt_val, line_dash="dash", line_color="#1A237E", line_width=2,
                annotation_text=f"True τ={true_cpt_val}" if row == 1 else "",
                annotation_position=true_ann_side,
                annotation_font=dict(color="#1A237E", size=10), row=row, col=1)

        if was_detected:
            close        = abs(stats["cpt"] - true_cpt_val) < 6
            det_ann_side = ("top left" if true_ann_side == "top right" else "top right") if close else true_ann_side
            for row in [1, 2]:
                fig.add_vline(x=stats["cpt"], line_dash="solid", line_color="crimson", line_width=2,
                    annotation_text=f"τ̂={stats['cpt']}" if row == 1 else "",
                    annotation_position=det_ann_side,
                    annotation_font=dict(color="crimson", size=10), row=row, col=1)

        fig.update_layout(
            height=560,
            title=f"{label} · T_max={stats['Tmax']:.2f} ({det_badge})",
            xaxis2_title="Month",
            yaxis_title="Indicator value",
            yaxis2_title="Difference",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, font=dict(size=10)),
            margin=dict(b=90),
        )
        return fig

    ch_cols = st.columns(len(_model_labels))
    for col, label in zip(ch_cols, _model_labels):
        mdata        = mr["models"][label]
        show_data    = mdata["all_data"][show_idx]
        show_stats   = mdata["all_stats"][show_idx]
        was_detected = mdata["detected_flags"][show_idx]
        crit_val     = mdata["crit_val"]
        with col:
            st.plotly_chart(
                _build_mini_fig(show_data, show_stats, was_detected, label,
                                npre_sim, npost_sim, delay_sim, true_cpt, crit_val),
                width="stretch",
            )
