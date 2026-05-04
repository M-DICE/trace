"""
SimRewilding — Distribution Change Detection using AMOC
Interactive analysis page for rewild_distribution_change_amoc results.
"""

import pickle
from datetime import datetime
from pathlib import Path
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from tracepy.simulation.distribution import ci_sim_cdf
from tracepy.stats.metrics import (
    wasserstein_distance_baci, wasserstein_distance_ba,
    auc_diff_ts, trend_stats_cdf,
)
from scipy.stats import gaussian_kde

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Distribution Change Detection (AMOC)",
    page_icon="📊",
    layout="wide",
)

# ── Simulation constants (mirror rewild_distribution_change_amoc.py) ──────────
NPRE = 24
NPOST_VEC = np.arange(24, 121, 3)
NPOST_MAX = 120
MU = 10.0
SIGMA = 1.0
NS = 200
BW = 0.3
ND = 50

TREND_INCREASE_MU = np.round(MU * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / 120, 4)
TREND_INCREASE_SIGMA = np.round(SIGMA * np.arange(0.5, 2.1, 0.5) / 120, 4)

EFFECT_SIZES_MU_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
EFFECT_SIZES_SIGMA_PCT = [50, 100, 150, 200]

RESULTS_PATH = Path(__file__).parent.parent / "results" / "distribution_amoc" / "sim_results.pkl"

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading distribution simulation results…")
def load_results(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Regenerating simulation run…")
def _regenerate_sim(seed: int, delay: int, trend_mu: float, trend_sigma: float):
    npre_delay  = NPRE + delay
    npost_delay = NPOST_MAX - delay
    return ci_sim_cdf(
        seed=seed, npre=npre_delay, npost=npost_delay,
        level=[MU, SIGMA], trend=[trend_mu, trend_sigma], ns=NS,
    )


def dist_label(trend_val, pct):
    return f"{pct}% ({trend_val:.4f}/mo)"


# ── Colour palettes ────────────────────────────────────────────────────────────
PALETTE_MU    = px.colors.sample_colorscale("Viridis", [i / 10 for i in range(11)])
PALETTE_SIGMA = px.colors.sample_colorscale("Plasma",  [i / 3  for i in range(4)])

# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("📊 Distribution Change Detection (AMOC)")

st.markdown("""
Rewilding interventions can affect not just the average abundance of a species, but also its
**variability** or the entire **shape of its distribution**.

This page analyses changes in distributions using distance measure **Wasserstein Distance** (BACI and BA).

We apply the AMOC (At Most One Change) method to the **time series of distances** to
identify when the distributions start to diverge.
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
    | **npre** | {NPRE} months | Pre-intervention period |
    | **npost_max** | {NPOST_MAX} months | Max post-intervention window |
    | **ns** | {NS} | Samples per distribution time point |
    | **dist_measure** | Wasserstein | Default distance |
    | **mu** | {MU} | Baseline mean |
    | **sigma** | {SIGMA} | Baseline standard deviation |
    | **Nsim** | 1,000 | Null simulations for critical value |
    | **simN** | 1,000 | Main simulations per effect size |
    """)

# ── Load data ──────────────────────────────────────────────────────────────────
data = load_results(RESULTS_PATH)

if data is None:
    st.error(
        f"Pre-computed results not found at `{RESULTS_PATH}`. "
        "Run `python rewild_distribution_change_amoc.py` first, "
        "or use the Run your own simulation section below."
    )
    results_available = False
else:
    cv        = data.get("critical_values")
    res_mu    = data.get("detection_results_mu", {})
    res_sigma = data.get("detection_results_sigma", {})
    if cv is None or not res_mu or not res_sigma:
        st.warning(
            "Saved results are incomplete (critical values or detection results missing). "
            "Re-run `python rewild_distribution_change_amoc.py` to regenerate."
        )
        results_available = False
    else:
        trends_mu    = sorted(res_mu.keys())
        trends_sigma = sorted(res_sigma.keys())
        mu_pct_to_trend    = dict(zip(EFFECT_SIZES_MU_PCT,    trends_mu))
        sigma_pct_to_trend = dict(zip(EFFECT_SIZES_SIGMA_PCT, trends_sigma))
        results_available = True

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
if results_available:
    st.header("📊 Results explorer")
    st.markdown(
        "Charts below are drawn from pre-computed simulations (1,000 runs per effect size)."
    )

    _tabs = ["Power curves", "Detection by delay", "Null distributions", "Estimation error"]
    tab_power, tab_delay, tab_null, tab_err = st.tabs(_tabs)

    # ── Tab 1: Power curves ────────────────────────────────────────────────────
    with tab_power:
        st.subheader("Power curves")
        st.markdown("""
        Each line shows the **probability of detecting the change** as the
        post-intervention monitoring window grows. Brighter/yellower lines are larger effect sizes.
        Dashed lines mark 80% and 95% power thresholds.
        """)

        power_choice = st.radio(
            "Change type",
            ["Mean shifts (Mu)", "Variance shifts (Sigma)", "Side-by-side"],
            horizontal=True,
            key="dist_power_choice",
        )

        def _power_fig_mu():
            fig = go.Figure()
            for i, (t, pct) in enumerate(zip(trends_mu, EFFECT_SIZES_MU_PCT)):
                fig.add_trace(go.Scatter(
                    x=NPOST_VEC, y=res_mu[t]["detection_rates"],
                    mode="lines", name=f"{pct}%",
                    line=dict(color=PALETTE_MU[i], width=2),
                    hovertemplate="npost: %{x} mo<br>Detection rate: %{y:.1%}<extra></extra>",
                ))
            fig.add_hline(y=0.80, line_dash="dash", line_color="grey",
                          annotation_text="80% power", annotation_position="right")
            fig.add_hline(y=0.95, line_dash="dot", line_color="grey",
                          annotation_text="95% power", annotation_position="right")
            fig.update_layout(
                title="Power curves — mean shifts (Mu)",
                xaxis_title="Post-intervention monitoring window (months)",
                yaxis_title="Detection rate",
                yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                legend_title="Effect size<br>(% of mean / 10 yr)",
                height=480,
            )
            return fig

        def _power_fig_sigma():
            fig = go.Figure()
            for i, (t, pct) in enumerate(zip(trends_sigma, EFFECT_SIZES_SIGMA_PCT)):
                fig.add_trace(go.Scatter(
                    x=NPOST_VEC, y=res_sigma[t]["detection_rates"],
                    mode="lines", name=f"{pct}%",
                    line=dict(color=PALETTE_SIGMA[i], width=2),
                    hovertemplate="npost: %{x} mo<br>Detection rate: %{y:.1%}<extra></extra>",
                ))
            fig.add_hline(y=0.80, line_dash="dash", line_color="grey",
                          annotation_text="80% power", annotation_position="right")
            fig.add_hline(y=0.95, line_dash="dot", line_color="grey",
                          annotation_text="95% power", annotation_position="right")
            fig.update_layout(
                title="Power curves — variance shifts (Sigma)",
                xaxis_title="Post-intervention monitoring window (months)",
                yaxis_title="Detection rate",
                yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                legend_title="Effect size<br>(% of sigma / 10 yr)",
                height=480,
            )
            return fig

        if power_choice == "Mean shifts (Mu)":
            st.plotly_chart(_power_fig_mu(), width="stretch")
        elif power_choice == "Variance shifts (Sigma)":
            st.plotly_chart(_power_fig_sigma(), width="stretch")
        else:
            cols = st.columns(2)
            with cols[0]:
                st.plotly_chart(_power_fig_mu(), width="stretch")
            with cols[1]:
                st.plotly_chart(_power_fig_sigma(), width="stretch")

    # ── Tab 2: Detection by delay ──────────────────────────────────────────────
    with tab_delay:
        st.subheader("Effect of intervention delay on detection")
        st.markdown("""
        The ecological response to a rewilding intervention rarely begins immediately.
        This tab shows how a **lag** between the formal intervention and the start of the
        distributional shift erodes detection probability.

        Choose a change type and effect size, then explore two views:
        - **Line chart** — detection rate vs delay at a chosen monitoring window, with ±1 SE bands.
        - **Heatmap** — the full 2-D picture: how delay and monitoring window interact.
        """)

        delay_c1, delay_c2, delay_c3 = st.columns(3)
        with delay_c1:
            delay_change_type = st.radio(
                "Change type",
                ["Mean (Mu)", "Variance (Sigma)"],
                horizontal=True,
                key="delay_change_type",
            )
        with delay_c2:
            if delay_change_type == "Mean (Mu)":
                delay_effect_pct = st.select_slider(
                    "Effect size", options=list(mu_pct_to_trend.keys()), value=30,
                    format_func=lambda x: f"{x}%", key="delay_effect_mu",
                )
                delay_results   = res_mu
                delay_pct_map   = mu_pct_to_trend
            else:
                delay_effect_pct = st.select_slider(
                    "Effect size", options=list(sigma_pct_to_trend.keys()), value=100,
                    format_func=lambda x: f"{x}%", key="delay_effect_sigma",
                )
                delay_results   = res_sigma
                delay_pct_map   = sigma_pct_to_trend
        with delay_c3:
            delay_view = st.radio(
                "View", ["Line chart", "Heatmap"],
                horizontal=True, key="delay_view",
            )

        delay_trend_key = delay_pct_map[delay_effect_pct]
        unique_delays   = np.arange(1, 21)
        _n_per_bin = int((np.array(delay_results[delay_trend_key]["delays"]) == 1).sum())

        def _delay_series(results, trend, npost_i):
            delays_arr   = np.array(results[trend]["delays"])
            detected_arr = results[trend]["detected_matrix"][:, npost_i]
            rates, lo, hi = [], [], []
            for d in unique_delays:
                mask = delays_arr == d
                n = mask.sum()
                r  = detected_arr[mask].mean() if n > 0 else np.nan
                se = np.sqrt(r * (1 - r) / n) if n > 1 and not np.isnan(r) else 0.0
                rates.append(r)
                lo.append(max(0.0, r - se))
                hi.append(min(1.0, r + se))
            return np.array(rates), np.array(lo), np.array(hi)

        if delay_view == "Line chart":
            delay_npost = st.select_slider(
                "Monitoring window (months)",
                options=[int(v) for v in NPOST_VEC],
                value=int(NPOST_VEC[0]),
                key="delay_npost_line",
            )
            npost_i_line = min(
                int(np.searchsorted(NPOST_VEC, delay_npost, side="left")),
                len(NPOST_VEC) - 1,
            )
            rates, lo, hi = _delay_series(delay_results, delay_trend_key, npost_i_line)

            colour      = "#2196F3"
            colour_fill = "rgba(33,150,243,0.15)"
            fig_dl = go.Figure()
            fig_dl.add_trace(go.Scatter(
                x=np.concatenate([unique_delays, unique_delays[::-1]]),
                y=np.concatenate([hi, lo[::-1]]),
                fill="toself", fillcolor=colour_fill,
                line=dict(width=0), showlegend=False, hoverinfo="skip",
            ))
            fig_dl.add_trace(go.Scatter(
                x=unique_delays, y=rates,
                mode="lines+markers",
                name=f"{delay_effect_pct}% effect",
                line=dict(color=colour, width=2.5),
                marker=dict(size=7),
                hovertemplate="Delay: %{x} mo<br>Detection: %{y:.1%}<extra></extra>",
            ))
            fig_dl.add_hline(y=0.80, line_dash="dash", line_color="grey",
                             annotation_text="80% power", annotation_position="right")
            fig_dl.update_layout(
                title=(
                    f"Detection rate vs intervention delay — {delay_effect_pct}% effect, "
                    f"npost = {delay_npost} mo<br>"
                    f"<sup>Shaded band = ±1 SE (≈{_n_per_bin} simulations per delay bin)</sup>"
                ),
                xaxis=dict(title="Intervention delay (months)", dtick=2),
                yaxis=dict(title="Detection rate", range=[0, 1.05], tickformat=".0%"),
                height=460,
            )
            st.plotly_chart(fig_dl, width="stretch")

            drop = float(np.nanmax(rates) - np.nanmin(rates))
            st.caption(
                f"Across delays 1–20 months, detection rate spans "
                f"**{float(np.nanmin(rates)):.0%} – {float(np.nanmax(rates)):.0%}** "
                f"(a drop of {drop:.0%}) for the {delay_effect_pct}% effect at npost = {delay_npost} mo."
            )

        else:
            delays_arr = np.array(delay_results[delay_trend_key]["delays"])
            det_mat    = delay_results[delay_trend_key]["detected_matrix"]
            z = np.full((len(unique_delays), len(NPOST_VEC)), np.nan)
            for di, d in enumerate(unique_delays):
                mask = delays_arr == d
                if mask.any():
                    z[di, :] = det_mat[mask, :].mean(axis=0)

            fig_heat = go.Figure()
            fig_heat.add_trace(go.Heatmap(
                x=NPOST_VEC, y=unique_delays, z=z,
                colorscale="Viridis", zmin=0, zmax=1,
                colorbar=dict(title="Detection rate", tickformat=".0%"),
                hovertemplate="npost: %{x} mo<br>Delay: %{y} mo<br>Detection: %{z:.1%}<extra></extra>",
            ))
            fig_heat.add_trace(go.Contour(
                x=NPOST_VEC, y=unique_delays, z=z,
                contours=dict(
                    coloring="none", showlabels=True,
                    start=0.80, end=0.95, size=0.15,
                    labelfont=dict(size=11, color="white"),
                ),
                line=dict(color="white", dash="dash"),
                showscale=False, hoverinfo="skip",
            ))
            fig_heat.update_layout(
                title=(
                    f"Detection rate: intervention delay × monitoring window — "
                    f"{delay_effect_pct}% effect<br>"
                    f"<sup>Contour lines at 80% and 95% power</sup>"
                ),
                xaxis_title="Post-intervention monitoring window (months)",
                yaxis=dict(title="Intervention delay (months)", dtick=2),
                height=500,
            )
            st.plotly_chart(fig_heat, width="stretch")
            st.caption(
                "**How to read this:** find your expected delay on the y-axis, then read "
                "across to where the colour reaches 80% (first white contour line) — "
                "that x-value is the minimum monitoring window needed."
            )

    # ── Tab 3: Null distributions ──────────────────────────────────────────────
    with tab_null:
        st.subheader("Null distributions of distance trend statistics")
        st.markdown("""
        The histograms show T_max under the null hypothesis (no change) for three monitoring
        window lengths. The crimson line marks the **95th percentile** — the critical value
        used to declare a changepoint.
        """)
        null_keys = [("24", "npost = 24 mo"), ("72", "npost = 72 mo"), ("120", "npost = 120 mo")]
        fig_null = make_subplots(
            rows=1, cols=3,
            subplot_titles=[title for _, title in null_keys],
        )
        for i, (label, _) in enumerate(null_keys):
            key = f"npost_{label}"
            if key not in cv:
                continue
            dist = cv[key]["null_dist"]
            crit = cv[key]["critical_value"]
            fig_null.add_trace(
                go.Histogram(
                    x=dist, nbinsx=50,
                    marker_color="#2196F3", opacity=0.75,
                    showlegend=False,
                    hovertemplate="T_max: %{x:.2f}<br>Count: %{y}<extra></extra>",
                ),
                row=1, col=i + 1,
            )
            fig_null.add_vline(
                x=crit, line_dash="dash", line_color="crimson",
                annotation_text=f"cv = {crit:.2f}",
                annotation_position="top right",
                row=1, col=i + 1,
            )
        fig_null.update_layout(
            height=420,
            title="Null distributions of T_max (1,000 simulations each)",
        )
        fig_null.update_xaxes(title_text="T_max")
        fig_null.update_yaxes(title_text="Count")
        st.plotly_chart(fig_null, width="stretch")

        st.markdown("**Critical values:**")
        cv_table = {
            "Scenario": [f"npost = {label} months" for label, _ in null_keys],
            "Critical value (95th pct)": [
                f"{cv[f'npost_{label}']['critical_value']:.3f}" if f"npost_{label}" in cv else "—"
                for label, _ in null_keys
            ],
            "Null mean": [
                f"{np.mean(cv[f'npost_{label}']['null_dist']):.3f}" if f"npost_{label}" in cv else "—"
                for label, _ in null_keys
            ],
            "Null SD": [
                f"{np.std(cv[f'npost_{label}']['null_dist']):.3f}" if f"npost_{label}" in cv else "—"
                for label, _ in null_keys
            ],
        }
        st.table(cv_table)

    # ── Tab 5: Estimation error ────────────────────────────────────────────────
    with tab_err:
        st.subheader("Changepoint localisation error")
        st.markdown("""
        The mean absolute error (months) between the detected changepoint and the true
        changepoint. Smaller = better. Error decreases as more post-intervention data arrives.
        """)
        err_type = st.radio(
            "Change type",
            ["Mean (Mu)", "Variance (Sigma)"],
            horizontal=True,
            key="err_type",
            captions=[
                "Localisation error when detecting a shift in the mean (μ) of the distribution.",
                "Localisation error when detecting a shift in the variance (σ) of the distribution.",
            ],
        )

        if err_type == "Mean (Mu)":
            err_results = res_mu
            err_trends  = trends_mu
            err_pcts    = EFFECT_SIZES_MU_PCT
            err_palette = PALETTE_MU
        else:
            err_results = res_sigma
            err_trends  = trends_sigma
            err_pcts    = EFFECT_SIZES_SIGMA_PCT
            err_palette = PALETTE_SIGMA

        fig_err = go.Figure()
        for i, (t, pct) in enumerate(zip(err_trends, err_pcts)):
            errs = err_results[t]["mean_errors"]
            fig_err.add_trace(go.Scatter(
                x=NPOST_VEC, y=errs,
                mode="lines",
                name=f"{pct}%",
                line=dict(color=err_palette[i], width=2),
                hovertemplate="npost: %{x} mo<br>Mean |error|: %{y:.1f} mo<extra></extra>",
            ))
        fig_err.update_layout(
            xaxis_title="Post-intervention monitoring window (months)",
            yaxis_title="Mean |τ̂ − τ_true| (months)",
            legend_title="Effect size",
            height=460,
        )
        st.plotly_chart(fig_err, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL RUN EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
if results_available:
    st.divider()
    st.header("🔍 Individual run explorer")
    st.markdown("Browse any of the 1,000 pre-computed simulations.")

    exp_c1, exp_c2, exp_c3 = st.columns(3)
    with exp_c1:
        exp_type = st.radio("Change type", ["Mean (Mu)", "Variance (Sigma)"],
                            horizontal=True, key="exp_type")
    with exp_c2:
        if exp_type == "Mean (Mu)":
            exp_pct = st.select_slider("Effect size", options=list(mu_pct_to_trend.keys()), value=30,
                                       format_func=lambda x: f"{x}%", key="exp_pct_mu")
            exp_trend   = mu_pct_to_trend[exp_pct]
            exp_results = res_mu
        else:
            exp_pct = st.select_slider("Effect size", options=list(sigma_pct_to_trend.keys()), value=100,
                                       format_func=lambda x: f"{x}%", key="exp_pct_sigma")
            exp_trend   = sigma_pct_to_trend[exp_pct]
            exp_results = res_sigma
    with exp_c3:
        exp_npost = st.select_slider(
            "Monitoring window (months)",
            options=[int(v) for v in NPOST_VEC],
            value=int(NPOST_VEC[0]),
            key="exp_npost",
        )

    npost_i_exp = int(np.searchsorted(NPOST_VEC, exp_npost, side="left"))
    npost_i_exp = min(npost_i_exp, len(NPOST_VEC) - 1)

    # ── Run navigator — slider + ◀ ▶ buttons ──────────────────────────────────
    if "exp_nav_idx" not in st.session_state:
        st.session_state["exp_nav_idx"] = 0
    if "exp_nav_slider" not in st.session_state:
        st.session_state["exp_nav_slider"] = 1

    n_runs = len(exp_results[exp_trend]["delays"])

    run_i = min(st.session_state.get("exp_nav_idx", 0), n_runs - 1)
    st.session_state["exp_nav_idx"]    = run_i
    st.session_state["exp_nav_slider"] = min(st.session_state.get("exp_nav_slider", 1), n_runs)

    def _exp_prev():
        new = max(0, st.session_state["exp_nav_idx"] - 1)
        st.session_state["exp_nav_idx"]    = new
        st.session_state["exp_nav_slider"] = new + 1

    def _exp_next():
        new = min(n_runs - 1, st.session_state["exp_nav_idx"] + 1)
        st.session_state["exp_nav_idx"]    = new
        st.session_state["exp_nav_slider"] = new + 1

    def _exp_on_slider():
        st.session_state["exp_nav_idx"] = st.session_state["exp_nav_slider"] - 1

    en1, en2, en3 = st.columns([1, 10, 1])
    with en1:
        st.button("◀", on_click=_exp_prev, key="exp_nav_prev", width="stretch")
    with en2:
        st.slider("Run", 1, n_runs, key="exp_nav_slider", on_change=_exp_on_slider)
    with en3:
        st.button("▶", on_click=_exp_next, key="exp_nav_next", width="stretch")

    _delay    = int(exp_results[exp_trend]["delays"][run_i])
    _seed     = int(exp_results[exp_trend]["seeds"][run_i])
    _true_cpt = NPRE + _delay
    _tmax     = float(exp_results[exp_trend]["tmax_matrix"][run_i, npost_i_exp])
    _cpt      = int(exp_results[exp_trend]["cpt_matrix"][run_i, npost_i_exp])
    _detected = bool(exp_results[exp_trend]["detected_matrix"][run_i, npost_i_exp])

    _trend_mu_val    = float(exp_trend) if exp_type == "Mean (Mu)"      else 0.0
    _trend_sigma_val = 0.0              if exp_type == "Mean (Mu)"      else float(exp_trend)
    _sim_data = _regenerate_sim(_seed, _delay, _trend_mu_val, _trend_sigma_val)

    # ── Metrics row ───────────────────────────────────────────────────────────
    crit_val = cv.get("npost_24", {}).get("critical_value")
    m1, m2, m3 = st.columns(3)
    with m1:
        _thresh_str = f"threshold {crit_val:.3f}" if crit_val is not None else "threshold —"
        st.metric("T_max", f"{_tmax:.3f}", delta=_thresh_str, delta_color="off")
    with m2:
        st.metric("Detected τ̂", f"month {_cpt}" if _detected else "✗ not detected")
    with m3:
        if _detected:
            _err = _cpt - _true_cpt
            _dir = "late" if _err > 0 else ("early" if _err < 0 else "exact")
            st.metric("Timing error", f"{_err:+d} mo ({_dir})",
                      help=f"True τ = month {_true_cpt} (pre={NPRE} + delay={_delay})")
        else:
            st.metric("Timing error", "—",
                      help=f"True τ = month {_true_cpt} (pre={NPRE} + delay={_delay})")

    # ── Distance + mean-difference time series ────────────────────────────────
    nt = NPRE + exp_npost
    dist_ts = wasserstein_distance_baci(
        _sim_data["sample_ctr"][:, :nt],
        _sim_data["sample_itv"][:, :nt],
    )
    t_ax = np.arange(1, nt + 1)
    mean_diff = (_sim_data["sample_itv"][:, :nt].mean(axis=0)
                 - _sim_data["sample_ctr"][:, :nt].mean(axis=0))

    fig_dist = make_subplots(
        rows=2, cols=1, row_heights=[0.55, 0.45],
        subplot_titles=[
            "Wasserstein distance (intervention vs control)",
            "Mean difference (intervention − control)",
        ],
        shared_xaxes=True, vertical_spacing=0.10,
    )
    for row in [1, 2]:
        fig_dist.add_vrect(x0=1, x1=NPRE, fillcolor="rgba(100,149,237,0.07)",
                           line_width=0, row=row, col=1)

    fig_dist.add_trace(go.Scatter(
        x=t_ax, y=dist_ts,
        mode="lines", name="Wasserstein distance",
        line=dict(color="black", width=1.5),
        hovertemplate="Month: %{x}<br>Distance: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    fig_dist.add_trace(go.Scatter(
        x=t_ax, y=mean_diff,
        mode="lines", name="Mean difference",
        line=dict(color="rgba(255,152,0,0.8)", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,152,0,0.10)",
        hovertemplate="Month: %{x}<br>Mean diff: %{y:.4f}<extra></extra>",
    ), row=2, col=1)
    fig_dist.add_hline(y=0, line_color="rgba(0,0,0,0.3)", line_width=1,
                       line_dash="dot", row=2, col=1)

    _true_ann_side = "top right" if _true_cpt < nt * 0.75 else "top left"
    for row in [1, 2]:
        fig_dist.add_vline(
            x=_true_cpt, line_dash="dash", line_color="#1A237E", line_width=2,
            annotation_text=f"True τ = {_true_cpt}" if row == 1 else "",
            annotation_position=_true_ann_side,
            annotation_font=dict(color="#1A237E", size=10),
            row=row, col=1,
        )
    if _detected:
        _close = abs(_cpt - _true_cpt) < 6
        _det_ann_side = ("top left" if _true_ann_side == "top right" else "top right") if _close else _true_ann_side
        for row in [1, 2]:
            fig_dist.add_vline(
                x=_cpt, line_dash="solid", line_color="crimson", line_width=2,
                annotation_text=f"τ̂ = {_cpt}" if row == 1 else "",
                annotation_position=_det_ann_side,
                annotation_font=dict(color="crimson", size=10),
                row=row, col=1,
            )

    fig_dist.update_layout(
        title=f"Run {run_i + 1} · {exp_pct}% effect · npost={exp_npost} mo",
        xaxis2_title="Month",
        yaxis_title="Wasserstein distance",
        yaxis2_title="Mean diff.",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, font=dict(size=11)),
        margin=dict(b=60),
    )
    st.plotly_chart(fig_dist, width="stretch")

    # ── Distribution snapshots ─────────────────────────────────────────────────
    st.markdown("**Distribution snapshots** — dashed = control, solid = intervention")
    snap_t_exp = st.slider(
        "Timepoint (month)", min_value=1, max_value=nt, value=min(10, nt), step=1,
        key="snap_t_exp",
    )
    _idx = snap_t_exp - 1
    _d_ctr = _sim_data["sample_ctr"][:, _idx]
    _d_itv = _sim_data["sample_itv"][:, _idx]
    _xr = np.linspace(min(_d_ctr.min(), _d_itv.min()) - 2, max(_d_ctr.max(), _d_itv.max()) + 2, 200)
    fig_snap = go.Figure()
    fig_snap.add_trace(go.Scatter(
        x=_xr, y=gaussian_kde(_d_ctr).evaluate(_xr),
        mode="lines", name="control",
        line=dict(color="#2196F3", dash="dash", width=2),
    ))
    fig_snap.add_trace(go.Scatter(
        x=_xr, y=gaussian_kde(_d_itv).evaluate(_xr),
        mode="lines", name="intervention",
        line=dict(color="#FF5722", width=2),
    ))
    fig_snap.update_layout(
        title=f"Distribution at t = {snap_t_exp}",
        xaxis_title="Value",
        yaxis_title="Density",
        height=380,
        legend_title="Group",
    )
    st.plotly_chart(fig_snap, width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# RUN YOUR OWN SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("🔬 Run your own simulation")

if "dist_pending_restore" in st.session_state:
    _pr = st.session_state.pop("dist_pending_restore")
    st.session_state["dist_p_n_sim"]      = _pr["n_sim"]
    st.session_state["dist_p_base_seed"]  = _pr["base_seed"]
    st.session_state["dist_p_npre"]       = _pr["s_npre"]
    st.session_state["dist_p_npost"]      = _pr["s_npost"]
    st.session_state["dist_p_mu_inc"]     = _pr["s_mu_inc"]
    st.session_state["dist_p_sigma_inc"]  = _pr["s_sigma_inc"]
    st.session_state["dist_p_ns"]         = _pr["s_ns"]

for _k, _v in [
    ("dist_p_n_sim", 30), ("dist_p_base_seed", 42), ("dist_p_npre", 24),
    ("dist_p_npost", 60), ("dist_p_mu_inc", 0.04), ("dist_p_sigma_inc", 0.0), ("dist_p_ns", 100),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

with st.expander("⚙️ Simulation parameters", expanded=True):
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        n_sim = st.slider(
            "N simulations",
            min_value=10, max_value=200, step=10,
            key="dist_p_n_sim",
            help=(
                "Each simulation generates a fresh (control, intervention) pair with different "
                "random noise. Detection rate = fraction of runs where T_max exceeds the critical value."
            ),
        )
        base_seed = st.number_input(
            "Random seed",
            min_value=0, max_value=99999, step=1,
            key="dist_p_base_seed",
            help="Simulation i uses seed = base_seed + i. Change to get a different draw of noise realisations.",
        )

    with col_b:
        s_npre  = st.slider("Pre-intervention (months)", 12, 48, key="dist_p_npre")
        s_npost = st.slider("Post-intervention (months)", 24, 120, key="dist_p_npost")

    with col_c:
        s_mu_inc    = st.slider("Mean increment per month (Mu trend)", 0.0, 0.10, step=0.005, key="dist_p_mu_inc")
        s_sigma_inc = st.slider("SD increment per month (Sigma trend)", 0.0, 0.01, step=0.001, key="dist_p_sigma_inc")
        s_ns        = st.slider("Samples per time point (ns)", 50, 500, step=50, key="dist_p_ns")

_cur_params = {
    "n_sim": n_sim, "base_seed": int(base_seed),
    "s_npre": s_npre, "s_npost": s_npost,
    "s_mu_inc": s_mu_inc, "s_sigma_inc": s_sigma_inc, "s_ns": s_ns,
}
if "dist_mini_last_params" in st.session_state:
    if st.session_state["dist_mini_last_params"] != _cur_params:
        for _k in ("dist_mini_runs", "dist_mini_nav_idx", "dist_mnav_slider"):
            st.session_state.pop(_k, None)
        st.session_state["dist_hist_sel_gen"] = st.session_state.get("dist_hist_sel_gen", 0) + 1

s_run = st.button("▶ Run simulation", type="primary")

_dist_history = st.session_state.get("dist_mini_history", [])
if _dist_history:
    _sel = st.selectbox(
        "Previous runs",
        options=range(len(_dist_history)),
        format_func=lambda i: _dist_history[i]["label"],
        index=None,
        placeholder="Select a previous run to restore…",
        key=f"dist_hist_sel_{st.session_state.get('dist_hist_sel_gen', 0)}",
    )
    if _sel is not None:
        _h  = _dist_history[_sel]
        _hp = _h["params"]
        st.session_state["dist_pending_restore"]      = _hp
        st.session_state["dist_mini_runs"]            = _h["data"]
        st.session_state["dist_mini_nav_idx"]         = _h["nav_idx"]
        st.session_state["dist_mnav_slider"]          = _h["nav_idx"] + 1
        st.session_state["dist_mini_last_params"]     = _hp
        st.session_state["dist_hist_sel_gen"] = st.session_state.get("dist_hist_sel_gen", 0) + 1

if s_run:
    _crit_val = cv.get("npost_24", {}).get("critical_value", 2.5) if results_available else 2.5

    all_sims: list       = []
    all_d_ts: list       = []
    all_stats_list: list = []

    progress_bar = st.progress(0, text="Starting…")

    for i in range(n_sim):
        sim    = ci_sim_cdf(seed=int(base_seed) + i, npre=s_npre, npost=s_npost,
                            level=[MU, SIGMA], trend=[s_mu_inc, s_sigma_inc], ns=s_ns)
        d_ts   = wasserstein_distance_baci(sim["sample_ctr"], sim["sample_itv"])
        s_stat = trend_stats_cdf(d_ts, nt=s_npre + s_npost)

        all_sims.append(sim)
        all_d_ts.append(d_ts)
        all_stats_list.append(s_stat)

        progress_bar.progress((i + 1) / n_sim, text=f"Simulation {i + 1}/{n_sim}…")

    progress_bar.empty()

    detected_flags = [s["Tmax"] > _crit_val for s in all_stats_list]
    first_det      = next((i for i, d in enumerate(detected_flags) if d), 0)

    st.session_state["dist_mini_runs"] = {
        "n_sim":        n_sim,
        "base_seed":    int(base_seed),
        "s_npre":       s_npre,
        "s_npost":      s_npost,
        "s_mu_inc":     s_mu_inc,
        "s_sigma_inc":  s_sigma_inc,
        "s_ns":         s_ns,
        "crit_val":     _crit_val,
        "all_sims":     all_sims,
        "all_d_ts":     all_d_ts,
        "all_stats":    all_stats_list,
        "detected_flags": detected_flags,
    }
    st.session_state["dist_mini_nav_idx"]    = first_det
    st.session_state["dist_mnav_slider"]     = first_det + 1

    _run_n    = len(st.session_state.get("dist_mini_history", [])) + 1
    _det_rate = sum(detected_flags) / n_sim
    _ts       = datetime.now().strftime("%H:%M")
    _hist_lbl = (
        f"#{_run_n} · {_ts} · N={n_sim} seed={int(base_seed)} pre={s_npre}mo post={s_npost}mo "
        f"μ={s_mu_inc:.3f} σ={s_sigma_inc:.3f} ns={s_ns} · det={_det_rate:.0%}"
    )
    if "dist_mini_history" not in st.session_state:
        st.session_state["dist_mini_history"] = []
    st.session_state["dist_mini_history"].insert(0, {
        "label":   _hist_lbl,
        "data":    st.session_state["dist_mini_runs"],
        "nav_idx": first_det,
        "params":  _cur_params,
    })
    st.session_state["dist_mini_last_params"] = _cur_params


# ── Results — persisted across rerenders via session state ────────────────────
if "dist_mini_runs" in st.session_state and "all_sims" in st.session_state["dist_mini_runs"]:
    mr             = st.session_state["dist_mini_runs"]
    detected_flags = mr["detected_flags"]
    n_detected     = sum(detected_flags)
    det_rate       = n_detected / mr["n_sim"]
    crit_val_mr    = mr["crit_val"]
    true_cpt       = mr["s_npre"]

    st.success("Simulation complete!")

    # ── Summary metrics ───────────────────────────────────────────────────────
    valid_cpts = [s["cpt"] for s, d in zip(mr["all_stats"], detected_flags) if d]
    sm1, sm2, sm3 = st.columns(3)
    with sm1:
        st.metric("Detection rate", f"{det_rate:.1%}",
                  help=f"{n_detected} of {mr['n_sim']} simulations")
        st.metric("Critical value", f"{crit_val_mr:.3f}")
    if valid_cpts:
        arr_cpts   = np.array(valid_cpts)
        mean_err   = float(np.mean(np.abs(arr_cpts - true_cpt)))
        median_lag = float(np.median(arr_cpts - true_cpt))
        lag_dir    = "late" if median_lag > 0 else ("early" if median_lag < 0 else "exact")
        with sm2:
            st.metric("Mean |timing error|", f"{mean_err:.1f} mo",
                      help="Average |τ̂ − true τ| across detected runs.")
        with sm3:
            st.metric("Median detection lag", f"{median_lag:+.1f} mo ({lag_dir})",
                      help="Median (τ̂ − true τ). Positive = declared later than true change.")

    # ── Run navigator ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("Browse simulation runs")

    n_sim_mr = mr["n_sim"]

    if "dist_mini_nav_idx" not in st.session_state:
        st.session_state["dist_mini_nav_idx"] = 0
    if "dist_mnav_slider" not in st.session_state:
        st.session_state["dist_mnav_slider"] = 1

    def _dist_nav_prev():
        new = max(0, st.session_state["dist_mini_nav_idx"] - 1)
        st.session_state["dist_mini_nav_idx"] = new
        st.session_state["dist_mnav_slider"]  = new + 1

    def _dist_nav_next():
        new = min(n_sim_mr - 1, st.session_state["dist_mini_nav_idx"] + 1)
        st.session_state["dist_mini_nav_idx"] = new
        st.session_state["dist_mnav_slider"]  = new + 1

    def _dist_on_slider():
        st.session_state["dist_mini_nav_idx"] = st.session_state["dist_mnav_slider"] - 1

    nc1, nc2, nc3 = st.columns([1, 10, 1])
    with nc1:
        st.button("◀", on_click=_dist_nav_prev, key="dist_mnav_prev", width="stretch")
    with nc2:
        st.slider("Run", 1, n_sim_mr, key="dist_mnav_slider", on_change=_dist_on_slider)
    with nc3:
        st.button("▶", on_click=_dist_nav_next, key="dist_mnav_next", width="stretch")

    det_indices = [i for i, d in enumerate(detected_flags) if d]
    st.caption(
        f"Detected a change in {n_detected} of {n_sim_mr} runs ({det_rate:.0%})."
        if det_indices else
        "No runs detected a change — try a larger mean/sigma increment or longer monitoring window."
    )

    show_idx  = st.session_state.get("dist_mini_nav_idx", 0)
    run_num   = show_idx + 1
    show_seed = mr["base_seed"] + show_idx

    st.subheader(
        f"Run #{run_num} of {n_sim_mr} · base seed {mr['base_seed']} · run seed {show_seed}"
    )

    sim          = mr["all_sims"][show_idx]
    d_ts         = mr["all_d_ts"][show_idx]
    s_stats      = mr["all_stats"][show_idx]
    was_detected = mr["detected_flags"][show_idx]
    _tmax        = s_stats["Tmax"]
    _cpt         = s_stats["cpt"]

    # ── Per-run metrics ───────────────────────────────────────────────────────
    mm1, mm2, mm3 = st.columns(3)
    with mm1:
        st.metric("T_max", f"{_tmax:.3f}", delta=f"threshold {crit_val_mr:.3f}", delta_color="off")
    with mm2:
        st.metric("Detected τ̂", f"month {_cpt}" if was_detected else "✗ not detected")
    with mm3:
        if was_detected:
            _err = _cpt - true_cpt
            _dir = "late" if _err > 0 else ("early" if _err < 0 else "exact")
            st.metric("Timing error", f"{_err:+d} mo ({_dir})",
                      help=f"True τ = month {true_cpt} (pre = {mr['s_npre']} months)")
        else:
            st.metric("Timing error", "—",
                      help=f"True τ = month {true_cpt} (pre = {mr['s_npre']} months)")

    # ── Distance + mean-difference time series ────────────────────────────────
    s_nt        = mr["s_npre"] + mr["s_npost"]
    t_sim       = np.arange(1, s_nt + 1)
    s_mean_diff = sim["sample_itv"].mean(axis=0) - sim["sample_ctr"].mean(axis=0)

    fig_s = make_subplots(
        rows=2, cols=1, row_heights=[0.55, 0.45],
        subplot_titles=[
            "Wasserstein distance (intervention vs control)",
            "Mean difference (intervention − control)",
        ],
        shared_xaxes=True, vertical_spacing=0.10,
    )
    for row in [1, 2]:
        fig_s.add_vrect(x0=1, x1=mr["s_npre"], fillcolor="rgba(100,149,237,0.07)",
                        line_width=0, row=row, col=1)

    fig_s.add_trace(go.Scatter(
        x=t_sim, y=d_ts,
        mode="lines", name="Wasserstein distance",
        line=dict(color="black", width=1.5),
        hovertemplate="Month: %{x}<br>Distance: %{y:.4f}<extra></extra>",
    ), row=1, col=1)

    fig_s.add_trace(go.Scatter(
        x=t_sim, y=s_mean_diff,
        mode="lines", name="Mean difference",
        line=dict(color="rgba(255,152,0,0.8)", width=1.5),
        fill="tozeroy", fillcolor="rgba(255,152,0,0.10)",
        hovertemplate="Month: %{x}<br>Mean diff: %{y:.4f}<extra></extra>",
    ), row=2, col=1)
    fig_s.add_hline(y=0, line_color="rgba(0,0,0,0.3)", line_width=1, line_dash="dot", row=2, col=1)

    _s_ann_side = "top right" if mr["s_npre"] < s_nt * 0.75 else "top left"
    for row in [1, 2]:
        fig_s.add_vline(
            x=mr["s_npre"], line_dash="dash", line_color="#1A237E", line_width=2,
            annotation_text=f"True τ = {mr['s_npre']}" if row == 1 else "",
            annotation_position=_s_ann_side,
            annotation_font=dict(color="#1A237E", size=10),
            row=row, col=1,
        )
    if was_detected:
        _det_ann_side = "top left" if _s_ann_side == "top right" else "top right"
        for row in [1, 2]:
            fig_s.add_vline(
                x=_cpt, line_dash="solid", line_color="crimson", line_width=2,
                annotation_text=f"τ̂ = {_cpt}" if row == 1 else "",
                annotation_position=_det_ann_side,
                annotation_font=dict(color="crimson", size=10),
                row=row, col=1,
            )

    fig_s.update_layout(
        title=f"Run #{run_num} · T_max={_tmax:.2f} ({'✓ detected' if was_detected else '✗ not detected'})",
        xaxis2_title="Month",
        yaxis_title="Wasserstein distance",
        yaxis2_title="Mean diff.",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, font=dict(size=11)),
        margin=dict(b=60),
    )
    st.plotly_chart(fig_s, width="stretch")

    # ── Distribution snapshots ─────────────────────────────────────────────────
    st.markdown("**Distribution snapshots** — dashed = control, solid = intervention")
    snap_t_mr = st.slider(
        "Timepoint (month)", min_value=1, max_value=s_nt, value=min(10, s_nt), step=1,
        key="snap_t_mr",
    )
    _idx = snap_t_mr - 1
    _d_ctr = sim["sample_ctr"][:, _idx]
    _d_itv = sim["sample_itv"][:, _idx]
    _xr = np.linspace(min(_d_ctr.min(), _d_itv.min()) - 2, max(_d_ctr.max(), _d_itv.max()) + 2, 200)
    fig_snap = go.Figure()
    fig_snap.add_trace(go.Scatter(
        x=_xr, y=gaussian_kde(_d_ctr).evaluate(_xr),
        mode="lines", name="control",
        line=dict(color="#2196F3", dash="dash", width=2),
    ))
    fig_snap.add_trace(go.Scatter(
        x=_xr, y=gaussian_kde(_d_itv).evaluate(_xr),
        mode="lines", name="intervention",
        line=dict(color="#FF5722", width=2),
    ))
    fig_snap.update_layout(
        title=f"Distribution at t = {snap_t_mr}",
        xaxis_title="Value",
        yaxis_title="Density",
        height=380,
        legend_title="Group",
    )
    st.plotly_chart(fig_snap, width="stretch")
