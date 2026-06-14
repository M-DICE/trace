"""
Distribution Change Detection using Forecast (Page-CUSUM)
Interactive analysis page for distribution_forecast results.
"""

import pickle
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from statsmodels.regression.linear_model import OLS

from tracepy.changepoint.amoc import load_crit_val_table, lookup_crit_val
from tracepy.changepoint.forecast import page_cusum
from tracepy.simulation.distribution import ci_sim_cdf
from tracepy.stats.metrics import wasserstein_distance_ba, wasserstein_distance_baci

warnings.filterwarnings("ignore")

_CRIT_VAL_TABLE = load_crit_val_table()
CRIT_VAL = lookup_crit_val(_CRIT_VAL_TABLE)

# Page config
st.set_page_config(
    page_title="Distribution Change Detection (Forecast)",
    page_icon="🌦️",
    layout="wide",
)

# Simulation constants
NPRE = 24
NPOST_VEC = np.arange(24, 121, 12)
NPOST_MAX = int(NPOST_VEC[-1])
NTT = NPRE + NPOST_MAX
MU = 10.0
SIGMA = 1.0
NS = 200

TREND_INCREASE_MU = np.round(MU * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / 120, 4)
EFFECT_SIZES_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

RESULTS_PATH = (
    Path(__file__).parent.parent / "results" / "distribution_forecast" / "sim_results.pkl"
)

PALETTE = px.colors.sample_colorscale("Viridis", [i / 10 for i in range(11)])


# Data loading
@st.cache_data(show_spinner="Loading distribution forecast results…")
def load_results(path: Path, mtime: float):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Regenerating simulation run…")
def _regenerate_dist_sim(seed: int, delay: int, trend_mu: float, ns: int):
    return ci_sim_cdf(
        seed=seed,
        npre=NPRE + delay,
        npost=NPOST_MAX - delay,
        level=[MU, SIGMA],
        trend=[trend_mu, 0],
        ns=ns,
    )


def _cusum_path_dist(dist_ts, npre, ntt, crit_val):
    """Page-CUSUM path on a Wasserstein distance time series (OLS, gamma=0)."""
    X = np.column_stack([np.ones(ntt), np.arange(1, ntt + 1)])
    ols = OLS(dist_ts[:npre], X[:npre]).fit()
    r = np.concatenate([ols.resid, X[npre:] @ ols.params - dist_ts[npre:]])

    train_mean = np.mean(r[:npre])
    sigma = np.std(r[:npre], ddof=1)

    c_upper = np.zeros(ntt)
    c_lower = np.zeros(ntt)
    threshold_curve = np.zeros(ntt)

    if sigma >= 1e-12:
        for k in range(1, ntt - npre + 1):
            inc = r[npre + k - 1] - train_mean
            c_upper[npre + k - 1] = max(0.0, c_upper[npre + k - 2] + inc)
            c_lower[npre + k - 1] = max(0.0, c_lower[npre + k - 2] - inc)
            weight = np.sqrt(npre) * (1.0 + k / npre)
            threshold_curve[npre + k - 1] = weight * crit_val * sigma

    return r, c_upper, c_lower, threshold_curve


def _detection_rates(results, trends, npost_vec):
    out = {}
    for t in trends:
        if t not in results:
            continue
        time_est_vec = np.array(results[t]["time_est_vec"], dtype=float)
        out[t] = np.array(
            [
                float(np.mean(np.isfinite(time_est_vec) & (time_est_vec <= npost)))
                for npost in npost_vec
            ]
        )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🌦️ Distribution Change Detection (Forecast)")

st.markdown("""
Rewilding interventions can affect not just the average abundance of a species, but also its
**variability** or the entire **shape of its distribution**.

This page analyses changes in distributions using **Wasserstein Distance**.
Unlike AMOC which tests the full series retrospectively, Forecast processes observations one at
a time and raises an alarm as soon as evidence accumulates. This makes it well-suited to real-time
monitoring of distributional shifts.
""")


# Sidebar
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
    | **dist_measure** | Wasserstein | Distance metric |
    | **mu** | {MU} | Baseline mean |
    | **sigma** | {SIGMA} | Baseline standard deviation |
    | **detector** | PageCUSUM | Changepoint method |
    | **gamma** | 0.0 | Page-CUSUM weighting |
    | **crit_val** | {CRIT_VAL:.6f} | Page-CUSUM critical value (PageCUSUM, γ=0, α=0.05) |
    | **simN** | 1,000 | Simulations per effect size |
    """)


# Load data
_mtime = RESULTS_PATH.stat().st_mtime if RESULTS_PATH.exists() else 0.0
data = load_results(RESULTS_PATH, _mtime)

if data is None:
    st.error(
        f"Pre-computed results not found at `{RESULTS_PATH}`. "
        "Run `uv run trace-sim distribution-forecast` first, "
        "or use the **Run your own simulation** section below."
    )
    results_available = False
else:
    res_baci = data.get("detection_results_baci", {})
    res_ba = data.get("detection_results_ba", {})

    if not res_baci or not res_ba:
        st.warning(
            "Saved results are incomplete "
            "(detection_results_baci or detection_results_ba missing). "
            "Re-run `uv run trace-sim distribution-forecast` to regenerate."
        )
        results_available = False
    else:
        trends = sorted(res_baci.keys())
        pct_to_trend = dict(zip(EFFECT_SIZES_PCT, trends))
        rates_baci = _detection_rates(res_baci, trends, NPOST_VEC)
        rates_ba = _detection_rates(res_ba, trends, NPOST_VEC)
        results_available = True


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
if results_available:
    st.header("📊 Results explorer")
    st.markdown(
        "Charts below are drawn from pre-computed simulations (1,000 runs per effect size)."
    )

    tab_power, tab_heatmap, tab_summary, tab_delay, tab_err, tab_bias = st.tabs(
        [
            "Power curves",
            "Detection heatmaps",
            "BACI vs BA summary",
            "Detection by delay",
            "Changepoint error",
            "Changepoint bias",
        ]
    )

    # Power curves
    with tab_power:
        st.subheader("Power curves")
        st.markdown("""
        Each line shows the **probability of detecting the distributional change** as the
        post-intervention monitoring window grows. Brighter/yellower lines are larger effect sizes.
        Dashed lines mark 80% and 95% power thresholds.
        """)

        power_view = st.radio(
            "Scenario",
            ["BACI", "BA", "Side-by-side"],
            horizontal=True,
            key="df_power_view",
        )

        def _power_fig(rates, title):
            fig = go.Figure()
            for i, (t, pct) in enumerate(zip(trends, EFFECT_SIZES_PCT)):
                if t not in rates:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=NPOST_VEC,
                        y=rates[t],
                        mode="lines",
                        name=f"{pct}%",
                        line=dict(color=PALETTE[i], width=2),
                        hovertemplate="npost: %{x} mo<br>Detection rate: %{y:.1%}<extra></extra>",
                    )
                )
            fig.add_hline(
                y=0.80,
                line_dash="dash",
                line_color="grey",
                annotation_text="80% power",
                annotation_position="right",
            )
            fig.add_hline(
                y=0.95,
                line_dash="dot",
                line_color="grey",
                annotation_text="95% power",
                annotation_position="right",
            )
            fig.update_layout(
                title=title,
                xaxis_title="Post-intervention monitoring window (months)",
                yaxis_title="Detection rate",
                yaxis=dict(range=[0, 1.05], tickformat=".0%"),
                legend_title="Effect size<br>(% of mean / 10 yr)",
                height=480,
            )
            return fig

        if power_view == "BACI":
            st.plotly_chart(_power_fig(rates_baci, "Power curves — BACI"), width="stretch")
        elif power_view == "BA":
            st.plotly_chart(_power_fig(rates_ba, "Power curves — BA"), width="stretch")
        else:
            _cols = st.columns(2)
            with _cols[0]:
                st.plotly_chart(_power_fig(rates_baci, "BACI"), width="stretch")
            with _cols[1]:
                st.plotly_chart(_power_fig(rates_ba, "BA"), width="stretch")

        st.markdown("""
        **How to read this:** each curve is one effect size (% of mean shift over 10 years).
        The x-axis shows how many months of post-intervention monitoring are available.
        A higher curve means the method detects the change more reliably.
        """)

    # Detection heatmaps
    with tab_heatmap:
        st.subheader("Detection heatmaps")
        st.markdown("""
        Each cell shows the detection rate for a given effect size and monitoring window.
        The heatmap reveals which combinations of effect size and observation period
        achieve reliable detection.
        """)

        heatmap_view = st.radio(
            "Scenario",
            ["BACI", "BA", "Side-by-side"],
            horizontal=True,
            key="df_heatmap_view",
        )

        def _heatmap_fig(rates, title):
            z = np.array([rates[t] for t in trends if t in rates])
            y_labels = [f"{p}%" for p, t in zip(EFFECT_SIZES_PCT, trends) if t in rates]
            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=NPOST_VEC,
                    y=y_labels,
                    z=z,
                    colorscale="Viridis",
                    zmin=0,
                    zmax=1,
                    colorbar=dict(title="Detection rate", tickformat=".0%"),
                    hovertemplate=(
                        "npost: %{x} mo<br>Effect: %{y}<br>Detection: %{z:.1%}<extra></extra>"
                    ),
                )
            )
            fig.add_trace(
                go.Contour(
                    x=NPOST_VEC,
                    y=y_labels,
                    z=z,
                    contours=dict(coloring="fill", start=0.795, end=1.005, size=0.21),
                    colorscale=[[0, "white"], [1, "white"]],
                    opacity=0.15,
                    showscale=False,
                    hoverinfo="skip",
                    line=dict(width=0),
                )
            )
            fig.add_trace(
                go.Contour(
                    x=NPOST_VEC,
                    y=y_labels,
                    z=z,
                    contours=dict(
                        coloring="none",
                        showlabels=True,
                        start=0.80,
                        end=0.95,
                        size=0.15,
                        labelfont=dict(size=11, color="#1a1a1a"),
                    ),
                    line=dict(color="#1a1a1a", width=2.5, dash="dash"),
                    showscale=False,
                    hoverinfo="skip",
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title="Post-intervention monitoring window (months)",
                yaxis_title="Effect size (% of mean / 10 yr)",
                yaxis=dict(autorange="reversed"),
                height=500,
            )
            return fig

        if heatmap_view == "BACI":
            st.plotly_chart(_heatmap_fig(rates_baci, "Detection heatmap — BACI"), width="stretch")
        elif heatmap_view == "BA":
            st.plotly_chart(_heatmap_fig(rates_ba, "Detection heatmap — BA"), width="stretch")
        else:
            _cols = st.columns(2)
            with _cols[0]:
                st.plotly_chart(_heatmap_fig(rates_baci, "BACI"), width="stretch")
            with _cols[1]:
                st.plotly_chart(_heatmap_fig(rates_ba, "BA"), width="stretch")

        st.caption(
            "**How to read this:** find your target effect size on the y-axis, then read "
            "across to where the colour reaches the 80% contour — that x-value is the "
            "minimum monitoring window needed for reliable detection."
        )

    # Summary tables
    with tab_summary:
        st.subheader("Detection summary")
        st.markdown("""
        - **Detection rate** — fraction of runs where any changepoint was declared
          (`np.isfinite(time_est)`).
        - **Delay** — `time_est + npre − true_cpt` (months). Positive = detected late.
          Only detected runs contribute to delay statistics.
        - **No detection** — number of runs where no changepoint was declared.
        """)

        def _build_summary_raw(results):
            rows = []
            for t, pct in zip(trends, EFFECT_SIZES_PCT):
                entry = results.get(t)
                if entry is None:
                    continue
                time_est_vec = np.array(entry["time_est_vec"], dtype=float)
                delays_arr = np.array(entry["delays"])
                true_cpts = NPRE + delays_arr
                finite_mask = np.isfinite(time_est_vec)
                delay_vec = time_est_vec[finite_mask] + NPRE - true_cpts[finite_mask]
                rows.append(
                    {
                        "pct": pct,
                        "det_rate": float(finite_mask.mean()),
                        "med_delay": float(np.median(delay_vec)) if len(delay_vec) > 0 else np.nan,
                        "mean_delay": float(np.mean(delay_vec)) if len(delay_vec) > 0 else np.nan,
                        "std_delay": float(np.std(delay_vec)) if len(delay_vec) > 0 else np.nan,
                        "no_det": int(np.sum(~finite_mask)),
                    }
                )
            return rows

        _raw_baci = _build_summary_raw(res_baci)
        _raw_ba = _build_summary_raw(res_ba)
        _labels = [f"{r['pct']}%" for r in _raw_baci]

        # Detection rate bar chart
        fig_summ_dr = go.Figure()
        fig_summ_dr.add_trace(
            go.Bar(
                x=_labels,
                y=[r["det_rate"] for r in _raw_baci],
                name="BACI",
                marker_color="rgba(33,150,243,0.82)",
                hovertemplate="%{x}<br>Detection rate: %{y:.1%}<extra>BACI</extra>",
            )
        )
        fig_summ_dr.add_trace(
            go.Bar(
                x=_labels,
                y=[r["det_rate"] for r in _raw_ba],
                name="BA",
                marker_color="rgba(255,152,0,0.82)",
                hovertemplate="%{x}<br>Detection rate: %{y:.1%}<extra>BA</extra>",
            )
        )
        fig_summ_dr.add_hline(
            y=0.80,
            line_dash="dash",
            line_color="grey",
            annotation_text="80%",
            annotation_position="right",
        )
        fig_summ_dr.add_hline(
            y=0.95,
            line_dash="dot",
            line_color="grey",
            annotation_text="95%",
            annotation_position="right",
        )
        fig_summ_dr.update_layout(
            barmode="group",
            title="Detection rate by effect size",
            xaxis_title="Effect size (% of mean / 10 yr)",
            yaxis=dict(title="Detection rate", range=[0, 1.05], tickformat=".0%"),
            legend_title="Scenario",
            height=380,
        )
        st.plotly_chart(fig_summ_dr, width="stretch")

        # Median delay bar chart
        fig_summ_dl = go.Figure()
        fig_summ_dl.add_trace(
            go.Bar(
                x=_labels,
                y=[r["med_delay"] for r in _raw_baci],
                name="BACI",
                marker_color="rgba(33,150,243,0.82)",
                hovertemplate="%{x}<br>Median delay: %{y:.1f} mo<extra>BACI</extra>",
            )
        )
        fig_summ_dl.add_trace(
            go.Bar(
                x=_labels,
                y=[r["med_delay"] for r in _raw_ba],
                name="BA",
                marker_color="rgba(255,152,0,0.82)",
                hovertemplate="%{x}<br>Median delay: %{y:.1f} mo<extra>BA</extra>",
            )
        )
        fig_summ_dl.add_hline(y=0, line_color="rgba(0,0,0,0.3)", line_width=1, line_dash="dot")
        fig_summ_dl.update_layout(
            barmode="group",
            title="Median detection delay by effect size",
            xaxis_title="Effect size (% of mean / 10 yr)",
            yaxis_title="Median delay (months, detected runs only)",
            legend_title="Scenario",
            height=380,
        )
        st.plotly_chart(fig_summ_dl, width="stretch")

        # Merged summary table
        _merged_rows = []
        for b, a in zip(_raw_baci, _raw_ba):
            _merged_rows.append(
                {
                    "Effect": f"{b['pct']}%",
                    "Det. rate (BACI)": f"{b['det_rate']:.1%}",
                    "Det. rate (BA)": f"{a['det_rate']:.1%}",
                    "Median delay (BACI)": f"{b['med_delay']:.1f}"
                    if np.isfinite(b["med_delay"])
                    else "n/a",
                    "Median delay (BA)": f"{a['med_delay']:.1f}"
                    if np.isfinite(a["med_delay"])
                    else "n/a",
                    "Mean delay (BACI)": f"{b['mean_delay']:.1f}"
                    if np.isfinite(b["mean_delay"])
                    else "n/a",
                    "Mean delay (BA)": f"{a['mean_delay']:.1f}"
                    if np.isfinite(a["mean_delay"])
                    else "n/a",
                    "Std delay (BACI)": f"{b['std_delay']:.1f}"
                    if np.isfinite(b["std_delay"])
                    else "n/a",
                    "Std delay (BA)": f"{a['std_delay']:.1f}"
                    if np.isfinite(a["std_delay"])
                    else "n/a",
                    "No det. (BACI)": b["no_det"],
                    "No det. (BA)": a["no_det"],
                }
            )
        st.dataframe(pd.DataFrame(_merged_rows), hide_index=True, width="stretch")

    # Detection by delay
    with tab_delay:
        st.subheader("Effect of intervention delay on detection")
        st.markdown("""
        The ecological response may start with a delay after the formal intervention date.
        A longer delay shifts the true changepoint deeper into the post-period, leaving
        fewer months for evidence to accumulate before the window closes — this is
        especially important for an online method like Forecast.
        """)

        dl_c1, dl_c2, dl_c3 = st.columns(3)
        with dl_c1:
            delay_eff_pct = st.select_slider(
                "Effect size",
                options=EFFECT_SIZES_PCT,
                value=30,
                format_func=lambda x: f"{x}%",
                key="df_delay_eff",
            )
        with dl_c2:
            delay_scenario = st.radio(
                "Scenario", ["BACI", "BA", "Both"], horizontal=True, key="df_delay_scenario"
            )
        with dl_c3:
            delay_view = st.radio(
                "View", ["Line chart", "Heatmap"], horizontal=True, key="df_delay_view"
            )

        _delay_trend = pct_to_trend[delay_eff_pct]
        _unique_delays = np.arange(1, 21)

        def _delay_rates(results, trend):
            dlys = np.array(results[trend]["delays"])
            det = np.isfinite(np.array(results[trend]["time_est_vec"], dtype=float))
            out = []
            for d in _unique_delays:
                mask = dlys == d
                out.append(float(det[mask].mean()) if mask.any() else np.nan)
            return np.array(out)

        if delay_view == "Line chart":
            fig_dl = go.Figure()
            _dl_configs = []
            if delay_scenario in ("BACI", "Both"):
                _dl_configs.append(("BACI", res_baci, "rgba(33,150,243,1)"))
            if delay_scenario in ("BA", "Both"):
                _dl_configs.append(("BA", res_ba, "rgba(255,152,0,1)"))
            for _lbl, _res, _col in _dl_configs:
                _rates = _delay_rates(_res, _delay_trend)
                fig_dl.add_trace(
                    go.Scatter(
                        x=_unique_delays,
                        y=_rates,
                        mode="lines+markers",
                        name=_lbl,
                        line=dict(color=_col, width=2.5),
                        marker=dict(size=7),
                        hovertemplate="Delay: %{x} mo<br>Detection: %{y:.1%}<extra>"
                        + _lbl
                        + "</extra>",
                    )
                )
            fig_dl.add_hline(
                y=0.80,
                line_dash="dash",
                line_color="grey",
                annotation_text="80%",
                annotation_position="right",
            )
            fig_dl.update_layout(
                title=f"Detection rate vs intervention delay — {delay_eff_pct}% effect",
                xaxis=dict(title="Intervention delay (months)", dtick=2),
                yaxis=dict(title="Detection rate", range=[0, 1.05], tickformat=".0%"),
                legend_title="Scenario",
                height=440,
            )
            st.plotly_chart(fig_dl, width="stretch")
        else:
            _hm_scen = delay_scenario if delay_scenario != "Both" else "BACI"
            if delay_scenario == "Both":
                _hm_scen = st.radio(
                    "Heatmap scenario", ["BACI", "BA"], horizontal=True, key="df_delay_hm_scen"
                )
            _hm_res = res_baci if _hm_scen == "BACI" else res_ba
            _z = np.full((len(_unique_delays), len(trends)), np.nan)
            for _ti, _t in enumerate(trends):
                _dlys = np.array(_hm_res[_t]["delays"])
                _det = np.isfinite(np.array(_hm_res[_t]["time_est_vec"], dtype=float))
                for _di, _d in enumerate(_unique_delays):
                    _mask = _dlys == _d
                    if _mask.any():
                        _z[_di, _ti] = float(_det[_mask].mean())
            fig_hm_dl = go.Figure()
            fig_hm_dl.add_trace(
                go.Heatmap(
                    x=[f"{p}%" for p in EFFECT_SIZES_PCT],
                    y=_unique_delays,
                    z=_z,
                    colorscale="Viridis",
                    zmin=0,
                    zmax=1,
                    colorbar=dict(title="Detection rate", tickformat=".0%"),
                    hovertemplate=(
                        "Effect: %{x}<br>Delay: %{y} mo<br>Detection: %{z:.1%}<extra></extra>"
                    ),
                )
            )
            fig_hm_dl.update_layout(
                title=f"Detection rate: effect × delay — {_hm_scen}",
                xaxis_title="Effect size",
                yaxis=dict(title="Intervention delay (months)", dtick=2),
                height=480,
            )
            st.plotly_chart(fig_hm_dl, width="stretch")

    # Changepoint error
    with tab_err:
        st.subheader("Changepoint localisation error")
        st.markdown("""
        When a changepoint is detected, AMOC on the residuals up to the detection time
        estimates when the change occurred. The mean absolute localisation error is
        `|cpt_est − true_cpt|` in months.

        Only detected runs are included. Smaller bars = more precise localisation.
        """)

        err_scenario = st.radio(
            "Scenario", ["BACI", "BA", "Side-by-side"], horizontal=True, key="df_err_scenario"
        )

        def _err_fig(results, title, colour):
            errors, pcts_valid = [], []
            for t, pct in zip(trends, EFFECT_SIZES_PCT):
                tv = np.array(results[t]["time_est_vec"], dtype=float)
                cv2 = np.array(results[t]["cpt_est_vec"], dtype=float)
                dlys = np.array(results[t]["delays"])
                true_cpts = NPRE + dlys
                det = np.isfinite(tv)
                if det.any():
                    errors.append(float(np.abs(cv2[det] - true_cpts[det]).mean()))
                    pcts_valid.append(pct)
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=[f"{p}%" for p in pcts_valid],
                    y=errors,
                    marker_color=colour,
                    hovertemplate="%{x} effect<br>Mean |error|: %{y:.2f} months<extra></extra>",
                )
            )
            fig.update_layout(
                title=title,
                xaxis_title="Effect size",
                yaxis_title="Mean |cpt_est − true_cpt| (months)",
                height=420,
            )
            return fig

        if err_scenario == "BACI":
            st.plotly_chart(
                _err_fig(res_baci, "Changepoint error — BACI", "rgba(33,150,243,0.82)"),
                width="stretch",
            )
        elif err_scenario == "BA":
            st.plotly_chart(
                _err_fig(res_ba, "Changepoint error — BA", "rgba(255,152,0,0.82)"), width="stretch"
            )
        else:
            _ec1, _ec2 = st.columns(2)
            with _ec1:
                st.plotly_chart(
                    _err_fig(res_baci, "BACI", "rgba(33,150,243,0.82)"), width="stretch"
                )
            with _ec2:
                st.plotly_chart(_err_fig(res_ba, "BA", "rgba(255,152,0,0.82)"), width="stretch")

    # Changepoint bias
    with tab_bias:
        st.subheader("Changepoint timing accuracy")
        st.markdown("""
        Signed timing error: *cpt_est − true_cpt*. Positive = declared late;
        negative = declared early. Only detected runs are included.

        The green band (±3 months) marks a practically acceptable timing window.
        """)

        bias_c1, bias_c2 = st.columns(2)
        with bias_c1:
            bias_scenario = st.radio(
                "Scenario", ["BACI", "BA", "Both"], horizontal=True, key="df_bias_scenario"
            )
        with bias_c2:
            bias_min_n = st.slider("Minimum detections to show box", 2, 50, 5, key="df_bias_min_n")

        _bias_configs = []
        if bias_scenario in ("BACI", "Both"):
            _bias_configs.append(("BACI", res_baci, "rgba(33,150,243,0.85)"))
        if bias_scenario in ("BA", "Both"):
            _bias_configs.append(("BA", res_ba, "rgba(255,152,0,0.85)"))

        fig_bias = go.Figure()
        fig_bias.add_hrect(
            y0=-3,
            y1=3,
            fillcolor="rgba(0,180,0,0.07)",
            line_width=0,
            annotation_text="±3 mo",
            annotation_position="top right",
            annotation_font=dict(color="green", size=10),
        )

        _n_table = {}
        for _lbl, _res, _col in _bias_configs:
            q1s, meds, q3s, p5s, p95s, ns_b = [], [], [], [], [], []
            for t, pct in zip(trends, EFFECT_SIZES_PCT):
                tv = np.array(_res[t]["time_est_vec"], dtype=float)
                cv2 = np.array(_res[t]["cpt_est_vec"], dtype=float)
                dlys = np.array(_res[t]["delays"])
                true_cpts = NPRE + dlys
                det = np.isfinite(tv)
                errs = cv2[det] - true_cpts[det]
                n = len(errs)
                ns_b.append(n)
                if n >= bias_min_n:
                    q1s.append(float(np.percentile(errs, 25)))
                    meds.append(float(np.median(errs)))
                    q3s.append(float(np.percentile(errs, 75)))
                    p5s.append(float(np.percentile(errs, 5)))
                    p95s.append(float(np.percentile(errs, 95)))
                else:
                    q1s.append(None)
                    meds.append(None)
                    q3s.append(None)
                    p5s.append(None)
                    p95s.append(None)
            _n_table[_lbl] = ns_b
            fig_bias.add_trace(
                go.Box(
                    x=[f"{p}%" for p in EFFECT_SIZES_PCT],
                    q1=q1s,
                    median=meds,
                    q3=q3s,
                    lowerfence=p5s,
                    upperfence=p95s,
                    name=_lbl,
                    marker_color=_col,
                    line_color=_col,
                    boxpoints=False,
                    hovertemplate=(
                        "<b>%{x} — " + _lbl + "</b><br>"
                        "Median: %{median} mo<br>IQR: %{q1}–%{q3} mo"
                        "<extra></extra>"
                    ),
                )
            )
        fig_bias.add_hline(
            y=0,
            line_color="black",
            line_width=1.5,
            line_dash="dot",
            annotation_text="True τ",
            annotation_position="top left",
            annotation_font=dict(size=10),
        )
        fig_bias.update_layout(
            boxmode="group",
            xaxis_title="Effect size",
            yaxis_title="Timing error (months)",
            height=460,
            title="Changepoint timing bias — detected runs only",
            legend=dict(orientation="h", yanchor="bottom", y=-0.25),
            margin=dict(l=70),
        )
        st.plotly_chart(fig_bias, width="stretch")

        _n_rows = {"Effect size": [f"{p}%" for p in EFFECT_SIZES_PCT]}
        for _lbl, _ns in _n_table.items():
            _n_rows[f"Detected ({_lbl})"] = _ns
        st.caption("Detections per scenario used to build each box:")
        st.dataframe(pd.DataFrame(_n_rows), hide_index=True, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL RUN EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
if results_available:
    st.divider()
    st.header("🔍 Individual run explorer")
    st.markdown("""
    Browse individual simulation runs from the pre-computed results.
    The lower panel shows the **Page-CUSUM path** on the Wasserstein distance time series —
    watch how the CUSUM statistic accumulates after the true changepoint and eventually
    crosses the detection threshold.
    """)

    ir_eff_pct = st.select_slider(
        "Effect size",
        options=EFFECT_SIZES_PCT,
        value=30,
        format_func=lambda x: f"{x}%",
        key="ir_eff",
    )

    ir_trend = pct_to_trend[ir_eff_pct]
    n_runs_ir = len(res_baci[ir_trend]["delays"])

    if "ir_nav_idx" not in st.session_state:
        st.session_state["ir_nav_idx"] = 0
    if "ir_nav_slider" not in st.session_state:
        st.session_state["ir_nav_slider"] = 1

    ir_run_i = min(st.session_state.get("ir_nav_idx", 0), n_runs_ir - 1)
    st.session_state["ir_nav_idx"] = ir_run_i
    st.session_state["ir_nav_slider"] = min(st.session_state.get("ir_nav_slider", 1), n_runs_ir)

    def _ir_prev():
        n = max(0, st.session_state["ir_nav_idx"] - 1)
        st.session_state["ir_nav_idx"] = n
        st.session_state["ir_nav_slider"] = n + 1

    def _ir_next():
        n = min(n_runs_ir - 1, st.session_state["ir_nav_idx"] + 1)
        st.session_state["ir_nav_idx"] = n
        st.session_state["ir_nav_slider"] = n + 1

    def _ir_slider():
        st.session_state["ir_nav_idx"] = st.session_state["ir_nav_slider"] - 1

    en1, en2, en3 = st.columns([1, 10, 1])
    with en1:
        st.button("◀", on_click=_ir_prev, key="ir_prev", width="stretch")
    with en2:
        st.slider("Run", 1, n_runs_ir, key="ir_nav_slider", on_change=_ir_slider)
    with en3:
        st.button("▶", on_click=_ir_next, key="ir_next", width="stretch")

    ir_delay = int(res_baci[ir_trend]["delays"][ir_run_i])
    ir_seed = int(res_baci[ir_trend]["seeds"][ir_run_i])
    ir_true_cpt = NPRE + ir_delay

    ir_sim = _regenerate_dist_sim(ir_seed, ir_delay, ir_trend, NS)

    ir_dist_baci = wasserstein_distance_baci(ir_sim["sample_ctr"], ir_sim["sample_itv"])
    ir_dist_ba = wasserstein_distance_ba(ir_sim["sample_itv"], NPRE)

    ir_r_baci, ir_c_upper_baci, ir_c_lower_baci, ir_thresh_baci = _cusum_path_dist(
        ir_dist_baci, NPRE, NTT, CRIT_VAL
    )
    ir_r_ba, ir_c_upper_ba, ir_c_lower_ba, ir_thresh_ba = _cusum_path_dist(
        ir_dist_ba, NPRE, NTT, CRIT_VAL
    )

    ir_time_est_baci = page_cusum(ir_r_baci, m=NPRE, crit_val=CRIT_VAL)
    ir_time_est_ba = page_cusum(ir_r_ba, m=NPRE, crit_val=CRIT_VAL)
    ir_detected_baci = bool(np.isfinite(ir_time_est_baci))
    ir_detected_ba = bool(np.isfinite(ir_time_est_ba))

    _X_ir = np.column_stack([np.ones(NTT), np.arange(1, NTT + 1)])
    ir_forecast_baci = _X_ir[NPRE:] @ OLS(ir_dist_baci[:NPRE], _X_ir[:NPRE]).fit().params
    ir_forecast_ba = _X_ir[NPRE:] @ OLS(ir_dist_ba[:NPRE], _X_ir[:NPRE]).fit().params

    t_full = np.arange(1, NTT + 1)
    t_post = t_full[NPRE:]
    _ann_side = "top right" if ir_true_cpt < NTT * 0.7 else "top left"

    ir_left, ir_right = st.columns(2)

    _ir_panel_data = [
        (
            "BACI",
            ir_left,
            ir_dist_baci,
            ir_forecast_baci,
            ir_c_upper_baci,
            ir_c_lower_baci,
            ir_thresh_baci,
            ir_time_est_baci,
            ir_detected_baci,
        ),
        (
            "BA",
            ir_right,
            ir_dist_ba,
            ir_forecast_ba,
            ir_c_upper_ba,
            ir_c_lower_ba,
            ir_thresh_ba,
            ir_time_est_ba,
            ir_detected_ba,
        ),
    ]

    for _scen, _col, _dist, _forecast, _c_up, _c_lo, _thresh, _te, _det in _ir_panel_data:
        with _col:
            _m1, _m2 = st.columns(2)
            with _m1:
                st.metric(
                    "True changepoint (τ)",
                    f"month {ir_true_cpt}",
                    help=f"npre ({NPRE}) + delay ({ir_delay})",
                )
            with _m2:
                if _det:
                    st.metric(
                        f"{_scen} detection",
                        f"month {NPRE + int(_te)}",
                        help="Post-period month when CUSUM crossed threshold",
                    )
                else:
                    st.metric(f"{_scen} detection", "✗ not detected")

            _det_t = NPRE + int(_te) if _det else None
            _close = _det and abs(_det_t - ir_true_cpt) < 6
            _det_side = (
                ("top left" if _ann_side == "top right" else "top right") if _close else _ann_side
            )

            fig_ir_s = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.50, 0.50],
                subplot_titles=[f"Wasserstein distance ({_scen})", "Page-CUSUM path"],
                shared_xaxes=True,
                vertical_spacing=0.12,
            )
            for row in [1, 2]:
                fig_ir_s.add_vrect(
                    x0=1, x1=NPRE, fillcolor="rgba(100,149,237,0.07)", line_width=0, row=row, col=1
                )

            fig_ir_s.add_trace(
                go.Scatter(
                    x=t_full,
                    y=_dist,
                    mode="lines",
                    name="Wasserstein dist.",
                    line=dict(color="black", width=1.5),
                    hovertemplate="Month: %{x}<br>Distance: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig_ir_s.add_trace(
                go.Scatter(
                    x=t_post,
                    y=_forecast,
                    mode="lines",
                    name="OLS forecast",
                    line=dict(color="#7B1FA2", width=2, dash="dot"),
                ),
                row=1,
                col=1,
            )

            fig_ir_s.add_trace(
                go.Scatter(
                    x=t_post,
                    y=_c_up[NPRE:],
                    mode="lines",
                    name="CUSUM (upper)",
                    line=dict(color="crimson", width=2),
                ),
                row=2,
                col=1,
            )
            fig_ir_s.add_trace(
                go.Scatter(
                    x=t_post,
                    y=_c_lo[NPRE:],
                    mode="lines",
                    name="CUSUM (lower)",
                    line=dict(color="steelblue", width=2, dash="dash"),
                ),
                row=2,
                col=1,
            )
            fig_ir_s.add_trace(
                go.Scatter(
                    x=t_post,
                    y=_thresh[NPRE:],
                    mode="lines",
                    name="Threshold T(k)",
                    line=dict(color="black", width=1.5, dash="dash"),
                    hovertemplate="k=%{x}<br>T(k)=%{y:.3f}<extra>Threshold</extra>",
                ),
                row=2,
                col=1,
            )

            for row in [1, 2]:
                fig_ir_s.add_vline(
                    x=ir_true_cpt,
                    line_dash="dash",
                    line_color="#1A237E",
                    line_width=2,
                    annotation_text=f"True τ={ir_true_cpt}" if row == 1 else "",
                    annotation_position=_ann_side,
                    annotation_font=dict(color="#1A237E", size=9),
                    row=row,
                    col=1,
                )
                if _det:
                    fig_ir_s.add_vline(
                        x=_det_t,
                        line_dash="solid",
                        line_color="crimson",
                        line_width=2,
                        annotation_text=f"Detected={_det_t}" if row == 1 else "",
                        annotation_position=_det_side,
                        annotation_font=dict(color="crimson", size=9),
                        row=row,
                        col=1,
                    )

            fig_ir_s.update_layout(
                height=520,
                title=(
                    f"Run {ir_run_i + 1} · {_scen} · {ir_eff_pct}% effect · "
                    f"delay={ir_delay} mo · "
                    f"{'✓ detected at month ' + str(_det_t) if _det else '✗ not detected'}"
                ),
                xaxis2_title="Month",
                yaxis_title="Wasserstein dist.",
                yaxis2_title="CUSUM",
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, font=dict(size=10)),
                margin=dict(b=80),
            )
            st.plotly_chart(fig_ir_s, width="stretch")

    # Distribution snapshots
    st.markdown("**Distribution snapshots** — dashed = control, solid = intervention")
    snap_t_ir = st.slider(
        "Timepoint (month)",
        min_value=1,
        max_value=NTT,
        value=min(10, NTT),
        step=1,
        key="snap_t_ir",
    )
    _idx_ir = snap_t_ir - 1
    _d_ctr = ir_sim["sample_ctr"][:, _idx_ir]
    _d_itv = ir_sim["sample_itv"][:, _idx_ir]
    _xr_ir = np.linspace(
        min(_d_ctr.min(), _d_itv.min()) - 2, max(_d_ctr.max(), _d_itv.max()) + 2, 200
    )
    fig_snap_ir = go.Figure()
    fig_snap_ir.add_trace(
        go.Scatter(
            x=_xr_ir,
            y=gaussian_kde(_d_ctr).evaluate(_xr_ir),
            mode="lines",
            name="control",
            line=dict(color="#2196F3", dash="dash", width=2),
        )
    )
    fig_snap_ir.add_trace(
        go.Scatter(
            x=_xr_ir,
            y=gaussian_kde(_d_itv).evaluate(_xr_ir),
            mode="lines",
            name="intervention",
            line=dict(color="#FF5722", width=2),
        )
    )
    fig_snap_ir.update_layout(
        title=f"Distribution at t = {snap_t_ir}",
        xaxis_title="Value",
        yaxis_title="Density",
        height=380,
        legend_title="Group",
    )
    st.plotly_chart(fig_snap_ir, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# RUN YOUR OWN SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("🔬 Run your own simulation")

if "df_pending_restore" in st.session_state:
    _pr = st.session_state.pop("df_pending_restore")
    st.session_state["df_p_n_sim"] = _pr["n_sim"]
    st.session_state["df_p_base_seed"] = _pr["base_seed"]
    st.session_state["df_p_npre"] = _pr["s_npre"]
    st.session_state["df_p_npost"] = _pr["s_npost"]
    st.session_state["df_p_mu_inc"] = _pr["s_mu_inc"]
    st.session_state["df_p_ns"] = _pr["s_ns"]

for _k, _v in [
    ("df_p_n_sim", 20),
    ("df_p_base_seed", 42),
    ("df_p_npre", 24),
    ("df_p_npost", 60),
    ("df_p_mu_inc", 0.04),
    ("df_p_ns", 100),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

with st.expander("⚙️ Simulation parameters", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        n_sim = st.slider(
            "N simulations",
            min_value=10,
            max_value=200,
            step=10,
            key="df_p_n_sim",
            help="Each simulation generates a fresh (control, intervention) pair.",
        )
        base_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=99999,
            step=1,
            key="df_p_base_seed",
            help="Simulation i uses seed = base_seed + i.",
        )
    with col_b:
        s_npre = st.slider("Pre-intervention (months)", 12, 48, key="df_p_npre")
        s_npost = st.slider("Post-intervention (months)", 24, 120, key="df_p_npost")
    with col_c:
        s_mu_inc = st.slider(
            "Mean increment per month (trend)", 0.0, 0.10, step=0.005, key="df_p_mu_inc"
        )
        s_ns = st.slider("Samples per time point (ns)", 50, 500, step=50, key="df_p_ns")

_cur_params = {
    "n_sim": n_sim,
    "base_seed": int(base_seed),
    "s_npre": s_npre,
    "s_npost": s_npost,
    "s_mu_inc": s_mu_inc,
    "s_ns": s_ns,
}
if "df_mini_last_params" in st.session_state:
    if st.session_state["df_mini_last_params"] != _cur_params:
        for _k in ("df_mini_runs", "df_mini_nav_idx", "df_mnav_slider"):
            st.session_state.pop(_k, None)
        st.session_state["df_hist_sel_gen"] = st.session_state.get("df_hist_sel_gen", 0) + 1

s_run = st.button("▶ Run simulation", type="primary")

_df_history = st.session_state.get("df_mini_history", [])
if _df_history:
    _sel = st.selectbox(
        "Previous runs",
        options=range(len(_df_history)),
        format_func=lambda i: _df_history[i]["label"],
        index=None,
        placeholder="Select a previous run to restore…",
        key=f"df_hist_sel_{st.session_state.get('df_hist_sel_gen', 0)}",
    )
    if _sel is not None:
        _h = _df_history[_sel]
        _hp = _h["params"]
        st.session_state["df_pending_restore"] = _hp
        st.session_state["df_mini_runs"] = _h["data"]
        st.session_state["df_mini_nav_idx"] = _h["nav_idx"]
        st.session_state["df_mnav_slider"] = _h["nav_idx"] + 1
        st.session_state["df_mini_last_params"] = _hp
        st.session_state["df_hist_sel_gen"] = st.session_state.get("df_hist_sel_gen", 0) + 1

if s_run:
    s_ntt = s_npre + s_npost
    run_baci = True
    run_ba = True

    all_sims_baci: list = []
    all_sims_ba: list = []

    progress_bar = st.progress(0, text="Starting…")

    for i in range(n_sim):
        _seed = int(base_seed) + i
        sim = ci_sim_cdf(
            seed=_seed,
            npre=s_npre,
            npost=s_npost,
            level=[MU, SIGMA],
            trend=[s_mu_inc, 0],
            ns=s_ns,
        )
        _X_m = np.column_stack([np.ones(s_ntt), np.arange(1, s_ntt + 1)])

        if run_baci:
            d_baci = wasserstein_distance_baci(sim["sample_ctr"], sim["sample_itv"])
            _ols_b = OLS(d_baci[:s_npre], _X_m[:s_npre]).fit()
            r_baci = np.concatenate([_ols_b.resid, _X_m[s_npre:] @ _ols_b.params - d_baci[s_npre:]])
            te_baci = page_cusum(r_baci, m=s_npre, crit_val=CRIT_VAL)
            all_sims_baci.append({"sim": sim, "dist_ts": d_baci, "r": r_baci, "time_est": te_baci})

        if run_ba:
            d_ba = wasserstein_distance_ba(sim["sample_itv"], s_npre)
            _ols_b2 = OLS(d_ba[:s_npre], _X_m[:s_npre]).fit()
            r_ba = np.concatenate([_ols_b2.resid, _X_m[s_npre:] @ _ols_b2.params - d_ba[s_npre:]])
            te_ba = page_cusum(r_ba, m=s_npre, crit_val=CRIT_VAL)
            all_sims_ba.append({"sim": sim, "dist_ts": d_ba, "r": r_ba, "time_est": te_ba})

        progress_bar.progress((i + 1) / n_sim, text=f"Simulation {i + 1}/{n_sim}…")

    progress_bar.empty()

    det_baci = [np.isfinite(x["time_est"]) for x in all_sims_baci]
    det_ba = [np.isfinite(x["time_est"]) for x in all_sims_ba]
    first_det = next((i for i, d in enumerate(det_baci or det_ba) if d), 0)

    st.session_state["df_mini_runs"] = {
        "n_sim": n_sim,
        "base_seed": int(base_seed),
        "s_npre": s_npre,
        "s_npost": s_npost,
        "s_ntt": s_ntt,
        "s_mu_inc": s_mu_inc,
        "s_ns": s_ns,
        "all_sims_baci": all_sims_baci,
        "all_sims_ba": all_sims_ba,
        "det_baci": det_baci,
        "det_ba": det_ba,
    }
    st.session_state["df_mini_nav_idx"] = first_det
    st.session_state["df_mnav_slider"] = first_det + 1

    _baci_rate = sum(det_baci) / n_sim if det_baci else None
    _ba_rate = sum(det_ba) / n_sim if det_ba else None
    _run_n = len(st.session_state.get("df_mini_history", [])) + 1
    _ts = datetime.now().strftime("%H:%M")
    _hist_lbl = (
        f"#{_run_n} · {_ts} · N={n_sim} seed={int(base_seed)} pre={s_npre}mo post={s_npost}mo "
        f"μ={s_mu_inc:.3f} ns={s_ns}"
        + (f" · baci={_baci_rate:.0%}" if _baci_rate is not None else "")
        + (f" · ba={_ba_rate:.0%}" if _ba_rate is not None else "")
    )
    if "df_mini_history" not in st.session_state:
        st.session_state["df_mini_history"] = []
    st.session_state["df_mini_history"].insert(
        0,
        {
            "label": _hist_lbl,
            "data": st.session_state["df_mini_runs"],
            "nav_idx": first_det,
            "params": _cur_params,
        },
    )
    st.session_state["df_mini_last_params"] = _cur_params


# Results — persisted across rerenders via session state
if "df_mini_runs" in st.session_state and (
    st.session_state["df_mini_runs"].get("all_sims_baci")
    or st.session_state["df_mini_runs"].get("all_sims_ba")
):
    mr = st.session_state["df_mini_runs"]
    det_baci = mr["det_baci"]
    det_ba = mr["det_ba"]
    run_baci = bool(mr["all_sims_baci"])
    run_ba = bool(mr["all_sims_ba"])

    st.success("Simulation complete!")

    # Summary metrics
    sm_cols = st.columns(4)
    _col_i = 0
    if run_baci:
        _baci_r = sum(det_baci) / mr["n_sim"]
        _baci_t = [x["time_est"] for x in mr["all_sims_baci"] if np.isfinite(x["time_est"])]
        with sm_cols[_col_i]:
            st.metric(
                "BACI detection rate",
                f"{_baci_r:.1%}",
                help=f"{sum(det_baci)} of {mr['n_sim']} simulations",
            )
        with sm_cols[_col_i + 1]:
            st.metric("BACI mean time to detect", f"{np.mean(_baci_t):.1f} mo" if _baci_t else "—")
        _col_i += 2
    if run_ba:
        _ba_r = sum(det_ba) / mr["n_sim"]
        _ba_t = [x["time_est"] for x in mr["all_sims_ba"] if np.isfinite(x["time_est"])]
        with sm_cols[_col_i]:
            st.metric(
                "BA detection rate",
                f"{_ba_r:.1%}",
                help=f"{sum(det_ba)} of {mr['n_sim']} simulations",
            )
        with sm_cols[_col_i + 1]:
            st.metric("BA mean time to detect", f"{np.mean(_ba_t):.1f} mo" if _ba_t else "—")

    # Timing error summary
    _true_cpt_custom = mr["s_npre"]
    _te_cols = st.columns(2 if (run_baci and run_ba) else 1)
    _te_pairs = []
    if run_baci and mr["all_sims_baci"]:
        _te_pairs.append(("BACI", mr["all_sims_baci"], det_baci))
    if run_ba and mr["all_sims_ba"]:
        _te_pairs.append(("BA", mr["all_sims_ba"], det_ba))
    for _te_col, (_te_lbl, _te_sims, _te_det) in zip(_te_cols, _te_pairs):
        _det_sims = [s for s, d in zip(_te_sims, _te_det) if d]
        if _det_sims:
            _mae = float(
                np.mean(
                    [abs(mr["s_npre"] + int(s["time_est"]) - _true_cpt_custom) for s in _det_sims]
                )
            )
            _med_lag = float(
                np.median([mr["s_npre"] + int(s["time_est"]) - _true_cpt_custom for s in _det_sims])
            )
            with _te_col:
                _ta, _tb = st.columns(2)
                with _ta:
                    st.metric(f"{_te_lbl} mean |timing error|", f"{_mae:.1f} mo")
                with _tb:
                    st.metric(f"{_te_lbl} median detection lag", f"{_med_lag:.1f} mo")

    # Run navigator
    st.divider()
    st.subheader("Browse simulation runs")

    n_sim_mr = mr["n_sim"]

    if "df_mini_nav_idx" not in st.session_state:
        st.session_state["df_mini_nav_idx"] = 0
    if "df_mnav_slider" not in st.session_state:
        st.session_state["df_mnav_slider"] = 1

    def _df_nav_prev():
        new = max(0, st.session_state["df_mini_nav_idx"] - 1)
        st.session_state["df_mini_nav_idx"] = new
        st.session_state["df_mnav_slider"] = new + 1

    def _df_nav_next():
        new = min(n_sim_mr - 1, st.session_state["df_mini_nav_idx"] + 1)
        st.session_state["df_mini_nav_idx"] = new
        st.session_state["df_mnav_slider"] = new + 1

    def _df_on_slider():
        st.session_state["df_mini_nav_idx"] = st.session_state["df_mnav_slider"] - 1

    nc1, nc2, nc3 = st.columns([1, 10, 1])
    with nc1:
        st.button("◀", on_click=_df_nav_prev, key="df_mnav_prev", width="stretch")
    with nc2:
        st.slider("Run", 1, n_sim_mr, key="df_mnav_slider", on_change=_df_on_slider)
    with nc3:
        st.button("▶", on_click=_df_nav_next, key="df_mnav_next", width="stretch")

    # Run caption
    _n_det_caption = max(sum(det_baci) if run_baci else 0, sum(det_ba) if run_ba else 0)
    st.caption(
        f"Detected a change in {_n_det_caption} of {n_sim_mr} runs "
        f"({_n_det_caption / n_sim_mr:.0%})."
    )

    show_idx = st.session_state.get("df_mini_nav_idx", 0)
    show_seed = mr["base_seed"] + show_idx
    st.subheader(
        f"Run #{show_idx + 1} of {n_sim_mr} · base seed {mr['base_seed']} · run seed {show_seed}"
    )

    s_ntt = mr["s_ntt"]
    t_sim = np.arange(1, s_ntt + 1)
    t_post_sim = t_sim[mr["s_npre"] :]
    _X_show = np.column_stack([np.ones(s_ntt), np.arange(1, s_ntt + 1)])

    _show_scenarios = []
    if run_baci:
        _show_scenarios.append(
            ("BACI", mr["all_sims_baci"][show_idx], det_baci[show_idx], "rgba(33,150,243,1)")
        )
    if run_ba:
        _show_scenarios.append(
            ("BA", mr["all_sims_ba"][show_idx], det_ba[show_idx], "rgba(255,152,0,1)")
        )

    show_cols = st.columns(len(_show_scenarios))

    for col, (scen_label, run_data, was_detected, colour) in zip(show_cols, _show_scenarios):
        te = run_data["time_est"]
        dist = run_data["dist_ts"]
        r = run_data["r"]

        _ols_show = OLS(dist[: mr["s_npre"]], _X_show[: mr["s_npre"]]).fit()
        _forecast_show = _X_show[mr["s_npre"] :] @ _ols_show.params

        # CUSUM path
        _train_mean = np.mean(r[: mr["s_npre"]])
        _sigma_r = np.std(r[: mr["s_npre"]], ddof=1)
        _c_up = np.zeros(s_ntt)
        _c_lo = np.zeros(s_ntt)
        _thresh = np.zeros(s_ntt)
        if _sigma_r >= 1e-12:
            for _k in range(1, mr["s_npost"] + 1):
                _inc = r[mr["s_npre"] + _k - 1] - _train_mean
                _c_up[mr["s_npre"] + _k - 1] = max(0.0, _c_up[mr["s_npre"] + _k - 2] + _inc)
                _c_lo[mr["s_npre"] + _k - 1] = max(0.0, _c_lo[mr["s_npre"] + _k - 2] - _inc)
                _w = np.sqrt(mr["s_npre"]) * (1.0 + _k / mr["s_npre"])
                _thresh[mr["s_npre"] + _k - 1] = _w * CRIT_VAL * _sigma_r

        det_text = f"✓ month {mr['s_npre'] + int(te)}" if was_detected else "✗ not detected"

        with col:
            # Per-run metrics
            mm1, mm2 = st.columns(2)
            with mm1:
                st.metric(f"{scen_label} — result", det_text)
            with mm2:
                st.metric("Critical value", f"{CRIT_VAL:.3f}")

            fig_m = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.50, 0.50],
                subplot_titles=["Wasserstein distance + forecast", "Page-CUSUM path"],
                shared_xaxes=True,
                vertical_spacing=0.12,
            )
            for row in [1, 2]:
                fig_m.add_vrect(
                    x0=1,
                    x1=mr["s_npre"],
                    fillcolor="rgba(100,149,237,0.07)",
                    line_width=0,
                    row=row,
                    col=1,
                )

            fig_m.add_trace(
                go.Scatter(
                    x=t_sim,
                    y=dist,
                    mode="lines",
                    name="Wasserstein dist.",
                    line=dict(color="black", width=1.5),
                    hovertemplate="Month: %{x}<br>Distance: %{y:.4f}<extra></extra>",
                ),
                row=1,
                col=1,
            )
            fig_m.add_trace(
                go.Scatter(
                    x=t_post_sim,
                    y=_forecast_show,
                    mode="lines",
                    name="OLS forecast",
                    line=dict(color="#7B1FA2", width=2, dash="dot"),
                ),
                row=1,
                col=1,
            )

            fig_m.add_trace(
                go.Scatter(
                    x=t_post_sim,
                    y=_c_up[mr["s_npre"] :],
                    mode="lines",
                    name="CUSUM (upper)",
                    line=dict(color="crimson", width=2),
                ),
                row=2,
                col=1,
            )
            fig_m.add_trace(
                go.Scatter(
                    x=t_post_sim,
                    y=_c_lo[mr["s_npre"] :],
                    mode="lines",
                    name="CUSUM (lower)",
                    line=dict(color="steelblue", width=2, dash="dash"),
                ),
                row=2,
                col=1,
            )
            fig_m.add_trace(
                go.Scatter(
                    x=t_post_sim,
                    y=_thresh[mr["s_npre"] :],
                    mode="lines",
                    name="Threshold T(k)",
                    line=dict(color="black", width=1.5, dash="dash"),
                    hovertemplate="k=%{x}<br>T(k)=%{y:.3f}<extra>Threshold</extra>",
                ),
                row=2,
                col=1,
            )

            for row in [1, 2]:
                fig_m.add_vline(
                    x=mr["s_npre"],
                    line_dash="dash",
                    line_color="#1A237E",
                    line_width=2,
                    row=row,
                    col=1,
                )
                if was_detected:
                    fig_m.add_vline(
                        x=mr["s_npre"] + int(te),
                        line_dash="solid",
                        line_color="crimson",
                        line_width=2,
                        row=row,
                        col=1,
                    )

            fig_m.update_layout(
                height=480,
                xaxis2_title="Month",
                yaxis_title="Distance",
                yaxis2_title="CUSUM",
                legend=dict(orientation="h", yanchor="bottom", y=-0.20, font=dict(size=9)),
                margin=dict(b=80),
            )
            st.plotly_chart(fig_m, width="stretch")

    # Distribution snapshots
    st.markdown("**Distribution snapshots** — dashed = control, solid = intervention")
    _ref_sim = (mr["all_sims_baci"] or mr["all_sims_ba"])[show_idx]["sim"]
    snap_t_df = st.slider(
        "Timepoint (month)",
        min_value=1,
        max_value=s_ntt,
        value=min(10, s_ntt),
        step=1,
        key="snap_t_df",
    )
    _idx_df = snap_t_df - 1
    _d_ctr = _ref_sim["sample_ctr"][:, _idx_df]
    _d_itv = _ref_sim["sample_itv"][:, _idx_df]
    _xr_df = np.linspace(
        min(_d_ctr.min(), _d_itv.min()) - 2, max(_d_ctr.max(), _d_itv.max()) + 2, 200
    )
    fig_snap_df = go.Figure()
    fig_snap_df.add_trace(
        go.Scatter(
            x=_xr_df,
            y=gaussian_kde(_d_ctr).evaluate(_xr_df),
            mode="lines",
            name="control",
            line=dict(color="#2196F3", dash="dash", width=2),
        )
    )
    fig_snap_df.add_trace(
        go.Scatter(
            x=_xr_df,
            y=gaussian_kde(_d_itv).evaluate(_xr_df),
            mode="lines",
            name="intervention",
            line=dict(color="#FF5722", width=2),
        )
    )
    fig_snap_df.update_layout(
        title=f"Distribution at t = {snap_t_df}",
        xaxis_title="Value",
        yaxis_title="Density",
        height=380,
        legend_title="Group",
    )
    st.plotly_chart(fig_snap_df, width="stretch")
