"""
SimRewilding — Trend Change Detection using Forecast (Page-CUSUM)
Interactive analysis page for rewild_trend_change_forecast results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import warnings
from datetime import datetime
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.arima.model import ARIMA

from amoc import ci_sim, ci_sim_ar, page_cusum_max_stat, trend_stats_forecast

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trend Change Detection (Forecast)",
    page_icon="🔮",
    layout="wide",
)

# ── Simulation constants (mirror rewild_trend_change_forecast.py) ──────────────
NPRE = 24
NPOST_YEARS = 10
NPOST_MONTHS = 12 * NPOST_YEARS
NPOST_MAX = NPOST_MONTHS
NTT = NPRE + NPOST_MAX
LEVEL = 10.0
TREND_CONTROL = 0.005
SIGMA = 0.05
PHI_DEFAULT = 0.8
ALPHA = 0.95

TREND_INCREASE = np.round(
    LEVEL * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / NPOST_MONTHS, 4
)
EFFECT_SIZES_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

RESULTS_PATH = Path(__file__).parent.parent / "results" / "trend_forecast" / "sim_results.pkl"
PALETTE = px.colors.sample_colorscale("Viridis", [i / 10 for i in range(11)])


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading pre-computed simulation results…")
def load_results(path: Path):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── CUSUM path helper (for individual run explorer) ───────────────────────────
def compute_cusum_path(y_itv, npre, ntt, phi=None):
    """Return (r, c_upper, c_lower) arrays over the full residual series."""
    y = np.asarray(y_itv, dtype=float)
    t_idx = np.arange(1, ntt + 1)
    X = np.column_stack([np.ones(ntt), t_idx])

    in_residuals = None
    out_errors = None

    if phi is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(y[:npre], exog=X[:npre], order=(1, 0, 0), trend="n")
                fit = model.fit(method="innovations_mle", disp=False)
            in_residuals = np.asarray(fit.resid, dtype=float)
            forecast_vals = fit.get_forecast(steps=ntt - npre, exog=X[npre:]).predicted_mean
            out_errors = np.asarray(forecast_vals, dtype=float) - y[npre:]
        except Exception:
            in_residuals = None

    if in_residuals is None:
        ols = OLS(y[:npre], X[:npre]).fit()
        in_residuals = np.asarray(ols.resid, dtype=float)
        out_errors = X[npre:] @ ols.params - y[npre:]

    r = np.concatenate([in_residuals, out_errors])
    sigma_hat = np.std(r[:npre], ddof=1)
    if sigma_hat < 1e-12:
        return r, np.zeros(len(r)), np.zeros(len(r))

    z = r / sigma_hat
    c_upper = np.zeros(len(r))
    c_lower = np.zeros(len(r))
    for t in range(npre, len(r)):
        c_upper[t] = max(0.0, c_upper[t - 1] + z[t])
        c_lower[t] = max(0.0, c_lower[t - 1] - z[t])

    return r, c_upper, c_lower


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔮 Trend Change Detection (Forecast)")

st.markdown("""
**Forecast (Page-CUSUM)** is an *online* changepoint detection method that processes
data one observation at a time and raises an alarm as soon as sufficient evidence
accumulates — without waiting for a fixed monitoring window to end.

Unlike AMOC which tests the full series retrospectively, Forecast declares detection as soon
as the evidence threshold is crossed. This makes it well-suited to real-time monitoring.
""")


# ── Sidebar ────────────────────────────────────────────────────────────────────
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
    | **npre** | {NPRE} months | Training period |
    | **npost_max** | {NPOST_MAX} months | Max observation window |
    | **level** | {LEVEL} | Baseline indicator value |
    | **trend_control** | {TREND_CONTROL}/mo | Pre-intervention slope |
    | **φ (phi)** | {PHI_DEFAULT} | AR(1) autocorrelation |
    | **Nsim** | 1,000 | Null sims for threshold calibration |
    | **simN** | 1,000 | Main simulations per effect size |
    """)


# ── Load data ──────────────────────────────────────────────────────────────────
data = load_results(RESULTS_PATH)

if data is None:
    st.error(
        f"Pre-computed results not found at `{RESULTS_PATH}`. "
        "Run `python rewild_trend_change_forecast.py` first."
    )
    results_available = False
else:
    thresholds = data.get("thresholds")
    res_iid    = data.get("detection_results_iid")
    res_ar     = data.get("detection_results_ar")

    if thresholds is None or res_iid is None or res_ar is None:
        missing = [k for k, v in [
            ("thresholds", thresholds),
            ("detection_results_iid", res_iid),
            ("detection_results_ar", res_ar),
        ] if v is None]
        st.warning(
            f"Results file exists but is incomplete (missing: {', '.join(missing)}). "
            "The simulation may still be running."
        )
        results_available = False
    else:
        trends = sorted(res_iid.keys())
        results_available = True


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
if results_available:
    st.header("📊 Results explorer")

    tab_det, tab_ttd, tab_null, tab_err, tab_delay, tab_bias = st.tabs([
        "Detection rates",
        "Time to detection",
        "Null distributions",
        "Changepoint error",
        "Detection by delay",
        "Changepoint bias",
    ])

    # ── Tab 1: Detection Rates ─────────────────────────────────────────────────
    with tab_det:
        st.subheader("Detection rate by effect size")
        st.markdown("""
        The **detection rate** is the fraction of simulations in which Page-CUSUM
        declared a changepoint within the maximum 10-year observation window.

        Since Forecast is an online method, this is a combined measure of *sensitivity*
        and *speed*: a run counts as detected whether the alarm fires at month 25 or
        month 143. Compare with the time-to-detection tab to separate these dimensions.
        """)

        iid_rates = [res_iid[t]["detection_rate"] for t in trends]
        ar_rates  = [res_ar[t]["detection_rate"]  for t in trends]
        labels    = [f"{p}%" for p in EFFECT_SIZES_PCT]

        fig_det = go.Figure()
        fig_det.add_trace(go.Bar(
            x=labels, y=iid_rates,
            name="i.i.d. BA",
            marker_color="rgba(33,150,243,0.82)",
            hovertemplate="%{x} effect<br>Detection: %{y:.1%}<extra>i.i.d. BA</extra>",
        ))
        fig_det.add_trace(go.Bar(
            x=labels, y=ar_rates,
            name="AR(1) BA",
            marker_color="rgba(255,152,0,0.82)",
            hovertemplate="%{x} effect<br>Detection: %{y:.1%}<extra>AR(1) BA</extra>",
        ))
        fig_det.add_hline(y=0.80, line_dash="dash", line_color="grey",
                          annotation_text="80%", annotation_position="right")
        fig_det.add_hline(y=0.95, line_dash="dot", line_color="grey",
                          annotation_text="95%", annotation_position="right")
        fig_det.update_layout(
            barmode="group",
            xaxis_title="Effect size (% of mean / 10 yr)",
            yaxis=dict(title="Detection rate", range=[0, 1.05], tickformat=".0%"),
            legend_title="Noise model",
            height=460,
            title="Detection rate within 10-year observation window",
        )
        st.plotly_chart(fig_det, width="stretch")

        st.info(
            "AR(1) autocorrelation reduces detection rates, especially for small effect sizes. "
            "The AR(1) threshold is calibrated separately (it must be higher to control false positives "
            "under autocorrelated noise), which lowers sensitivity for weak signals.",
            icon="ℹ️",
        )

    # ── Tab 2: Time to Detection ───────────────────────────────────────────────
    with tab_ttd:
        st.subheader("Time to detection distribution")
        st.markdown("""
        For each detected run, *time_est* is the number of post-intervention months
        that elapsed before Page-CUSUM crossed the threshold. Shorter = faster detection.

        Only detected runs are shown. Non-detected runs are excluded (they contribute
        to the detection rate gap shown in the previous tab).
        """)

        ttd_view = st.radio("View", ["Mean by effect size", "Distribution"],
                            horizontal=True, key="ttd_view")

        IID_COLOUR = "rgba(33,150,243,1)"
        AR_COLOUR  = "rgba(255,152,0,1)"
        TTD_CONFIGS = [("i.i.d. BA", res_iid, IID_COLOUR), ("AR(1) BA", res_ar, AR_COLOUR)]

        def _ttd_means(results):
            means = []
            for t in trends:
                tv = results[t]["time_est_vec"]
                detected = np.isfinite(tv)
                means.append(float(tv[detected].mean()) if detected.any() else np.nan)
            return means

        def mean_ttd_fig(configs, title):
            fig = go.Figure()
            for label, results, colour in configs:
                fig.add_trace(go.Scatter(
                    x=[f"{p}%" for p in EFFECT_SIZES_PCT], y=_ttd_means(results),
                    mode="lines+markers",
                    name=label,
                    marker=dict(size=8),
                    line=dict(color=colour, width=2),
                    hovertemplate=f"%{{x}} effect<br>Mean time: %{{y:.1f}} months<extra>{label}</extra>",
                ))
            fig.update_layout(
                title=title,
                xaxis_title="Effect size",
                yaxis_title="Mean time to detection (months)",
                legend_title="Noise model",
                height=420,
            )
            return fig

        def box_ttd_fig(configs, title):
            fig = go.Figure()
            for label, results, colour in configs:
                q1s, meds, q3s, p5s, p95s = [], [], [], [], []
                for t in trends:
                    tv = results[t]["time_est_vec"]
                    vals = tv[np.isfinite(tv)]
                    if len(vals) >= 5:
                        q1s.append(float(np.percentile(vals, 25)))
                        meds.append(float(np.median(vals)))
                        q3s.append(float(np.percentile(vals, 75)))
                        p5s.append(float(np.percentile(vals, 5)))
                        p95s.append(float(np.percentile(vals, 95)))
                    else:
                        q1s.append(None); meds.append(None); q3s.append(None)
                        p5s.append(None); p95s.append(None)
                fig.add_trace(go.Box(
                    x=[f"{p}%" for p in EFFECT_SIZES_PCT],
                    q1=q1s, median=meds, q3=q3s,
                    lowerfence=p5s, upperfence=p95s,
                    name=label,
                    marker_color=colour,
                    line_color=colour,
                    boxpoints=False,
                    hovertemplate=(
                        "<b>%{x} — " + label + "</b><br>"
                        "Median: %{median:.0f} mo<br>IQR: %{q1:.0f}–%{q3:.0f} mo"
                        "<extra></extra>"
                    ),
                ))
            fig.update_layout(
                title=title,
                boxmode="group",
                xaxis_title="Effect size",
                yaxis_title="Time to detection (months post-intervention)",
                legend_title="Noise model",
                height=460,
            )
            return fig

        make_fig = box_ttd_fig if ttd_view == "Distribution (box)" else mean_ttd_fig
        st.plotly_chart(
            make_fig(TTD_CONFIGS, "Time to detection — i.i.d. vs AR(1) (BA)"),
            width="stretch",
        )

    # ── Tab 3: Null Distributions ──────────────────────────────────────────────
    with tab_null:
        st.subheader("Null distributions of max Page-CUSUM statistic")
        st.markdown("""
        The histograms show the max CUSUM under the null hypothesis (no change) for i.i.d. and AR(1) noise.
        The crimson line marks the **95th percentile** — the threshold used to declare a detection.
        """)

        fig_null = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                "i.i.d. BA — max CUSUM under null",
                "AR(1) BA — max CUSUM under null",
            ],
        )
        for col_i, (key, colour) in enumerate([("iid", "#2196F3"), ("ar1", "#FF9800")], start=1):
            entry = thresholds[key]
            dist = entry["null_dist"]
            h    = entry["threshold"]
            fig_null.add_trace(
                go.Histogram(
                    x=dist, nbinsx=50,
                    marker_color=colour, opacity=0.75,
                    showlegend=False,
                    hovertemplate="max CUSUM: %{x:.1f}<br>Count: %{y}<extra></extra>",
                ),
                row=1, col=col_i,
            )
            fig_null.add_vline(
                x=h, line_dash="dash", line_color="crimson",
                annotation_text=f"h = {h:.2f}",
                annotation_position="top right",
                row=1, col=col_i,
            )

        fig_null.update_layout(
            height=400,
            title="Null distribution of max Page-CUSUM statistic (1,000 simulations each)",
        )
        fig_null.update_xaxes(title_text="max CUSUM")
        fig_null.update_yaxes(title_text="Count")
        st.plotly_chart(fig_null, width="stretch")

        cv_tbl = {
            "Noise model":       ["i.i.d. BA", "AR(1) BA"],
            "Threshold (95th pct)": [
                f"{thresholds['iid']['threshold']:.3f}",
                f"{thresholds['ar1']['threshold']:.3f}",
            ],
            "Null mean": [
                f"{np.mean(thresholds['iid']['null_dist']):.3f}",
                f"{np.mean(thresholds['ar1']['null_dist']):.3f}",
            ],
            "Null SD": [
                f"{np.std(thresholds['iid']['null_dist']):.3f}",
                f"{np.std(thresholds['ar1']['null_dist']):.3f}",
            ],
        }
        st.table(cv_tbl)

    # ── Tab 4: Changepoint Error ───────────────────────────────────────────────
    with tab_err:
        st.subheader("Changepoint localisation error")
        st.markdown("""
        When a changepoint is detected, we use AMOC on the residuals up to the
        detection time to locate when the change occurred. The mean
        absolute localisation error is `|cpt_est − true_cpt|`.

        Only runs that successfully detected a changepoint are included. Error is measured
        in months.
        """)

        err_noise = st.radio("Noise model", ["i.i.d. BA", "AR(1) BA", "Comparison"],
                             horizontal=True, key="err_noise")

        def err_fig(results, title, colour):
            errors, pcts_valid = [], []
            for t, pct in zip(trends, EFFECT_SIZES_PCT):
                tv  = results[t]["time_est_vec"]
                cv2 = results[t]["cpt_est_vec"]
                dlys = np.array(results[t]["delays"])
                true_cpts = NPRE + dlys
                det = np.isfinite(tv)
                if det.any():
                    errors.append(float(np.abs(cv2[det] - true_cpts[det]).mean()))
                    pcts_valid.append(pct)

            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[f"{p}%" for p in pcts_valid],
                y=errors,
                marker_color=colour,
                hovertemplate="%{x} effect<br>Mean |error|: %{y:.2f} months<extra></extra>",
            ))
            fig.update_layout(
                title=title,
                xaxis_title="Effect size",
                yaxis_title="Mean |cpt_est − true_cpt| (months)",
                height=420,
            )
            return fig

        if err_noise == "i.i.d. BA":
            st.plotly_chart(err_fig(res_iid, "Changepoint error — i.i.d. BA",
                                    "rgba(33,150,243,0.82)"), width="stretch")
        elif err_noise == "AR(1) BA":
            st.plotly_chart(err_fig(res_ar, "Changepoint error — AR(1) BA",
                                    "rgba(255,152,0,0.82)"), width="stretch")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(err_fig(res_iid, "i.i.d. BA",
                                         "rgba(33,150,243,0.82)"), width="stretch")
            with c2:
                st.plotly_chart(err_fig(res_ar, "AR(1) BA",
                                         "rgba(255,152,0,0.82)"), width="stretch")

    # ── Tab 5: Detection by Delay ──────────────────────────────────────────────
    with tab_delay:
        st.subheader("Effect of intervention delay on detection")
        st.markdown("""
        The ecological response to an intervention may start with a delay after the
        formal intervention date. A delay shifts the true changepoint deeper into the
        post-period, giving less time for evidence to accumulate before the window ends.
        """)

        dl_c1, dl_c2, dl_c3 = st.columns(3)
        with dl_c1:
            delay_eff_pct = st.select_slider(
                "Effect size", options=EFFECT_SIZES_PCT, value=20,
                format_func=lambda x: f"{x}%", key="delay_eff",
            )
        with dl_c2:
            delay_noise = st.radio("Noise model", ["i.i.d. BA", "AR(1) BA", "Both"],
                                   horizontal=True, key="delay_noise")
        with dl_c3:
            delay_view = st.radio("View", ["Line chart", "Heatmap"],
                                  horizontal=True, key="delay_view")

        delay_idx   = EFFECT_SIZES_PCT.index(delay_eff_pct)
        delay_trend = trends[delay_idx]
        unique_delays = np.arange(1, 21)

        def delay_rates(results, trend):
            dlys = np.array(results[trend]["delays"])
            det  = np.isfinite(results[trend]["time_est_vec"])
            rates = []
            for d in unique_delays:
                mask = dlys == d
                rates.append(float(det[mask].mean()) if mask.any() else np.nan)
            return np.array(rates)

        if delay_view == "Line chart":
            fig_dl = go.Figure()
            configs = []
            if delay_noise in ("i.i.d. BA", "Both"):
                configs.append(("i.i.d. BA", res_iid, "rgba(33,150,243,1)"))
            if delay_noise in ("AR(1) BA", "Both"):
                configs.append(("AR(1) BA", res_ar, "rgba(255,152,0,1)"))

            for label, results, colour in configs:
                rates = delay_rates(results, delay_trend)
                fig_dl.add_trace(go.Scatter(
                    x=unique_delays, y=rates,
                    mode="lines+markers",
                    name=label,
                    line=dict(color=colour, width=2.5),
                    marker=dict(size=7),
                    hovertemplate="Delay: %{x} mo<br>Detection: %{y:.1%}<extra>" + label + "</extra>",
                ))

            fig_dl.add_hline(y=0.80, line_dash="dash", line_color="grey",
                             annotation_text="80%", annotation_position="right")
            fig_dl.update_layout(
                title=f"Detection rate vs intervention delay — {delay_eff_pct}% effect",
                xaxis=dict(title="Intervention delay (months)", dtick=2),
                yaxis=dict(title="Detection rate", range=[0, 1.05], tickformat=".0%"),
                legend_title="Noise model",
                height=440,
            )
            st.plotly_chart(fig_dl, width="stretch")

        else:
            # Heatmap: delay × effect size
            _hm_noise = delay_noise if delay_noise != "Both" else "i.i.d. BA"
            if delay_noise == "Both":
                _hm_noise = st.radio("Heatmap noise model", ["i.i.d. BA", "AR(1) BA"],
                                     horizontal=True, key="delay_hm_noise")
            hm_res = res_iid if _hm_noise == "i.i.d. BA" else res_ar
            z = np.full((len(unique_delays), len(trends)), np.nan)
            for ti, t in enumerate(trends):
                dlys = np.array(hm_res[t]["delays"])
                det  = np.isfinite(hm_res[t]["time_est_vec"])
                for di, d in enumerate(unique_delays):
                    mask = dlys == d
                    if mask.any():
                        z[di, ti] = float(det[mask].mean())

            fig_hm = go.Figure()
            fig_hm.add_trace(go.Heatmap(
                x=[f"{p}%" for p in EFFECT_SIZES_PCT],
                y=unique_delays,
                z=z,
                colorscale="Viridis", zmin=0, zmax=1,
                colorbar=dict(title="Detection rate", tickformat=".0%"),
                hovertemplate="Effect: %{x}<br>Delay: %{y} mo<br>Detection: %{z:.1%}<extra></extra>",
            ))
            fig_hm.update_layout(
                title=f"Detection rate: effect size × delay — {_hm_noise} noise",
                xaxis_title="Effect size",
                yaxis=dict(title="Intervention delay (months)", dtick=2),
                height=480,
            )
            st.plotly_chart(fig_hm, width="stretch")

    # ── Tab 6: Changepoint Bias ────────────────────────────────────────────────
    with tab_bias:
        st.subheader("Changepoint timing accuracy")
        st.markdown("""
        Signed detection timing error: *cpt_est − true_cpt*. Positive = declared late;
        negative = declared early. Only detected runs are included.

        The green band (±3 months) marks a practically acceptable timing window.
        """)

        bias_c1, bias_c2 = st.columns(2)
        with bias_c1:
            bias_noise = st.radio("Noise model", ["i.i.d. BA", "AR(1) BA", "Both"],
                                  horizontal=True, key="bias_noise")
        with bias_c2:
            bias_min_n = st.slider("Minimum detections to show box", 5, 50, 10,
                                   key="bias_min_n")

        bias_configs = []
        if bias_noise in ("i.i.d. BA", "Both"):
            bias_configs.append(("i.i.d. BA", res_iid, "rgba(33,150,243,0.85)"))
        if bias_noise in ("AR(1) BA", "Both"):
            bias_configs.append(("AR(1) BA", res_ar, "rgba(255,152,0,0.85)"))

        fig_bias = go.Figure()
        fig_bias.add_hrect(y0=-3, y1=3,
                           fillcolor="rgba(0,180,0,0.07)", line_width=0,
                           annotation_text="±3 mo", annotation_position="top right",
                           annotation_font=dict(color="green", size=10))

        n_table = {}
        for label, results, colour in bias_configs:
            q1s, meds, q3s, p5s, p95s = [], [], [], [], []
            ns = []
            for t, pct in zip(trends, EFFECT_SIZES_PCT):
                tv   = results[t]["time_est_vec"]
                cv2  = results[t]["cpt_est_vec"]
                dlys = np.array(results[t]["delays"])
                true_cpts = NPRE + dlys
                det  = np.isfinite(tv)
                errs = cv2[det] - true_cpts[det]
                n    = len(errs)
                ns.append(n)
                if n >= bias_min_n:
                    q1s.append(float(np.percentile(errs, 25)))
                    meds.append(float(np.median(errs)))
                    q3s.append(float(np.percentile(errs, 75)))
                    p5s.append(float(np.percentile(errs, 5)))
                    p95s.append(float(np.percentile(errs, 95)))
                else:
                    q1s.append(None); meds.append(None); q3s.append(None)
                    p5s.append(None); p95s.append(None)
            n_table[label] = ns
            fig_bias.add_trace(go.Box(
                x=[f"{p}%" for p in EFFECT_SIZES_PCT],
                q1=q1s, median=meds, q3=q3s,
                lowerfence=p5s, upperfence=p95s,
                name=label,
                marker_color=colour, line_color=colour, boxpoints=False,
                hovertemplate=(
                    "<b>%{x} — " + label + "</b><br>"
                    "Median: %{median} mo<br>IQR: %{q1}–%{q3} mo"
                    "<extra></extra>"
                ),
            ))

        fig_bias.add_hline(y=0, line_color="black", line_width=1.5, line_dash="dot",
                           annotation_text="True τ", annotation_position="top left",
                           annotation_font=dict(size=10))
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

        n_rows = {"Effect size": [f"{p}%" for p in EFFECT_SIZES_PCT]}
        for label, ns in n_table.items():
            n_rows[f"Detected ({label})"] = ns
        st.caption("Detections per 1,000 simulations used to build each box:")
        st.dataframe(n_rows, hide_index=True, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL RUN EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
if results_available:
    st.divider()
    st.header("🔍 Individual run explorer")
    st.markdown("""
    Browse individual simulation runs. The lower panel shows the **Page-CUSUM path** —
    the running upper and lower CUSUM statistics — alongside the detection threshold.
    Watch how evidence accumulates after the true changepoint and the CUSUM eventually
    crosses the threshold.
    """)

    ir_c1, ir_c2 = st.columns(2)
    with ir_c1:
        ir_eff_pct = st.select_slider(
            "Effect size", options=EFFECT_SIZES_PCT, value=30,
            format_func=lambda x: f"{x}%", key="ir_eff",
        )
    with ir_c2:
        ir_noise = st.radio("Noise model", ["i.i.d. BA", "AR(1) BA"],
                            horizontal=True, key="ir_noise")

    if "ir_nav_idx" not in st.session_state:
        st.session_state["ir_nav_idx"] = 0
    if "ir_nav_slider" not in st.session_state:
        st.session_state["ir_nav_slider"] = 1

    def _ir_prev():
        n = max(0, st.session_state["ir_nav_idx"] - 1)
        st.session_state["ir_nav_idx"]    = n
        st.session_state["ir_nav_slider"] = n + 1

    def _ir_next():
        n = min(999, st.session_state["ir_nav_idx"] + 1)
        st.session_state["ir_nav_idx"]    = n
        st.session_state["ir_nav_slider"] = n + 1

    def _ir_slider():
        st.session_state["ir_nav_idx"] = st.session_state["ir_nav_slider"] - 1

    en1, en2, en3 = st.columns([1, 10, 1])
    with en1:
        st.button("◀", on_click=_ir_prev, key="ir_prev", width="stretch")
    with en2:
        st.slider("Run", 1, 1000, key="ir_nav_slider", on_change=_ir_slider)
    with en3:
        st.button("▶", on_click=_ir_next, key="ir_next", width="stretch")

    ir_trend    = trends[EFFECT_SIZES_PCT.index(ir_eff_pct)]
    ir_run_i    = st.session_state.get("ir_nav_idx", 0)
    ir_results  = res_iid if ir_noise == "i.i.d. BA" else res_ar

    if ir_run_i >= len(ir_results[ir_trend]["delays"]):
        st.warning(
            "It looks like you have run fewer simulations than expected. "
            "Please re-run the full simulation."
        )
        st.stop()
    ir_phi      = None if ir_noise == "i.i.d. BA" else PHI_DEFAULT
    ir_h        = thresholds["iid"]["threshold"] if ir_noise == "i.i.d. BA" else thresholds["ar1"]["threshold"]

    ir_delay    = int(ir_results[ir_trend]["delays"][ir_run_i])
    ir_true_cpt = NPRE + ir_delay
    ir_time_est = ir_results[ir_trend]["time_est_vec"][ir_run_i]
    ir_cpt_est  = ir_results[ir_trend]["cpt_est_vec"][ir_run_i]
    ir_detected = np.isfinite(ir_time_est)
    ir_seed      = ir_results[ir_trend]["seeds"][ir_run_i]
    _npre_delay  = NPRE + ir_delay
    _npost_delay = NPOST_MAX - ir_delay
    if ir_phi is None:
        ir_sim_data = ci_sim(seed=ir_seed, npre=_npre_delay, npost=_npost_delay,
                             level=LEVEL, trend=[TREND_CONTROL, ir_trend], sigma=SIGMA)
    else:
        ir_sim_data = ci_sim_ar(seed=ir_seed, npre=_npre_delay, npost=_npost_delay,
                                level=LEVEL, trend=[TREND_CONTROL, ir_trend],
                                phi=ir_phi, sigma=SIGMA)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("True changepoint (τ)", f"month {ir_true_cpt}",
                  help=f"npre ({NPRE}) + delay ({ir_delay})")
    with m2:
        st.metric("Detection time", f"month {NPRE + int(ir_time_est)}" if ir_detected else "✗ not detected",
                  help="Post-period month when CUSUM crossed threshold")
    with m3:
        st.metric("Estimated τ̂", f"month {int(ir_cpt_est)}" if ir_detected else "—")
    with m4:
        if ir_detected:
            err = int(ir_cpt_est) - ir_true_cpt
            st.metric("Timing error", f"{err:+d} mo",
                      delta=f"{'late' if err > 0 else 'early' if err < 0 else 'exact'}")
        else:
            st.metric("Timing error", "—")

    # Compute CUSUM path for this run
    y_itv_full = ir_sim_data["y_itv"]
    r, c_upper, c_lower = compute_cusum_path(y_itv_full, NPRE, NTT, phi=ir_phi)
    t_full = np.arange(1, NTT + 1)

    # Build 3-panel figure
    fig_run = make_subplots(
        rows=3, cols=1,
        row_heights=[0.38, 0.28, 0.34],
        subplot_titles=[
            "Intervention series (training ↔ forecast region)",
            "Residual series (in-sample ∥ out-of-sample forecast errors)",
            "Page-CUSUM path",
        ],
        shared_xaxes=True,
        vertical_spacing=0.09,
    )

    # Shaded pre-period
    for row in [1, 2, 3]:
        fig_run.add_vrect(x0=1, x1=NPRE, fillcolor="rgba(100,149,237,0.07)",
                          line_width=0, row=row, col=1)

    # Panel 1: time series
    fig_run.add_trace(go.Scatter(
        x=t_full, y=y_itv_full, mode="lines", name="Intervention",
        line=dict(color="rgba(76,175,80,0.8)", width=1.5),
    ), row=1, col=1)
    # Forecast overlay: OLS/ARIMA fitted on pre-period
    t_idx = np.arange(1, NTT + 1)
    X_all = np.column_stack([np.ones(NTT), t_idx])
    ols_fit = OLS(y_itv_full[:NPRE], X_all[:NPRE]).fit()
    y_fitted_pre  = X_all[:NPRE] @ ols_fit.params
    y_forecast    = X_all[NPRE:] @ ols_fit.params
    fig_run.add_trace(go.Scatter(
        x=t_full[:NPRE], y=y_fitted_pre, mode="lines", name="OLS fit (pre)",
        line=dict(color="#1565C0", width=2, dash="dash"),
    ), row=1, col=1)
    fig_run.add_trace(go.Scatter(
        x=t_full[NPRE:], y=y_forecast, mode="lines", name="Forecast",
        line=dict(color="#7B1FA2", width=2, dash="dot"),
    ), row=1, col=1)

    # Panel 2: residuals
    fig_run.add_trace(go.Scatter(
        x=t_full, y=r, mode="lines", name="Residuals",
        line=dict(color="rgba(255,152,0,0.8)", width=1.2),
        fill="tozeroy", fillcolor="rgba(255,152,0,0.07)",
        showlegend=True,
    ), row=2, col=1)
    fig_run.add_hline(y=0, line_color="rgba(0,0,0,0.25)", line_width=1,
                      line_dash="dot", row=2, col=1)

    # Panel 3: CUSUM
    t_post = t_full[NPRE:]
    fig_run.add_trace(go.Scatter(
        x=t_post, y=c_upper[NPRE:], mode="lines", name="CUSUM (upper)",
        line=dict(color="crimson", width=2),
    ), row=3, col=1)
    fig_run.add_trace(go.Scatter(
        x=t_post, y=c_lower[NPRE:], mode="lines", name="CUSUM (lower)",
        line=dict(color="steelblue", width=2, dash="dash"),
    ), row=3, col=1)
    fig_run.add_hline(y=ir_h, line_dash="dash", line_color="black", line_width=1.5,
                      annotation_text=f"h = {ir_h:.1f}",
                      annotation_position="top right",
                      row=3, col=1)

    # Vertical lines: true τ, detection time
    for row in [1, 2, 3]:
        if ir_delay > 0:
            fig_run.add_vline(x=NPRE, line_dash="dot", line_color="steelblue",
                              line_width=1.5,
                              annotation_text="Intervention" if row == 1 else "",
                              annotation_position="top right",
                              annotation_font=dict(color="steelblue", size=9),
                              row=row, col=1)
        ann_side = "top right" if ir_true_cpt < NTT * 0.7 else "top left"
        fig_run.add_vline(x=ir_true_cpt, line_dash="dash", line_color="#1A237E",
                          line_width=2,
                          annotation_text=f"True τ={ir_true_cpt}" if row == 1 else "",
                          annotation_position=ann_side,
                          annotation_font=dict(color="#1A237E", size=9),
                          row=row, col=1)
        if ir_detected:
            det_t = NPRE + int(ir_time_est)
            close = abs(det_t - ir_true_cpt) < 6
            det_side = ("top left" if ann_side == "top right" else "top right") if close else ann_side
            fig_run.add_vline(x=det_t, line_dash="solid", line_color="crimson",
                              line_width=2,
                              annotation_text=f"Detected={det_t}" if row == 1 else "",
                              annotation_position=det_side,
                              annotation_font=dict(color="crimson", size=9),
                              row=row, col=1)

    fig_run.update_layout(
        height=750,
        title=(
            f"Run {ir_run_i + 1} · {ir_noise} · {ir_eff_pct}% effect · "
            f"delay={ir_delay} mo · "
            f"{'✓ detected at month ' + str(NPRE + int(ir_time_est)) if ir_detected else '✗ not detected'}"
        ),
        xaxis3_title="Month",
        yaxis_title="Indicator",
        yaxis2_title="Residual",
        yaxis3_title="CUSUM",
        legend=dict(orientation="h", yanchor="bottom", y=-0.12, font=dict(size=10)),
        margin=dict(b=100),
    )
    st.plotly_chart(fig_run, width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# MINI-SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("🔬 Run your own simulation")
st.markdown("""
Run Forecast detection interactively in the browser. Both i.i.d. BA and AR(1) BA
models run in parallel so you can compare their CUSUM paths and detection outcomes.
""")

if "fcst_pending_restore" in st.session_state:
    _pr = st.session_state.pop("fcst_pending_restore")
    st.session_state["fcst_p_n_sim"]      = _pr["n_sim_mini"]
    st.session_state["fcst_p_base_seed"]  = _pr["base_seed_mini"]
    st.session_state["fcst_p_effect_pct"] = _pr["effect_pct_mini"]
    st.session_state["fcst_p_phi"]        = _pr["phi_mini"]
    st.session_state["fcst_p_npre"]       = _pr["npre_mini"]
    st.session_state["fcst_p_delay"]      = _pr["delay_mini"]
    st.session_state["fcst_p_npost"]      = _pr["npost_mini"]

for _k, _v in [
    ("fcst_p_n_sim", 20), ("fcst_p_base_seed", 42), ("fcst_p_effect_pct", 30),
    ("fcst_p_phi", PHI_DEFAULT), ("fcst_p_npre", NPRE), ("fcst_p_delay", 10), ("fcst_p_npost", 60),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

with st.expander("⚙️ Simulation parameters", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        n_sim_mini = st.slider("N simulations", 5, 100, step=5,
                               key="fcst_p_n_sim",
                               help="Number of independent realisations to run.")
        base_seed_mini = st.number_input("Random seed", 0, 99999, step=1,
                                         key="fcst_p_base_seed")
    with col_b:
        effect_pct_mini = st.select_slider(
            "Effect size (% of mean / 10 yr)", options=EFFECT_SIZES_PCT,
            key="fcst_p_effect_pct",
        )
        phi_mini = st.slider("φ (AR(1))", 0.0, 0.95, step=0.05,
                             key="fcst_p_phi",
                             help="AR(1) autocorrelation coefficient.")
    with col_c:
        npre_mini = st.slider("Pre-period (months)", 6, 60, step=6, key="fcst_p_npre")
        delay_mini = st.slider("Intervention delay (months)", 0, 20, key="fcst_p_delay")
        npost_mini = st.slider("Post-period (months)", 12, NPOST_MAX, step=6, key="fcst_p_npost")

_cur_params = {
    "n_sim_mini": n_sim_mini, "base_seed_mini": int(base_seed_mini),
    "effect_pct_mini": effect_pct_mini, "phi_mini": phi_mini,
    "npre_mini": npre_mini, "delay_mini": delay_mini, "npost_mini": npost_mini,
}
if "mini_fcst_last_params" in st.session_state:
    if st.session_state["mini_fcst_last_params"] != _cur_params:
        for _k in ("mini_fcst", "mini_fcst_idx", "mini_fcst_slider"):
            st.session_state.pop(_k, None)
        st.session_state["mini_fcst_hist_gen"] = st.session_state.get("mini_fcst_hist_gen", 0) + 1

run_mini_btn = st.button("▶ Run simulation", type="primary")

_fcst_history = st.session_state.get("mini_fcst_history", [])
if _fcst_history:
    _sel = st.selectbox(
        "Previous runs",
        options=range(len(_fcst_history)),
        format_func=lambda i: _fcst_history[i]["label"],
        index=None,
        placeholder="Select a previous run to restore…",
        key=f"mini_fcst_hist_sel_{st.session_state.get('mini_fcst_hist_gen', 0)}",
    )
    if _sel is not None:
        _h  = _fcst_history[_sel]
        _hp = _h["params"]
        st.session_state["fcst_pending_restore"]   = _hp
        st.session_state["mini_fcst"]              = _h["data"]
        st.session_state["mini_fcst_idx"]          = _h["nav_idx"]
        st.session_state["mini_fcst_slider"]       = _h["nav_idx"] + 1
        st.session_state["mini_fcst_last_params"]  = _hp
        st.session_state["mini_fcst_hist_gen"] = st.session_state.get("mini_fcst_hist_gen", 0) + 1

if run_mini_btn:
    trend_delta_mini  = LEVEL * (effect_pct_mini / 100) / NPOST_MONTHS
    trend_interv_mini = TREND_CONTROL + trend_delta_mini

    effective_npost = npost_mini - delay_mini
    if effective_npost < 4:
        st.error("Delay must be at least 4 months shorter than the post-period.")
        st.stop()

    true_cpt_mini = npre_mini + delay_mini
    ntt_mini      = npre_mini + npost_mini

    if results_available:
        h_iid_mini = thresholds["iid"]["threshold"]
        h_ar_mini  = thresholds["ar1"]["threshold"]
    else:
        # Approximate thresholds from a small calibration
        h_iid_mini = h_ar_mini = 5.0

    mini_iid_runs, mini_ar_runs = [], []
    prog = st.progress(0, text="Running…")

    for i in range(n_sim_mini):
        seed = int(base_seed_mini) + i

        sim_iid = ci_sim(seed=seed, npre=true_cpt_mini, npost=effective_npost,
                         level=LEVEL, trend=[TREND_CONTROL, trend_interv_mini], sigma=SIGMA)
        res_iid_mini = trend_stats_forecast(
            sim_iid["y_itv"], npre=npre_mini, ntt=ntt_mini, phi=None, h=h_iid_mini
        )
        mini_iid_runs.append((sim_iid, res_iid_mini, seed))

        sim_ar = ci_sim_ar(seed=seed, npre=true_cpt_mini, npost=effective_npost,
                           level=LEVEL, trend=[TREND_CONTROL, trend_interv_mini],
                           phi=phi_mini, sigma=SIGMA)
        res_ar_mini = trend_stats_forecast(
            sim_ar["y_itv"], npre=npre_mini, ntt=ntt_mini, phi=phi_mini, h=h_ar_mini
        )
        mini_ar_runs.append((sim_ar, res_ar_mini, seed))

        prog.progress((i + 1) / n_sim_mini, text=f"Simulation {i+1}/{n_sim_mini}…")

    prog.empty()

    first_det_idx = next(
        (j for j, (_, r, _) in enumerate(mini_iid_runs) if np.isfinite(r["time_est"])),
        0,
    )
    st.session_state["mini_fcst"] = {
        "iid_runs": mini_iid_runs, "ar_runs": mini_ar_runs,
        "n_sim": n_sim_mini, "true_cpt": true_cpt_mini,
        "npre": npre_mini, "npost": npost_mini, "ntt": ntt_mini,
        "delay": delay_mini, "effect_pct": effect_pct_mini,
        "phi": phi_mini, "h_iid": h_iid_mini, "h_ar": h_ar_mini,
        "base_seed": int(base_seed_mini),
    }
    st.session_state["mini_fcst_idx"] = first_det_idx
    st.session_state["mini_fcst_slider"] = first_det_idx + 1

    _iid_det  = float(np.mean([np.isfinite(r["time_est"]) for _, r, _ in mini_iid_runs]))
    _ar_det   = float(np.mean([np.isfinite(r["time_est"]) for _, r, _ in mini_ar_runs]))
    _run_n    = len(st.session_state.get("mini_fcst_history", [])) + 1
    _ts       = datetime.now().strftime("%H:%M")
    _hist_lbl = (
        f"#{_run_n} · {_ts} · N={n_sim_mini} seed={int(base_seed_mini)} eff={effect_pct_mini}% "
        f"φ={phi_mini:.2f} pre={npre_mini}mo delay={delay_mini}mo post={npost_mini}mo "
        f"· iid={_iid_det:.0%} ar={_ar_det:.0%}"
    )
    if "mini_fcst_history" not in st.session_state:
        st.session_state["mini_fcst_history"] = []
    st.session_state["mini_fcst_history"].insert(0, {
        "label":   _hist_lbl,
        "data":    st.session_state["mini_fcst"],
        "nav_idx": first_det_idx,
        "params":  _cur_params,
    })
    st.session_state["mini_fcst_last_params"] = _cur_params

if "mini_fcst" in st.session_state:
    mf = st.session_state["mini_fcst"]

    # Summary metrics
    iid_det_rate = np.mean([np.isfinite(r["time_est"]) for _, r, _ in mf["iid_runs"]])
    ar_det_rate  = np.mean([np.isfinite(r["time_est"]) for _, r, _ in mf["ar_runs"]])
    iid_times    = [r["time_est"] for _, r, _ in mf["iid_runs"] if np.isfinite(r["time_est"])]
    ar_times     = [r["time_est"] for _, r, _ in mf["ar_runs"]  if np.isfinite(r["time_est"])]

    st.success("Simulation complete!")
    ms1, ms2, ms3, ms4 = st.columns(4)
    with ms1:
        st.metric("i.i.d. detection rate", f"{iid_det_rate:.1%}",
                  help=f"{int(iid_det_rate*mf['n_sim'])} of {mf['n_sim']} runs")
    with ms2:
        st.metric("i.i.d. mean time", f"{np.mean(iid_times):.1f} mo" if iid_times else "—")
    with ms3:
        st.metric("AR(1) detection rate", f"{ar_det_rate:.1%}",
                  help=f"{int(ar_det_rate*mf['n_sim'])} of {mf['n_sim']} runs")
    with ms4:
        st.metric("AR(1) mean time", f"{np.mean(ar_times):.1f} mo" if ar_times else "—")

    # Navigator
    st.divider()
    n_mini = mf["n_sim"]

    def _mini_prev():
        n = max(0, st.session_state["mini_fcst_idx"] - 1)
        st.session_state["mini_fcst_idx"]    = n
        st.session_state["mini_fcst_slider"] = n + 1

    def _mini_next():
        n = min(n_mini - 1, st.session_state["mini_fcst_idx"] + 1)
        st.session_state["mini_fcst_idx"]    = n
        st.session_state["mini_fcst_slider"] = n + 1

    def _mini_slider():
        st.session_state["mini_fcst_idx"] = st.session_state["mini_fcst_slider"] - 1

    nc1, nc2, nc3 = st.columns([1, 10, 1])
    with nc1:
        st.button("◀", on_click=_mini_prev, key="mini_prev", width="stretch")
    with nc2:
        st.slider("Run", 1, n_mini, key="mini_fcst_slider", on_change=_mini_slider)
    with nc3:
        st.button("▶", on_click=_mini_next, key="mini_next", width="stretch")

    show_i = st.session_state.get("mini_fcst_idx", 0)
    sim_iid_show, res_iid_show, seed_show = mf["iid_runs"][show_i]
    sim_ar_show,  res_ar_show,  _        = mf["ar_runs"][show_i]

    mini_cols = st.columns(2)
    for col, (label, sim_show, res_show, phi_val, h_val) in zip(
        mini_cols,
        [
            ("i.i.d. BA", sim_iid_show, res_iid_show, None,          mf["h_iid"]),
            ("AR(1) BA",  sim_ar_show,  res_ar_show,  mf["phi"],     mf["h_ar"]),
        ]
    ):
        detected = np.isfinite(res_show["time_est"])
        det_text = f"✓ month {mf['npre'] + int(res_show['time_est'])}" if detected else "✗ not detected"
        with col:
            st.subheader(f"{label} — Run #{show_i+1} · seed {seed_show} · {det_text}")
            r_m, c_up_m, c_lo_m = compute_cusum_path(sim_show["y_itv"], mf["npre"],
                                                       mf["ntt"], phi=phi_val)
            t_m   = np.arange(1, mf["ntt"] + 1)
            t_post_m = t_m[mf["npre"]:]

            fig_m = make_subplots(
                rows=2, cols=1, row_heights=[0.55, 0.45],
                subplot_titles=["Time series + forecast", "Page-CUSUM path"],
                shared_xaxes=True, vertical_spacing=0.12,
            )
            for row in [1, 2]:
                fig_m.add_vrect(x0=1, x1=mf["npre"], fillcolor="rgba(100,149,237,0.07)",
                                line_width=0, row=row, col=1)

            fig_m.add_trace(go.Scatter(
                x=t_m, y=sim_show["y_itv"], mode="lines", name="Intervention",
                line=dict(color="rgba(76,175,80,0.8)", width=1.5), showlegend=True,
            ), row=1, col=1)
            X_m = np.column_stack([np.ones(mf["ntt"]), np.arange(1, mf["ntt"]+1)])
            ols_m = OLS(sim_show["y_itv"][:mf["npre"]], X_m[:mf["npre"]]).fit()
            fig_m.add_trace(go.Scatter(
                x=t_m[mf["npre"]:], y=X_m[mf["npre"]:] @ ols_m.params,
                mode="lines", name="Forecast",
                line=dict(color="#7B1FA2", width=2, dash="dot"),
            ), row=1, col=1)

            fig_m.add_trace(go.Scatter(
                x=t_post_m, y=c_up_m[mf["npre"]:], mode="lines", name="CUSUM upper",
                line=dict(color="crimson", width=2),
            ), row=2, col=1)
            fig_m.add_trace(go.Scatter(
                x=t_post_m, y=c_lo_m[mf["npre"]:], mode="lines", name="CUSUM lower",
                line=dict(color="steelblue", width=2, dash="dash"),
            ), row=2, col=1)
            fig_m.add_hline(y=h_val, line_dash="dash", line_color="black",
                            annotation_text=f"h={h_val:.1f}",
                            annotation_position="top right", row=2, col=1)

            for row in [1, 2]:
                if mf["delay"] > 0:
                    fig_m.add_vline(x=mf["npre"], line_dash="dot", line_color="steelblue",
                                    line_width=1.5, row=row, col=1)
                fig_m.add_vline(x=mf["true_cpt"], line_dash="dash",
                                line_color="#1A237E", line_width=2, row=row, col=1)
                if detected:
                    det_t = mf["npre"] + int(res_show["time_est"])
                    fig_m.add_vline(x=det_t, line_dash="solid",
                                    line_color="crimson", line_width=2, row=row, col=1)

            fig_m.update_layout(
                height=520,
                xaxis2_title="Month",
                yaxis_title="Indicator",
                yaxis2_title="CUSUM",
                legend=dict(orientation="h", yanchor="bottom", y=-0.18, font=dict(size=9)),
                margin=dict(b=80),
            )
            st.plotly_chart(fig_m, width="stretch")
