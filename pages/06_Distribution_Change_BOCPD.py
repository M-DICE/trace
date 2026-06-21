"""
Distribution Change Detection using BOCPD
Interactive analysis page for distribution_bocpd results.

BOCPD (Bayesian Online Changepoint Detection) processes the Wasserstein distance
series one month at a time and stops at the first declared changepoint. There is
no monitoring-window (npost) dimension — every replication runs once over the
full window, so all charts derive from per-run outcome lists.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

from tracepy.plotting.styles import CLR_TAU_DET, CLR_TAU_TRUE
from tracepy.simulation.distribution import ci_sim_cdf
from tracepy.stats.metrics import wasserstein_distance_baci
from webapp.components import render_run_navigator
from webapp.constants import EFFECT_SIZES_PCT, NPOST_MONTHS, NPRE, PALETTE
from webapp.figures import add_power_thresholds, add_pre_shading, add_timing_band
from webapp.loaders import load_results

warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Distribution Change Detection (BOCPD)",
    page_icon="🔄📊",
    layout="wide",
)

# Simulation constants (mirror config/default_params.yaml → distribution)
NPOST_MAX = NPOST_MONTHS
MU = 10.0
SIGMA = 1.0
NS = 200
DELAY_MAX = 20
N_TRENDS = 11
MSL = 10
PTR = 0.02

# NB: the /120 denominator is literal in the CLI, not /NPOST_MONTHS.
TREND_INCREASE_MU = np.round(MU * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / 120, 4)

RESULTS_PATH = Path(__file__).parent.parent / "results" / "distribution_bocpd" / "sim_results.pkl"


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


# Per-run derived quantities (tolerant of partial / --quick dicts)
def _run_arrays(entry):
    """Return aligned numpy arrays for one effect-size entry.

    Computes true changepoint, detection mask, timing error (cpt − true) and
    detection lag (time_est − true) purely from the saved per-run lists, so it
    works for partial or --quick (simN=10) results without assuming 1,000 runs.
    """
    cpt_est = np.array([np.nan if c is None else c for c in entry["cpt_est"]], dtype=float)
    time_est = np.array([np.nan if t is None else t for t in entry["time_est"]], dtype=float)
    delays = np.array(entry["delay"], dtype=float)
    true_cpt = NPRE + delays
    detected = np.isfinite(cpt_est)
    timing_err = cpt_est - true_cpt  # nan where not detected
    det_lag = time_est - true_cpt  # nan where not detected
    return {
        "cpt_est": cpt_est,
        "time_est": time_est,
        "delays": delays,
        "true_cpt": true_cpt,
        "detected": detected,
        "timing_err": timing_err,
        "det_lag": det_lag,
        "n": len(delays),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔄📊 Distribution Change Detection (BOCPD)")

st.markdown("""
Rewilding interventions can shift the **shape of a species' distribution**, not just its
average. Here the intervention shifts the distribution's **mean** after the changepoint.

Each month is summarised by two 200-sample distributions (control vs intervention). We
collapse them to a single number — their **Wasserstein-1 distance** (earth-mover's distance,
in the same units as the indicator) — giving a univariate distance series. Before the
changepoint the two distributions coincide, so the distance is small and noise-driven; after
it, the intervention mean drifts away and the distance rises.

**BOCPD (Bayesian Online Changepoint Detection)** watches this distance series one observation
at a time, maintaining a posterior over the *location of the most recent changepoint* via a
particle filter (a resampled set of candidate changepoint histories). At each step it takes the
maximum a posteriori (MAP) estimate of that location and declares a changepoint the moment the
first τ>0 break enters the MAP path. Unlike AMOC and Forecast it needs no null distribution or
critical value — there is a **single** simulation phase. Each replication runs once over the full
144-month window and stops at the first alarm.
""")

# Load data
_mtime = RESULTS_PATH.stat().st_mtime if RESULTS_PATH.exists() else 0.0
with st.spinner("Loading distribution BOCPD results…"):
    data = load_results(RESULTS_PATH, _mtime)

simN = None
if data is None:
    st.error(
        f"Pre-computed results not found at `{RESULTS_PATH}`. "
        "Run `uv run trace-sim distribution-bocpd` first (or `--quick` for a fast smoke run), "
        "or use the **Run your own simulation** section below."
    )
    results_available = False
else:
    res = data.get("detection_results", {})
    if not res:
        st.warning(
            "Saved results are incomplete (`detection_results` missing or empty). "
            "Re-run `uv run trace-sim distribution-bocpd` to regenerate."
        )
        results_available = False
    else:
        trends = sorted(res.keys())
        # Map sorted increments positionally to effect-size percentages. Tolerate
        # partial dicts (fewer than 11 increments saved so far).
        pct_to_trend = {}
        for t in trends:
            try:
                _idx = int(np.argmin(np.abs(TREND_INCREASE_MU - t)))
                pct_to_trend[EFFECT_SIZES_PCT[_idx]] = t
            except (IndexError, ValueError):
                continue
        avail_pcts = sorted(pct_to_trend.keys(), key=lambda p: EFFECT_SIZES_PCT.index(p))
        run_arrays = {t: _run_arrays(res[t]) for t in trends}
        simN = max((a["n"] for a in run_arrays.values()), default=None)
        results_available = True
        if len(trends) < N_TRENDS:
            st.info(
                f"Partial results: {len(trends)} of {N_TRENDS} effect sizes saved so far. "
                "Charts show what is available."
            )


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
    _simN_label = f"{simN:,}" if simN else "1,000"
    st.markdown(f"""
    | Parameter | Value | Meaning |
    |-----------|-------|---------|
    | **npre** | {NPRE} months | Pre-intervention period |
    | **npost_max** | {NPOST_MAX} months | Post-intervention window |
    | **mu** | {MU} | Baseline distribution mean |
    | **sigma** | {SIGMA} | Baseline standard deviation |
    | **ns** | {NS} | Samples per month |
    | **dist_measure** | Wasserstein | Distance metric |
    | **msl** | {MSL} | Minimum segment length |
    | **ptr** | {PTR} | Run-length hazard rate |
    | **simN** | {_simN_label} | Replications per effect size |
    """)

# ══════════════════════════════════════════════════════════════════════════════
# RESULTS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
if results_available:
    st.header("📊 Results explorer")
    st.markdown(
        f"Charts below are drawn from the pre-computed BOCPD simulations "
        f"(up to {simN:,} runs per effect size). Every chart derives from per-run "
        "outcomes — BOCPD has no monitoring-window axis."
    )

    tab_sens, tab_ttd, tab_loc = st.tabs(
        [
            "Sensitivity curve",
            "Time to detection",
            "Localisation error",
        ]
    )

    # Sensitivity (detection power) curve
    with tab_sens:
        st.subheader("Sensitivity (detection power) curve")
        st.markdown("""
        The headline result: **probability that BOCPD ever fires within the 10-year window**
        as the mean-shift effect size grows. Because BOCPD sees the whole window, this is the
        probability of *eventually* declaring a changepoint, not a power-vs-window curve.
        Dashed/dotted lines mark the 80% and 95% thresholds.
        """)

        _x_pct, _y_rate, _n_det, _n_tot = [], [], [], []
        for pct in avail_pcts:
            t = pct_to_trend[pct]
            a = run_arrays[t]
            n_det = int(a["detected"].sum())
            _x_pct.append(pct)
            _y_rate.append(n_det / a["n"] if a["n"] else 0.0)
            _n_det.append(n_det)
            _n_tot.append(a["n"])

        fig_sens = go.Figure()
        fig_sens.add_trace(
            go.Scatter(
                x=_x_pct,
                y=_y_rate,
                mode="lines+markers",
                line=dict(color=PALETTE[5], width=2.5),
                marker=dict(size=8, color=[PALETTE[EFFECT_SIZES_PCT.index(p)] for p in _x_pct]),
                customdata=np.column_stack([_n_det, _n_tot]),
                hovertemplate=(
                    "Effect: %{x}%<br>Detection rate: %{y:.1%}"
                    "<br>%{customdata[0]} of %{customdata[1]} detected<extra></extra>"
                ),
                name="Detection rate",
            )
        )
        add_power_thresholds(fig_sens)
        fig_sens.update_layout(
            title="Detection rate vs mean-shift effect size",
            xaxis_title="Effect size (% of mean shift over 10 yr)",
            yaxis=dict(title="Detection rate (ever fired)", range=[0, 1.05], tickformat=".0%"),
            height=480,
        )
        st.plotly_chart(fig_sens, width="stretch")
        st.caption(
            "Because BOCPD processes the full window online, this is the probability of "
            "*ever* firing within 10 years — not a function of monitoring-window length."
        )

    # Time to detection
    with tab_ttd:
        st.subheader("Time to detection")
        st.markdown("""
        Among **detected** runs, how long after the true onset does the alarm fire?
        Boxes show the IQR (whiskers = 5th/95th percentile).
        **Detection lag** — `time_est − (npre + delay)`, months between the true onset
        and the alarm.
        """)

        fig_ttd = go.Figure()
        _n_caption = []
        for pct in avail_pcts:
            t = pct_to_trend[pct]
            a = run_arrays[t]
            det = a["detected"]
            vals = a["det_lag"][det]
            vals = vals[np.isfinite(vals)]
            n = len(vals)
            _n_caption.append(n)
            # Skip degenerate boxes: percentiles on a handful of detections collapse
            # to a flat line that misleadingly reads as real spread.
            if n >= 5:
                fig_ttd.add_trace(
                    go.Box(
                        x=[f"{pct}%"] * n,
                        q1=[float(np.percentile(vals, 25))],
                        median=[float(np.median(vals))],
                        q3=[float(np.percentile(vals, 75))],
                        lowerfence=[float(np.percentile(vals, 5))],
                        upperfence=[float(np.percentile(vals, 95))],
                        name=f"{pct}%",
                        marker_color=PALETTE[EFFECT_SIZES_PCT.index(pct)],
                        line_color=PALETTE[EFFECT_SIZES_PCT.index(pct)],
                        boxpoints=False,
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{pct}% (n={n})</b><br>"
                            "Median: %{median:.1f} mo<br>IQR: %{q1:.1f}–%{q3:.1f} mo<extra></extra>"
                        ),
                    )
                )
        fig_ttd.update_layout(
            title="Time to detection — detected runs only",
            xaxis_title="Effect size (% of mean shift over 10 yr)",
            yaxis_title="Detection lag (months after true onset)",
            height=460,
        )
        st.plotly_chart(fig_ttd, width="stretch")
        _n_tbl = pd.DataFrame(
            {"Effect size": [f"{p}%" for p in avail_pcts], "Detected (n)": _n_caption}
        )
        st.caption(
            "Detected runs per effect size used to build each box. Effect sizes with "
            "fewer than 5 detections are omitted (their boxes would be degenerate)."
        )
        st.dataframe(_n_tbl, hide_index=True, width="stretch")

    # Changepoint localisation error
    with tab_loc:
        st.subheader("Changepoint localisation error")
        st.markdown("""
        Signed timing error of the declared changepoint: `cpt_est − (npre + delay)`.
        Positive = declared **late** ▲, negative = **early** ▼. Only detected runs count.
        """)

        loc_min_n = st.slider("Minimum detections to show box", 2, 50, 5, key="bd_loc_min_n")
        fig_loc = go.Figure()
        add_timing_band(fig_loc)
        _loc_n = []
        for pct in avail_pcts:
            t = pct_to_trend[pct]
            a = run_arrays[t]
            errs = a["timing_err"][a["detected"]]
            errs = errs[np.isfinite(errs)]
            n = len(errs)
            _loc_n.append(n)
            if n >= loc_min_n:
                fig_loc.add_trace(
                    go.Box(
                        x=[f"{pct}%"] * n,
                        q1=[float(np.percentile(errs, 25))],
                        median=[float(np.median(errs))],
                        q3=[float(np.percentile(errs, 75))],
                        lowerfence=[float(np.percentile(errs, 5))],
                        upperfence=[float(np.percentile(errs, 95))],
                        name=f"{pct}%",
                        marker_color=PALETTE[EFFECT_SIZES_PCT.index(pct)],
                        line_color=PALETTE[EFFECT_SIZES_PCT.index(pct)],
                        boxpoints=False,
                        showlegend=False,
                        hovertemplate=(
                            f"<b>{pct}% (n={n})</b><br>"
                            "Median: %{median:.1f} mo<br>"
                            "IQR: %{q1:.1f}–%{q3:.1f} mo<extra></extra>"
                        ),
                    )
                )
        fig_loc.add_hline(
            y=0,
            line_color="black",
            line_width=1.5,
            line_dash="dot",
            annotation_text="True changepoint",
            annotation_position="top left",
            annotation_font=dict(size=10),
        )
        fig_loc.update_layout(
            title="Changepoint timing error — detected runs only",
            xaxis_title="Effect size (% of mean shift over 10 yr)",
            yaxis_title="Timing error (months)  ▲ late / ▼ early",
            height=470,
        )
        st.plotly_chart(fig_loc, width="stretch")
        st.caption("Detections per effect size used to build each box:")
        st.dataframe(
            pd.DataFrame({"Effect size": [f"{p}%" for p in avail_pcts], "Detected (n)": _loc_n}),
            hide_index=True,
            width="stretch",
        )


# ══════════════════════════════════════════════════════════════════════════════
# INDIVIDUAL RUN EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
if results_available:
    st.divider()
    st.header("🔍 Individual run explorer")
    st.markdown("""
    Browse individual pre-computed runs. The recorded outcome — the declared changepoint
    `cpt_est` and alarm step `time_est` — comes straight from the saved results. The distance
    series itself is **re-drawn in Python** (no R) with the saved effect size and onset delay;
    because R and NumPy draw from different random streams it is **not the exact realisation
    BOCPD scored**, but a representative example of that parameter combination. The τ̂ / alarm
    markers therefore need not align with features of this particular draw.

    *Caveat:* `run_bocpd` returns only the declared changepoint and alarm step, **not** the full
    run-length posterior — so this view shows the distance series with the declared vs true
    changepoint, not a run-length heatmap.
    """)

    ir_eff_pct = st.select_slider(
        "Effect size",
        options=avail_pcts,
        value=avail_pcts[min(3, len(avail_pcts) - 1)],
        format_func=lambda x: f"{x}%",
        key="bd_ir_eff",
    )
    ir_trend = pct_to_trend[ir_eff_pct]
    ir_entry = res[ir_trend]
    n_runs_ir = len(ir_entry["delay"])

    # Run navigator — slider + ◀ ▶ buttons
    ir_run_i = render_run_navigator(
        n_runs_ir, "bd_ir_nav_idx", "bd_ir_nav_slider", "bd_ir_prev", "bd_ir_next"
    )

    ir_delay = int(ir_entry["delay"][ir_run_i])
    ir_trend_idx = EFFECT_SIZES_PCT.index(ir_eff_pct) + 1
    ir_seed = (ir_run_i + 1) * N_TRENDS + ir_trend_idx
    ir_true_cpt = NPRE + ir_delay
    _cpt_raw = ir_entry["cpt_est"][ir_run_i]
    _time_raw = ir_entry["time_est"][ir_run_i]
    ir_detected = _cpt_raw is not None
    ir_cpt = int(_cpt_raw) if _cpt_raw is not None else None
    ir_t_alarm = int(_time_raw) if _time_raw is not None else None

    ir_sim = _regenerate_dist_sim(ir_seed, ir_delay, float(ir_trend), NS)
    ir_dist_ts = wasserstein_distance_baci(ir_sim["sample_ctr"], ir_sim["sample_itv"])
    nt = len(ir_dist_ts)
    t_ax = np.arange(1, nt + 1)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Detected", "✓ yes" if ir_detected else "✗ no")
    with m2:
        st.metric("Detected τ̂", f"month {ir_cpt}" if ir_detected else "✗ not detected")
    with m3:
        if ir_detected:
            _err = ir_cpt - ir_true_cpt
            _dir = "late" if _err > 0 else ("early" if _err < 0 else "exact")
            st.metric(
                "Timing error",
                f"{_err:+d} mo ({_dir})",
                help=f"True τ = month {ir_true_cpt} (pre={NPRE} + delay={ir_delay})",
            )
        else:
            st.metric(
                "Timing error",
                "—",
                help=f"True τ = month {ir_true_cpt} (pre={NPRE} + delay={ir_delay})",
            )
    with m4:
        if ir_detected and ir_t_alarm is not None:
            _lag = ir_t_alarm - ir_true_cpt
            st.metric(
                "Alarm step",
                f"month {ir_t_alarm}",
                delta=f"lag {_lag:+d} mo",
                delta_color="off",
                help="Step at which BOCPD first declared the changepoint.",
            )
        else:
            st.metric("Alarm step", "—")

    # 2-panel figure: distance series + PDF snapshots
    _ann_side = "top right" if ir_true_cpt < nt * 0.75 else "top left"

    # Pick snapshot months: one pre, one near onset, two post.
    _snap_months = sorted(
        {
            m
            for m in [
                max(1, NPRE // 2),
                ir_true_cpt,
                min(nt, ir_true_cpt + 24),
                nt,
            ]
            if 1 <= m <= nt
        }
    )
    _snap_colors = px.colors.sample_colorscale(
        "Viridis", np.linspace(0, 1, max(len(_snap_months), 2))
    )

    fig_ir = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.55, 0.45],
        subplot_titles=[
            "Wasserstein distance series (intervention vs control)",
            "Distribution snapshots (dashed = control, solid = intervention)",
        ],
        vertical_spacing=0.12,
    )

    # Top: distance series
    add_pre_shading(fig_ir, NPRE, row=1, col=1)
    fig_ir.add_trace(
        go.Scatter(
            x=t_ax,
            y=ir_dist_ts,
            mode="lines",
            name="Wasserstein distance",
            line=dict(color="black", width=1.5),
            hovertemplate="Month: %{x}<br>Distance: %{y:.4f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig_ir.add_vline(
        x=ir_true_cpt,
        line_dash="solid",
        line_color=CLR_TAU_TRUE,
        line_width=2,
        annotation_text=f"True τ={ir_true_cpt}",
        annotation_position=_ann_side,
        annotation_font=dict(color=CLR_TAU_TRUE, size=10),
        row=1,
        col=1,
    )
    if ir_detected:
        _close = abs(ir_cpt - ir_true_cpt) < 6
        _det_side = (
            ("top left" if _ann_side == "top right" else "top right") if _close else _ann_side
        )
        fig_ir.add_vline(
            x=ir_cpt,
            line_dash="dash",
            line_color=CLR_TAU_DET,
            line_width=2,
            annotation_text=f"Detected={ir_cpt}",
            annotation_position=_det_side,
            annotation_font=dict(color=CLR_TAU_DET, size=10),
            row=1,
            col=1,
        )
    if ir_detected and ir_t_alarm is not None and ir_t_alarm <= nt:
        fig_ir.add_trace(
            go.Scatter(
                x=[ir_t_alarm],
                y=[ir_dist_ts[ir_t_alarm - 1]],
                mode="markers",
                name="Alarm raised",
                marker=dict(color="crimson", size=11, symbol="x"),
                hovertemplate=f"Alarm at month {ir_t_alarm}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Bottom: PDF snapshots
    for color, tp in zip(_snap_colors, _snap_months):
        idx = tp - 1
        d_ctr = ir_sim["sample_ctr"][:, idx]
        d_itv = ir_sim["sample_itv"][:, idx]
        vmin = min(d_ctr.min(), d_itv.min()) - 1
        vmax = max(d_ctr.max(), d_itv.max()) + 1
        xr = np.linspace(vmin, vmax, 200)
        fig_ir.add_trace(
            go.Scatter(
                x=xr,
                y=gaussian_kde(d_ctr).evaluate(xr),
                mode="lines",
                line=dict(color=color, dash="dash", width=1.5),
                opacity=0.7,
                name=f"control t={tp}",
                legendgroup=f"t{tp}",
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=1,
        )
        fig_ir.add_trace(
            go.Scatter(
                x=xr,
                y=gaussian_kde(d_itv).evaluate(xr),
                mode="lines",
                line=dict(color=color, width=2),
                name=f"t={tp}",
                legendgroup=f"t{tp}",
                hovertemplate=f"t={tp}<br>Value: %{{x:.2f}}<br>Density: %{{y:.3f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig_ir.update_layout(
        title=f"Run {ir_run_i + 1} · {ir_eff_pct}% effect · delay={ir_delay} mo · seed={ir_seed}",
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, font=dict(size=10)),
        margin=dict(b=70),
    )
    fig_ir.update_xaxes(title_text="Month", row=1, col=1)
    fig_ir.update_yaxes(title_text="Wasserstein distance", row=1, col=1)
    fig_ir.update_xaxes(title_text="Value", row=2, col=1)
    fig_ir.update_yaxes(title_text="Density", row=2, col=1)
    st.plotly_chart(fig_ir, width="stretch")
    st.caption(
        "Top: an illustrative distance series for this effect size and delay (solid navy = "
        "true onset, dashed crimson = declared changepoint, ✗ = alarm step from the saved "
        "run). Bottom: control vs intervention densities at selected months — they coincide "
        "before the onset and diverge after."
    )

# ══════════════════════════════════════════════════════════════════════════════
# RUN YOUR OWN SIMULATION  (live BOCPD — only part that needs R/rpy2)
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("🔬 Run your own simulation")
st.markdown("""
Simulate a fresh distance series and run **live BOCPD** on it. This is the only part of the
page that needs **R + rpy2**; the results explorer and individual-run explorer above are pure
Python.
""")

_BOCPD_KWARGS = dict(ptr=PTR, msl=MSL)


@st.cache_data(show_spinner="Running live BOCPD…")
def _live_bocpd(seed: int, delay: int, trend_mu: float, ns: int, npre: int, npost: int):
    """Regenerate a distance series and run BOCPD once. Needs R/rpy2."""
    from tracepy.changepoint.bocpd import run_bocpd

    sim = ci_sim_cdf(
        seed=seed,
        npre=npre,
        npost=npost,
        level=[MU, SIGMA],
        trend=[trend_mu, 0],
        ns=ns,
    )
    dist_ts = wasserstein_distance_baci(sim["sample_ctr"], sim["sample_itv"])
    out = run_bocpd(dist_ts, **_BOCPD_KWARGS)
    return {
        "dist_ts": dist_ts,
        "sample_ctr": sim["sample_ctr"],
        "sample_itv": sim["sample_itv"],
        "cpt_est": out["cpt_est"],
        "time_est": out["time_est"],
    }


with st.expander("⚙️ Simulation parameters", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        live_seed = st.number_input(
            "Random seed", min_value=0, max_value=99999, value=42, step=1, key="bd_live_seed"
        )
        live_ns = st.slider("Samples per month (ns)", 50, 500, 200, step=50, key="bd_live_ns")
    with col_b:
        live_npre = st.slider("Pre-intervention (months)", 12, 48, NPRE, key="bd_live_npre")
        live_delay = st.slider("Onset delay (months)", 1, DELAY_MAX, 6, key="bd_live_delay")
    with col_c:
        live_effect = st.select_slider(
            "Effect size",
            options=EFFECT_SIZES_PCT,
            value=30,
            format_func=lambda x: f"{x}%",
            key="bd_live_effect",
        )

live_trend = float(TREND_INCREASE_MU[EFFECT_SIZES_PCT.index(live_effect)])
live_npost = NPOST_MAX - live_delay
live_true_cpt = live_npre + live_delay

if st.button("▶ Run simulation", type="primary", key="bd_live_run"):
    try:
        st.session_state["bd_live_result"] = _live_bocpd(
            int(live_seed),
            int(live_delay),
            live_trend,
            int(live_ns),
            int(live_npre),
            int(live_npost),
        )
        st.session_state["bd_live_meta"] = {
            "true_cpt": live_true_cpt,
            "npre": int(live_npre),
            "delay": int(live_delay),
            "effect": live_effect,
        }
    except ImportError:
        st.error(
            "R + rpy2 are required for live BOCPD. The pre-computed explorer above needs neither."
        )
    except Exception as exc:  # noqa: BLE001 — surface R/rpy2 runtime issues gracefully
        st.error(f"Live BOCPD failed: {exc}")

if "bd_live_result" in st.session_state and "bd_live_meta" in st.session_state:
    _r = st.session_state["bd_live_result"]
    _meta = st.session_state["bd_live_meta"]
    _dist = _r["dist_ts"]
    _nt = len(_dist)
    _t = np.arange(1, _nt + 1)
    _true = _meta["true_cpt"]
    _cpt = _r["cpt_est"]
    _talarm = _r["time_est"]
    _det = _cpt is not None

    st.success("BOCPD complete!")
    lm1, lm2, lm3 = st.columns(3)
    with lm1:
        st.metric("Detected τ̂", f"month {int(_cpt)}" if _det else "✗ not detected")
    with lm2:
        if _det:
            _e = int(_cpt) - _true
            _d = "late" if _e > 0 else ("early" if _e < 0 else "exact")
            st.metric("Timing error", f"{_e:+d} mo ({_d})", help=f"True τ = month {_true}")
        else:
            st.metric("Timing error", "—", help=f"True τ = month {_true}")
    with lm3:
        st.metric("Alarm step", f"month {int(_talarm)}" if _talarm is not None else "—")

    fig_live = go.Figure()
    add_pre_shading(fig_live, _meta["npre"])
    fig_live.add_trace(
        go.Scatter(
            x=_t,
            y=_dist,
            mode="lines",
            name="Wasserstein distance",
            line=dict(color="black", width=1.5),
            hovertemplate="Month: %{x}<br>Distance: %{y:.4f}<extra></extra>",
        )
    )
    fig_live.add_vline(
        x=_true,
        line_dash="solid",
        line_color=CLR_TAU_TRUE,
        line_width=2,
        annotation_text=f"True τ={_true}",
        annotation_position="top right",
        annotation_font=dict(color=CLR_TAU_TRUE, size=10),
    )
    if _det:
        fig_live.add_vline(
            x=int(_cpt),
            line_dash="dash",
            line_color=CLR_TAU_DET,
            line_width=2,
            annotation_text=f"Detected={int(_cpt)}",
            annotation_position="top left",
            annotation_font=dict(color=CLR_TAU_DET, size=10),
        )
    fig_live.update_layout(
        title=(
            f"Live BOCPD · {_meta['effect']}% effect · delay={_meta['delay']} mo · "
            f"{'✓ detected' if _det else '✗ not detected'}"
        ),
        xaxis_title="Month",
        yaxis_title="Wasserstein distance",
        height=420,
    )
    st.plotly_chart(fig_live, width="stretch")
