"""
Trend Change Detection using BOCPD
Interactive analysis page for the pre-computed trend_bocpd simulation results.

BOCPD (Bayesian Online Changepoint Detection) is an *online* detector run as a
single pass over the full 144-month difference series, stopping at the first
declared changepoint. Unlike AMOC/Forecast there is no monitoring-window
(npost) dimension: each replication produces exactly one outcome.
"""

import pickle
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.stats import linregress

from tracepy.simulation.trend import ci_sim

# Page config
st.set_page_config(
    page_title="Trend Change Detection (BOCPD)",
    page_icon="🔄",
    layout="wide",
)

# Simulation constants (mirror config/default_params.yaml)
NPRE = 24
NPOST_YEARS = 10
NPOST_MONTHS = 12 * NPOST_YEARS  # 120
NPOST_MAX = NPOST_MONTHS
NTT = NPRE + NPOST_MAX  # 144
LEVEL = 10.0
TREND_CONTROL = 0.005
SIGMA = 0.05
DELAY_MAX = 20
N_TRENDS = 11
MSL = 10  # min segment length (bocpd.msl)
PTR = 0.02  # run-length hazard (bocpd.ptr)

TREND_INCREASE = np.round(
    LEVEL * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / NPOST_MONTHS, 4
)
EFFECT_SIZES_PCT = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]  # % of mean / 10 yr

RESULTS_PATH = Path(__file__).parent.parent / "results" / "trend_bocpd" / "sim_results.pkl"

# Colour palette (one per effect size)
PALETTE = px.colors.sample_colorscale("Viridis", [i / 10 for i in range(11)])


# Data loading
@st.cache_data(show_spinner="Loading pre-computed simulation results…")
def load_results(path: Path, mtime: float):
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Regenerating simulation run…")
def _regenerate_iid(seed: int, delay: int, trend_inc: float):
    """Regenerate the (control, intervention) pair for a saved run.

    Pure-Python ci_sim — no R required.  The true changepoint in the difference
    series sits at ``NPRE + delay`` (1-indexed).
    """
    npre_delay = NPRE + delay
    npost_delay = NPOST_MAX - delay
    return ci_sim(
        seed=seed,
        npre=npre_delay,
        npost=npost_delay,
        level=LEVEL,
        trend=[TREND_CONTROL, TREND_CONTROL + trend_inc],
        sigma=SIGMA,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔄 Trend Change Detection (BOCPD)")

st.markdown("""
After a rewilding intervention, we want to know, as the data arrives, whether
an ecological indicator has changed trajectory.

**BOCPD (Bayesian Online Changepoint Detection)** processes the series **one
observation at a time**, maintaining a posterior over the **location of the most
recent changepoint** with a **particle filter**.
""")


# Load data
_mtime = RESULTS_PATH.stat().st_mtime if RESULTS_PATH.exists() else 0.0
data = load_results(RESULTS_PATH, _mtime)

results_available = False
simN = None
if data is None:
    st.error(
        f"Pre-computed results not found at `{RESULTS_PATH}`.\n\n"
        "Run `uv run trace-sim trend-bocpd` first (or `--quick` for a fast smoke run)."
    )
else:
    res = data.get("detection_results", {})
    if not res:
        st.warning(
            "Saved results are incomplete (no detection results found). "
            "Re-run `uv run trace-sim trend-bocpd` to regenerate."
        )
    else:
        trends = sorted(res.keys())
        # Map each saved increment to its nearest effect-size percentage so labels
        # stay correct even when only a subset of increments was saved (rather than
        # assuming the saved increments are a prefix of the full 11-value grid).
        pct_list = [EFFECT_SIZES_PCT[int(np.argmin(np.abs(TREND_INCREASE - t)))] for t in trends]
        simN = max(len(res[t]["delay"]) for t in trends)
        results_available = True


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

    _simN_label = f"{simN:,}" if simN else "1,000"
    st.markdown(f"""
    | Parameter | Value | Meaning |
    |-----------|-------|---------|
    | **npre** | {NPRE} mo | Pre-intervention monitoring |
    | **npost_max** | {NPOST_MAX} mo | Post-intervention window |
    | **level** | {LEVEL:g} | Baseline indicator value |
    | **trend_control** | {TREND_CONTROL}/mo | Pre-intervention slope |
    | **sigma** | {SIGMA} | Noise SD |
    | **msl** | {MSL} | Minimum segment length |
    | **ptr** | {PTR} | Run-length hazard rate |
    | **simN** | {_simN_label} | Replications per effect size |
    """)


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS EXPLORER  (only shown when data is available)
# ══════════════════════════════════════════════════════════════════════════════
if results_available:
    st.header("📊 Results explorer")
    st.markdown(
        f"Charts below are drawn from pre-computed simulations "
        f"(up to {simN:,} runs per effect size). Every chart is computed from the "
        "per-run lists — BOCPD has no monitoring-window axis."
    )

    # helpers shared across tabs
    def _detected_mask(trend):
        """Boolean mask of runs that fired (cpt_est is not None)."""
        return np.array([c is not None for c in res[trend]["cpt_est"]], dtype=bool)

    def _timing_errors(trend):
        """cpt_est − (NPRE + delay) for detected runs (positive = declared late)."""
        cpt = res[trend]["cpt_est"]
        delay = res[trend]["delay"]
        return np.array(
            [c - (NPRE + d) for c, d in zip(cpt, delay) if c is not None],
            dtype=float,
        )

    def _detection_lags(trend):
        """time_est − (NPRE + delay) for runs where the alarm fired."""
        t_est = res[trend]["time_est"]
        delay = res[trend]["delay"]
        return np.array(
            [t - (NPRE + d) for t, d in zip(t_est, delay) if t is not None],
            dtype=float,
        )

    def _abs_alarm_times(trend):
        return np.array([t for t in res[trend]["time_est"] if t is not None], dtype=float)

    tab_power, tab_ttd, tab_err = st.tabs(
        [
            "Sensitivity curve",
            "Time to detection",
            "Localisation error",
        ]
    )

    # Tab: Sensitivity (detection power) curve
    with tab_power:
        st.subheader("Sensitivity curve: detection rate vs effect size")
        st.markdown("""
        Each point is the fraction of replications in which
        BOCPD **ever fired** within the 10-year monitoring window, for a given effect
        size. Brighter/yellower markers are larger effects.

        The dashed line marks **80% power** and the dotted line **95% power**.
        """)

        rates, n_det, n_tot = [], [], []
        for t in trends:
            mask = _detected_mask(t)
            n_tot.append(int(mask.size))
            n_det.append(int(mask.sum()))
            rates.append(float(mask.mean()) if mask.size else np.nan)

        fig_power = go.Figure()
        fig_power.add_trace(
            go.Scatter(
                x=pct_list,
                y=rates,
                mode="lines+markers",
                line=dict(color="#1f9e89", width=2),
                marker=dict(
                    size=11,
                    color=PALETTE[: len(trends)],
                    line=dict(color="rgba(0,0,0,0.4)", width=1),
                ),
                customdata=np.stack([n_det, n_tot], axis=-1),
                hovertemplate=(
                    "Effect: %{x}%<br>Detection rate: %{y:.1%}<br>"
                    "n detected: %{customdata[0]} / %{customdata[1]}<extra></extra>"
                ),
                name="Detection rate",
            )
        )
        fig_power.add_hline(
            y=0.80,
            line_dash="dash",
            line_color="grey",
            annotation_text="80% power",
            annotation_position="right",
        )
        fig_power.add_hline(
            y=0.95,
            line_dash="dot",
            line_color="grey",
            annotation_text="95% power",
            annotation_position="right",
        )
        fig_power.update_layout(
            title="Detection rate vs effect size (BOCPD, full 10-year window)",
            xaxis=dict(title="Effect size (% of mean / 10 yr)", dtick=10),
            yaxis=dict(title="Detection rate", range=[0, 1.05], tickformat=".0%"),
            height=480,
        )
        st.plotly_chart(fig_power, width="stretch")
        st.caption(
            "Because BOCPD sees the whole window in a single online pass, this is the "
            "probability of the detector *ever* firing within 10 years — not a "
            "function of monitoring duration."
        )

    # Time to detection
    with tab_ttd:
        st.subheader("Time to detection (detected runs only)")
        st.markdown("""
        Among runs that fired, how long after the true onset does the alarm ring?
        """)

        use_lag = True

        x_labels = [f"{p}%" for p in pct_list]
        ns = []
        q1s, meds, q3s, p5s, p95s = [], [], [], [], []
        for t in trends:
            err = _timing_errors(t)
            n = len(err)
            ns.append(n)
            if n >= 5:
                q1s.append(float(np.percentile(err, 25)))
                meds.append(float(np.median(err)))
                q3s.append(float(np.percentile(err, 75)))
                p5s.append(float(np.percentile(err, 5)))
                p95s.append(float(np.percentile(err, 95)))
            else:
                q1s.append(None)
                meds.append(None)
                q3s.append(None)
                p5s.append(None)
                p95s.append(None)

        fig_err = go.Figure()
        fig_err.add_hrect(
            y0=-3,
            y1=3,
            fillcolor="rgba(0,180,0,0.07)",
            line_width=0,
            annotation_text="± 3 months",
            annotation_position="top right",
            annotation_font=dict(color="green", size=10),
        )
        fig_err.add_trace(
            go.Box(
                x=x_labels,
                q1=q1s,
                median=meds,
                q3=q3s,
                lowerfence=p5s,
                upperfence=p95s,
                marker_color="rgba(94,79,162,0.85)",
                line_color="rgba(94,79,162,0.85)",
                boxpoints=False,
                name="Timing error",
                hovertemplate=(
                    "<b>%{x} effect</b><br>"
                    "Median: %{median} mo<br>"
                    "IQR: %{q1} – %{q3} mo<br>"
                    "5th–95th pct: %{lowerfence} – %{upperfence} mo<extra></extra>"
                ),
            )
        )
        fig_err.add_hline(
            y=0,
            line_color="black",
            line_width=1.5,
            line_dash="dot",
            annotation_text="True changepoint",
            annotation_position="top left",
            annotation_font=dict(size=10),
        )
        fig_err.add_annotation(
            x=-0.07,
            y=0.75,
            xref="paper",
            yref="paper",
            text="▲ Late",
            showarrow=False,
            font=dict(size=10, color="crimson"),
            textangle=-90,
        )
        fig_err.add_annotation(
            x=-0.07,
            y=0.25,
            xref="paper",
            yref="paper",
            text="▼ Early",
            showarrow=False,
            font=dict(size=10, color="steelblue"),
            textangle=-90,
        )
        fig_err.update_layout(
            xaxis=dict(title="Effect size (% of mean / 10 yr)", tickvals=x_labels),
            yaxis=dict(title="Timing error: cpt_est − true (months)", zeroline=False),
            height=480,
            margin=dict(l=70),
            title=(
                "Changepoint timing error by effect size<br>"
                "<sup>Box = IQR · whiskers = 5th–95th percentile · detected runs only</sup>"
            ),
        )
        st.plotly_chart(fig_err, width="stretch")
        n_caption = " · ".join(f"{p}%: {n}" for p, n in zip(pct_list, ns))
        st.caption(
            f"**n detected per effect size** — {n_caption}. Boxes with fewer than "
            "5 detections are omitted."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # INDIVIDUAL RUN EXPLORER
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.header("🔍 Individual run explorer")
    st.markdown(f"""
    Browse any of the {simN:,} pre-computed BOCPD replications. The recorded
    outcome — whether it fired, the declared changepoint `cpt_est`, and the alarm
    step `time_est` — comes straight from the saved results. The series itself is
    **re-drawn in Python** with `ci_sim` for illustration (no R needed): it shares
    the same effect size and onset delay as the saved run, but because R and NumPy
    use different random streams it is **not the exact realisation BOCPD scored**.
    Treat it as a representative example of that parameter combination — the τ̂ and
    alarm markers need not line up with features of this particular draw.
    """)

    exp_effect_pct = st.select_slider(
        "Effect size",
        options=pct_list,
        value=pct_list[min(3, len(pct_list) - 1)],
        format_func=lambda x: f"{x}%",
        key="bocpd_exp_effect",
    )

    if "exp_nav_idx" not in st.session_state:
        st.session_state["exp_nav_idx"] = 0
    if "exp_nav_slider" not in st.session_state:
        st.session_state["exp_nav_slider"] = 1

    exp_trend = trends[pct_list.index(exp_effect_pct)]
    n_runs_exp = len(res[exp_trend]["delay"])

    # Clamp any carried-over navigation state to the current effect size's run
    # count before the slider widget is drawn (effect sizes can differ in length).
    st.session_state["exp_nav_idx"] = min(
        st.session_state.get("exp_nav_idx", 0), max(n_runs_exp - 1, 0)
    )
    st.session_state["exp_nav_slider"] = min(
        st.session_state.get("exp_nav_slider", 1), max(n_runs_exp, 1)
    )

    def _exp_prev():
        new = max(0, st.session_state["exp_nav_idx"] - 1)
        st.session_state["exp_nav_idx"] = new
        st.session_state["exp_nav_slider"] = new + 1

    def _exp_next():
        new = min(n_runs_exp - 1, st.session_state["exp_nav_idx"] + 1)
        st.session_state["exp_nav_idx"] = new
        st.session_state["exp_nav_slider"] = new + 1

    def _exp_on_slider():
        st.session_state["exp_nav_idx"] = st.session_state["exp_nav_slider"] - 1

    en1, en2, en3 = st.columns([1, 10, 1])
    with en1:
        st.button("◀", on_click=_exp_prev, key="exp_nav_prev", width="stretch")
    with en2:
        st.slider("Run", 1, max(n_runs_exp, 1), key="exp_nav_slider", on_change=_exp_on_slider)
    with en3:
        st.button("▶", on_click=_exp_next, key="exp_nav_next", width="stretch")

    run_i = st.session_state.get("exp_nav_idx", 0)
    if run_i >= n_runs_exp:
        run_i = n_runs_exp - 1
        st.session_state["exp_nav_idx"] = run_i
        st.session_state["exp_nav_slider"] = run_i + 1

    # Reconstruct the run
    # 1-based index into the *full* effect-size grid — this is the ``trend_idx``
    # (``m``) the CLI used in ``seed = s * n_trends + trend_idx``.
    trend_idx = EFFECT_SIZES_PCT.index(exp_effect_pct) + 1
    s = run_i + 1
    seed = s * N_TRENDS + trend_idx
    delay = int(res[exp_trend]["delay"][run_i])
    true_cpt = NPRE + delay
    cpt = res[exp_trend]["cpt_est"][run_i]
    t_alarm = res[exp_trend]["time_est"][run_i]
    detected = cpt is not None

    sim_data = _regenerate_iid(seed, delay, float(exp_trend))
    y_ctr = sim_data["y_ctr"]
    y_itv = sim_data["y_itv"]
    diff = y_itv - y_ctr
    nt = len(diff)
    t = np.arange(1, nt + 1)

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Detected?", "✓ yes" if detected else "✗ no")
    with m2:
        st.metric(
            "Detected τ̂",
            f"month {cpt}" if detected else "✗ not detected",
            help=f"True τ = month {true_cpt} (npre={NPRE} + delay={delay})",
        )
    with m3:
        if detected:
            _err = cpt - true_cpt
            _dir = "late" if _err > 0 else ("early" if _err < 0 else "exact")
            st.metric("Timing error", f"{_err:+d} mo ({_dir})")
        else:
            st.metric("Timing error", "—")
    with m4:
        if t_alarm is not None:
            _lag = t_alarm - true_cpt
            st.metric(
                "Alarm step",
                f"month {t_alarm}",
                delta=f"lag {_lag:+d} mo",
                delta_color="off",
            )
        else:
            st.metric("Alarm step", "—")

    # Figure
    t_pre = t[:true_cpt]
    t_post = t[true_cpt:]
    fit_pre = fit_post = None
    if len(t_pre) >= 2:
        sl, ic, *_ = linregress(t_pre, y_itv[:true_cpt])
        fit_pre = sl * t_pre + ic
    if len(t_post) >= 2:
        sl, ic, *_ = linregress(t_post, y_itv[true_cpt:])
        fit_post = sl * t_post + ic

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.62, 0.38],
        subplot_titles=[
            "Control vs Intervention — raw series + fitted slopes",
            "Difference (intervention − control) — the kind of series BOCPD runs on",
        ],
        shared_xaxes=True,
        vertical_spacing=0.10,
    )
    for row in [1, 2]:
        fig.add_vrect(
            x0=1, x1=NPRE, fillcolor="rgba(100,149,237,0.07)", line_width=0, row=row, col=1
        )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y_ctr,
            mode="lines",
            name="Control",
            line=dict(color="rgba(96,125,139,0.7)", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=y_itv,
            mode="lines",
            name="Intervention",
            line=dict(color="rgba(76,175,80,0.7)", width=1.5),
        ),
        row=1,
        col=1,
    )
    if fit_pre is not None:
        fig.add_trace(
            go.Scatter(
                x=t_pre,
                y=fit_pre,
                mode="lines",
                name="Fitted (pre)",
                line=dict(color="#1565C0", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )
    if fit_post is not None:
        fig.add_trace(
            go.Scatter(
                x=t_post,
                y=fit_post,
                mode="lines",
                name="Fitted (post)",
                line=dict(color="#2E7D32", width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=t,
            y=diff,
            mode="lines",
            name="Difference",
            line=dict(color="rgba(255,152,0,0.8)", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255,152,0,0.10)",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line_color="rgba(0,0,0,0.3)", line_width=1, line_dash="dot", row=2, col=1)

    true_ann_side = "top right" if true_cpt < nt * 0.75 else "top left"
    for row in [1, 2]:
        fig.add_vline(
            x=true_cpt,
            line_dash="solid",
            line_color="#1A237E",
            line_width=2,
            annotation_text=f"True τ={true_cpt}" if row == 1 else "",
            annotation_position=true_ann_side,
            annotation_font=dict(color="#1A237E", size=10),
            row=row,
            col=1,
        )
    if detected:
        close = abs(cpt - true_cpt) < 6
        det_ann_side = (
            ("top left" if true_ann_side == "top right" else "top right")
            if close
            else true_ann_side
        )
        for row in [1, 2]:
            fig.add_vline(
                x=cpt,
                line_dash="dash",
                line_color="crimson",
                line_width=2,
                annotation_text=f"τ̂={cpt}" if row == 1 else "",
                annotation_position=det_ann_side,
                annotation_font=dict(color="crimson", size=10),
                row=row,
                col=1,
            )
    if t_alarm is not None:
        for row in [1, 2]:
            fig.add_vline(
                x=t_alarm,
                line_dash="dot",
                line_color="rgba(255,87,34,0.6)",
                line_width=1.5,
                annotation_text="alarm" if row == 2 else "",
                annotation_position="bottom right",
                annotation_font=dict(color="rgba(255,87,34,0.9)", size=9),
                row=row,
                col=1,
            )

    fig.update_layout(
        height=580,
        title=(
            f"Run {run_i + 1} · {exp_effect_pct}% effect · delay={delay} mo · "
            f"seed={seed} · {'✓ detected' if detected else '✗ not detected'}"
        ),
        xaxis2_title="Month",
        yaxis_title="Indicator value",
        yaxis2_title="Difference",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, font=dict(size=11)),
        margin=dict(b=80),
    )
    st.plotly_chart(fig, width="stretch")

    st.info(
        "**Note:** the run-length posterior is not stored — `run_bocpd` returns only "
        "`cpt_est`/`time_est`, not the full per-step posterior matrix. This explorer "
        "therefore shows an illustrative series with the **declared vs true** changepoint "
        "(and the alarm step), not a run-length heatmap.",
        icon="ℹ️",
    )


# ══════════════════════════════════════════════════════════════════════════════
# (OPTIONAL) RUN YOUR OWN SIMULATION — the only part that needs R/rpy2
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.header("🔬 Run your own simulation")
st.markdown("""
Pick an effect size and onset delay, then run **live BOCPD** on a freshly simulated
difference series. This is the **only** part of the page that needs R + rpy2 — the
pre-computed explorer above needs neither.
""")

with st.expander("⚙️ Live BOCPD (requires R + rpy2)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        live_effect_pct = st.select_slider(
            "Effect size (% of mean / 10 yr)",
            options=EFFECT_SIZES_PCT,
            value=30,
            key="bocpd_live_effect",
        )
    with c2:
        live_delay = st.slider(
            "Onset delay (months)",
            min_value=1,
            max_value=DELAY_MAX,
            value=10,
            step=1,
            key="bocpd_live_delay",
        )
    with c3:
        live_seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=99999,
            value=42,
            step=1,
            key="bocpd_live_seed",
        )

    run_live = st.button("▶ Run live BOCPD", type="primary", width="content")


@st.cache_data(show_spinner="Running live BOCPD (R)…")
def _run_live_bocpd(seed: int, delay: int, trend_inc: float):
    """Run BOCPD once on a freshly simulated difference series. Needs R/rpy2."""
    from tracepy.changepoint.bocpd import run_bocpd

    npre_delay = NPRE + delay
    npost_delay = NPOST_MAX - delay
    sim = ci_sim(
        seed=seed,
        npre=npre_delay,
        npost=npost_delay,
        level=LEVEL,
        trend=[TREND_CONTROL, TREND_CONTROL + trend_inc],
        sigma=SIGMA,
    )
    x = sim["y_itv"] - sim["y_ctr"]
    out = run_bocpd(
        x,
        prior_alpha=(0.0, 0.05),
        prior_sigma2=0.0025,
        sig_prior=(100.0, 1.0),
        pm=1.0,
        ptr=PTR,
        maxp=50,
        np_=40,
        msl=MSL,
    )
    return {"y_ctr": sim["y_ctr"], "y_itv": sim["y_itv"], **out}


if run_live:
    trend_inc_live = float(np.round(LEVEL * (live_effect_pct / 100) / NPOST_MONTHS, 4))
    try:
        live = _run_live_bocpd(int(live_seed), int(live_delay), trend_inc_live)
    except ImportError:
        st.warning(
            "R + rpy2 required for live BOCPD; the pre-computed explorer above needs "
            "neither. Install R (≥ 4.x) and rpy2 to use this section.",
            icon="⚠️",
        )
    else:
        y_ctr = live["y_ctr"]
        y_itv = live["y_itv"]
        diff = y_itv - y_ctr
        nt = len(diff)
        t = np.arange(1, nt + 1)
        true_cpt = NPRE + live_delay
        cpt = live["cpt_est"]
        t_alarm = live["time_est"]
        detected = cpt is not None

        lc1, lc2, lc3 = st.columns(3)
        with lc1:
            st.metric("Detected?", "✓ yes" if detected else "✗ no")
        with lc2:
            st.metric(
                "Detected τ̂",
                f"month {cpt}" if detected else "✗ not detected",
                help=f"True τ = month {true_cpt}",
            )
        with lc3:
            if detected:
                _err = cpt - true_cpt
                _dir = "late" if _err > 0 else ("early" if _err < 0 else "exact")
                st.metric("Timing error", f"{_err:+d} mo ({_dir})")
            else:
                st.metric("Timing error", "—")

        fig_live = go.Figure()
        fig_live.add_vrect(x0=1, x1=NPRE, fillcolor="rgba(100,149,237,0.07)", line_width=0)
        fig_live.add_trace(
            go.Scatter(
                x=t,
                y=diff,
                mode="lines",
                name="Difference",
                line=dict(color="rgba(255,152,0,0.85)", width=1.5),
                fill="tozeroy",
                fillcolor="rgba(255,152,0,0.10)",
            )
        )
        fig_live.add_hline(y=0, line_color="rgba(0,0,0,0.3)", line_width=1, line_dash="dot")
        fig_live.add_vline(
            x=true_cpt,
            line_dash="solid",
            line_color="#1A237E",
            line_width=2,
            annotation_text=f"True τ={true_cpt}",
            annotation_position="top right",
        )
        if detected:
            fig_live.add_vline(
                x=cpt,
                line_dash="dash",
                line_color="crimson",
                line_width=2,
                annotation_text=f"τ̂={cpt}",
                annotation_position="top left",
            )
        if t_alarm is not None:
            fig_live.add_vline(
                x=t_alarm,
                line_dash="dot",
                line_color="rgba(255,87,34,0.6)",
                line_width=1.5,
                annotation_text="alarm",
                annotation_position="bottom right",
            )
        fig_live.update_layout(
            title=(
                f"Live BOCPD · {live_effect_pct}% effect · delay={live_delay} mo · "
                f"seed={int(live_seed)} · {'✓ detected' if detected else '✗ not detected'}"
            ),
            xaxis_title="Month",
            yaxis_title="Difference (intervention − control)",
            height=460,
        )
        st.plotly_chart(fig_live, width="stretch")
