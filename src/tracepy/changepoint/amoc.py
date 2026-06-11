"""
AMOC (At Most One Change) orchestration for trend and distribution changepoint detection.

Contains:
- Critical value table utilities (load_crit_val_table, lookup_crit_val)
- Trend change AMOC orchestration (calculate_critical_values, run_main_simulation*)
- Distribution change AMOC orchestration
  (calculate_critical_values_cdf, run_main_simulation_mu/sigma)

Simulation workers and the shared Monte Carlo loop live in
tracepy.simulation.runners; this module is responsible only for orchestration
(parameter wiring, aggregation, and result packaging).
"""

import json
from pathlib import Path

import numpy as np

from tracepy.params.manager import load_params
from tracepy.simulation.runners import (
    _main_sim_worker,
    _main_sim_worker_ar,
    _main_sim_worker_cdf,
    _null_sim_worker,
    _null_sim_worker_cdf,
    _run_parallel,
    _simulation_loop,
)


def _find_project_root():
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not find project root")


_CRIT_VAL_TABLE_PATH = _find_project_root() / "data" / "CritValTable.json"


def load_crit_val_table(path=_CRIT_VAL_TABLE_PATH):
    """Load the critical value lookup table from *path* (default: data/CritValTable.json)."""
    with open(path) as f:
        return json.load(f)


def lookup_crit_val(table, detector="PageCUSUM", gamma=0.0, alpha=0.05):
    """
    Return the critical value for a given detector configuration.

    Parameters
    ----------
    table : list[dict]
        Loaded via load_crit_val_table().
    detector : str
        Detector name (e.g. ``'PageCUSUM'``).
    gamma : float
        Weight exponent used in the detector.
    alpha : float
        Significance level (e.g. 0.05 for 95th percentile).

    Returns
    -------
    float
        Critical value matching the requested configuration.

    Raises
    ------
    KeyError
        If no table row matches the requested combination.
    """
    for row in table:
        if row["Detector"] == detector and row["Gamma"] == gamma and row["Alpha"] == alpha:
            return row["CritVal"]
    raise KeyError(f"No entry for detector={detector}, gamma={gamma}, alpha={alpha}")


# ============================================================================
# Module-level constants
# ============================================================================

_cfg = load_params()

CRITICAL_VALUE_NPOST_SHORT = _cfg["stats"]["critical_value_npost_short"]
CRITICAL_VALUE_NPOST_LONG = _cfg["stats"]["critical_value_npost_long"]


# ============================================================================
# Null Distribution (Critical Values) — Trend Change
# ============================================================================


def _run_null_simulations(
    Nsim, npre, npost, level, trend_control, sigma, phi, use_ar, label, ba=False
):
    """
    Run *Nsim* null simulations (no trend change) and return max test statistics.

    Delegates to _null_sim_worker via _run_parallel.  The label is printed as
    ``  [label] N sims ... done in Xs`` on a single line.

    Parameters
    ----------
    Nsim : int
        Number of null simulations.
    npre : int
        Pre-intervention length (months).
    npost : int
        Post-intervention length (months).
    level : float
        Base level of the time series.
    trend_control : float
        Control trend (identical before and after under the null).
    sigma : float
        Noise standard deviation.
    phi : float
        AR(1) coefficient (used only when use_ar=True).
    use_ar : bool
        If True, use AR(1) noise; if False, use i.i.d. noise.
    label : str
        Human-readable label shown in the progress line.
    ba : bool
        If True and use_ar=False, use BA design (intervention only).
        Ignored when use_ar=True (AR path is always BA).

    Returns
    -------
    ndarray, shape (Nsim,)
        Maximum test statistic from each null simulation.
    """
    args_list = [
        (i, npre, npost, level, trend_control, sigma, phi, use_ar, ba) for i in range(1, Nsim + 1)
    ]
    results = _run_parallel(_null_sim_worker, args_list, label=label)
    return np.array(results)


def calculate_critical_values(Nsim, npre, level, trend_control, sigma, phi, alpha=0.95):
    """
    Compute critical values under the null hypothesis (no trend change) at two
    post-intervention lengths and for two noise types, matching R's getCritical.

    R's getCritical uses npost.vec[c(3, 7)] from seq(24, 120, by=12), which
    resolves to:
      - 48 months  (CRITICAL_VALUE_NPOST_SHORT)
      - 96 months  (CRITICAL_VALUE_NPOST_LONG)

    For each npost length, i.i.d. BACI, i.i.d. BA, and AR(1) critical values
    are estimated from *Nsim* null simulations each.

    Parameters
    ----------
    Nsim : int
        Number of null simulations per combination.
    npre : int
        Pre-intervention period (months).
    level : float
        Base level of the time series.
    trend_control : float
        Control trend (same before and after under the null).
    sigma : float
        Noise standard deviation.
    phi : float
        AR(1) coefficient (used only for AR noise simulations).
    alpha : float
        Quantile of the null distribution used as critical value (default 0.95).

    Returns
    -------
    dict
        Keys: ``'iid_48'``, ``'iid_96'``, ``'iid_ba_48'``, ``'iid_ba_96'``,
              ``'ar1_48'``, ``'ar1_96'``.
        Each value is a dict with ``'critical_value'`` (float) and
        ``'null_dist'`` (ndarray).
    """
    critical_values = {}

    for npost, tag in [
        (CRITICAL_VALUE_NPOST_SHORT, str(CRITICAL_VALUE_NPOST_SHORT)),
        (CRITICAL_VALUE_NPOST_LONG, str(CRITICAL_VALUE_NPOST_LONG)),
    ]:
        print(f"  npost = {npost} months:")

        null_iid = _run_null_simulations(
            Nsim,
            npre,
            npost,
            level,
            trend_control,
            sigma,
            phi,
            use_ar=False,
            label=f"i.i.d. BACI, {npost}mo",
        )
        cv_iid = np.percentile(null_iid, alpha * 100)
        critical_values[f"iid_{tag}"] = {"critical_value": cv_iid, "null_dist": null_iid}
        print(f"    i.i.d. BACI cv = {cv_iid:.4f}")

        null_iid_ba = _run_null_simulations(
            Nsim,
            npre,
            npost,
            level,
            trend_control,
            sigma,
            phi,
            use_ar=False,
            ba=True,
            label=f"i.i.d. BA, {npost}mo",
        )
        cv_iid_ba = np.percentile(null_iid_ba, alpha * 100)
        critical_values[f"iid_ba_{tag}"] = {"critical_value": cv_iid_ba, "null_dist": null_iid_ba}
        print(f"    i.i.d. BA   cv = {cv_iid_ba:.4f}")

        null_ar = _run_null_simulations(
            Nsim,
            npre,
            npost,
            level,
            trend_control,
            sigma,
            phi,
            use_ar=True,
            label=f"AR(1), {npost}mo",
        )
        cv_ar = np.percentile(null_ar, alpha * 100)
        critical_values[f"ar1_{tag}"] = {"critical_value": cv_ar, "null_dist": null_ar}
        print(f"    AR(1)       cv = {cv_ar:.4f}")

    return critical_values


# ============================================================================
# Main Simulation (Alternative Hypothesis) — Trend Change
# ============================================================================


def _amoc_aggregate(raw, critical_value, npre):
    """
    Aggregate raw worker output into the standard AMOC result dict.

    Parameters
    ----------
    raw : list of (delay, tmax_vec, cpt_vec, seed)
        Direct output from _main_sim_worker or _main_sim_worker_ar.
    critical_value : float
        Threshold used to declare detection (Tmax > critical_value).
    npre : int
        Pre-intervention length; used to compute true changepoint locations
        as ``npre + delay``.

    Returns
    -------
    dict
        Keys: ``tmax_matrix``, ``cpt_matrix``, ``detected_matrix``, ``delays``,
        ``detection_rates``, ``mean_errors``, ``seeds``.
    """
    delays = [r[0] for r in raw]
    tmax_matrix = np.array([r[1] for r in raw])
    cpt_matrix = np.array([r[2] for r in raw])
    seeds = [r[3] for r in raw]

    detected_matrix = tmax_matrix > critical_value
    detection_rates = detected_matrix.mean(axis=0)

    true_cpts = np.array([npre + d for d in delays])
    error_matrix = np.abs(cpt_matrix - true_cpts[:, np.newaxis])
    mean_errors = error_matrix.mean(axis=0)

    return {
        "tmax_matrix": tmax_matrix,
        "cpt_matrix": cpt_matrix,
        "detected_matrix": detected_matrix,
        "delays": delays,
        "detection_rates": detection_rates,
        "mean_errors": mean_errors,
        "seeds": seeds,
    }


def _fmt_amoc_progress(result):
    """Return progress suffix showing detection rate and mean error at the last npost."""
    return (
        f"detect={float(result['detection_rates'][-1]):.2%}  "
        f"err={float(result['mean_errors'][-1]):.1f}mo"
    )


def run_main_simulation(
    simN,
    trend_increase,
    critical_value,
    npre,
    npost_max,
    npost_vec,
    level,
    trend_control,
    sigma,
    delay_set,
    existing_results=None,
    on_trend_done=None,
):
    """
    Run i.i.d. BACI main simulations (with trend change) for all trend increments.

    Uses a growing-window approach: for each effect size, *simN* time series are
    generated at *npost_max* and tested at every length in *npost_vec* by
    truncation, matching R's outer/inner sapply structure.

    Parameters
    ----------
    simN : int
        Number of Monte Carlo replications per trend increment.
    trend_increase : sequence of float
        Effect sizes to simulate (trend increment above trend_control).
    critical_value : float
        AMOC threshold; detections are flagged where Tmax > critical_value.
    npre : int
        Pre-intervention length (months).
    npost_max : int
        Maximum post-intervention length (months); sets the generated series length.
    npost_vec : array-like of int
        Growing-window evaluation lengths (months).
    level : float
        Base level of the time series.
    trend_control : float
        Control (and pre-intervention) trend slope.
    sigma : float
        Noise standard deviation.
    delay_set : array-like of int
        Pool of intervention-onset delays (months); one is sampled per simulation.
    existing_results : dict or None
        Pre-computed results keyed by trend increment; matching keys are skipped.
    on_trend_done : callable or None
        Called with the full results dict after each trend increment completes.

    Returns
    -------
    dict
        Mapping trend_inc → result dict (see _amoc_aggregate for keys).
    """

    def make_args(trend_val, trend_idx, n_trends):
        trend_interv = trend_control + trend_val
        return [
            (
                sim_idx,
                trend_interv,
                trend_idx,
                n_trends,
                npre,
                npost_max,
                npost_vec,
                level,
                trend_control,
                sigma,
                delay_set,
                False,
            )
            for sim_idx in range(1, simN + 1)
        ]

    return _simulation_loop(
        _main_sim_worker,
        make_args,
        trend_increase,
        "trend_inc",
        simN,
        aggregate_fn=lambda raw: _amoc_aggregate(raw, critical_value, npre),
        fmt_progress=_fmt_amoc_progress,
        existing_results=existing_results,
        on_trend_done=on_trend_done,
    )


def run_main_simulation_iid_ba(
    simN,
    trend_increase,
    critical_value,
    npre,
    npost_max,
    npost_vec,
    level,
    trend_control,
    sigma,
    delay_set,
    existing_results=None,
    on_trend_done=None,
):
    """
    Run i.i.d. BA main simulations (with trend change) for all trend increments.

    Equivalent to run_main_simulation() with BA design: only the intervention
    series is passed to trend_stats (no control series), matching R's
    simResult.v2 from AMOC.R.  The i.i.d. BA critical value (``iid_ba_48``)
    should be passed as *critical_value*.

    Parameters
    ----------
    simN : int
        Number of Monte Carlo replications per trend increment.
    trend_increase : sequence of float
        Effect sizes to simulate (trend increment above trend_control).
    critical_value : float
        AMOC threshold; detections are flagged where Tmax > critical_value.
    npre : int
        Pre-intervention length (months).
    npost_max : int
        Maximum post-intervention length (months).
    npost_vec : array-like of int
        Growing-window evaluation lengths (months).
    level : float
        Base level of the time series.
    trend_control : float
        Control (and pre-intervention) trend slope.
    sigma : float
        Noise standard deviation.
    delay_set : array-like of int
        Pool of intervention-onset delays (months).
    existing_results : dict or None
        Pre-computed results; matching keys are skipped.
    on_trend_done : callable or None
        Called with the full results dict after each trend increment completes.

    Returns
    -------
    dict
        Mapping trend_inc → result dict (see _amoc_aggregate for keys).
    """

    def make_args(trend_val, trend_idx, n_trends):
        trend_interv = trend_control + trend_val
        return [
            (
                sim_idx,
                trend_interv,
                trend_idx,
                n_trends,
                npre,
                npost_max,
                npost_vec,
                level,
                trend_control,
                sigma,
                delay_set,
                True,
            )
            for sim_idx in range(1, simN + 1)
        ]

    return _simulation_loop(
        _main_sim_worker,
        make_args,
        trend_increase,
        "trend_inc",
        simN,
        aggregate_fn=lambda raw: _amoc_aggregate(raw, critical_value, npre),
        fmt_progress=_fmt_amoc_progress,
        existing_results=existing_results,
        on_trend_done=on_trend_done,
    )


def run_main_simulation_ar(
    simN,
    trend_increase,
    critical_value,
    npre,
    npost_max,
    npost_vec,
    level,
    trend_control,
    sigma,
    phi,
    delay_set,
    existing_results=None,
    on_trend_done=None,
):
    """
    Run AR(1) BA main simulations (with trend change) for all trend increments.

    Equivalent to run_main_simulation_iid_ba() with AR(1) noise and an
    ARIMA(1,0,0)-based test statistic.  Uses ci_sim_ar and trend_stats_ar,
    always in BA design.  The AR(1) critical value (``ar1_48``) should be
    passed as *critical_value* so detection is calibrated to the autocorrelated
    null distribution.

    Parameters
    ----------
    simN : int
        Number of Monte Carlo replications per trend increment.
    trend_increase : sequence of float
        Effect sizes to simulate (trend increment above trend_control).
    critical_value : float
        AMOC threshold; detections are flagged where Tmax > critical_value.
    npre : int
        Pre-intervention length (months).
    npost_max : int
        Maximum post-intervention length (months).
    npost_vec : array-like of int
        Growing-window evaluation lengths (months).
    level : float
        Base level of the time series.
    trend_control : float
        Control (and pre-intervention) trend slope.
    sigma : float
        AR(1) innovation standard deviation.
    phi : float
        AR(1) autocorrelation coefficient.
    delay_set : array-like of int
        Pool of intervention-onset delays (months).
    existing_results : dict or None
        Pre-computed results; matching keys are skipped.
    on_trend_done : callable or None
        Called with the full results dict after each trend increment completes.

    Returns
    -------
    dict
        Mapping trend_inc → result dict (see _amoc_aggregate for keys).
    """

    def make_args(trend_val, trend_idx, n_trends):
        trend_interv = trend_control + trend_val
        return [
            (
                sim_idx,
                trend_interv,
                trend_idx,
                n_trends,
                npre,
                npost_max,
                npost_vec,
                level,
                trend_control,
                sigma,
                phi,
                delay_set,
            )
            for sim_idx in range(1, simN + 1)
        ]

    return _simulation_loop(
        _main_sim_worker_ar,
        make_args,
        trend_increase,
        "trend_inc",
        simN,
        aggregate_fn=lambda raw: _amoc_aggregate(raw, critical_value, npre),
        fmt_progress=_fmt_amoc_progress,
        existing_results=existing_results,
        on_trend_done=on_trend_done,
    )


# ============================================================================
# Distribution Change AMOC
# ============================================================================

CRITICAL_VALUE_NPOST_SHORT_CDF = _cfg["distribution"]["critical_value_npost_short"]
CRITICAL_VALUE_NPOST_MEDIUM_CDF = _cfg["distribution"]["critical_value_npost_medium"]
CRITICAL_VALUE_NPOST_LONG_CDF = _cfg["distribution"]["critical_value_npost_long"]


def calculate_critical_values_cdf(Nsim, npre, mu, sigma, ns, dist_measure, bw, nd, alpha=0.95):
    """
    Compute critical values under the null hypothesis (no distribution change) at
    three post-intervention lengths matching the distribution pipeline's npost grid.

    Parameters
    ----------
    Nsim : int
        Number of null simulations per npost length.
    npre : int
        Pre-intervention period (months).
    mu : float
        Baseline distribution mean.
    sigma : float
        Baseline distribution standard deviation.
    ns : int
        Number of samples per time point.
    dist_measure : str
        Distance measure: ``'wasserstein'`` or ``'auc_diff'``.
    bw : float
        Bandwidth for auc_diff_ts (unused when dist_measure='wasserstein').
    nd : int
        Density grid points for auc_diff_ts (unused when dist_measure='wasserstein').
    alpha : float
        Quantile of the null distribution used as critical value (default 0.95).

    Returns
    -------
    dict
        Keys: ``'npost_24'``, ``'npost_72'``, ``'npost_120'``.
        Each value is a dict with ``'critical_value'`` (float) and
        ``'null_dist'`` (ndarray).
    """
    critical_values = {}

    for npost, label in [
        (CRITICAL_VALUE_NPOST_SHORT_CDF, "24"),
        (CRITICAL_VALUE_NPOST_MEDIUM_CDF, "72"),
        (CRITICAL_VALUE_NPOST_LONG_CDF, "120"),
    ]:
        args_list = [
            (i, npre, npost, mu, sigma, ns, dist_measure, bw, nd) for i in range(1, Nsim + 1)
        ]
        results = _run_parallel(
            _null_sim_worker_cdf, args_list, label=f"npost={npost}mo, {Nsim} sims"
        )
        cv = np.percentile(results, alpha * 100)
        critical_values[f"npost_{label}"] = {
            "critical_value": cv,
            "null_dist": np.array(results),
        }
        print(f"    cv = {cv:.4f}")

    return critical_values


def run_main_simulation_mu(
    simN,
    trend_increase_mu,
    critical_value,
    npre,
    npost_max,
    npost_vec,
    mu,
    sigma,
    ns,
    dist_measure,
    bw,
    nd,
    delay_set,
    ba=False,
    existing_results=None,
    on_trend_done=None,
):
    """
    Run CDF AMOC simulations for a mean (mu) shift at each effect size.

    Parameters
    ----------
    simN : int
        Number of Monte Carlo replications per effect size.
    trend_increase_mu : sequence of float
        Mean-shift magnitudes to simulate.
    critical_value : float
        AMOC threshold; detections flagged where Tmax > critical_value.
    npre : int
        Pre-intervention length (months).
    npost_max : int
        Maximum post-intervention length (months).
    npost_vec : array-like of int
        Growing-window evaluation lengths (months).
    mu : float
        Baseline distribution mean.
    sigma : float
        Baseline distribution standard deviation.
    ns : int
        Number of samples per time point.
    dist_measure : str
        Distance measure for BACI design: ``'wasserstein'`` or ``'auc_diff'``.
        Ignored when ba=True.
    bw : float
        Bandwidth for auc_diff_ts (ignored when ba=True).
    nd : int
        Density grid points for auc_diff_ts (ignored when ba=True).
    delay_set : array-like of int
        Pool of intervention-onset delays (months).
    ba : bool
        If True, use BA design (wasserstein_distance_ba on intervention only).
        If False, use BACI design selected by dist_measure.
    existing_results : dict or None
        Pre-computed results; matching keys are skipped.
    on_trend_done : callable or None
        Called with the full results dict after each effect size completes.

    Returns
    -------
    dict
        Mapping trend_mu → result dict (see _amoc_aggregate for keys).
    """

    def make_args(trend_val, trend_idx, n_trends):
        return [
            (
                sim_idx,
                trend_val,
                0.0,
                ba,
                trend_idx,
                n_trends,
                npre,
                npost_max,
                npost_vec,
                mu,
                sigma,
                ns,
                dist_measure,
                bw,
                nd,
                delay_set,
            )
            for sim_idx in range(1, simN + 1)
        ]

    return _simulation_loop(
        _main_sim_worker_cdf,
        make_args,
        trend_increase_mu,
        "trend_mu",
        simN,
        aggregate_fn=lambda raw: _amoc_aggregate(raw, critical_value, npre),
        fmt_progress=None,
        existing_results=existing_results,
        on_trend_done=on_trend_done,
    )


def run_main_simulation_sigma(
    simN,
    trend_increase_sigma,
    critical_value,
    npre,
    npost_max,
    npost_vec,
    mu,
    sigma,
    ns,
    dist_measure,
    bw,
    nd,
    delay_set,
    existing_results=None,
    on_trend_done=None,
):
    """
    Run CDF AMOC simulations for a spread (sigma) shift at each effect size.

    Parameters
    ----------
    simN : int
        Number of Monte Carlo replications per effect size.
    trend_increase_sigma : sequence of float
        Sigma-shift magnitudes to simulate.
    critical_value : float
        AMOC threshold; detections flagged where Tmax > critical_value.
    npre : int
        Pre-intervention length (months).
    npost_max : int
        Maximum post-intervention length (months).
    npost_vec : array-like of int
        Growing-window evaluation lengths (months).
    mu : float
        Baseline distribution mean.
    sigma : float
        Baseline distribution standard deviation.
    ns : int
        Number of samples per time point.
    dist_measure : str
        Distance measure: ``'wasserstein'`` or ``'auc_diff'``.
    bw : float
        Bandwidth for auc_diff_ts.
    nd : int
        Density grid points for auc_diff_ts.
    delay_set : array-like of int
        Pool of intervention-onset delays (months).
    existing_results : dict or None
        Pre-computed results; matching keys are skipped.
    on_trend_done : callable or None
        Called with the full results dict after each effect size completes.

    Returns
    -------
    dict
        Mapping trend_sigma → result dict (see _amoc_aggregate for keys).
    """

    def make_args(trend_val, trend_idx, n_trends):
        return [
            (
                sim_idx,
                0.0,
                trend_val,
                False,
                trend_idx,
                n_trends,
                npre,
                npost_max,
                npost_vec,
                mu,
                sigma,
                ns,
                dist_measure,
                bw,
                nd,
                delay_set,
            )
            for sim_idx in range(1, simN + 1)
        ]

    return _simulation_loop(
        _main_sim_worker_cdf,
        make_args,
        trend_increase_sigma,
        "trend_sigma",
        simN,
        aggregate_fn=lambda raw: _amoc_aggregate(raw, critical_value, npre),
        fmt_progress=None,
        existing_results=existing_results,
        on_trend_done=on_trend_done,
    )
