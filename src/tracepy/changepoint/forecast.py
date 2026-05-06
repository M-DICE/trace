"""
Forecast (Page-CUSUM) orchestration for trend and distribution changepoint detection.

Contains:
- page_cusum            — weighted two-sided Page-CUSUM detector
- trend_stats_forecast  — two-stage forecast-based changepoint detection
- Trend forecast workers and orchestrators (i.i.d. and AR(1))
- Distribution forecast workers and orchestrators (BACI and BA)

Simulation workers for AMOC live in tracepy.simulation.runners.  The shared
Monte Carlo loop (_simulation_loop) and timing helper (_fmt_elapsed) are also
imported from the simulation package.
"""

import warnings

import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.arima.model import ARIMA

from tracepy.simulation.distribution import ci_sim_cdf
from tracepy.simulation.runners import _simulation_loop
from tracepy.simulation.trend import ci_sim, ci_sim_ar
from tracepy.stats.metrics import trend_stats, wasserstein_distance_ba, wasserstein_distance_baci


def page_cusum(errors, m, crit_val, gamma=0.0):
    """
    Weighted two-sided Page-CUSUM detector.

    Accumulates raw centered errors against a time-varying threshold
    T(k) = w(k) * crit_val * sigma, where
    w(k) = sqrt(m) * (1 + k/m) * (k / (k + m))^gamma.
    *crit_val* must be resolved from CritValTable.json via lookup_crit_val().

    Parameters
    ----------
    errors : array-like, shape (m + n,)
        Concatenated in-sample residuals (length m) and out-of-sample errors
        (length n).
    m : int
        Training period length (number of in-sample points).
    crit_val : float
        Page-CUSUM critical value from CritValTable.json.
    gamma : float
        Weight exponent (default 0.0 for unweighted CUSUM).

    Returns
    -------
    int or float
        1-indexed detection step within the post-period, or ``np.inf`` if
        no threshold crossing occurred.
    """
    errors = np.asarray(errors, dtype=float)
    n = len(errors) - m
    train_mean = np.mean(errors[:m])
    sigma = np.std(errors[:m], ddof=1)
    if sigma < 1e-12:
        return np.inf
    c_upper = c_lower = 0.0
    for k in range(1, n + 1):
        inc = errors[m + k - 1] - train_mean
        c_upper = max(0.0, c_upper + inc)
        c_lower = max(0.0, c_lower - inc)
        weight = np.sqrt(m) * (1.0 + k / m) * ((k / (k + m)) ** gamma)
        threshold = weight * crit_val * sigma
        if c_upper > threshold or c_lower > threshold:
            return k
    return np.inf


def trend_stats_forecast(y_itv, npre, ntt, phi=None, crit_val=2.1705321342):
    """
    Two-stage forecast-based changepoint detection (BA design).

    Stage 1 — fit a linear model (OLS or ARIMA(1,0,0)) on the pre-period,
    produce multi-step-ahead forecasts, build the residual series, and run
    Page-CUSUM to find *when* a change is first declared (time_est).

    Stage 2 — if detection occurred, run AMOC trend_stats on the residuals
    truncated to length ``npre + time_est`` to locate *where* the changepoint
    is (cpt_est).

    Sign convention matches R: in-sample residuals are (actual − fitted);
    out-of-sample errors are (predicted − actual).  The two-sided CUSUM
    detects shifts in either direction.

    Parameters
    ----------
    y_itv : array-like, shape (ntt,)
        Intervention time series (full length npre + npost_max).
    npre : int
        Training period length (months).
    ntt : int
        Total time series length (npre + npost_max).  Used to build the
        design matrix for forecasting all post-period steps at once.
    phi : float or None
        If provided, fit ARIMA(1,0,0) with this AR coefficient; if None,
        fall back to OLS.
    crit_val : float
        Page-CUSUM critical value from CritValTable.json (default: PageCUSUM,
        gamma=0, alpha=0.05).

    Returns
    -------
    dict
        ``{'time_est': int or inf, 'cpt_est': int or inf}``

        time_est : detection time (1-indexed, relative to post-period start).
        cpt_est  : estimated changepoint location (1-indexed absolute month).
    """
    y = np.asarray(y_itv, dtype=float)
    t_idx = np.arange(1, ntt + 1)
    X = np.column_stack([np.ones(ntt), t_idx])

    in_residuals = None
    out_errors = None

    if phi is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = ARIMA(y[:npre], exog=X[:npre], order=(1, 0, 0), trend='n')
                fit = model.fit(method='innovations_mle', disp=False)
            in_residuals = np.asarray(fit.resid, dtype=float)
            forecast_vals = fit.get_forecast(steps=ntt - npre, exog=X[npre:]).predicted_mean
            out_errors = np.asarray(forecast_vals, dtype=float) - y[npre:]
        except Exception:
            in_residuals = None

    if in_residuals is None:
        # OLS fallback (also used when phi is None)
        ols = OLS(y[:npre], X[:npre]).fit()
        in_residuals = np.asarray(ols.resid, dtype=float)
        out_errors = X[npre:] @ ols.params - y[npre:]

    r = np.concatenate([in_residuals, out_errors])
    time_est = page_cusum(r, m=npre, crit_val=crit_val)

    if np.isfinite(time_est):
        time_est_int = int(time_est)
        nt = npre + time_est_int
        # trend_stats requires nt >= mint + 4 = 28; fall back to npre if too short
        if nt >= 28:
            stats = trend_stats(y_ctr=None, y_itv=r[:nt], nt=nt)
            cpt_est = stats['cpt']
        else:
            cpt_est = npre
        return {'time_est': time_est_int, 'cpt_est': cpt_est}
    else:
        return {'time_est': np.inf, 'cpt_est': np.inf}


# ============================================================================
# Trend Forecast workers
# ============================================================================

def _forecast_sim_worker_iid(args):
    """
    i.i.d. BA trend forecast worker — one run of Forecast detection.

    Top-level for pickling by multiprocessing on macOS (spawn start method).

    Parameters
    ----------
    args : tuple
        (sim_idx, trend_interv, trend_idx, n_trends, npre, ntt, npost_max,
         level, trend_control, sigma, delay_set, crit_val)

    Returns
    -------
    tuple
        (cpt_est, time_est, delay, seed)
    """
    (sim_idx, trend_interv, trend_idx, n_trends, npre, ntt, npost_max,
     level, trend_control, sigma, delay_set, crit_val) = args
    warnings.filterwarnings('ignore')

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))

    sim_data = ci_sim(seed=seed, npre=npre + delay, npost=npost_max - delay,
                      level=level, trend=[trend_control, trend_interv], sigma=sigma)

    result = trend_stats_forecast(sim_data['y_itv'], npre=npre, ntt=ntt,
                                  phi=None, crit_val=crit_val)
    return result['cpt_est'], result['time_est'], delay, seed


def _forecast_sim_worker_ar(args):
    """
    AR(1) BA trend forecast worker — one run of Forecast detection.

    Top-level for pickling by multiprocessing on macOS (spawn start method).

    Parameters
    ----------
    args : tuple
        (sim_idx, trend_interv, trend_idx, n_trends, npre, ntt, npost_max,
         level, trend_control, sigma, phi, delay_set, crit_val)

        phi : float — AR(1) coefficient passed to both ci_sim_ar and
                      trend_stats_forecast.

    Returns
    -------
    tuple
        (cpt_est, time_est, delay, seed)
    """
    (sim_idx, trend_interv, trend_idx, n_trends, npre, ntt, npost_max,
     level, trend_control, sigma, phi, delay_set, crit_val) = args
    warnings.filterwarnings('ignore')

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))

    sim_data = ci_sim_ar(seed=seed, npre=npre + delay, npost=npost_max - delay,
                         level=level, trend=[trend_control, trend_interv],
                         phi=phi, sigma=sigma)

    result = trend_stats_forecast(sim_data['y_itv'], npre=npre, ntt=ntt,
                                  phi=phi, crit_val=crit_val)
    return result['cpt_est'], result['time_est'], delay, seed


# ============================================================================
# Distribution Forecast workers
# ============================================================================

def _forecast_sim_worker_cdf(args):
    """
    Distribution forecast worker for BACI and BA designs.

    Replaces the former ``_forecast_sim_worker_cdf_baci`` and
    ``_forecast_sim_worker_cdf_ba``.  A single ``ba`` flag selects which
    distance measure is applied; all other logic (OLS fit, Page-CUSUM,
    AMOC trend_stats for changepoint location) is shared.

    Top-level for pickling by multiprocessing on macOS (spawn start method).

    Parameters
    ----------
    args : tuple
        (sim_idx, trend_mu, trend_idx, n_trends, npre, ntt, npost_max,
         mu, sigma, ns, delay_set, crit_val, ba)

        ba : bool — True  → wasserstein_distance_ba (intervention series only);
                    False → wasserstein_distance_baci (control and intervention).

    Returns
    -------
    tuple
        (cpt_est, time_est, delay, seed)

        cpt_est  : float or inf — estimated changepoint (1-indexed absolute month)
        time_est : int or inf   — detection time (1-indexed, relative to post-period)
        delay    : int          — intervention-onset delay applied in this simulation
        seed     : int          — simulation seed for reproducibility
    """
    (sim_idx, trend_mu, trend_idx, n_trends, npre, ntt, npost_max,
     mu, sigma, ns, delay_set, crit_val, ba) = args
    warnings.filterwarnings('ignore')

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))

    sim = ci_sim_cdf(seed=seed, npre=npre + delay, npost=npost_max - delay,
                     level=[mu, sigma], trend=[trend_mu, 0], ns=ns)

    if ba:
        dist_ts = wasserstein_distance_ba(sim['sample_itv'], npre)
    else:
        dist_ts = wasserstein_distance_baci(sim['sample_ctr'], sim['sample_itv'])

    X = np.column_stack([np.ones(ntt), np.arange(1, ntt + 1)])
    ols = OLS(dist_ts[:npre], X[:npre]).fit()
    r = np.concatenate([ols.resid, X[npre:] @ ols.params - dist_ts[npre:]])

    time_est = page_cusum(r, m=npre, crit_val=crit_val)

    if np.isfinite(time_est):
        time_est_int = int(time_est)
        nt = npre + time_est_int
        if nt >= 28:
            stats = trend_stats(y_ctr=None, y_itv=r[:nt], nt=nt)
            cpt_est = stats['cpt']
        else:
            cpt_est = npre
    else:
        cpt_est = np.inf

    return cpt_est, time_est, delay, seed


# ============================================================================
# Result aggregation
# ============================================================================

def _collect_results(raw, npre):
    """
    Aggregate per-simulation tuples from forecast workers into summary arrays.

    Parameters
    ----------
    raw : list of (cpt_est, time_est, delay, seed)
        Direct output from any ``_forecast_sim_worker_*`` function.
    npre : int
        Pre-intervention length; used to compute true changepoints as
        ``npre + delay``.

    Returns
    -------
    dict
        cpt_est_vec    : ndarray(simN,) — estimated changepoints (inf for non-detections)
        time_est_vec   : ndarray(simN,) — detection times (inf for non-detections)
        delays         : list[int]      — intervention-onset delay per simulation
        detected       : ndarray(simN,) of bool — True where time_est is finite
        detection_rate : float          — fraction of simulations with a detection
        mean_time      : float or nan   — mean detection time among detections
        mean_error     : float or nan   — mean |cpt_est - true_cpt| among detections
        seeds          : list[int]      — random seeds for reproducibility
    """
    cpt_est_vec  = np.array([r[0] for r in raw], dtype=float)
    time_est_vec = np.array([r[1] for r in raw], dtype=float)
    delays       = [r[2] for r in raw]
    seeds        = [r[3] for r in raw]

    detected = np.isfinite(time_est_vec)
    detection_rate = float(detected.mean())

    true_cpts = np.array([npre + d for d in delays], dtype=float)
    errors    = np.abs(cpt_est_vec[detected] - true_cpts[detected])
    mean_error = float(errors.mean())        if detected.any() else np.nan
    mean_time  = float(time_est_vec[detected].mean()) if detected.any() else np.nan

    return {
        'cpt_est_vec':    cpt_est_vec,
        'time_est_vec':   time_est_vec,
        'delays':         delays,
        'detected':       detected,
        'detection_rate': detection_rate,
        'mean_time':      mean_time,
        'mean_error':     mean_error,
        'seeds':          seeds,
    }


def _fmt_forecast_progress(result):
    """Return progress suffix showing detection rate and mean error."""
    return (f"detect={result['detection_rate']:.2%}  "
            f"err={result['mean_error']:.1f}mo")


# ============================================================================
# Trend Forecast orchestrators
# ============================================================================

def run_simulation_iid(simN, trend_increase, crit_val, npre, ntt, npost_max,
                       level, trend_control, sigma, delay_set,
                       existing_results=None, on_trend_done=None):
    """
    Run i.i.d. BA Forecast simulations for all trend increments.

    Parameters
    ----------
    simN : int
        Number of Monte Carlo replications per trend increment.
    trend_increase : sequence of float
        Effect sizes to simulate (trend increment above trend_control).
    crit_val : float
        Page-CUSUM critical value from CritValTable.json.
    npre : int
        Pre-intervention (training) length (months).
    ntt : int
        Total time series length (npre + npost_max).
    npost_max : int
        Maximum post-intervention length (months).
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
        Mapping trend_inc → result dict (see _collect_results for keys).
    """
    def make_args(trend_val, trend_idx, n_trends):
        trend_interv = trend_control + trend_val
        return [
            (sim_idx, trend_interv, trend_idx, n_trends, npre, ntt, npost_max,
             level, trend_control, sigma, delay_set, crit_val)
            for sim_idx in range(1, simN + 1)
        ]

    return _simulation_loop(
        _forecast_sim_worker_iid, make_args, trend_increase, 'trend_inc', simN,
        aggregate_fn=lambda raw: _collect_results(raw, npre),
        fmt_progress=_fmt_forecast_progress,
        existing_results=existing_results,
        on_trend_done=on_trend_done,
    )


def run_simulation_ar(simN, trend_increase, crit_val, npre, ntt, npost_max,
                      level, trend_control, sigma, phi, delay_set,
                      existing_results=None, on_trend_done=None):
    """
    Run AR(1) BA Forecast simulations for all trend increments.

    Parameters
    ----------
    simN : int
        Number of Monte Carlo replications per trend increment.
    trend_increase : sequence of float
        Effect sizes to simulate (trend increment above trend_control).
    crit_val : float
        Page-CUSUM critical value from CritValTable.json.
    npre : int
        Pre-intervention (training) length (months).
    ntt : int
        Total time series length (npre + npost_max).
    npost_max : int
        Maximum post-intervention length (months).
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
        Mapping trend_inc → result dict (see _collect_results for keys).
    """
    def make_args(trend_val, trend_idx, n_trends):
        trend_interv = trend_control + trend_val
        return [
            (sim_idx, trend_interv, trend_idx, n_trends, npre, ntt, npost_max,
             level, trend_control, sigma, phi, delay_set, crit_val)
            for sim_idx in range(1, simN + 1)
        ]

    return _simulation_loop(
        _forecast_sim_worker_ar, make_args, trend_increase, 'trend_inc', simN,
        aggregate_fn=lambda raw: _collect_results(raw, npre),
        fmt_progress=_fmt_forecast_progress,
        existing_results=existing_results,
        on_trend_done=on_trend_done,
    )


# ============================================================================
# Distribution Forecast orchestrators
# ============================================================================

def run_simulation_cdf_baci(simN: int, trend_increase_mu, crit_val: float,
                            npre: int, ntt: int, npost_max: int,
                            mu: float, sigma: float, ns: int, delay_set,
                            existing_results=None, on_trend_done=None) -> dict:
    """
    Run BACI distribution Forecast simulations for all mean-shift effect sizes.

    Uses wasserstein_distance_baci (control and intervention series), fitting
    an OLS model on the pre-period distance time series and applying Page-CUSUM
    to the residuals.

    Parameters
    ----------
    simN : int
        Number of Monte Carlo replications per effect size.
    trend_increase_mu : sequence of float
        Mean-shift magnitudes to simulate.
    crit_val : float
        Page-CUSUM critical value from CritValTable.json.
    npre : int
        Pre-intervention (training) length (months).
    ntt : int
        Total time series length (npre + npost_max).
    npost_max : int
        Maximum post-intervention length (months).
    mu : float
        Baseline distribution mean.
    sigma : float
        Baseline distribution standard deviation.
    ns : int
        Number of samples per time point.
    delay_set : array-like of int
        Pool of intervention-onset delays (months).
    existing_results : dict or None
        Pre-computed results; matching keys are skipped.
    on_trend_done : callable or None
        Called with the full results dict after each effect size completes.

    Returns
    -------
    dict
        Mapping trend_mu → result dict (see _collect_results for keys).
    """
    def make_args(trend_val, trend_idx, n_trends):
        return [
            (sim_idx, trend_val, trend_idx, n_trends, npre, ntt, npost_max,
             mu, sigma, ns, delay_set, crit_val, False)
            for sim_idx in range(1, simN + 1)
        ]

    return _simulation_loop(
        _forecast_sim_worker_cdf, make_args, trend_increase_mu, 'trend_mu', simN,
        aggregate_fn=lambda raw: _collect_results(raw, npre),
        fmt_progress=_fmt_forecast_progress,
        existing_results=existing_results,
        on_trend_done=on_trend_done,
    )


def run_simulation_cdf_ba(simN: int, trend_increase_mu, crit_val: float,
                          npre: int, ntt: int, npost_max: int,
                          mu: float, sigma: float, ns: int, delay_set,
                          existing_results=None, on_trend_done=None) -> dict:
    """
    Run BA distribution Forecast simulations for all mean-shift effect sizes.

    Uses wasserstein_distance_ba (intervention series only), fitting an OLS
    model on the pre-period distance time series and applying Page-CUSUM to
    the residuals.

    Parameters
    ----------
    simN : int
        Number of Monte Carlo replications per effect size.
    trend_increase_mu : sequence of float
        Mean-shift magnitudes to simulate.
    crit_val : float
        Page-CUSUM critical value from CritValTable.json.
    npre : int
        Pre-intervention (training) length (months).
    ntt : int
        Total time series length (npre + npost_max).
    npost_max : int
        Maximum post-intervention length (months).
    mu : float
        Baseline distribution mean.
    sigma : float
        Baseline distribution standard deviation.
    ns : int
        Number of samples per time point.
    delay_set : array-like of int
        Pool of intervention-onset delays (months).
    existing_results : dict or None
        Pre-computed results; matching keys are skipped.
    on_trend_done : callable or None
        Called with the full results dict after each effect size completes.

    Returns
    -------
    dict
        Mapping trend_mu → result dict (see _collect_results for keys).
    """
    def make_args(trend_val, trend_idx, n_trends):
        return [
            (sim_idx, trend_val, trend_idx, n_trends, npre, ntt, npost_max,
             mu, sigma, ns, delay_set, crit_val, True)
            for sim_idx in range(1, simN + 1)
        ]

    return _simulation_loop(
        _forecast_sim_worker_cdf, make_args, trend_increase_mu, 'trend_mu', simN,
        aggregate_fn=lambda raw: _collect_results(raw, npre),
        fmt_progress=_fmt_forecast_progress,
        existing_results=existing_results,
        on_trend_done=on_trend_done,
    )
