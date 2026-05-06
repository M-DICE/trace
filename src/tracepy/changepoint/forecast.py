"""
Forecast (Page-CUSUM) orchestration for trend changepoint detection.

Contains:
- page_cusum — weighted two-sided Page-CUSUM detector
- trend_stats_forecast — two-stage forecast-based changepoint detection
- run_simulation_iid, run_simulation_ar — orchestration functions
"""

import os
import time
import warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.arima.model import ARIMA

from tracepy.simulation.trend import ci_sim, ci_sim_ar
from tracepy.simulation.distribution import ci_sim_cdf
from tracepy.stats.metrics import trend_stats, wasserstein_distance_baci, wasserstein_distance_ba
from tracepy.changepoint.amoc import load_crit_val_table, lookup_crit_val


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def page_cusum(errors, m, crit_val, gamma=0.0):
    """
    Weighted two-sided Page-CUSUM detector.

    Accumulates raw centered errors against a time-varying threshold
    T(k) = w(k) * crit_val * sigma, where w(k) = sqrt(m) * (1 + k/m) * (k/(k+m))^gamma.
    crit_val must be resolved from CritValTable.json via lookup_crit_val().

    Returns 1-indexed detection step in the post-period, or np.inf.
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
    truncated to length npre + time_est to locate *where* the changepoint
    is (cpt_est).

    Sign convention matches R: in-sample residuals are (actual - fitted);
    out-of-sample errors are (predicted - actual). The two-sided CUSUM
    detects shifts in either direction.

    Parameters
    ----------
    y_itv : array-like (ntt,)
        Intervention time series (full length npre + npost_max).
    npre : int
        Training period length (months).
    ntt : int
        Total time series length (npre + npost_max). Used to build the design
        matrix for forecasting all post-period steps at once.
    phi : float or None
        If provided, fit ARIMA(1,0,0) with AR coefficient; if None, use OLS.
    crit_val : float
        Page-CUSUM critical value from CritValTable.json (default: PageCUSUM,
        gamma=0, alpha=0.05).

    Returns
    -------
    dict
        {'time_est': int or inf, 'cpt_est': int or inf}
        time_est: detection time (1-indexed, relative to post-period start).
        cpt_est:  estimated changepoint location (1-indexed absolute month).
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
# Main Simulation (Alternative Hypothesis) — Forecast
# ============================================================================

def _forecast_sim_worker_iid(args):
    """
    i.i.d. BA simulation worker — one run of Forecast detection.

    Top-level for pickling by multiprocessing on macOS.

    Returns
    -------
    tuple: (cpt_est, time_est, delay, seed)
    """
    (sim_idx, trend_interv, trend_idx, n_trends, npre, ntt, npost_max,
     level, trend_control, sigma, delay_set, crit_val) = args
    warnings.filterwarnings('ignore')

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))
    npre_delay = npre + delay
    npost_delay = npost_max - delay

    sim_data = ci_sim(seed=seed, npre=npre_delay, npost=npost_delay, level=level,
                      trend=[trend_control, trend_interv], sigma=sigma)

    result = trend_stats_forecast(sim_data['y_itv'], npre=npre, ntt=ntt,
                                  phi=None, crit_val=crit_val)
    return result['cpt_est'], result['time_est'], delay, seed


def _forecast_sim_worker_ar(args):
    """
    AR(1) BA simulation worker — one run of Forecast detection.

    Top-level for pickling by multiprocessing on macOS.

    Returns
    -------
    tuple: (cpt_est, time_est, delay, seed)
    """
    (sim_idx, trend_interv, trend_idx, n_trends, npre, ntt, npost_max,
     level, trend_control, sigma, phi, delay_set, crit_val) = args
    warnings.filterwarnings('ignore')

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))
    npre_delay = npre + delay
    npost_delay = npost_max - delay

    sim_data = ci_sim_ar(seed=seed, npre=npre_delay, npost=npost_delay, level=level,
                         trend=[trend_control, trend_interv], phi=phi, sigma=sigma)

    result = trend_stats_forecast(sim_data['y_itv'], npre=npre, ntt=ntt,
                                  phi=phi, crit_val=crit_val)
    return result['cpt_est'], result['time_est'], delay, seed


def _collect_results(raw, npre):
    """
    Aggregate raw per-simulation tuples into summary arrays and metrics.

    Parameters
    ----------
    raw : list of (cpt_est, time_est, delay, seed)
    npre : int

    Returns
    -------
    dict
    """
    cpt_est_vec  = np.array([r[0] for r in raw], dtype=float)
    time_est_vec = np.array([r[1] for r in raw], dtype=float)
    delays       = [r[2] for r in raw]
    seeds        = [r[3] for r in raw]

    detected = np.isfinite(time_est_vec)
    detection_rate = float(detected.mean())

    true_cpts = np.array([npre + d for d in delays], dtype=float)
    errors = np.abs(cpt_est_vec[detected] - true_cpts[detected])
    mean_error = float(errors.mean()) if detected.any() else np.nan
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


def run_simulation_iid(simN, trend_increase, crit_val, npre, ntt, npost_max,
                       level, trend_control, sigma, delay_set,
                       existing_results=None, on_trend_done=None):
    """Run i.i.d. BA simulations for all trend increments."""
    detection_results = dict(existing_results or {})
    n_trends = len(trend_increase)
    n_workers = os.cpu_count() or 1

    for trend_idx, trend_inc in enumerate(trend_increase, start=1):
        if trend_inc in detection_results:
            print(f"  [{trend_idx}/{n_trends}] trend_inc={trend_inc:.4f}  (cached, skipping)")
            continue

        trend_interv = trend_control + trend_inc
        t0 = time.perf_counter()
        print(f"  [{trend_idx}/{n_trends}] trend_inc={trend_inc:.4f}  ({simN} sims) ...",
              end="", flush=True)

        args_list = [
            (sim_idx, trend_interv, trend_idx, n_trends, npre, ntt, npost_max,
             level, trend_control, sigma, delay_set, crit_val)
            for sim_idx in range(1, simN + 1)
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            raw = list(executor.map(_forecast_sim_worker_iid, args_list))

        res = _collect_results(raw, npre)
        print(f" done in {_fmt_elapsed(time.perf_counter() - t0)}  "
              f"detect={res['detection_rate']:.2%}  "
              f"err={res['mean_error']:.1f}mo", flush=True)

        detection_results[trend_inc] = res

        if on_trend_done:
            on_trend_done(detection_results)

    return detection_results


def run_simulation_ar(simN, trend_increase, crit_val, npre, ntt, npost_max,
                      level, trend_control, sigma, phi, delay_set,
                      existing_results=None, on_trend_done=None):
    """Run AR(1) BA simulations for all trend increments."""
    detection_results = dict(existing_results or {})
    n_trends = len(trend_increase)
    n_workers = os.cpu_count() or 1

    for trend_idx, trend_inc in enumerate(trend_increase, start=1):
        if trend_inc in detection_results:
            print(f"  [{trend_idx}/{n_trends}] trend_inc={trend_inc:.4f}  (cached, skipping)")
            continue

        trend_interv = trend_control + trend_inc
        t0 = time.perf_counter()
        print(f"  [{trend_idx}/{n_trends}] trend_inc={trend_inc:.4f}  ({simN} sims) ...",
              end="", flush=True)

        args_list = [
            (sim_idx, trend_interv, trend_idx, n_trends, npre, ntt, npost_max,
             level, trend_control, sigma, phi, delay_set, crit_val)
            for sim_idx in range(1, simN + 1)
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            raw = list(executor.map(_forecast_sim_worker_ar, args_list))

        res = _collect_results(raw, npre)
        print(f" done in {_fmt_elapsed(time.perf_counter() - t0)}  "
              f"detect={res['detection_rate']:.2%}  "
              f"err={res['mean_error']:.1f}mo", flush=True)

        detection_results[trend_inc] = res

        if on_trend_done:
            on_trend_done(detection_results)

    return detection_results


# ============================================================================
# Distribution Forecast — BACI and BA workers + orchestrators
# ============================================================================

def _forecast_sim_worker_cdf_baci(args):
    """
    BACI distribution forecast worker — one run per (sim, trend) pair.

    Top-level for pickling by multiprocessing on macOS.

    Returns
    -------
    tuple: (cpt_est, time_est, delay, seed)
    """
    (sim_idx, trend_mu, trend_idx, n_trends, npre, ntt, npost_max,
     mu, sigma, ns, delay_set, crit_val) = args
    warnings.filterwarnings('ignore')

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))

    sim = ci_sim_cdf(seed=seed, npre=npre + delay, npost=npost_max - delay,
                     level=[mu, sigma], trend=[trend_mu, 0], ns=ns)

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


def _forecast_sim_worker_cdf_ba(args):
    """
    BA distribution forecast worker — intervention series only.

    Top-level for pickling by multiprocessing on macOS.

    Returns
    -------
    tuple: (cpt_est, time_est, delay, seed)
    """
    (sim_idx, trend_mu, trend_idx, n_trends, npre, ntt, npost_max,
     mu, sigma, ns, delay_set, crit_val) = args
    warnings.filterwarnings('ignore')

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))

    sim = ci_sim_cdf(seed=seed, npre=npre + delay, npost=npost_max - delay,
                     level=[mu, sigma], trend=[trend_mu, 0], ns=ns)

    dist_ts = wasserstein_distance_ba(sim['sample_itv'], npre)

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


def run_simulation_cdf_baci(simN: int, trend_increase_mu, crit_val: float,
                            npre: int, ntt: int, npost_max: int,
                            mu: float, sigma: float, ns: int, delay_set,
                            existing_results=None, on_trend_done=None) -> dict:
    """Run BACI distribution forecast simulations for all mean trend increments."""
    detection_results = dict(existing_results or {})
    n_trends = len(trend_increase_mu)
    n_workers = os.cpu_count() or 1

    for trend_idx, trend_mu in enumerate(trend_increase_mu, start=1):
        if trend_mu in detection_results:
            print(f"  [{trend_idx}/{n_trends}] trend_mu={trend_mu:.4f}  (cached, skipping)")
            continue

        t0 = time.perf_counter()
        print(f"  [{trend_idx}/{n_trends}] trend_mu={trend_mu:.4f}  ({simN} sims) ...",
              end="", flush=True)

        args_list = [
            (sim_idx, trend_mu, trend_idx, n_trends, npre, ntt, npost_max,
             mu, sigma, ns, delay_set, crit_val)
            for sim_idx in range(1, simN + 1)
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            raw = list(executor.map(_forecast_sim_worker_cdf_baci, args_list))

        res = _collect_results(raw, npre)
        print(f" done in {_fmt_elapsed(time.perf_counter() - t0)}  "
              f"detect={res['detection_rate']:.2%}  "
              f"err={res['mean_error']:.1f}mo", flush=True)

        detection_results[trend_mu] = res

        if on_trend_done:
            on_trend_done(detection_results)

    return detection_results


def run_simulation_cdf_ba(simN: int, trend_increase_mu, crit_val: float,
                          npre: int, ntt: int, npost_max: int,
                          mu: float, sigma: float, ns: int, delay_set,
                          existing_results=None, on_trend_done=None) -> dict:
    """Run BA distribution forecast simulations for all mean trend increments."""
    detection_results = dict(existing_results or {})
    n_trends = len(trend_increase_mu)
    n_workers = os.cpu_count() or 1

    for trend_idx, trend_mu in enumerate(trend_increase_mu, start=1):
        if trend_mu in detection_results:
            print(f"  [{trend_idx}/{n_trends}] trend_mu={trend_mu:.4f}  (cached, skipping)")
            continue

        t0 = time.perf_counter()
        print(f"  [{trend_idx}/{n_trends}] trend_mu={trend_mu:.4f}  ({simN} sims) ...",
              end="", flush=True)

        args_list = [
            (sim_idx, trend_mu, trend_idx, n_trends, npre, ntt, npost_max,
             mu, sigma, ns, delay_set, crit_val)
            for sim_idx in range(1, simN + 1)
        ]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            raw = list(executor.map(_forecast_sim_worker_cdf_ba, args_list))

        res = _collect_results(raw, npre)
        print(f" done in {_fmt_elapsed(time.perf_counter() - t0)}  "
              f"detect={res['detection_rate']:.2%}  "
              f"err={res['mean_error']:.1f}mo", flush=True)

        detection_results[trend_mu] = res

        if on_trend_done:
            on_trend_done(detection_results)

    return detection_results
