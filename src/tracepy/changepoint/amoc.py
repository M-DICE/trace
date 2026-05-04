"""
AMOC (At Most One Change) orchestration for trend and distribution changepoint detection.

Contains:
- Critical value table utilities (load_crit_val_table, lookup_crit_val)
- Trend change AMOC orchestration (from rewild_trend_change_amoc.py)
- Distribution change AMOC orchestration (from rewild_distribution_change_amoc.py)
"""

import json
import os
import time
import warnings
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from tracepy.simulation.trend import ci_sim, ci_sim_ar
from tracepy.stats.metrics import trend_stats, trend_stats_ar
from tracepy.simulation.distribution import ci_sim_cdf
from tracepy.stats.metrics import trend_stats_cdf, wasserstein_distance_baci, wasserstein_distance_ba, auc_diff_ts
from tracepy.params.manager import load_params


def _find_project_root():
    p = Path(__file__).resolve()
    while p != p.parent:
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    raise RuntimeError("Could not find project root")


_CRIT_VAL_TABLE_PATH = _find_project_root() / "data" / "CritValTable.json"


def load_crit_val_table(path=_CRIT_VAL_TABLE_PATH):
    with open(path) as f:
        return json.load(f)


def lookup_crit_val(table, detector="PageCUSUM", gamma=0.0, alpha=0.05):
    for row in table:
        if row["Detector"] == detector and row["Gamma"] == gamma and row["Alpha"] == alpha:
            return row["CritVal"]
    raise KeyError(f"No entry for detector={detector}, gamma={gamma}, alpha={alpha}")


# ============================================================================
# Module-level constants (from rewild_trend_change_amoc.py)
# ============================================================================

_cfg = load_params()

# R uses npost.vec[c(3, 7)] (1-based indices).
# npost.vec = seq(24, 120, by=12) so:
#   npost.vec[3] = 24 + (3-1)*12 = 48 months  (≈ 4 years post-intervention)
#   npost.vec[7] = 24 + (7-1)*12 = 96 months  (≈ 8 years post-intervention)
CRITICAL_VALUE_NPOST_SHORT = _cfg["stats"]["critical_value_npost_short"]
CRITICAL_VALUE_NPOST_LONG  = _cfg["stats"]["critical_value_npost_long"]


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


# ============================================================================
# Phase 1: Calculate Critical Values (Null Distribution) — Trend Change
# ============================================================================

def _null_sim_worker(args):
    """
    Run one null simulation (no trend change) and return the max test statistic.

    Top-level so it can be pickled by multiprocessing on macOS (spawn start method).
    """
    i, npre, npost, level, trend_control, sigma, phi, use_ar, ba = args
    warnings.filterwarnings('ignore')
    if use_ar:
        sim_data = ci_sim_ar(seed=i, npre=npre, npost=npost, level=level,
                             trend=[trend_control, trend_control], phi=phi, sigma=sigma)
        stats = trend_stats_ar(y_itv=sim_data['y_itv'], nt=npre + npost)  # BA: intervention only
    elif ba:
        sim_data = ci_sim(seed=i, npre=npre, npost=npost, level=level,
                          trend=[trend_control, trend_control], sigma=sigma)
        stats = trend_stats(y_itv=sim_data['y_itv'], nt=npre + npost)  # BA: intervention only
    else:
        sim_data = ci_sim(seed=i, npre=npre, npost=npost, level=level,
                          trend=[trend_control, trend_control], sigma=sigma)
        stats = trend_stats(y_ctr=sim_data['y_ctr'], y_itv=sim_data['y_itv'],
                            nt=npre + npost)
    return stats['Tmax']


def _main_sim_worker_iid_ba(args):
    """
    i.i.d. BA equivalent of _main_sim_worker.

    Matches R's simResult.v2 from AMOC.R: uses ci_sim (i.i.d. noise) but calls
    trend_stats without y_ctr (intervention series only, BA design).

    Top-level so it can be pickled by multiprocessing on macOS (spawn start method).

    Returns
    -------
    tuple:
        delay    : int              — months of delay applied to intervention effect
        tmax_vec : array(n_npost,)  — Tmax at each npost length in npost_vec
        cpt_vec  : array(n_npost,)  — detected changepoint at each npost length
        seed     : int              — simulation seed for reproducible on-demand regeneration
    """
    (sim_idx, trend_interv, trend_idx, n_trends, npre, npost_max, npost_vec,
     level, trend_control, sigma, delay_set) = args
    warnings.filterwarnings('ignore')

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))
    npre_delay = npre + delay
    npost_delay = npost_max - delay

    sim_data = ci_sim(seed=seed, npre=npre_delay, npost=npost_delay, level=level,
                      trend=[trend_control, trend_interv], sigma=sigma)

    n_npost = len(npost_vec)
    tmax_vec = np.zeros(n_npost)
    cpt_vec  = np.zeros(n_npost, dtype=int)

    for j, npost in enumerate(npost_vec):
        nt = npre + npost
        stats = trend_stats(
            y_itv=sim_data['y_itv'][:nt],  # BA: intervention only
            nt=nt,
        )
        tmax_vec[j] = stats['Tmax']
        cpt_vec[j]  = stats['cpt']

    return delay, tmax_vec, cpt_vec, seed


def _main_sim_worker(args):
    """
    Run one main simulation (with trend change) using a growing window approach.

    Generates one long time series at npost_max, then tests it at every npost
    length in npost_vec by truncating — matching R's inner sapply over npost.vec.

    Top-level so it can be pickled by multiprocessing on macOS (spawn start method).

    Delay is seeded deterministically from sim_idx and trend_idx so results are
    reproducible whether the loop runs sequentially or in parallel.

    Returns
    -------
    tuple:
        delay    : int              — months of delay applied to intervention effect
        tmax_vec : array(n_npost,)  — Tmax at each npost length in npost_vec
        cpt_vec  : array(n_npost,)  — detected changepoint at each npost length
        seed     : int              — simulation seed for reproducible on-demand regeneration
    """
    (sim_idx, trend_interv, trend_idx, n_trends, npre, npost_max, npost_vec,
     level, trend_control, sigma, delay_set) = args
    warnings.filterwarnings('ignore')

    # Seed delay sampling deterministically (same seed as ci_sim for reproducibility)
    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))
    npre_delay = npre + delay        # actual changepoint location in the series
    npost_delay = npost_max - delay  # remaining post-intervention samples

    # Generate the full time series once at npost_max
    sim_data = ci_sim(seed=seed, npre=npre_delay, npost=npost_delay, level=level,
                      trend=[trend_control, trend_interv], sigma=sigma)

    # Growing window: truncate to npre + npost for each npost in npost_vec,
    # matching R's: trend.stats(y.ctr=sim.ts$y.ctr[1:nt], ..., nt=npre+npost)
    n_npost = len(npost_vec)
    tmax_vec = np.zeros(n_npost)
    cpt_vec  = np.zeros(n_npost, dtype=int)

    for j, npost in enumerate(npost_vec):
        nt = npre + npost   # truncation length (uses original npre, not npre_delay)
        stats = trend_stats(
            y_ctr=sim_data['y_ctr'][:nt],
            y_itv=sim_data['y_itv'][:nt],
            nt=nt,
        )
        tmax_vec[j] = stats['Tmax']
        cpt_vec[j]  = stats['cpt']

    return delay, tmax_vec, cpt_vec, seed


def _main_sim_worker_ar(args):
    """
    AR(1) equivalent of _main_sim_worker.

    Uses ci_sim_ar (AR(1) noise generation) and trend_stats_ar (ARIMA(1,0,0)-based
    test statistic) instead of the i.i.d. variants. Everything else — growing window
    loop, delay seeding, return signature — is identical to _main_sim_worker.

    Top-level so it can be pickled by multiprocessing on macOS (spawn start method).

    Returns
    -------
    tuple:
        delay    : int              — months of delay applied to intervention effect
        tmax_vec : array(n_npost,)  — Tmax at each npost length in npost_vec
        cpt_vec  : array(n_npost,)  — detected changepoint at each npost length
        seed     : int              — simulation seed for reproducible on-demand regeneration
    """
    (sim_idx, trend_interv, trend_idx, n_trends, npre, npost_max, npost_vec,
     level, trend_control, sigma, phi, delay_set) = args
    warnings.filterwarnings('ignore')

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))
    npre_delay = npre + delay
    npost_delay = npost_max - delay

    # Generate full AR(1) time series once at npost_max
    sim_data = ci_sim_ar(seed=seed, npre=npre_delay, npost=npost_delay, level=level,
                         trend=[trend_control, trend_interv], phi=phi, sigma=sigma)

    # Growing window: truncate and apply AR(1)-aware test at each npost length
    n_npost = len(npost_vec)
    tmax_vec = np.zeros(n_npost)
    cpt_vec  = np.zeros(n_npost, dtype=int)

    for j, npost in enumerate(npost_vec):
        nt = npre + npost
        stats = trend_stats_ar(
            y_itv=sim_data['y_itv'][:nt],  # BA: intervention only, matching AMOC_AR1.R
            nt=nt,
        )
        tmax_vec[j] = stats['Tmax']
        cpt_vec[j]  = stats['cpt']

    return delay, tmax_vec, cpt_vec, seed


def _run_null_simulations(Nsim, npre, npost, level, trend_control, sigma, phi,
                          use_ar, label, ba=False):
    """
    Run Nsim null simulations (no trend change) and return the max test statistics.

    Parameters
    ----------
    use_ar : bool
        If True, use AR(1) noise (ci_sim_ar + trend_stats_ar).
        If False, use i.i.d. noise (ci_sim + trend_stats).
    label : str
        Human-readable label for progress printing.
    ba : bool
        If True and use_ar=False, use BA design (intervention series only).
        Ignored when use_ar=True (AR(1) always uses BA).
    """
    n_workers = os.cpu_count() or 1
    args_list = [
        (i, npre, npost, level, trend_control, sigma, phi, use_ar, ba)
        for i in range(1, Nsim + 1)
    ]
    t0 = time.perf_counter()
    print(f"  [{label}] {Nsim} sims ...", end="", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_null_sim_worker, args_list))
    print(f" done in {_fmt_elapsed(time.perf_counter() - t0)}", flush=True)
    return np.array(results)


def calculate_critical_values(Nsim, npre, level, trend_control, sigma, phi, alpha=0.95):
    """
    Compute critical values under the null hypothesis (no trend change) at two
    post-intervention lengths and for two noise types, matching R's getCritical.

    R's getCritical uses npost.vec[c(3, 7)] from seq(24, 120, by=12), which resolves to:
      - 48 months  (CRITICAL_VALUE_NPOST_SHORT)
      - 96 months  (CRITICAL_VALUE_NPOST_LONG)

    For each npost length, both i.i.d. and AR(1) critical values are calculated.

    Parameters
    ----------
    Nsim : int
        Number of null simulations per combination
    npre : int
        Pre-intervention period (months)
    level : float
        Base level of time series
    trend_control : float
        Control trend (same before and after under null)
    sigma : float
        Noise level
    phi : float
        AR(1) coefficient (used only for AR noise simulations)
    alpha : float
        Significance level (default 0.95 = 95th percentile)

    Returns
    -------
    critical_values : dict
        Keys: 'iid_48', 'iid_96', 'ar1_48', 'ar1_96', 'iid_ba_48', 'iid_ba_96'
        Each value is a dict with 'critical_value' (float) and 'null_dist' (array)
    """
    critical_values = {}

    for npost, tag in [(CRITICAL_VALUE_NPOST_SHORT, str(CRITICAL_VALUE_NPOST_SHORT)),
                       (CRITICAL_VALUE_NPOST_LONG,  str(CRITICAL_VALUE_NPOST_LONG))]:
        print(f"  npost = {npost} months:")

        null_iid = _run_null_simulations(Nsim, npre, npost, level, trend_control,
                                         sigma, phi, use_ar=False,
                                         label=f"i.i.d. BACI, {npost}mo")
        cv_iid = np.percentile(null_iid, alpha * 100)
        critical_values[f'iid_{tag}'] = {'critical_value': cv_iid, 'null_dist': null_iid}
        print(f"    i.i.d. BACI cv = {cv_iid:.4f}")

        null_iid_ba = _run_null_simulations(Nsim, npre, npost, level, trend_control,
                                            sigma, phi, use_ar=False, ba=True,
                                            label=f"i.i.d. BA, {npost}mo")
        cv_iid_ba = np.percentile(null_iid_ba, alpha * 100)
        critical_values[f'iid_ba_{tag}'] = {'critical_value': cv_iid_ba, 'null_dist': null_iid_ba}
        print(f"    i.i.d. BA   cv = {cv_iid_ba:.4f}")

        null_ar = _run_null_simulations(Nsim, npre, npost, level, trend_control,
                                        sigma, phi, use_ar=True,
                                        label=f"AR(1), {npost}mo")
        cv_ar = np.percentile(null_ar, alpha * 100)
        critical_values[f'ar1_{tag}'] = {'critical_value': cv_ar, 'null_dist': null_ar}
        print(f"    AR(1)       cv = {cv_ar:.4f}")

    return critical_values


# ============================================================================
# Phase 2: Main Simulation (Alternative Hypothesis) — Trend Change
# ============================================================================

def run_main_simulation(simN, trend_increase, critical_value, npre, npost_max,
                        npost_vec, level, trend_control, sigma, delay_set,
                        existing_results=None, on_trend_done=None):
    """
    Run main simulation (i.i.d. BACI) with intervention effect and calculate detection rates.
    """
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
            (sim_idx, trend_interv, trend_idx, n_trends, npre, npost_max, npost_vec,
             level, trend_control, sigma, delay_set)
            for sim_idx in range(1, simN + 1)
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_main_sim_worker, args_list))

        delays       = [r[0] for r in results]
        tmax_matrix  = np.array([r[1] for r in results])
        cpt_matrix   = np.array([r[2] for r in results])
        seeds        = [r[3] for r in results]

        detected_matrix = tmax_matrix > critical_value
        detection_rates = detected_matrix.mean(axis=0)

        true_cpts    = np.array([npre + d for d in delays])
        error_matrix = np.abs(cpt_matrix - true_cpts[:, np.newaxis])
        mean_errors  = error_matrix.mean(axis=0)

        print(f" done in {_fmt_elapsed(time.perf_counter() - t0)}  "
              f"detect={float(detection_rates[-1]):.2%}  "
              f"err={float(mean_errors[-1]):.1f}mo", flush=True)

        detection_results[trend_inc] = {
            'tmax_matrix':     tmax_matrix,
            'cpt_matrix':      cpt_matrix,
            'detected_matrix': detected_matrix,
            'delays':          delays,
            'detection_rates': detection_rates,
            'mean_errors':     mean_errors,
            'seeds':           seeds,
        }

        if on_trend_done:
            on_trend_done(detection_results)

    return detection_results


def run_main_simulation_iid_ba(simN, trend_increase, critical_value, npre, npost_max,
                               npost_vec, level, trend_control, sigma, delay_set,
                               existing_results=None, on_trend_done=None):
    """
    i.i.d. BA equivalent of run_main_simulation().

    Dispatches to _main_sim_worker_iid_ba, which uses i.i.d. noise but tests the
    intervention series only (BA design), matching R's simResult.v2 from AMOC.R.
    The i.i.d. BA critical value (iid_ba_48) should be passed.
    """
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
            (sim_idx, trend_interv, trend_idx, n_trends, npre, npost_max, npost_vec,
             level, trend_control, sigma, delay_set)
            for sim_idx in range(1, simN + 1)
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_main_sim_worker_iid_ba, args_list))

        delays      = [r[0] for r in results]
        tmax_matrix = np.array([r[1] for r in results])
        cpt_matrix  = np.array([r[2] for r in results])
        seeds       = [r[3] for r in results]

        detected_matrix = tmax_matrix > critical_value
        detection_rates = detected_matrix.mean(axis=0)

        true_cpts    = np.array([npre + d for d in delays])
        error_matrix = np.abs(cpt_matrix - true_cpts[:, np.newaxis])
        mean_errors  = error_matrix.mean(axis=0)

        print(f" done in {_fmt_elapsed(time.perf_counter() - t0)}  "
              f"detect={float(detection_rates[-1]):.2%}  "
              f"err={float(mean_errors[-1]):.1f}mo", flush=True)

        detection_results[trend_inc] = {
            'tmax_matrix':     tmax_matrix,
            'cpt_matrix':      cpt_matrix,
            'detected_matrix': detected_matrix,
            'delays':          delays,
            'detection_rates': detection_rates,
            'mean_errors':     mean_errors,
            'seeds':           seeds,
        }

        if on_trend_done:
            on_trend_done(detection_results)

    return detection_results


def run_main_simulation_ar(simN, trend_increase, critical_value, npre, npost_max,
                           npost_vec, level, trend_control, sigma, phi, delay_set,
                           existing_results=None, on_trend_done=None):
    """
    AR(1) equivalent of run_main_simulation().

    Dispatches to _main_sim_worker_ar, which generates time series with AR(1)
    noise and applies the ARIMA(1,0,0)-based test statistic at every npost length.
    The AR(1) critical value (ar1_48) should be passed so detection is calibrated
    to the autocorrelated null distribution.
    """
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
            (sim_idx, trend_interv, trend_idx, n_trends, npre, npost_max, npost_vec,
             level, trend_control, sigma, phi, delay_set)
            for sim_idx in range(1, simN + 1)
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_main_sim_worker_ar, args_list))

        delays      = [r[0] for r in results]
        tmax_matrix = np.array([r[1] for r in results])
        cpt_matrix  = np.array([r[2] for r in results])
        seeds       = [r[3] for r in results]

        detected_matrix = tmax_matrix > critical_value
        detection_rates = detected_matrix.mean(axis=0)

        true_cpts    = np.array([npre + d for d in delays])
        error_matrix = np.abs(cpt_matrix - true_cpts[:, np.newaxis])
        mean_errors  = error_matrix.mean(axis=0)

        print(f" done in {_fmt_elapsed(time.perf_counter() - t0)}  "
              f"detect={float(detection_rates[-1]):.2%}  "
              f"err={float(mean_errors[-1]):.1f}mo", flush=True)

        detection_results[trend_inc] = {
            'tmax_matrix':     tmax_matrix,
            'cpt_matrix':      cpt_matrix,
            'detected_matrix': detected_matrix,
            'delays':          delays,
            'detection_rates': detection_rates,
            'mean_errors':     mean_errors,
            'seeds':           seeds,
        }

        if on_trend_done:
            on_trend_done(detection_results)

    return detection_results


# ============================================================================
# Distribution Change AMOC (from rewild_distribution_change_amoc.py)
# ============================================================================

# Critical value npost lengths (R uses [48, 84] — npost.vec[c(3, 7)])
# DEVIATION: Python uses [24, 72, 120] to cover full range at step=3
CRITICAL_VALUE_NPOST_SHORT_CDF  = _cfg["distribution"]["critical_value_npost_short"]
CRITICAL_VALUE_NPOST_MEDIUM_CDF = _cfg["distribution"]["critical_value_npost_medium"]
CRITICAL_VALUE_NPOST_LONG_CDF   = _cfg["distribution"]["critical_value_npost_long"]


def _null_sim_worker_cdf(args):
    i, npre, npost, mu, sigma, ns, dist_measure, bw, nd = args
    sim = ci_sim_cdf(seed=i, npre=npre, npost=npost,
                     level=[mu, sigma], trend=[0, 0], ns=ns)
    if dist_measure == "wasserstein":
        dist_ts = wasserstein_distance_baci(sim['sample_ctr'], sim['sample_itv'])
    else:
        dist_ts = auc_diff_ts(sim['sample_ctr'], sim['sample_itv'], bw=bw, nd=nd)
    stats = trend_stats_cdf(dist_ts, nt=npre + npost)
    return stats['Tmax']


def _main_sim_worker_cdf_mu(args):
    (sim_idx, trend_mu, trend_idx, n_trends, npre, npost_max, npost_vec,
     mu, sigma, ns, dist_measure, bw, nd, delay_set) = args

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))
    npre_delay = npre + delay
    npost_delay = npost_max - delay

    sim = ci_sim_cdf(seed=seed, npre=npre_delay, npost=npost_delay,
                     level=[mu, sigma], trend=[trend_mu, 0], ns=ns)

    tmax_vec = np.zeros(len(npost_vec))
    cpt_vec  = np.zeros(len(npost_vec), dtype=int)
    for j, npost in enumerate(npost_vec):
        nt = npre + npost
        if dist_measure == "wasserstein":
            dist_ts = wasserstein_distance_baci(sim['sample_ctr'][:, :nt], sim['sample_itv'][:, :nt])
        else:
            dist_ts = auc_diff_ts(sim['sample_ctr'][:, :nt], sim['sample_itv'][:, :nt], bw=bw, nd=nd)
        stats = trend_stats_cdf(dist_ts, nt=nt)
        tmax_vec[j] = stats['Tmax']
        cpt_vec[j]  = stats['cpt']

    return delay, tmax_vec, cpt_vec, seed


def _main_sim_worker_cdf_mu_ba(args):
    (sim_idx, trend_mu, trend_idx, n_trends, npre, npost_max, npost_vec,
     mu, sigma, ns, dist_measure, bw, nd, delay_set) = args

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))
    npre_delay = npre + delay
    npost_delay = npost_max - delay

    sim = ci_sim_cdf(seed=seed, npre=npre_delay, npost=npost_delay,
                     level=[mu, sigma], trend=[trend_mu, 0], ns=ns)

    tmax_vec = np.zeros(len(npost_vec))
    cpt_vec  = np.zeros(len(npost_vec), dtype=int)
    for j, npost in enumerate(npost_vec):
        nt = npre + npost
        # BA uses nominal npre (24) for baseline
        if dist_measure == "wasserstein":
            dist_ts = wasserstein_distance_ba(sim['sample_itv'][:, :nt], npre)
        else:
            # Fallback to BA Wasserstein
            dist_ts = wasserstein_distance_ba(sim['sample_itv'][:, :nt], npre)
        stats = trend_stats_cdf(dist_ts, nt=nt)
        tmax_vec[j] = stats['Tmax']
        cpt_vec[j]  = stats['cpt']

    return delay, tmax_vec, cpt_vec, seed


def _main_sim_worker_cdf_sigma(args):
    (sim_idx, trend_sigma, trend_idx, n_trends, npre, npost_max, npost_vec,
     mu, sigma, ns, dist_measure, bw, nd, delay_set) = args

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))
    npre_delay = npre + delay
    npost_delay = npost_max - delay

    sim = ci_sim_cdf(seed=seed, npre=npre_delay, npost=npost_delay,
                     level=[mu, sigma], trend=[0, trend_sigma], ns=ns)

    tmax_vec = np.zeros(len(npost_vec))
    cpt_vec  = np.zeros(len(npost_vec), dtype=int)
    for j, npost in enumerate(npost_vec):
        nt = npre + npost
        if dist_measure == "wasserstein":
            dist_ts = wasserstein_distance_baci(sim['sample_ctr'][:, :nt], sim['sample_itv'][:, :nt])
        else:
            dist_ts = auc_diff_ts(sim['sample_ctr'][:, :nt], sim['sample_itv'][:, :nt], bw=bw, nd=nd)
        stats = trend_stats_cdf(dist_ts, nt=nt)
        tmax_vec[j] = stats['Tmax']
        cpt_vec[j]  = stats['cpt']

    return delay, tmax_vec, cpt_vec, seed


def calculate_critical_values_cdf(Nsim, npre, mu, sigma, ns, dist_measure, bw, nd, alpha=0.95):
    critical_values = {}
    n_workers = os.cpu_count() or 1

    for npost, label in [(CRITICAL_VALUE_NPOST_SHORT_CDF, "24"),
                         (CRITICAL_VALUE_NPOST_MEDIUM_CDF, "72"),
                         (CRITICAL_VALUE_NPOST_LONG_CDF, "120")]:
        t0 = time.perf_counter()
        print(f"  npost = {npost} months ({Nsim} null sims) ...", end="", flush=True)
        args_list = [(i, npre, npost, mu, sigma, ns, dist_measure, bw, nd) for i in range(1, Nsim + 1)]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_null_sim_worker_cdf, args_list))

        cv = np.percentile(results, alpha * 100)
        critical_values[f'npost_{label}'] = {'critical_value': cv, 'null_dist': np.array(results)}
        print(f" done in {_fmt_elapsed(time.perf_counter() - t0)}  cv = {cv:.4f}")

    return critical_values


def run_main_simulation_mu(simN, trend_increase_mu, critical_value, npre, npost_max, npost_vec,
                           mu, sigma, ns, dist_measure, bw, nd, delay_set, ba=False,
                           existing_results=None, on_trend_done=None):
    detection_results = dict(existing_results or {})
    n_trends = len(trend_increase_mu)
    n_workers = os.cpu_count() or 1
    worker_func = _main_sim_worker_cdf_mu_ba if ba else _main_sim_worker_cdf_mu

    for trend_idx, trend_mu in enumerate(trend_increase_mu, start=1):
        if trend_mu in detection_results:
            print(f"  [{trend_idx}/{n_trends}] trend_mu={trend_mu:.4f}  (cached, skipping)")
            continue

        t0 = time.perf_counter()
        print(f"  [{trend_idx}/{n_trends}] trend_mu={trend_mu:.4f}  ({simN} sims) ...", end="", flush=True)
        args_list = [(sim_idx, trend_mu, trend_idx, n_trends, npre, npost_max, npost_vec,
                      mu, sigma, ns, dist_measure, bw, nd, delay_set)
                     for sim_idx in range(1, simN + 1)]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(worker_func, args_list))

        delays = [r[0] for r in results]
        tmax_matrix = np.array([r[1] for r in results])
        cpt_matrix = np.array([r[2] for r in results])
        seeds = [r[3] for r in results]

        detected_matrix = tmax_matrix > critical_value
        detection_rates = detected_matrix.mean(axis=0)

        true_cpts = np.array([npre + d for d in delays])
        error_matrix = np.abs(cpt_matrix - true_cpts[:, np.newaxis])
        mean_errors = error_matrix.mean(axis=0)

        detection_results[trend_mu] = {
            'tmax_matrix': tmax_matrix,
            'cpt_matrix': cpt_matrix,
            'detected_matrix': detected_matrix,
            'delays': delays,
            'detection_rates': detection_rates,
            'mean_errors': mean_errors,
            'seeds': seeds,
        }
        print(f" done in {_fmt_elapsed(time.perf_counter() - t0)}")

        if on_trend_done:
            on_trend_done(detection_results)

    return detection_results


def run_main_simulation_sigma(simN, trend_increase_sigma, critical_value, npre, npost_max, npost_vec,
                              mu, sigma, ns, dist_measure, bw, nd, delay_set,
                              existing_results=None, on_trend_done=None):
    detection_results = dict(existing_results or {})
    n_trends = len(trend_increase_sigma)
    n_workers = os.cpu_count() or 1

    for trend_idx, trend_sigma in enumerate(trend_increase_sigma, start=1):
        if trend_sigma in detection_results:
            print(f"  [{trend_idx}/{n_trends}] trend_sigma={trend_sigma:.4f}  (cached, skipping)")
            continue

        t0 = time.perf_counter()
        print(f"  [{trend_idx}/{n_trends}] trend_sigma={trend_sigma:.4f}  ({simN} sims) ...", end="", flush=True)
        args_list = [(sim_idx, trend_sigma, trend_idx, n_trends, npre, npost_max, npost_vec,
                      mu, sigma, ns, dist_measure, bw, nd, delay_set)
                     for sim_idx in range(1, simN + 1)]

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_main_sim_worker_cdf_sigma, args_list))

        delays = [r[0] for r in results]
        tmax_matrix = np.array([r[1] for r in results])
        cpt_matrix = np.array([r[2] for r in results])
        seeds = [r[3] for r in results]

        detected_matrix = tmax_matrix > critical_value
        detection_rates = detected_matrix.mean(axis=0)

        true_cpts = np.array([npre + d for d in delays])
        error_matrix = np.abs(cpt_matrix - true_cpts[:, np.newaxis])
        mean_errors = error_matrix.mean(axis=0)

        detection_results[trend_sigma] = {
            'tmax_matrix': tmax_matrix,
            'cpt_matrix': cpt_matrix,
            'detected_matrix': detected_matrix,
            'delays': delays,
            'detection_rates': detection_rates,
            'mean_errors': mean_errors,
            'seeds': seeds,
        }
        print(f" done in {_fmt_elapsed(time.perf_counter() - t0)}")

        if on_trend_done:
            on_trend_done(detection_results)

    return detection_results
