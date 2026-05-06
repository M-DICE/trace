"""
Parallel simulation workers and Monte Carlo orchestration.

This module centralises all simulation workers and the shared loop used by
both the AMOC and Forecast changepoint pipelines.  Keeping workers here
avoids duplicating the ProcessPoolExecutor boilerplate and makes each
worker independently testable.

Public helpers
--------------
_run_parallel      : low-level parallel dispatch with optional timing output.
_simulation_loop   : Monte Carlo loop over an effect-size grid.

Trend AMOC workers
------------------
_null_sim_worker   : one null (no-change) simulation for the trend pipeline.
_main_sim_worker   : one alternative (trend-change) simulation; supports both
                     BACI and BA designs via a single ``ba`` flag.
_main_sim_worker_ar: AR(1) variant of _main_sim_worker (BA only).

CDF AMOC workers
----------------
_null_sim_worker_cdf : one null simulation for the distribution pipeline.
_main_sim_worker_cdf : one alternative simulation for distribution shifts;
                       supports mu-shift, sigma-shift, BACI, and BA in a
                       single worker controlled by its argument tuple.
"""

import os
import time
import warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from tracepy.simulation.trend import ci_sim, ci_sim_ar
from tracepy.simulation.distribution import ci_sim_cdf
from tracepy.stats.metrics import (
    trend_stats,
    trend_stats_ar,
    trend_stats_cdf,
    wasserstein_distance_baci,
    wasserstein_distance_ba,
    auc_diff_ts,
)
from tracepy.simulation.utils import _fmt_elapsed


# ============================================================================
# Low-level parallel dispatch
# ============================================================================

def _run_parallel(worker_fn, args_list, label=None):
    """
    Dispatch *worker_fn* over *args_list* using all available CPUs.

    Parameters
    ----------
    worker_fn : callable
        Top-level picklable function that receives one element of *args_list*.
    args_list : list[tuple]
        Argument tuples passed one-per-call to *worker_fn*.
    label : str or None
        If provided, print ``  [label] N sims ...`` before and timing after
        on the same line.

    Returns
    -------
    list
        Results in the same order as *args_list*.
    """
    n_workers = os.cpu_count() or 1
    if label:
        print(f"  [{label}] {len(args_list)} sims ...", end="", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(worker_fn, args_list))
    if label:
        print(f" done in {_fmt_elapsed(time.perf_counter() - t0)}", flush=True)
    return results


# ============================================================================
# Shared Monte Carlo loop
# ============================================================================

def _simulation_loop(
    worker_fn,
    make_args,
    trend_values,
    label_key,
    simN,
    aggregate_fn,
    fmt_progress=None,
    existing_results=None,
    on_trend_done=None,
):
    """
    Core Monte Carlo loop over an ordered grid of effect-size values.

    For each value in *trend_values* that is not already cached, the loop:

    1. Builds per-simulation args via ``make_args(trend_val, trend_idx, n_trends)``.
    2. Dispatches *worker_fn* over all *simN* arg tuples in parallel.
    3. Aggregates raw results via ``aggregate_fn(raw)``.
    4. Prints elapsed time with an optional progress suffix from
       ``fmt_progress(result)`` (omitted when *fmt_progress* is None).
    5. Calls ``on_trend_done(results)`` when provided (e.g. to checkpoint).

    Parameters
    ----------
    worker_fn : callable
        Top-level picklable worker.  Must be defined at module level so it
        can be pickled by child processes (macOS uses the ``spawn`` start method).
    make_args : callable
        Signature ``(trend_val, trend_idx, n_trends) -> list[tuple]`` of
        length *simN*.  Runs in the main process, so closures are fine.
    trend_values : sequence
        Ordered effect-size values (e.g. trend increments) to loop over.
    label_key : str
        Variable name shown in progress output (e.g. ``'trend_inc'``,
        ``'trend_mu'``).
    simN : int
        Number of Monte Carlo replications per effect-size value.
    aggregate_fn : callable
        ``raw -> dict`` — maps the flat list of per-simulation return tuples
        to a result dict stored under ``detection_results[trend_val]``.
    fmt_progress : callable or None
        ``result -> str`` — returns a suffix appended to the "done in …"
        progress line (e.g. ``"detect=80.00%  err=3.2mo"``).
        Pass None for no extra output.
    existing_results : dict or None
        Pre-computed results; matching keys are skipped (used as cache).
    on_trend_done : callable or None
        Called with the full results dict after each effect-size completes.

    Returns
    -------
    dict
        Mapping *trend_val* → result dict for every value in *trend_values*.
    """
    detection_results = dict(existing_results or {})
    n_trends = len(trend_values)
    n_workers = os.cpu_count() or 1

    for trend_idx, trend_val in enumerate(trend_values, start=1):
        if trend_val in detection_results:
            print(f"  [{trend_idx}/{n_trends}] {label_key}={trend_val:.4f}  (cached, skipping)")
            continue

        t0 = time.perf_counter()
        print(f"  [{trend_idx}/{n_trends}] {label_key}={trend_val:.4f}  ({simN} sims) ...",
              end="", flush=True)

        args_list = make_args(trend_val, trend_idx, n_trends)

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            raw = list(executor.map(worker_fn, args_list))

        result = aggregate_fn(raw)
        suffix = f"  {fmt_progress(result)}" if fmt_progress else ""
        print(f" done in {_fmt_elapsed(time.perf_counter() - t0)}{suffix}", flush=True)

        detection_results[trend_val] = result
        if on_trend_done:
            on_trend_done(detection_results)

    return detection_results


# ============================================================================
# Trend AMOC workers
# ============================================================================

def _null_sim_worker(args):
    """
    Run one null (no trend change) simulation and return the max test statistic.

    Supports i.i.d. and AR(1) noise, and both BACI and BA designs, so it
    serves all three null-distribution estimation calls in the trend pipeline.

    Top-level so it can be pickled by multiprocessing on macOS (spawn start method).

    Parameters
    ----------
    args : tuple
        (i, npre, npost, level, trend_control, sigma, phi, use_ar, ba)

        i             : int   — simulation index (used as random seed)
        use_ar        : bool  — True → AR(1) noise via ci_sim_ar + trend_stats_ar;
                                False → i.i.d. noise via ci_sim + trend_stats
        ba            : bool  — True → BA design (intervention only, y_ctr unused);
                                False → BACI design (both series passed to trend_stats).
                                Ignored when use_ar=True (AR path is always BA).

    Returns
    -------
    float
        Maximum CUSUM test statistic Tmax over the simulated null series.
    """
    i, npre, npost, level, trend_control, sigma, phi, use_ar, ba = args
    warnings.filterwarnings('ignore')

    if use_ar:
        sim_data = ci_sim_ar(seed=i, npre=npre, npost=npost, level=level,
                             trend=[trend_control, trend_control], phi=phi, sigma=sigma)
        stats = trend_stats_ar(y_itv=sim_data['y_itv'], nt=npre + npost)
    elif ba:
        sim_data = ci_sim(seed=i, npre=npre, npost=npost, level=level,
                          trend=[trend_control, trend_control], sigma=sigma)
        stats = trend_stats(y_itv=sim_data['y_itv'], nt=npre + npost)
    else:
        sim_data = ci_sim(seed=i, npre=npre, npost=npost, level=level,
                          trend=[trend_control, trend_control], sigma=sigma)
        stats = trend_stats(y_ctr=sim_data['y_ctr'], y_itv=sim_data['y_itv'],
                            nt=npre + npost)

    return stats['Tmax']


def _main_sim_worker(args):
    """
    Run one alternative (trend-change) simulation using a growing window.

    Generates a single long time series at *npost_max*, then tests it at
    every npost length in *npost_vec* by truncating — matching R's inner
    sapply over npost.vec.  A single ``ba`` flag selects between BACI (control
    and intervention) and BA (intervention only) designs, so this worker
    replaces the former ``_main_sim_worker`` and ``_main_sim_worker_iid_ba``.

    Delay is seeded deterministically from *sim_idx* and *trend_idx* so
    results are reproducible whether the loop runs sequentially or in parallel.

    Top-level so it can be pickled by multiprocessing on macOS (spawn start method).

    Parameters
    ----------
    args : tuple
        (sim_idx, trend_interv, trend_idx, n_trends, npre, npost_max,
         npost_vec, level, trend_control, sigma, delay_set, ba)

        trend_interv : float — post-intervention trend (trend_control + trend_inc)
        ba           : bool  — True → BA design (y_itv only); False → BACI design

    Returns
    -------
    tuple
        (delay, tmax_vec, cpt_vec, seed)

        delay    : int               — months of delay applied to the intervention effect
        tmax_vec : ndarray(n_npost,) — Tmax at each npost length in npost_vec
        cpt_vec  : ndarray(n_npost,) — detected changepoint at each npost length
        seed     : int               — simulation seed for on-demand regeneration
    """
    (sim_idx, trend_interv, trend_idx, n_trends, npre, npost_max, npost_vec,
     level, trend_control, sigma, delay_set, ba) = args
    warnings.filterwarnings('ignore')

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))
    npre_delay  = npre + delay
    npost_delay = npost_max - delay

    sim_data = ci_sim(seed=seed, npre=npre_delay, npost=npost_delay, level=level,
                      trend=[trend_control, trend_interv], sigma=sigma)

    n_npost  = len(npost_vec)
    tmax_vec = np.zeros(n_npost)
    cpt_vec  = np.zeros(n_npost, dtype=int)

    for j, npost in enumerate(npost_vec):
        nt = npre + npost  # truncation length (uses original npre, not npre_delay)
        if ba:
            stats = trend_stats(y_itv=sim_data['y_itv'][:nt], nt=nt)
        else:
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
    AR(1) variant of _main_sim_worker (BA design only).

    Uses ci_sim_ar (AR(1) noise generation) and trend_stats_ar (ARIMA(1,0,0)-based
    test statistic) instead of the i.i.d. variants.  Delay seeding, growing-window
    loop, and return signature are identical to _main_sim_worker.

    Always uses BA design (intervention series only), matching AMOC_AR1.R.

    Top-level so it can be pickled by multiprocessing on macOS (spawn start method).

    Parameters
    ----------
    args : tuple
        (sim_idx, trend_interv, trend_idx, n_trends, npre, npost_max,
         npost_vec, level, trend_control, sigma, phi, delay_set)

        phi : float — AR(1) autocorrelation coefficient

    Returns
    -------
    tuple
        (delay, tmax_vec, cpt_vec, seed) — same shape as _main_sim_worker.
    """
    (sim_idx, trend_interv, trend_idx, n_trends, npre, npost_max, npost_vec,
     level, trend_control, sigma, phi, delay_set) = args
    warnings.filterwarnings('ignore')

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))
    npre_delay  = npre + delay
    npost_delay = npost_max - delay

    sim_data = ci_sim_ar(seed=seed, npre=npre_delay, npost=npost_delay, level=level,
                         trend=[trend_control, trend_interv], phi=phi, sigma=sigma)

    n_npost  = len(npost_vec)
    tmax_vec = np.zeros(n_npost)
    cpt_vec  = np.zeros(n_npost, dtype=int)

    for j, npost in enumerate(npost_vec):
        nt = npre + npost
        stats = trend_stats_ar(y_itv=sim_data['y_itv'][:nt], nt=nt)
        tmax_vec[j] = stats['Tmax']
        cpt_vec[j]  = stats['cpt']

    return delay, tmax_vec, cpt_vec, seed


# ============================================================================
# CDF AMOC workers
# ============================================================================

def _null_sim_worker_cdf(args):
    """
    Run one null (no distribution change) simulation and return Tmax.

    Top-level so it can be pickled by multiprocessing on macOS (spawn start method).

    Parameters
    ----------
    args : tuple
        (i, npre, npost, mu, sigma, ns, dist_measure, bw, nd)

        i            : int   — simulation index (used as random seed)
        mu, sigma    : float — baseline distribution parameters
        ns           : int   — samples per time point
        dist_measure : str   — ``'wasserstein'`` or ``'auc_diff'``
        bw           : float — bandwidth for auc_diff_ts (unused for wasserstein)
        nd           : int   — density grid points for auc_diff_ts (unused for wasserstein)

    Returns
    -------
    float
        Maximum CUSUM test statistic Tmax over the simulated null series.
    """
    i, npre, npost, mu, sigma, ns, dist_measure, bw, nd = args
    warnings.filterwarnings('ignore')

    sim = ci_sim_cdf(seed=i, npre=npre, npost=npost,
                     level=[mu, sigma], trend=[0, 0], ns=ns)
    if dist_measure == "wasserstein":
        dist_ts = wasserstein_distance_baci(sim['sample_ctr'], sim['sample_itv'])
    else:
        dist_ts = auc_diff_ts(sim['sample_ctr'], sim['sample_itv'], bw=bw, nd=nd)

    stats = trend_stats_cdf(dist_ts, nt=npre + npost)
    return stats['Tmax']


def _main_sim_worker_cdf(args):
    """
    CDF AMOC worker for distribution-shift detection (mu and/or sigma).

    A single worker that replaces the three former ``_main_sim_worker_cdf_mu``,
    ``_main_sim_worker_cdf_mu_ba``, and ``_main_sim_worker_cdf_sigma`` functions.
    The combination of *trend_mu*, *trend_sigma*, and *ba* selects the
    experimental condition:

    - mu-shift BACI:  ``trend_mu != 0, trend_sigma=0.0, ba=False``
    - mu-shift BA:    ``trend_mu != 0, trend_sigma=0.0, ba=True``
    - sigma-shift:    ``trend_mu=0.0,  trend_sigma != 0, ba=False``

    Generates one long time series at *npost_max*, then tests it at every npost
    length in *npost_vec* via a growing window — matching R's inner sapply over
    npost.vec.

    Top-level so it can be pickled by multiprocessing on macOS (spawn start method).

    Parameters
    ----------
    args : tuple
        (sim_idx, trend_mu, trend_sigma, ba, trend_idx, n_trends,
         npre, npost_max, npost_vec, mu, sigma, ns,
         dist_measure, bw, nd, delay_set)

        trend_mu     : float — post-intervention shift in mean (pass 0.0 for sigma-only)
        trend_sigma  : float — post-intervention shift in spread (pass 0.0 for mu-only)
        ba           : bool  — True → BA design (wasserstein_distance_ba on intervention
                               series only); False → BACI design (wasserstein_distance_baci
                               or auc_diff_ts on both series, selected by dist_measure)
        dist_measure : str   — ``'wasserstein'`` or ``'auc_diff'`` (ignored when ba=True)
        bw           : float — bandwidth for auc_diff_ts (ignored when ba=True)
        nd           : int   — density grid points for auc_diff_ts (ignored when ba=True)

    Returns
    -------
    tuple
        (delay, tmax_vec, cpt_vec, seed)

        delay    : int               — months of delay applied to the intervention effect
        tmax_vec : ndarray(n_npost,) — Tmax at each npost length in npost_vec
        cpt_vec  : ndarray(n_npost,) — detected changepoint at each npost length
        seed     : int               — simulation seed for on-demand regeneration
    """
    (sim_idx, trend_mu, trend_sigma, ba, trend_idx, n_trends,
     npre, npost_max, npost_vec, mu, sigma, ns,
     dist_measure, bw, nd, delay_set) = args
    warnings.filterwarnings('ignore')

    seed = sim_idx * n_trends + trend_idx
    rng = np.random.default_rng(seed)
    delay = int(rng.choice(delay_set))
    npre_delay  = npre + delay
    npost_delay = npost_max - delay

    sim = ci_sim_cdf(seed=seed, npre=npre_delay, npost=npost_delay,
                     level=[mu, sigma], trend=[trend_mu, trend_sigma], ns=ns)

    n_npost  = len(npost_vec)
    tmax_vec = np.zeros(n_npost)
    cpt_vec  = np.zeros(n_npost, dtype=int)

    for j, npost in enumerate(npost_vec):
        nt = npre + npost
        if ba:
            dist_ts = wasserstein_distance_ba(sim['sample_itv'][:, :nt], npre)
        elif dist_measure == "wasserstein":
            dist_ts = wasserstein_distance_baci(sim['sample_ctr'][:, :nt],
                                                sim['sample_itv'][:, :nt])
        else:
            dist_ts = auc_diff_ts(sim['sample_ctr'][:, :nt],
                                  sim['sample_itv'][:, :nt], bw=bw, nd=nd)

        stats = trend_stats_cdf(dist_ts, nt=nt)
        tmax_vec[j] = stats['Tmax']
        cpt_vec[j]  = stats['cpt']

    return delay, tmax_vec, cpt_vec, seed
