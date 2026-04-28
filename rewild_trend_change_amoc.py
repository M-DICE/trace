"""
Trend Change Detection using AMOC (At Most One Change)

This script demonstrates changepoint detection for univariate time series
with trend changes using the AMOC approach (offline batch method).

Equivalent to: Rewild_trend_change_AMOC.R
"""

import argparse
import numpy as np
import warnings
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import file_utils
from amoc import ci_sim, ci_sim_ar, trend_stats, trend_stats_ar
from plotting import (plot_time_series, plot_simulation_results, plot_power_curves,
                      plot_power_curves_comparison, plot_time_to_detection,
                      plot_detection_heatmap, plot_mean_error_curves,
                      plot_null_distributions, plot_tmax_signal_vs_noise,
                      plot_detection_by_delay, plot_changepoint_bias)

warnings.filterwarnings('ignore')


# ============================================================================
# Simulation Parameters
# ============================================================================
if __name__ == "__main__":
    print("Initialising parameters for simulation")

# Time series structure

## Pre-intervention
npre = 24                                    # 24 months, 2 years before intervention
## Post-intervention
npost_years=10
npost_months=12*npost_years
npost_vec = np.arange(npre, npost_months + 1, 12) # Annual steps (12 months at a time)

npost_max = npost_vec[-1]                   # Maximum post-intervention period
ntt = npre + npost_max                      # Total time series length

# Level and trend parameters
level = 10
trend_control = 0.005

# Trend increment as percentage change to mean before intervention
trend_increase = np.round(
    level * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / npost_months,
    4
)

# Noise and autocorrelation
sigma = 0.05                                # Noise level
phi = 0.5                                   # Auto-correlation strength (AR(1) coefficient)

# Delay in effect, up to 20 time points
delay_set = np.arange(1, 21)

## Simulation to get the critical values of the trend difference test
alpha = 0.95   # significance level
Nsim = 1000 # Simulate n-datasets where NO change occurs
simN = 1000 # Simulate n-datasets WITH intervention effect

# ---- Critical value npost lengths ----
# R uses npost.vec[c(3, 7)] (1-based indices).
# npost.vec = seq(24, 120, by=12) so:
#   npost.vec[3] = 24 + (3-1)*12 = 48 months  (≈ 4 years post-intervention)
#   npost.vec[7] = 24 + (7-1)*12 = 96 months  (≈ 8 years post-intervention)
# We use explicit named constants for clarity.
CRITICAL_VALUE_NPOST_SHORT = 48
CRITICAL_VALUE_NPOST_LONG  = 96

if __name__ == "__main__":
    print("Finished initialising parameters for simulation")


# ============================================================================
# Phase 1: Calculate Critical Values (Null Distribution)
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
        delay     : int          — months of delay applied to intervention effect
        tmax_vec  : array(n_npost,) — Tmax at each npost length in npost_vec
        cpt_vec   : array(n_npost,) — detected changepoint at each npost length
        sim_data  : dict         — full time series (y_ctr, y_itv) at npost_max
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

    return delay, tmax_vec, cpt_vec, sim_data


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
        delay     : int          — months of delay applied to intervention effect
        tmax_vec  : array(n_npost,) — Tmax at each npost length in npost_vec
        cpt_vec   : array(n_npost,) — detected changepoint at each npost length
        sim_data  : dict         — full time series (y_ctr, y_itv) at npost_max
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

    return delay, tmax_vec, cpt_vec, sim_data


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
        delay     : int             — months of delay applied to intervention effect
        tmax_vec  : array(n_npost,) — Tmax at each npost length in npost_vec
        cpt_vec   : array(n_npost,) — detected changepoint at each npost length
        sim_data  : dict            — full AR(1) time series (y_ctr, y_itv) at npost_max
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

    return delay, tmax_vec, cpt_vec, sim_data


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
    print(f"  [{label}] {Nsim} simulations across {n_workers} cores...", flush=True)
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_null_sim_worker, args_list))
    print(f"  [{label}] Done.", flush=True)
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
        print(f"  npost = {npost} months ({tag}):")

        # i.i.d. noise, BACI design
        null_iid = _run_null_simulations(Nsim, npre, npost, level, trend_control,
                                         sigma, phi, use_ar=False,
                                         label=f"i.i.d. BACI, {npost}mo")
        cv_iid = np.percentile(null_iid, alpha * 100)
        critical_values[f'iid_{tag}'] = {'critical_value': cv_iid, 'null_dist': null_iid}
        print(f"    i.i.d. BACI critical value ({int(alpha*100)}th pct): {cv_iid:.4f}")

        # i.i.d. noise, BA design (matches R's getCritical.v2)
        null_iid_ba = _run_null_simulations(Nsim, npre, npost, level, trend_control,
                                            sigma, phi, use_ar=False, ba=True,
                                            label=f"i.i.d. BA, {npost}mo")
        cv_iid_ba = np.percentile(null_iid_ba, alpha * 100)
        critical_values[f'iid_ba_{tag}'] = {'critical_value': cv_iid_ba, 'null_dist': null_iid_ba}
        print(f"    i.i.d. BA   critical value ({int(alpha*100)}th pct): {cv_iid_ba:.4f}")

        # AR(1) noise, BA design
        null_ar = _run_null_simulations(Nsim, npre, npost, level, trend_control,
                                        sigma, phi, use_ar=True,
                                        label=f"AR(1), {npost}mo")
        cv_ar = np.percentile(null_ar, alpha * 100)
        critical_values[f'ar1_{tag}'] = {'critical_value': cv_ar, 'null_dist': null_ar}
        print(f"    AR(1)       critical value ({int(alpha*100)}th pct): {cv_ar:.4f}")

    return critical_values


# ============================================================================
# Phase 2: Main Simulation (Alternative Hypothesis)
# ============================================================================

def run_main_simulation(simN, trend_increase, critical_value, npre, npost_max,
                        npost_vec, level, trend_control, sigma, delay_set):
    """
    Run main simulation with intervention effect and calculate detection rates.

    Parameters
    ----------
    simN : int
        Number of main simulations
    trend_increase : array
        Array of trend increments to test
    critical_value : float
        Critical threshold from Phase 1
    npre : int
        Pre-intervention period
    npost_max : int
        Maximum post-intervention period
    npost_vec : array
        Vector of post-intervention lengths to test
    level : float
        Base level of time series
    trend_control : float
        Control trend
    sigma : float
        Noise level
    delay_set : array
        Set of possible delay values

    Returns
    -------
    detection_results : dict
        Dictionary mapping trend increments to detection rates
    """
    detection_results = {}
    n_trends = len(trend_increase)
    n_workers = os.cpu_count() or 1

    for trend_idx, trend_inc in enumerate(trend_increase, start=1):
        trend_interv = trend_control + trend_inc
        print(f"\n  Trend increment {trend_idx}/{n_trends}: {trend_inc:.4f} "
              f"(intervention trend: {trend_interv:.4f})", flush=True)

        # npost_vec passed into each worker so it can run the growing window loop
        args_list = [
            (sim_idx, trend_interv, trend_idx, n_trends, npre, npost_max, npost_vec,
             level, trend_control, sigma, delay_set)
            for sim_idx in range(1, simN + 1)
        ]

        print(f"    {simN} simulations × {len(npost_vec)} npost lengths "
              f"across {n_workers} cores...", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_main_sim_worker, args_list))

        # Unzip results — order preserved by executor.map, so index = sim_idx - 1
        delays         = [r[0] for r in results]
        # tmax_matrix shape: (simN, len(npost_vec)) — Tmax at each npost per simulation
        tmax_matrix    = np.array([r[1] for r in results])
        # cpt_matrix shape:  (simN, len(npost_vec)) — detected τ̂ at each npost
        cpt_matrix     = np.array([r[2] for r in results])
        simulated_data = [r[3] for r in results]

        # Detection at each npost length: Tmax > critical_value (uses iid_short threshold)
        detected_matrix   = tmax_matrix > critical_value   # shape (simN, len(npost_vec))
        detection_rates   = detected_matrix.mean(axis=0)   # shape (len(npost_vec),)

        # Error = |τ̂ - true τ| where true τ = npre + delay (varies per simulation)
        true_cpts  = np.array([npre + d for d in delays])                  # (simN,)
        error_matrix = np.abs(cpt_matrix - true_cpts[:, np.newaxis])       # (simN, len(npost_vec))
        mean_errors  = error_matrix.mean(axis=0)                           # (len(npost_vec),)

        # Summary at npost_max (last column) for quick reporting
        rate_at_max  = float(detection_rates[-1])
        error_at_max = float(mean_errors[-1])
        print(f"    At npost_max: detection rate = {rate_at_max:.2%}, "
              f"mean error = {error_at_max:.2f} months", flush=True)

        detection_results[trend_inc] = {
            # Raw matrices — one row per simulation, one column per npost length
            'tmax_matrix':      tmax_matrix,     # (simN, len(npost_vec))
            'cpt_matrix':       cpt_matrix,       # (simN, len(npost_vec))
            'detected_matrix':  detected_matrix,  # (simN, len(npost_vec))
            'delays':           delays,            # list[int], length simN

            # Aggregated over simulations — one value per npost length
            'detection_rates':  detection_rates,  # (len(npost_vec),)
            'mean_errors':      mean_errors,       # (len(npost_vec),)

            # Full time series stored for overlay plotting
            'simulated_data':   simulated_data,   # list[dict], length simN
        }

    return detection_results


def run_main_simulation_iid_ba(simN, trend_increase, critical_value, npre, npost_max,
                               npost_vec, level, trend_control, sigma, delay_set):
    """
    i.i.d. BA equivalent of run_main_simulation().

    Dispatches to _main_sim_worker_iid_ba, which uses i.i.d. noise but tests the
    intervention series only (BA design), matching R's simResult.v2 from AMOC.R.
    The i.i.d. BA critical value (iid_ba_48) should be passed.

    Parameters and return value are identical to run_main_simulation().
    """
    detection_results = {}
    n_trends = len(trend_increase)
    n_workers = os.cpu_count() or 1

    for trend_idx, trend_inc in enumerate(trend_increase, start=1):
        trend_interv = trend_control + trend_inc
        print(f"\n  Trend increment {trend_idx}/{n_trends}: {trend_inc:.4f} "
              f"(intervention trend: {trend_interv:.4f})", flush=True)

        args_list = [
            (sim_idx, trend_interv, trend_idx, n_trends, npre, npost_max, npost_vec,
             level, trend_control, sigma, delay_set)
            for sim_idx in range(1, simN + 1)
        ]

        print(f"    {simN} simulations × {len(npost_vec)} npost lengths "
              f"across {n_workers} cores...", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_main_sim_worker_iid_ba, args_list))

        delays         = [r[0] for r in results]
        tmax_matrix    = np.array([r[1] for r in results])
        cpt_matrix     = np.array([r[2] for r in results])
        simulated_data = [r[3] for r in results]

        detected_matrix = tmax_matrix > critical_value
        detection_rates = detected_matrix.mean(axis=0)

        true_cpts    = np.array([npre + d for d in delays])
        error_matrix = np.abs(cpt_matrix - true_cpts[:, np.newaxis])
        mean_errors  = error_matrix.mean(axis=0)

        rate_at_max  = float(detection_rates[-1])
        error_at_max = float(mean_errors[-1])
        print(f"    At npost_max: detection rate = {rate_at_max:.2%}, "
              f"mean error = {error_at_max:.2f} months", flush=True)

        detection_results[trend_inc] = {
            'tmax_matrix':     tmax_matrix,
            'cpt_matrix':      cpt_matrix,
            'detected_matrix': detected_matrix,
            'delays':          delays,
            'detection_rates': detection_rates,
            'mean_errors':     mean_errors,
            'simulated_data':  simulated_data,
        }

    return detection_results


def run_main_simulation_ar(simN, trend_increase, critical_value, npre, npost_max,
                           npost_vec, level, trend_control, sigma, phi, delay_set):
    """
    AR(1) equivalent of run_main_simulation().

    Dispatches to _main_sim_worker_ar, which generates time series with AR(1)
    noise and applies the ARIMA(1,0,0)-based test statistic at every npost length.
    The AR(1) critical value (ar1_48) should be passed so detection is calibrated
    to the autocorrelated null distribution.

    Parameters
    ----------
    simN : int
    trend_increase : array
    critical_value : float       — AR(1) critical value from Phase 1 (e.g. ar1_48)
    npre : int
    npost_max : int
    npost_vec : array
    level : float
    trend_control : float
    sigma : float
    phi : float                  — AR(1) coefficient passed to ci_sim_ar
    delay_set : array

    Returns
    -------
    detection_results : dict     — same structure as run_main_simulation()
    """
    detection_results = {}
    n_trends = len(trend_increase)
    n_workers = os.cpu_count() or 1

    for trend_idx, trend_inc in enumerate(trend_increase, start=1):
        trend_interv = trend_control + trend_inc
        print(f"\n  Trend increment {trend_idx}/{n_trends}: {trend_inc:.4f} "
              f"(intervention trend: {trend_interv:.4f})", flush=True)

        args_list = [
            (sim_idx, trend_interv, trend_idx, n_trends, npre, npost_max, npost_vec,
             level, trend_control, sigma, phi, delay_set)
            for sim_idx in range(1, simN + 1)
        ]

        print(f"    {simN} simulations × {len(npost_vec)} npost lengths "
              f"across {n_workers} cores...", flush=True)
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_main_sim_worker_ar, args_list))

        delays         = [r[0] for r in results]
        tmax_matrix    = np.array([r[1] for r in results])
        cpt_matrix     = np.array([r[2] for r in results])
        simulated_data = [r[3] for r in results]

        detected_matrix = tmax_matrix > critical_value
        detection_rates = detected_matrix.mean(axis=0)

        true_cpts    = np.array([npre + d for d in delays])
        error_matrix = np.abs(cpt_matrix - true_cpts[:, np.newaxis])
        mean_errors  = error_matrix.mean(axis=0)

        rate_at_max  = float(detection_rates[-1])
        error_at_max = float(mean_errors[-1])
        print(f"    At npost_max: detection rate = {rate_at_max:.2%}, "
              f"mean error = {error_at_max:.2f} months", flush=True)

        detection_results[trend_inc] = {
            'tmax_matrix':     tmax_matrix,
            'cpt_matrix':      cpt_matrix,
            'detected_matrix': detected_matrix,
            'delays':          delays,
            'detection_rates': detection_rates,
            'mean_errors':     mean_errors,
            'simulated_data':  simulated_data,
        }

    return detection_results


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AMOC trend-change simulation")
    parser.add_argument(
        "-p", "--plots-only",
        action="store_true",
        help="Skip simulation phases and generate plots from saved results only",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run with Nsim=10, simN=10 for fast smoke-testing",
    )
    args = parser.parse_args()

    if args.quick:
        Nsim = 10
        simN = 10

    # Setup
    plots_dir = file_utils.setup_plots_directory("trend_amoc")
    file_utils.setup_results_directory("trend_amoc")

    print("Trend Change AMOC Detection")
    print("=" * 70)
    print(f"Pre-intervention period: {npre} months")
    print(f"Post-intervention range: {npost_vec[0]} to {npost_max} months")
    print(f"Total time series length: {ntt} months")
    print(f"Number of trend increments to test: {len(trend_increase)}")
    print(f"Noise level (sigma): {sigma}")
    print(f"AR(1) coefficient (phi): {phi}")
    print(f"Delay range: {delay_set[0]} to {delay_set[-1]} time points")
    print(f"Critical value npost lengths: {CRITICAL_VALUE_NPOST_SHORT} and {CRITICAL_VALUE_NPOST_LONG} months")
    print()

    # Check if we can load previous results
    print("Checking for saved simulation results...")
    loaded_data = file_utils.load_simulation_results("trend_amoc")

    if args.plots_only and loaded_data is None:
        print("ERROR: --plots-only requested but no saved results found in results/.")
        print("Run without --plots-only first to generate simulation results.")
        raise SystemExit(1)

    if (loaded_data is not None
            and 'critical_values' in loaded_data
            and 'detection_results_ar' in loaded_data):
        critical_values            = loaded_data['critical_values']
        detection_results          = loaded_data['detection_results']
        detection_results_ar       = loaded_data['detection_results_ar']
        detection_results_iid_ba   = loaded_data.get('detection_results_iid_ba')
        test_stats                 = loaded_data.get('test_stats', {})
        use_saved = True
        print("Using previously saved results. Skipping simulations.")
        print()

        if test_stats:
            print("Saved Test Statistics (i.i.d. example, seed=42):")
            print(f"  Test statistic: {test_stats['Tmax']:.4f}")
            print(f"  Detected changepoint: {test_stats['cpt']} months (true: {test_stats['npre']})")
            print()

        print("Saved Critical Values:")
        for key, cv_data in critical_values.items():
            cv = cv_data['critical_value']
            nd = cv_data['null_dist']
            print(f"  [{key}] {int(alpha*100)}th pct = {cv:.4f}  "
                  f"(null dist: mean={np.mean(nd):.4f}, sd={np.std(nd):.4f})")
        print()
    else:
        use_saved = False

    example_trend = trend_increase[3]  # 4th trend increment — used throughout

    if not use_saved:
        # Run a concrete example for plotting and to verify data generation
        sim_data_test = ci_sim(seed=42, npre=npre, npost=npost_max, level=level,
                               trend=[trend_control, trend_control + example_trend], sigma=sigma)
        stats_test = trend_stats(y_ctr=sim_data_test['y_ctr'], y_itv=sim_data_test['y_itv'],
                                 nt=npre + npost_max)

        print("Testing data generation and trend statistics (i.i.d., seed=42):")
        print(f"  Test statistic: {stats_test['Tmax']:.4f}")
        print(f"  Detected changepoint: {stats_test['cpt']} months (true: {npre})")
        print()

        # ====================================================================
        # PHASE 1: Calculate Critical Values (Null Distribution)
        # Matches R's getCritical: 2 npost lengths × 2 noise types = 4 values
        # ====================================================================
        print("=" * 70)
        print("PHASE 1: Calculating Critical Values (Null Distribution)")
        print("=" * 70)
        print(f"Simulating {Nsim} null datasets at npost={CRITICAL_VALUE_NPOST_SHORT} and "
              f"{CRITICAL_VALUE_NPOST_LONG} months, for i.i.d. and AR(1) noise...")
        print()
        critical_values = calculate_critical_values(
            Nsim=Nsim,
            npre=npre,
            level=level,
            trend_control=trend_control,
            sigma=sigma,
            phi=phi,
            alpha=alpha
        )
        print()
        print("Phase 1 complete. Critical values:")
        for key, cv_data in critical_values.items():
            cv = cv_data['critical_value']
            nd = cv_data['null_dist']
            print(f"  [{key}] {int(alpha*100)}th pct = {cv:.4f}  "
                  f"(null dist: mean={np.mean(nd):.4f}, sd={np.std(nd):.4f})")
        print()

        # ====================================================================
        # PHASE 2: Main Simulation (Alternative Hypothesis, i.i.d. noise)
        # Uses the i.i.d. critical value at the short npost length (48 months)
        # ====================================================================
        print("=" * 70)
        print("PHASE 2: Main Simulation (WITH Trend Change, i.i.d. noise)")
        print("=" * 70)
        print(f"Simulating {simN} datasets for each of {len(trend_increase)} trend increments...")
        print(f"Using critical value: iid_{CRITICAL_VALUE_NPOST_SHORT} = "
              f"{critical_values[f'iid_{CRITICAL_VALUE_NPOST_SHORT}']['critical_value']:.4f}")
        print()

        detection_results = run_main_simulation(
            simN=simN,
            trend_increase=trend_increase,
            critical_value=critical_values[f'iid_{CRITICAL_VALUE_NPOST_SHORT}']['critical_value'],
            npre=npre,
            npost_max=npost_max,
            npost_vec=npost_vec,
            level=level,
            trend_control=trend_control,
            sigma=sigma,
            delay_set=delay_set
        )
        print()

        # Prepare metadata to save alongside results
        test_stats = {
            'Tmax': stats_test['Tmax'],
            'cpt': stats_test['cpt'],
            'npre': npre,
            'alpha': alpha
        }

        # ====================================================================
        # PHASE 3: Main Simulation (Alternative Hypothesis, AR(1) noise)
        # Uses the AR(1) critical value at the short npost length (48 months)
        # ====================================================================
        print("=" * 70)
        print("PHASE 3: Main Simulation (WITH Trend Change, AR(1) noise)")
        print("=" * 70)
        print(f"Simulating {simN} datasets for each of {len(trend_increase)} trend increments...")
        print(f"Using critical value: ar1_{CRITICAL_VALUE_NPOST_SHORT} = "
              f"{critical_values[f'ar1_{CRITICAL_VALUE_NPOST_SHORT}']['critical_value']:.4f}")
        print()

        detection_results_ar = run_main_simulation_ar(
            simN=simN,
            trend_increase=trend_increase,
            critical_value=critical_values[f'ar1_{CRITICAL_VALUE_NPOST_SHORT}']['critical_value'],
            npre=npre,
            npost_max=npost_max,
            npost_vec=npost_vec,
            level=level,
            trend_control=trend_control,
            sigma=sigma,
            phi=phi,
            delay_set=delay_set
        )
        print()

        # ====================================================================
        # PHASE 4: Main Simulation (Alternative Hypothesis, i.i.d. noise, BA)
        # Matches R's simResult.v2 from AMOC.R
        # Uses the i.i.d. BA critical value at the short npost length (48 months)
        # ====================================================================
        print("=" * 70)
        print("PHASE 4: Main Simulation (WITH Trend Change, i.i.d. noise, BA design)")
        print("=" * 70)
        print(f"Simulating {simN} datasets for each of {len(trend_increase)} trend increments...")
        print(f"Using critical value: iid_ba_{CRITICAL_VALUE_NPOST_SHORT} = "
              f"{critical_values[f'iid_ba_{CRITICAL_VALUE_NPOST_SHORT}']['critical_value']:.4f}")
        print()

        detection_results_iid_ba = run_main_simulation_iid_ba(
            simN=simN,
            trend_increase=trend_increase,
            critical_value=critical_values[f'iid_ba_{CRITICAL_VALUE_NPOST_SHORT}']['critical_value'],
            npre=npre,
            npost_max=npost_max,
            npost_vec=npost_vec,
            level=level,
            trend_control=trend_control,
            sigma=sigma,
            delay_set=delay_set
        )
        print()

        file_utils.save_simulation_results("trend_amoc", {
            'critical_values':           critical_values,
            'detection_results':         detection_results,
            'detection_results_ar':      detection_results_ar,
            'detection_results_iid_ba':  detection_results_iid_ba,
            'test_stats':                test_stats,
        })
        n_iid    = sum(len(detection_results[t]['simulated_data']) for t in detection_results)
        n_ar1    = sum(len(detection_results_ar[t]['simulated_data']) for t in detection_results_ar)
        n_iid_ba = sum(len(detection_results_iid_ba[t]['simulated_data']) for t in detection_results_iid_ba)
        print(f"  (i.i.d. BACI: {n_iid}, AR(1): {n_ar1}, i.i.d. BA: {n_iid_ba} simulations)")

    # ========================================================================
    # Results Summary — detection rate and mean error at npost_max, all scenarios
    # ========================================================================
    has_ba = detection_results_iid_ba is not None
    col = (f"{'Trend':>8}  {'iid detect':>11}  {'iid error':>10}  "
           f"{'ar1 detect':>11}  {'ar1 error':>10}"
           + (f"  {'ba detect':>10}  {'ba error':>9}" if has_ba else ""))
    sep = "-" * len(col)
    print("=" * len(col))
    print(f"RESULTS SUMMARY — at npost_max ({npost_max} months)")
    print(f"  i.i.d. BACI critical value: {critical_values[f'iid_{CRITICAL_VALUE_NPOST_SHORT}']['critical_value']:.4f} (iid_{CRITICAL_VALUE_NPOST_SHORT})")
    print(f"  AR(1)  BA   critical value: {critical_values[f'ar1_{CRITICAL_VALUE_NPOST_SHORT}']['critical_value']:.4f} (ar1_{CRITICAL_VALUE_NPOST_SHORT})")
    if has_ba:
        print(f"  i.i.d. BA   critical value: {critical_values[f'iid_ba_{CRITICAL_VALUE_NPOST_SHORT}']['critical_value']:.4f} (iid_ba_{CRITICAL_VALUE_NPOST_SHORT})")
    print("=" * len(col))
    print(col)
    print(sep)
    for trend_inc in trend_increase:
        iid = detection_results[trend_inc]
        ar1 = detection_results_ar[trend_inc]
        iid_rate  = iid['detection_rates'][-1]
        iid_error = iid['mean_errors'][-1]
        ar1_rate  = ar1['detection_rates'][-1]
        ar1_error = ar1['mean_errors'][-1]
        line = (f"{trend_inc:>8.4f}  {iid_rate:>11.2%}  {iid_error:>9.2f}m  "
                f"{ar1_rate:>11.2%}  {ar1_error:>9.2f}m")
        if has_ba:
            ba = detection_results_iid_ba[trend_inc]
            ba_rate  = ba['detection_rates'][-1]
            ba_error = ba['mean_errors'][-1]
            line += f"  {ba_rate:>10.2%}  {ba_error:>8.2f}m"
        print(line)
    print(sep)
    print()

    # ========================================================================
    # Generate Plots
    # ========================================================================

    # Ensure we have an example time series for plot 2 (not stored in results pkl)
    if use_saved:
        sim_data_test = ci_sim(seed=42, npre=npre, npost=npost_max, level=level,
                               trend=[trend_control, trend_control + example_trend], sigma=sigma)
        stats_test = trend_stats(y_ctr=sim_data_test['y_ctr'], y_itv=sim_data_test['y_itv'],
                                 nt=npre + npost_max)

    print("Generating plots...")
    print()

    # Plot 1: Sensitivity analysis at npost_max
    print("  1. Sensitivity analysis plot (at npost_max)...")
    detection_rates_at_max = [detection_results[t]['detection_rates'][-1] for t in trend_increase]
    fig, ax = plot_simulation_results(
        detection_rates_at_max,
        trend_increase,
        savefile=f"{plots_dir}/sensitivity_analysis.png"
    )
    print()

    # Plot 2: Example time series (uses sim_data_test computed above, seed=42)
    print("  2. Example time series plot...")
    print(f"     [Verification] Detected changepoint: {stats_test['cpt']} months (true: {npre})")
    fig, axes = plot_time_series(sim_data_test, npre, stats=stats_test,
                                  savefile=f"{plots_dir}/example_time_series.png")
    print()

    # Plot 5: Power curves — i.i.d. only
    print("  5. Power curves — i.i.d. noise (detection rate vs npost length)...")
    fig, ax = plot_power_curves(
        detection_results, trend_increase, npost_vec,
        savefile=f"{plots_dir}/power_curves_iid.png"
    )
    print()

    # Plot 6: Power curves — AR(1) only
    print("  6. Power curves — AR(1) noise...")
    fig, ax = plot_power_curves(
        detection_results_ar, trend_increase, npost_vec,
        savefile=f"{plots_dir}/power_curves_ar1.png"
    )
    print()

    # Plot 7: Side-by-side comparison
    print("  7. Power curves comparison — i.i.d. vs AR(1)...")
    fig, axes = plot_power_curves_comparison(
        detection_results, detection_results_ar, trend_increase, npost_vec,
        savefile=f"{plots_dir}/power_curves_comparison.png"
    )
    print()

    # Plot 8: Time-to-detection curve
    print("  8. Time-to-detection curve...")
    fig, ax = plot_time_to_detection(
        detection_results, detection_results_ar, trend_increase, npost_vec,
        savefile=f"{plots_dir}/time_to_detection.png"
    )
    print()

    # Plot 9: Detection rate heatmap
    print("  9. Detection rate heatmap (trend × npost)...")
    fig, axes = plot_detection_heatmap(
        detection_results, detection_results_ar, trend_increase, npost_vec,
        savefile=f"{plots_dir}/detection_heatmap.png"
    )
    print()

    # Plot 10: Mean error curves
    print("  10. Mean error curves vs npost length...")
    fig, axes = plot_mean_error_curves(
        detection_results, detection_results_ar, trend_increase, npost_vec,
        savefile=f"{plots_dir}/mean_error_curves.png"
    )
    print()

    # Plot 11: Null distribution histograms
    print("  11. Null distribution histograms...")
    fig, axes = plot_null_distributions(
        critical_values,
        npost_short=CRITICAL_VALUE_NPOST_SHORT,
        npost_long=CRITICAL_VALUE_NPOST_LONG,
        savefile=f"{plots_dir}/null_distributions.png"
    )
    print()

    # Plot 12: Tmax signal vs noise
    print("  12. Tmax signal vs noise distributions...")
    fig, axes = plot_tmax_signal_vs_noise(
        critical_values, detection_results, detection_results_ar,
        trend_increase, npost_vec,
        npost_short=CRITICAL_VALUE_NPOST_SHORT,
        savefile=f"{plots_dir}/tmax_signal_vs_noise.png"
    )
    print()

    # Plot 13: Detection rate by delay
    print("  13. Detection rate by delay...")
    fig, axes = plot_detection_by_delay(
        detection_results, detection_results_ar, trend_increase,
        savefile=f"{plots_dir}/detection_by_delay.png"
    )
    print()

    # Plot 14: Changepoint bias (signed error)
    print("  14. Changepoint bias (signed error)...")
    fig, axes = plot_changepoint_bias(
        detection_results, detection_results_ar, trend_increase, npre,
        savefile=f"{plots_dir}/changepoint_bias.png"
    )
    print()

    print("=" * 70)
    print(f"✓ All plots saved to: {plots_dir}/")
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
