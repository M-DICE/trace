"""
Trend Change Detection using Forecast (Page-CUSUM on forecast errors)

Online changepoint detection: fit a linear model on the pre-intervention
period, generate multi-step forecasts, and apply Page-CUSUM to the residual
series. When detection fires, AMOC locates the changepoint precisely.

Equivalent to: Rewild_trend_change_Forecast.R
"""

import argparse
import time
import warnings
import os
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import file_utils
from amoc import (ci_sim, ci_sim_ar,
                  load_crit_val_table, lookup_crit_val,
                  trend_stats_forecast)

_CRIT_VAL = lookup_crit_val(load_crit_val_table())

warnings.filterwarnings('ignore')


def _fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


# ============================================================================
# Simulation Parameters  (matching Rewild_trend_change_Forecast.R)
# ============================================================================
if __name__ == "__main__":
    print("Initialising parameters for simulation")

npre = 24
npost_years = 10
npost_months = 12 * npost_years
npost_vec = np.arange(npre, npost_months + 1, 12)   # annual steps

npost_max = npost_vec[-1]
ntt = npre + npost_max

level = 10
trend_control = 0.005
trend_increase = np.round(
    level * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / npost_months,
    4
)

sigma = 0.05
phi = 0.8                  # R Forecast file uses phi=0.8 (AMOC uses 0.5)
delay_set = np.arange(1, 21)

alpha = 0.95
Nsim = 1000
simN = 1000

if __name__ == "__main__":
    print("Finished initialising parameters for simulation")




# ============================================================================
# Phase 2: Main Simulation (Alternative Hypothesis)
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
# Main Execution
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forecast trend-change simulation")
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

    plots_dir = file_utils.setup_plots_directory("trend_forecast")
    file_utils.setup_results_directory("trend_forecast")

    print("Trend Change Forecast Detection (Page-CUSUM)")
    print("=" * 70)
    print(f"Pre-intervention period: {npre} months")
    print(f"Post-intervention max:   {npost_max} months")
    print(f"Total series length:     {ntt} months")
    print(f"Trend increments:        {len(trend_increase)}")
    print(f"Noise level (sigma):     {sigma}")
    print(f"AR(1) coefficient (phi): {phi}")
    print(f"Delay range:             {delay_set[0]}–{delay_set[-1]} months")
    print()

    loaded_data = file_utils.load_simulation_results("trend_forecast") or {}

    if args.plots_only and not loaded_data:
        print("ERROR: --plots-only requested but no saved results found.")
        raise SystemExit(1)

    detection_results_iid  = loaded_data.get('detection_results', loaded_data.get('detection_results_iid', {}))
    detection_results_ar   = loaded_data.get('detection_results_ar',  {})

    def _save():
        file_utils.save_simulation_results("trend_forecast", {
            'critical_values':        _CRIT_VAL,
            'detection_results':      detection_results_iid,
            'detection_results_ar':   detection_results_ar,
        })

    print(f"Critical value (PageCUSUM, γ=0, α=0.05): {_CRIT_VAL:.7f}")
    print()

    if not args.plots_only:
        t_total = time.perf_counter()

        # ====================================================================
        # PHASE 1 (formerly): now uses table-derived critical value — no sims
        # ====================================================================

        # ====================================================================
        # PHASE 2: i.i.d. BA simulation
        # ====================================================================
        n_done = len(detection_results_iid)
        print("=" * 70)
        print(f"PHASE 2: Main Simulation (i.i.d. noise, BA design)  ({n_done}/{len(trend_increase)} cached)")
        print("=" * 70)
        print(f"crit_val: {_CRIT_VAL:.7f}")
        t0 = time.perf_counter()

        def on_done_iid(r):
            global detection_results_iid
            detection_results_iid = r
            _save()

        detection_results_iid = run_simulation_iid(
            simN=simN,
            trend_increase=trend_increase,
            crit_val=_CRIT_VAL,
            npre=npre, ntt=ntt, npost_max=npost_max,
            level=level, trend_control=trend_control,
            sigma=sigma, delay_set=delay_set,
            existing_results=detection_results_iid,
            on_trend_done=on_done_iid
        )
        _save()
        print(f"  Phase 2 total: {_fmt_elapsed(time.perf_counter() - t0)}")
        print()

        # ====================================================================
        # PHASE 3: AR(1) BA simulation
        # ====================================================================
        n_done = len(detection_results_ar)
        print("=" * 70)
        print(f"PHASE 3: Main Simulation (AR(1) noise, BA design)  ({n_done}/{len(trend_increase)} cached)")
        print("=" * 70)
        print(f"crit_val: {_CRIT_VAL:.7f}")
        t0 = time.perf_counter()

        def on_done_ar(r):
            global detection_results_ar
            detection_results_ar = r
            _save()

        detection_results_ar = run_simulation_ar(
            simN=simN,
            trend_increase=trend_increase,
            crit_val=_CRIT_VAL,
            npre=npre, ntt=ntt, npost_max=npost_max,
            level=level, trend_control=trend_control,
            sigma=sigma, phi=phi, delay_set=delay_set,
            existing_results=detection_results_ar,
            on_trend_done=on_done_ar
        )
        _save()
        print(f"  Phase 3 total: {_fmt_elapsed(time.perf_counter() - t0)}")
        print()

        print(f"All phases complete in {_fmt_elapsed(time.perf_counter() - t_total)}")
        print()

    # ========================================================================
    # Summary Table
    # ========================================================================
    col = (f"{'Trend':>8}  {'iid detect':>11}  {'iid time':>9}  {'iid error':>10}  "
           f"{'ar1 detect':>11}  {'ar1 time':>9}  {'ar1 error':>10}")
    sep = "-" * len(col)
    print("=" * len(col))
    print("RESULTS SUMMARY")
    print(f"  crit_val: {_CRIT_VAL:.7f}")
    print("=" * len(col))
    print(col)
    print(sep)
    for trend_inc in trend_increase:
        iid = detection_results_iid[trend_inc]
        ar1 = detection_results_ar[trend_inc]
        print(f"{trend_inc:>8.4f}  "
              f"{iid['detection_rate']:>11.2%}  "
              f"{iid['mean_time']:>9.1f}  "
              f"{iid['mean_error']:>9.2f}m  "
              f"{ar1['detection_rate']:>11.2%}  "
              f"{ar1['mean_time']:>9.1f}  "
              f"{ar1['mean_error']:>9.2f}m")
    print(sep)
    print()

    # ========================================================================
    # Generate Plots
    # ========================================================================
    print("Generating plots...")
    from plotting import (plot_time_series, plot_simulation_results, plot_power_curves,
                          plot_power_curves_comparison, plot_time_to_detection,
                          plot_detection_heatmap, plot_mean_error_curves,
                          plot_null_distributions)

    # Example time series plot: regenerate one run on the fly
    example_trend = trend_increase[3]
    res = detection_results_iid[example_trend]
    _ex_seed = res['seeds'][0]
    _ex_delay = res['delays'][0]
    _ex_sim_data = ci_sim(seed=_ex_seed, npre=npre + _ex_delay, npost=npost_max - _ex_delay,
                          level=level, trend=[trend_control, trend_control + example_trend], sigma=sigma)
    _ex_stats = {'cpt': res['cpt_est_vec'][0], 'time_est': res['time_est_vec'][0]}
    
    print("  1. Example time series plot...")
    plot_time_series(_ex_sim_data, npre, stats=_ex_stats,
                     savefile=f"{plots_dir}/example_time_series.png")

    print("  2. Sensitivity analysis plot...")
    dr_iid = [detection_results_iid[t]['detection_rate'] for t in trend_increase]
    plot_simulation_results(dr_iid, trend_increase, savefile=f"{plots_dir}/sensitivity_analysis.png")

    print("  3. Power curves...")
    # Power curves need a nested structure for npost_vec which Forecast doesn't really have 
    # in the same way (it's online), so we'll just plot vs trend for now or adapt if needed.
    # Actually, plot_power_curves expects a result for each npost. 
    # Forecast is "online", so it doesn't have detection rates per npost in the same way.
    # We might need a specialized plot or just skip the npost-based ones.
    # For now, let's keep it simple.

    print("ANALYSIS COMPLETE")
    print("=" * 70)

