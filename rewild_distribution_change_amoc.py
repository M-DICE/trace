"""
Distribution Change Detection using AMOC

This script detects distribution changes (shifts in mean or variance) 
in paired time series using the AMOC offline batch method.

Equivalent to: Rewild_distribution_change_AMOC.R
"""

import argparse
import numpy as np
import warnings
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import file_utils

from amoc import (ci_sim_cdf, wasserstein_distance_baci, wasserstein_distance_ba,
                  auc_diff_ts, trend_stats_cdf)
from plotting import (plot_simulation_results, plot_power_curves,
                      plot_detection_heatmap, plot_mean_error_curves,
                      plot_null_distributions, plot_distance_time_series,
                      plot_distribution_difference, plot_power_curves_mu_sigma)

warnings.filterwarnings('ignore')


# ============================================================================
# Simulation Parameters
# ============================================================================

if __name__ == "__main__":
    print("Initialising parameters for simulation")

npre = 24
# DEVIATION: Python uses step=3 (33 values) for quarterly resolution. 
# R uses seq(24, 120, by=12) (9 values).
npost_vec = np.arange(24, 121, 3)          
npost_max = npost_vec[-1]                   # 120
ntt = npre + npost_max                      # 144

mu = 10
sigma = 1
ns = 200                                    # samples per distribution time point
bw = 0.3                                    # KDE bandwidth
nd = 50                                     # KDE grid points
dist_measure = "wasserstein"                # or "auc"

# Mean change: 5% to 100% over 120 months → 11 values
trend_increase_mu = np.round(mu * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / 120, 4)

# EXTENSION: Variance change simulation has no R equivalent
# Variance change: 50% to 200% over 120 months → 4 values
trend_increase_sigma = np.round(sigma * np.arange(0.5, 2.1, 0.5) / 120, 4)

delay_set = np.arange(1, 21)
alpha = 0.95
Nsim = 1000
simN = 1000

# Critical value npost lengths (R uses [48, 84] — npost.vec[c(3, 7)])
# DEVIATION: Python uses [24, 72, 120] to cover full range at step=3
CRITICAL_VALUE_NPOST_SHORT  = 24    # npost_vec[0]
CRITICAL_VALUE_NPOST_MEDIUM = 72    # npost_vec[16]
CRITICAL_VALUE_NPOST_LONG   = 120   # npost_vec[32]

FOLDER = "distribution_amoc"

if __name__ == "__main__":
    print("Finished initialising parameters for simulation")

# ============================================================================
# Phase 1: Calculate Critical Values (Null Distribution)
# ============================================================================

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

# ============================================================================
# Phase 2: Main Simulation (Alternative Hypothesis)
# ============================================================================

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

    return delay, tmax_vec, cpt_vec, sim

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

    return delay, tmax_vec, cpt_vec, sim

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

    return delay, tmax_vec, cpt_vec, sim

# Orchestration functions

def calculate_critical_values_cdf(Nsim, npre, mu, sigma, ns, dist_measure, bw, nd, alpha=0.95):
    critical_values = {}
    n_workers = os.cpu_count() or 1
    
    for npost, label in [(CRITICAL_VALUE_NPOST_SHORT, "24"), 
                         (CRITICAL_VALUE_NPOST_MEDIUM, "72"), 
                         (CRITICAL_VALUE_NPOST_LONG, "120")]:
        print(f"  npost = {npost} months:")
        args_list = [(i, npre, npost, mu, sigma, ns, dist_measure, bw, nd) for i in range(1, Nsim + 1)]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_null_sim_worker_cdf, args_list))
        
        cv = np.percentile(results, alpha * 100)
        critical_values[f'npost_{label}'] = {'critical_value': cv, 'null_dist': np.array(results)}
        print(f"    Critical value: {cv:.4f}")
        
    return critical_values

def run_main_simulation_mu(simN, trend_increase_mu, critical_value, npre, npost_max, npost_vec,
                           mu, sigma, ns, dist_measure, bw, nd, delay_set, ba=False):
    detection_results = {}
    n_trends = len(trend_increase_mu)
    n_workers = os.cpu_count() or 1
    worker_func = _main_sim_worker_cdf_mu_ba if ba else _main_sim_worker_cdf_mu

    for trend_idx, trend_mu in enumerate(trend_increase_mu, start=1):
        print(f"  Trend mu increment {trend_idx}/{n_trends}: {trend_mu:.4f}")
        args_list = [(sim_idx, trend_mu, trend_idx, n_trends, npre, npost_max, npost_vec,
                      mu, sigma, ns, dist_measure, bw, nd, delay_set)
                     for sim_idx in range(1, simN + 1)]
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(worker_func, args_list))
            
        delays = [r[0] for r in results]
        tmax_matrix = np.array([r[1] for r in results])
        cpt_matrix = np.array([r[2] for r in results])
        simulated_data = [r[3] for r in results]
        
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
            'simulated_data': simulated_data
        }
    return detection_results

def run_main_simulation_sigma(simN, trend_increase_sigma, critical_value, npre, npost_max, npost_vec,
                              mu, sigma, ns, dist_measure, bw, nd, delay_set):
    detection_results = {}
    n_trends = len(trend_increase_sigma)
    n_workers = os.cpu_count() or 1

    for trend_idx, trend_sigma in enumerate(trend_increase_sigma, start=1):
        print(f"  Trend sigma increment {trend_idx}/{n_trends}: {trend_sigma:.4f}")
        args_list = [(sim_idx, trend_sigma, trend_idx, n_trends, npre, npost_max, npost_vec,
                      mu, sigma, ns, dist_measure, bw, nd, delay_set)
                     for sim_idx in range(1, simN + 1)]
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_main_sim_worker_cdf_sigma, args_list))
            
        delays = [r[0] for r in results]
        tmax_matrix = np.array([r[1] for r in results])
        cpt_matrix = np.array([r[2] for r in results])
        simulated_data = [r[3] for r in results]
        
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
            'simulated_data': simulated_data
        }
    return detection_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distribution change AMOC simulation")
    parser.add_argument("--plots-only", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run with small Nsim/simN")
    args = parser.parse_args()

    if args.quick:
        Nsim = 10
        simN = 10

    plots_dir = file_utils.setup_plots_directory(FOLDER)
    file_utils.setup_results_directory(FOLDER)
    loaded_data = file_utils.load_simulation_results(FOLDER)

    if args.plots_only and loaded_data is None:
        print("Error: No saved results found.")
        exit(1)

    if loaded_data and 'detection_results_mu' in loaded_data:
        critical_values = loaded_data['critical_values']
        detection_results_mu = loaded_data['detection_results_mu']
        detection_results_sigma = loaded_data['detection_results_sigma']
        detection_results_mu_ba = loaded_data.get('detection_results_mu_ba')
        use_saved = True
    else:
        use_saved = False

    if not use_saved:
        print("PHASE 1: Critical Values")
        critical_values = calculate_critical_values_cdf(Nsim, npre, mu, sigma, ns, dist_measure, bw, nd, alpha)
        
        cv = critical_values['npost_24']['critical_value']
        
        print("\nPHASE 2A: Mean Change (BACI)")
        detection_results_mu = run_main_simulation_mu(simN, trend_increase_mu, cv, npre, npost_max, npost_vec,
                                                    mu, sigma, ns, dist_measure, bw, nd, delay_set)
        
        print("\nPHASE 2A': Mean Change (BA)")
        detection_results_mu_ba = run_main_simulation_mu(simN, trend_increase_mu, cv, npre, npost_max, npost_vec,
                                                       mu, sigma, ns, dist_measure, bw, nd, delay_set, ba=True)

        print("\nPHASE 2B: Variance Change")
        detection_results_sigma = run_main_simulation_sigma(simN, trend_increase_sigma, cv, npre, npost_max, npost_vec,
                                                         mu, sigma, ns, dist_measure, bw, nd, delay_set)
        
        file_utils.save_simulation_results(FOLDER, {
            'critical_values':          critical_values,
            'detection_results_mu':     detection_results_mu,
            'detection_results_sigma':  detection_results_sigma,
            'detection_results_mu_ba':  detection_results_mu_ba,
        })

    # Summary table
    print("\nRESULTS SUMMARY (at npost_max)")
    print(f"{'Trend Mu':>10} | {'Detect Rate':>12}")
    for t in trend_increase_mu:
        print(f"{t:10.4f} | {detection_results_mu[t]['detection_rates'][-1]:12.2%}")
    
    print(f"\n{'Trend Sigma':>10} | {'Detect Rate':>12}")
    for t in trend_increase_sigma:
        print(f"{t:10.4f} | {detection_results_sigma[t]['detection_rates'][-1]:12.2%}")

    # Plotting
    print("\nGenerating plots...")
    # Example distance TS
    example_mu = trend_increase_mu[5]
    res = detection_results_mu[example_mu]
    sim_data = res['simulated_data'][0]
    nt = npre + npost_max
    if dist_measure == "wasserstein":
        dist_ts = wasserstein_distance_baci(sim_data['sample_ctr'], sim_data['sample_itv'])
    else:
        dist_ts = auc_diff_ts(sim_data['sample_ctr'], sim_data['sample_itv'], bw=bw, nd=nd)
    
    stats = {'Tmax': res['tmax_matrix'][0, -1], 'cpt': res['cpt_matrix'][0, -1]}
    
    plot_distance_time_series(sim_data, npre, dist_ts, stats=stats, dist_measure=dist_measure,
                              savefile=f"{plots_dir}/example_distance_time_series.png")
    plot_distribution_difference(sim_data, npre, dist_ts, stats=stats, dist_measure=dist_measure,
                                 savefile=f"{plots_dir}/example_distribution_difference.png")
    
    # Sensitivity
    dr_mu = [detection_results_mu[t]['detection_rates'][-1] for t in trend_increase_mu]
    plot_simulation_results(dr_mu, trend_increase_mu, savefile=f"{plots_dir}/sensitivity_analysis_mu.png")
    
    dr_sigma = [detection_results_sigma[t]['detection_rates'][-1] for t in trend_increase_sigma]
    plot_simulation_results(dr_sigma, trend_increase_sigma, savefile=f"{plots_dir}/sensitivity_analysis_sigma.png")

    # Power curves
    plot_power_curves(detection_results_mu, trend_increase_mu, npost_vec, 
                      savefile=f"{plots_dir}/power_curves_mu.png")
    plot_power_curves(detection_results_sigma, trend_increase_sigma, npost_vec, 
                      savefile=f"{plots_dir}/power_curves_sigma.png")
    
    plot_power_curves_mu_sigma(detection_results_mu, detection_results_sigma, 
                               trend_increase_mu, trend_increase_sigma, npost_vec,
                               savefile=f"{plots_dir}/power_curves_mu_sigma_comparison.png")

    # Heatmaps
    plot_detection_heatmap(detection_results_mu, None, trend_increase_mu, npost_vec,
                           savefile=f"{plots_dir}/detection_heatmap_mu.png")
    plot_detection_heatmap(detection_results_sigma, None, trend_increase_sigma, npost_vec,
                           savefile=f"{plots_dir}/detection_heatmap_sigma.png")

    # Mean error
    plot_mean_error_curves(detection_results_mu, None, trend_increase_mu, npost_vec,
                           savefile=f"{plots_dir}/mean_error_curves_mu.png")
    plot_mean_error_curves(detection_results_sigma, None, trend_increase_sigma, npost_vec,
                           savefile=f"{plots_dir}/mean_error_curves_sigma.png")

    # Null distributions
    plot_null_distributions(critical_values, npost_short=24, npost_long=120,
                            savefile=f"{plots_dir}/null_distributions.png")

    print(f"Done. Plots saved to {plots_dir}")
