import time
import warnings

import numpy as np

from tracepy.changepoint.amoc import (
    CRITICAL_VALUE_NPOST_LONG_CDF,
    CRITICAL_VALUE_NPOST_SHORT_CDF,
    calculate_critical_values_cdf,
    run_main_simulation_mu,
    run_main_simulation_sigma,
)
from tracepy.cli._utils import fmt_elapsed
from tracepy.params.manager import load_params
from tracepy.plotting.reports import (
    detection_summary_table,
    plot_detection_heatmap,
    plot_distance_time_series,
    plot_distribution_difference,
    plot_fdr_heatmap,
    plot_mean_error_curves,
    plot_null_distributions,
    plot_power_curves,
    plot_power_curves_mu_sigma,
    plot_simulation_results,
)
from tracepy.simulation.distribution import ci_sim_cdf
from tracepy.stats.metrics import auc_diff_ts, wasserstein_distance_baci
from tracepy.store.persistence import (
    load_simulation_results,
    save_simulation_results,
    setup_plots_directory,
    setup_results_directory,
)

warnings.filterwarnings("ignore")

FOLDER = "distribution_amoc"


def run(quick: bool, plots_only: bool) -> None:
    cfg = load_params()
    sim = cfg["simulation"]
    runs = cfg["runs"]
    dist = cfg["distribution"]
    stats_cfg = cfg["stats"]

    npre = sim["npre"]
    npost_vec = np.arange(24, 121, dist["npost_step"])
    npost_max = int(npost_vec[-1])
    mu = dist["level_mu"]
    sigma = dist["level_sigma"]
    ns = dist["ns"]
    bw = dist["bw"]
    nd = dist["nd"]
    dist_measure = dist["dist_measure"]
    delay_set = np.arange(1, sim["delay_max"] + 1)
    alpha = stats_cfg["alpha"]
    Nsim = 10 if quick else runs["Nsim"]
    simN = 10 if quick else runs["simN"]

    trend_increase_mu = np.round(mu * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / 120, 4)
    trend_increase_sigma = np.round(sigma * np.arange(0.5, 2.1, 0.5) / 120, 4)

    plots_dir = setup_plots_directory(FOLDER)
    setup_results_directory(FOLDER)

    print("Distribution Change AMOC Detection")
    print("=" * 70)
    print(f"Pre-intervention period:  {npre} months")
    print(f"npost_vec:                {npost_vec[0]} to {npost_max} (step={dist['npost_step']})")
    print(f"Distribution mean (mu):   {mu}")
    print(f"Distribution std (sigma): {sigma}")
    print(f"Distance measure:         {dist_measure}")
    print(f"Samples per time point:   {ns}")
    print(f"Delay range:              {delay_set[0]}–{delay_set[-1]} months")
    print()

    loaded = load_simulation_results(FOLDER) or {}

    if plots_only and not loaded:
        print("ERROR: --plots-only requested but no saved results found.")
        raise SystemExit(1)

    critical_values = loaded.get("critical_values")
    detection_results_mu = loaded.get("detection_results_mu", {})
    detection_results_mu_ba = loaded.get("detection_results_mu_ba", {})
    detection_results_sigma = loaded.get("detection_results_sigma", {})

    def _save():
        save_simulation_results(
            FOLDER,
            {
                "critical_values": critical_values,
                "detection_results_mu": detection_results_mu,
                "detection_results_mu_ba": detection_results_mu_ba,
                "detection_results_sigma": detection_results_sigma,
            },
        )

    if not plots_only:
        t_total = time.perf_counter()

        if critical_values is not None:
            print("PHASE 1: Critical Values (loaded from cache, skipping)")
        else:
            print("=" * 70)
            print("PHASE 1: Critical Values (Null Distribution)")
            print("=" * 70)
            t0 = time.perf_counter()
            critical_values = calculate_critical_values_cdf(
                Nsim,
                npre,
                mu,
                sigma,
                ns,
                dist_measure,
                bw,
                nd,
                alpha,
            )
            print(f"  Phase 1 total: {fmt_elapsed(time.perf_counter() - t0)}")
            _save()

        cv = critical_values[f"npost_{CRITICAL_VALUE_NPOST_SHORT_CDF}"]["critical_value"]

        print("=" * 70)
        print(
            f"PHASE 2A: Mean Change BACI  ({len(detection_results_mu)}/{len(trend_increase_mu)} cached)"
        )
        print("=" * 70)
        t0 = time.perf_counter()

        def on_done_mu(r):
            nonlocal detection_results_mu
            detection_results_mu = r
            _save()

        detection_results_mu = run_main_simulation_mu(
            simN,
            trend_increase_mu,
            cv,
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
            existing_results=detection_results_mu,
            on_trend_done=on_done_mu,
        )
        _save()
        print(f"  Phase 2A total: {fmt_elapsed(time.perf_counter() - t0)}")

        print("=" * 70)
        print(
            f"PHASE 2A': Mean Change BA  ({len(detection_results_mu_ba)}/{len(trend_increase_mu)} cached)"
        )
        print("=" * 70)
        t0 = time.perf_counter()

        def on_done_mu_ba(r):
            nonlocal detection_results_mu_ba
            detection_results_mu_ba = r
            _save()

        detection_results_mu_ba = run_main_simulation_mu(
            simN,
            trend_increase_mu,
            cv,
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
            ba=True,
            existing_results=detection_results_mu_ba,
            on_trend_done=on_done_mu_ba,
        )
        _save()
        print(f"  Phase 2A' total: {fmt_elapsed(time.perf_counter() - t0)}")

        print("=" * 70)
        print(
            f"PHASE 2B: Variance Change  ({len(detection_results_sigma)}/{len(trend_increase_sigma)} cached)"
        )
        print("=" * 70)
        t0 = time.perf_counter()

        def on_done_sigma(r):
            nonlocal detection_results_sigma
            detection_results_sigma = r
            _save()

        detection_results_sigma = run_main_simulation_sigma(
            simN,
            trend_increase_sigma,
            cv,
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
            existing_results=detection_results_sigma,
            on_trend_done=on_done_sigma,
        )
        _save()
        print(f"  Phase 2B total: {fmt_elapsed(time.perf_counter() - t0)}")

        print(f"\nAll phases complete in {fmt_elapsed(time.perf_counter() - t_total)}")

    _print_summary(
        detection_results_mu, detection_results_sigma, trend_increase_mu, trend_increase_sigma
    )
    _generate_plots(
        plots_dir,
        critical_values,
        detection_results_mu,
        detection_results_mu_ba,
        detection_results_sigma,
        trend_increase_mu,
        trend_increase_sigma,
        npost_vec,
        npre,
        npost_max,
        mu,
        sigma,
        ns,
        dist_measure,
        bw,
        nd,
    )


def _print_summary(
    detection_results_mu, detection_results_sigma, trend_increase_mu, trend_increase_sigma
):
    print("\nRESULTS SUMMARY (at npost_max)")
    print(f"{'Trend Mu':>10} | {'Detect Rate':>12}")
    for t in trend_increase_mu:
        print(f"{t:10.4f} | {detection_results_mu[t]['detection_rates'][-1]:12.2%}")
    print(f"\n{'Trend Sigma':>10} | {'Detect Rate':>12}")
    for t in trend_increase_sigma:
        print(f"{t:10.4f} | {detection_results_sigma[t]['detection_rates'][-1]:12.2%}")


def _generate_plots(
    plots_dir,
    critical_values,
    detection_results_mu,
    detection_results_mu_ba,
    detection_results_sigma,
    trend_increase_mu,
    trend_increase_sigma,
    npost_vec,
    npre,
    npost_max,
    mu,
    sigma,
    ns,
    dist_measure,
    bw,
    nd,
):
    print("\nGenerating plots...")
    example_mu = trend_increase_mu[5]
    res = detection_results_mu[example_mu]
    sim_data = ci_sim_cdf(
        seed=res["seeds"][0],
        npre=npre + res["delays"][0],
        npost=npost_max - res["delays"][0],
        level=[mu, sigma],
        trend=[example_mu, 0],
        ns=ns,
    )
    if dist_measure == "wasserstein":
        dist_ts = wasserstein_distance_baci(sim_data["sample_ctr"], sim_data["sample_itv"])
    else:
        dist_ts = auc_diff_ts(sim_data["sample_ctr"], sim_data["sample_itv"], bw=bw, nd=nd)
    ex_stats = {"Tmax": res["tmax_matrix"][0, -1], "cpt": res["cpt_matrix"][0, -1]}

    plot_distance_time_series(
        sim_data,
        npre,
        dist_ts,
        stats=ex_stats,
        dist_measure=dist_measure,
        savefile=f"{plots_dir}/example_distance_time_series.png",
    )
    plot_distribution_difference(
        sim_data,
        npre,
        dist_ts,
        stats=ex_stats,
        dist_measure=dist_measure,
        savefile=f"{plots_dir}/example_distribution_difference.png",
    )
    plot_simulation_results(
        [detection_results_mu[t]["detection_rates"][-1] for t in trend_increase_mu],
        trend_increase_mu,
        savefile=f"{plots_dir}/sensitivity_analysis_mu.png",
    )
    plot_simulation_results(
        [detection_results_sigma[t]["detection_rates"][-1] for t in trend_increase_sigma],
        trend_increase_sigma,
        savefile=f"{plots_dir}/sensitivity_analysis_sigma.png",
    )
    plot_power_curves(
        detection_results_mu,
        trend_increase_mu,
        npost_vec,
        savefile=f"{plots_dir}/power_curves_mu.png",
    )
    plot_power_curves(
        detection_results_sigma,
        trend_increase_sigma,
        npost_vec,
        savefile=f"{plots_dir}/power_curves_sigma.png",
    )
    plot_power_curves_mu_sigma(
        detection_results_mu,
        detection_results_sigma,
        trend_increase_mu,
        trend_increase_sigma,
        npost_vec,
        savefile=f"{plots_dir}/power_curves_mu_sigma_comparison.png",
    )
    plot_detection_heatmap(
        detection_results_mu,
        None,
        trend_increase_mu,
        npost_vec,
        savefile=f"{plots_dir}/detection_heatmap_mu.png",
    )
    plot_detection_heatmap(
        detection_results_sigma,
        None,
        trend_increase_sigma,
        npost_vec,
        savefile=f"{plots_dir}/detection_heatmap_sigma.png",
    )
    plot_mean_error_curves(
        detection_results_mu,
        None,
        trend_increase_mu,
        npost_vec,
        savefile=f"{plots_dir}/mean_error_curves_mu.png",
    )
    plot_mean_error_curves(
        detection_results_sigma,
        None,
        trend_increase_sigma,
        npost_vec,
        savefile=f"{plots_dir}/mean_error_curves_sigma.png",
    )
    plot_null_distributions(
        critical_values,
        npost_short=CRITICAL_VALUE_NPOST_SHORT_CDF,
        npost_long=CRITICAL_VALUE_NPOST_LONG_CDF,
        savefile=f"{plots_dir}/null_distributions.png",
    )
    plot_fdr_heatmap(
        detection_results_mu,
        None,
        trend_increase_mu,
        npost_vec,
        npre,
        savefile=f"{plots_dir}/fdr_heatmap_mu.png",
    )
    plot_fdr_heatmap(
        detection_results_sigma,
        None,
        trend_increase_sigma,
        npost_vec,
        npre,
        savefile=f"{plots_dir}/fdr_heatmap_sigma.png",
    )
    df_mu = detection_summary_table(detection_results_mu, trend_increase_mu, npre)
    df_sigma = detection_summary_table(detection_results_sigma, trend_increase_sigma, npre)
    print("Mean-change (Mu) summary:")
    print(df_mu.to_string(index=False))
    print("\nVariance-change (Sigma) summary:")
    print(df_sigma.to_string(index=False))
    print(f"All plots saved to: {plots_dir}/")
    print("ANALYSIS COMPLETE")
    print("=" * 70)
