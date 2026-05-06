import time
import warnings

import numpy as np

from tracepy.changepoint.amoc import load_crit_val_table, lookup_crit_val
from tracepy.changepoint.forecast import run_simulation_cdf_ba, run_simulation_cdf_baci
from tracepy.cli._utils import fmt_elapsed
from tracepy.params.manager import load_params
from tracepy.plotting.reports import (
    plot_detection_heatmap,
    plot_forecast_detection_summary,
    plot_power_curves,
)
from tracepy.store.persistence import (
    load_simulation_results,
    save_simulation_results,
    setup_plots_directory,
    setup_results_directory,
)

warnings.filterwarnings("ignore")

FOLDER = "distribution_forecast"


def run(quick: bool, plots_only: bool) -> None:
    cfg = load_params()
    sim = cfg["simulation"]
    runs = cfg["runs"]
    dist = cfg["distribution"]

    npre = sim["npre"]
    mu = dist["level_mu"]
    sigma = dist["level_sigma"]
    ns = dist["ns"]
    delay_set = np.arange(1, sim["delay_max"] + 1)
    simN = 10 if quick else runs["simN"]

    npost_vec = np.arange(24, 121, 12)
    npost_max = int(npost_vec[-1])
    ntt = npre + npost_max

    trend_increase_mu = np.round(mu * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / 120, 4)

    crit_val = lookup_crit_val(load_crit_val_table(), detector="PageCUSUM", gamma=0.0, alpha=0.05)

    plots_dir = setup_plots_directory(FOLDER)
    setup_results_directory(FOLDER)

    print("Distribution Change Forecast Detection (Page-CUSUM)")
    print("=" * 70)
    print(f"Pre-intervention period:  {npre} months")
    print(f"Post-intervention max:    {npost_max} months")
    print(f"Total series length:      {ntt} months")
    print(f"Distribution mean (mu):   {mu}")
    print(f"Distribution std (sigma): {sigma}")
    print(f"Samples per time point:   {ns}")
    print(f"Delay range:              {delay_set[0]}–{delay_set[-1]} months")
    print(f"Critical value:           {crit_val:.7f}")
    print()

    loaded = load_simulation_results(FOLDER) or {}

    if plots_only and not loaded:
        print("ERROR: --plots-only requested but no saved results found.")
        raise SystemExit(1)

    detection_results_baci = loaded.get("detection_results_baci", {})
    detection_results_ba = loaded.get("detection_results_ba", {})

    def _save():
        save_simulation_results(
            FOLDER,
            {
                "detection_results_baci": detection_results_baci,
                "detection_results_ba": detection_results_ba,
            },
        )

    if not plots_only:
        t_total = time.perf_counter()

        print("=" * 70)
        print(f"PHASE 1: BACI  ({len(detection_results_baci)}/{len(trend_increase_mu)} cached)")
        print("=" * 70)
        t0 = time.perf_counter()

        def on_done_baci(r):
            nonlocal detection_results_baci
            detection_results_baci = r
            _save()

        detection_results_baci = run_simulation_cdf_baci(
            simN=simN,
            trend_increase_mu=trend_increase_mu,
            crit_val=crit_val,
            npre=npre,
            ntt=ntt,
            npost_max=npost_max,
            mu=mu,
            sigma=sigma,
            ns=ns,
            delay_set=delay_set,
            existing_results=detection_results_baci,
            on_trend_done=on_done_baci,
        )
        _save()
        print(f"  Phase 1 total: {fmt_elapsed(time.perf_counter() - t0)}")
        print()

        print("=" * 70)
        print(f"PHASE 2: BA  ({len(detection_results_ba)}/{len(trend_increase_mu)} cached)")
        print("=" * 70)
        t0 = time.perf_counter()

        def on_done_ba(r):
            nonlocal detection_results_ba
            detection_results_ba = r
            _save()

        detection_results_ba = run_simulation_cdf_ba(
            simN=simN,
            trend_increase_mu=trend_increase_mu,
            crit_val=crit_val,
            npre=npre,
            ntt=ntt,
            npost_max=npost_max,
            mu=mu,
            sigma=sigma,
            ns=ns,
            delay_set=delay_set,
            existing_results=detection_results_ba,
            on_trend_done=on_done_ba,
        )
        _save()
        print(f"  Phase 2 total: {fmt_elapsed(time.perf_counter() - t0)}")
        print()

        print(f"All phases complete in {fmt_elapsed(time.perf_counter() - t_total)}")
        print()

    _print_summary(detection_results_baci, detection_results_ba, trend_increase_mu)
    _generate_plots(
        plots_dir, detection_results_baci, detection_results_ba, trend_increase_mu, npost_vec, npre
    )


def _print_summary(detection_results_baci, detection_results_ba, trend_increase_mu):
    col = f"{'Trend Mu':>10} | {'BACI Detect':>12} | {'BA Detect':>10}"
    sep = "-" * len(col)
    print("\nRESULTS SUMMARY (at npost_max)")
    print(sep)
    print(col)
    print(sep)
    for t in trend_increase_mu:
        baci_rate = detection_results_baci[t]["detection_rate"]
        ba_rate = detection_results_ba[t]["detection_rate"]
        print(f"{t:10.4f} | {baci_rate:12.2%} | {ba_rate:10.2%}")
    print(sep)
    print()


def _generate_plots(
    plots_dir, detection_results_baci, detection_results_ba, trend_increase_mu, npost_vec, npre
):
    print("Generating plots...")

    for label, results in [("baci", detection_results_baci), ("ba", detection_results_ba)]:
        for t in trend_increase_mu:
            time_est_vec = np.array(results[t]["time_est_vec"], dtype=float)
            results[t]["detection_rates"] = np.array(
                [
                    float(np.mean(np.isfinite(time_est_vec) & (time_est_vec <= npost)))
                    for npost in npost_vec
                ]
            )

    plot_power_curves(
        detection_results_baci,
        trend_increase_mu,
        npost_vec,
        savefile=f"{plots_dir}/power_curves_baci.png",
    )
    plot_power_curves(
        detection_results_ba,
        trend_increase_mu,
        npost_vec,
        savefile=f"{plots_dir}/power_curves_ba.png",
    )
    plot_detection_heatmap(
        detection_results_baci,
        None,
        trend_increase_mu,
        npost_vec,
        savefile=f"{plots_dir}/detection_heatmap_baci.png",
    )
    plot_detection_heatmap(
        detection_results_ba,
        None,
        trend_increase_mu,
        npost_vec,
        savefile=f"{plots_dir}/detection_heatmap_ba.png",
    )

    plot_forecast_detection_summary(
        detection_results_baci,
        trend_increase_mu,
        npre,
        title="Distribution Forecast BACI",
        savefile=f"{plots_dir}/summary_table_baci.csv",
    )
    plot_forecast_detection_summary(
        detection_results_ba,
        trend_increase_mu,
        npre,
        title="Distribution Forecast BA",
        savefile=f"{plots_dir}/summary_table_ba.csv",
    )

    print(f"All plots saved to: {plots_dir}/")
    print("ANALYSIS COMPLETE")
    print("=" * 70)
