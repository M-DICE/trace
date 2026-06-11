import time
import warnings

import numpy as np

from tracepy.changepoint.amoc import load_crit_val_table, lookup_crit_val
from tracepy.changepoint.forecast import run_simulation_ar, run_simulation_iid
from tracepy.cli._utils import fmt_elapsed
from tracepy.params.manager import load_params
from tracepy.plotting.reports import plot_simulation_results, plot_time_series
from tracepy.simulation.trend import ci_sim
from tracepy.store.persistence import (
    load_simulation_results,
    save_simulation_results,
    setup_plots_directory,
    setup_results_directory,
)

warnings.filterwarnings("ignore")

FOLDER = "trend_forecast"


def run(quick: bool, plots_only: bool, no_cache: bool = False) -> None:
    cfg = load_params()
    sim = cfg["simulation"]
    runs = cfg["runs"]
    fcst = cfg["forecast"]

    npre = sim["npre"]
    npost_months = 12 * sim["npost_years"]
    npost_max = npost_months
    ntt = npre + npost_max
    level = sim["level"]
    trend_control = sim["trend_control"]
    sigma = sim["sigma"]
    phi = fcst["phi"]
    delay_set = np.arange(1, sim["delay_max"] + 1)
    simN = 10 if quick else runs["simN"]

    crit_val = lookup_crit_val(load_crit_val_table(), gamma=fcst["gamma"])

    trend_increase = np.round(
        level * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / npost_months, 4
    )

    plots_dir = setup_plots_directory(FOLDER)
    setup_results_directory(FOLDER)

    print("Trend Change Forecast Detection (Page-CUSUM)")
    print("=" * 70)
    print(f"Pre-intervention period: {npre} months")
    print(f"Post-intervention max:   {npost_max} months")
    print(f"Total series length:     {ntt} months")
    print(f"Trend increments:        {len(trend_increase)}")
    print(f"Noise level (sigma):     {sigma}")
    print(f"AR(1) coefficient (phi): {phi}")
    print(f"Delay range:             {delay_set[0]}–{delay_set[-1]} months")
    print(f"Critical value:          {crit_val:.7f}")
    print()

    loaded = {} if no_cache else (load_simulation_results(FOLDER) or {})

    if plots_only and not loaded:
        print("ERROR: --plots-only requested but no saved results found.")
        raise SystemExit(1)

    detection_results_iid = loaded.get("detection_results", loaded.get("detection_results_iid", {}))
    detection_results_ar = loaded.get("detection_results_ar", {})

    def _save():
        save_simulation_results(
            FOLDER,
            {
                "critical_values": crit_val,
                "detection_results": detection_results_iid,
                "detection_results_ar": detection_results_ar,
            },
        )

    if not plots_only:
        t_total = time.perf_counter()

        print("=" * 70)
        print(f"PHASE 2: i.i.d. BA  ({len(detection_results_iid)}/{len(trend_increase)} cached)")
        print("=" * 70)
        t0 = time.perf_counter()

        def on_done_iid(r):
            nonlocal detection_results_iid
            detection_results_iid = r
            _save()

        detection_results_iid = run_simulation_iid(
            simN=simN,
            trend_increase=trend_increase,
            crit_val=crit_val,
            npre=npre,
            ntt=ntt,
            npost_max=npost_max,
            level=level,
            trend_control=trend_control,
            sigma=sigma,
            delay_set=delay_set,
            existing_results=detection_results_iid,
            on_trend_done=on_done_iid,
        )
        _save()
        print(f"  Phase 2 total: {fmt_elapsed(time.perf_counter() - t0)}")
        print()

        print("=" * 70)
        print(f"PHASE 3: AR(1) BA  ({len(detection_results_ar)}/{len(trend_increase)} cached)")
        print("=" * 70)
        t0 = time.perf_counter()

        def on_done_ar(r):
            nonlocal detection_results_ar
            detection_results_ar = r
            _save()

        detection_results_ar = run_simulation_ar(
            simN=simN,
            trend_increase=trend_increase,
            crit_val=crit_val,
            npre=npre,
            ntt=ntt,
            npost_max=npost_max,
            level=level,
            trend_control=trend_control,
            sigma=sigma,
            phi=phi,
            delay_set=delay_set,
            existing_results=detection_results_ar,
            on_trend_done=on_done_ar,
        )
        _save()
        print(f"  Phase 3 total: {fmt_elapsed(time.perf_counter() - t0)}")
        print()

        print(f"All phases complete in {fmt_elapsed(time.perf_counter() - t_total)}")
        print()

    _print_summary(detection_results_iid, detection_results_ar, trend_increase, crit_val)
    _generate_plots(
        plots_dir,
        detection_results_iid,
        detection_results_ar,
        trend_increase,
        npre,
        npost_max,
        level,
        trend_control,
        sigma,
    )


def _print_summary(detection_results_iid, detection_results_ar, trend_increase, crit_val):
    col = (
        f"{'Trend':>8}  {'iid detect':>11}  {'iid time':>9}  {'iid error':>10}  "
        f"{'ar1 detect':>11}  {'ar1 time':>9}  {'ar1 error':>10}"
    )
    sep = "-" * len(col)
    print("=" * len(col))
    print("RESULTS SUMMARY")
    print(f"  crit_val: {crit_val:.7f}")
    print("=" * len(col))
    print(col)
    print(sep)
    for trend_inc in trend_increase:
        iid = detection_results_iid[trend_inc]
        ar1 = detection_results_ar[trend_inc]
        print(
            f"{trend_inc:>8.4f}  "
            f"{iid['detection_rate']:>11.2%}  "
            f"{iid['mean_time']:>9.1f}  "
            f"{iid['mean_error']:>9.2f}m  "
            f"{ar1['detection_rate']:>11.2%}  "
            f"{ar1['mean_time']:>9.1f}  "
            f"{ar1['mean_error']:>9.2f}m"
        )
    print(sep)
    print()


def _generate_plots(
    plots_dir,
    detection_results_iid,
    detection_results_ar,
    trend_increase,
    npre,
    npost_max,
    level,
    trend_control,
    sigma,
):
    print("Generating plots...")
    example_trend = trend_increase[3]
    res = detection_results_iid[example_trend]
    ex_sim_data = ci_sim(
        seed=res["seeds"][0],
        npre=npre + res["delays"][0],
        npost=npost_max - res["delays"][0],
        level=level,
        trend=[trend_control, trend_control + example_trend],
        sigma=sigma,
    )
    ex_stats = {"cpt": res["cpt_est_vec"][0], "time_est": res["time_est_vec"][0]}
    plot_time_series(
        ex_sim_data, npre, stats=ex_stats, savefile=f"{plots_dir}/example_time_series.png"
    )
    plot_simulation_results(
        [detection_results_iid[t]["detection_rate"] for t in trend_increase],
        trend_increase,
        savefile=f"{plots_dir}/sensitivity_analysis.png",
    )
    print(f"All plots saved to: {plots_dir}/")
    print("ANALYSIS COMPLETE")
    print("=" * 70)
