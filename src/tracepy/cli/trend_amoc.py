import time
import warnings

import numpy as np

from tracepy.changepoint.amoc import (
    CRITICAL_VALUE_NPOST_LONG,
    CRITICAL_VALUE_NPOST_SHORT,
    calculate_critical_values,
    run_main_simulation,
    run_main_simulation_ar,
    run_main_simulation_iid_ba,
)
from tracepy.cli._utils import fmt_elapsed
from tracepy.params.manager import load_params
from tracepy.plotting.reports import (
    detection_summary_table,
    plot_changepoint_bias,
    plot_detection_by_delay,
    plot_detection_heatmap,
    plot_fdr_heatmap,
    plot_mean_error_curves,
    plot_null_distributions,
    plot_power_curves,
    plot_power_curves_comparison,
    plot_simulation_results,
    plot_time_series,
    plot_time_to_detection,
    plot_tmax_signal_vs_noise,
)
from tracepy.simulation.trend import ci_sim
from tracepy.stats.metrics import trend_stats
from tracepy.store.persistence import (
    load_simulation_results,
    save_simulation_results,
    setup_plots_directory,
    setup_results_directory,
)

warnings.filterwarnings("ignore")

FOLDER = "trend_amoc"


def run(quick: bool, plots_only: bool) -> None:
    cfg = load_params()
    sim = cfg["simulation"]
    runs = cfg["runs"]
    stats = cfg["stats"]

    npre = sim["npre"]
    npost_months = 12 * sim["npost_years"]
    npost_vec = np.arange(npre, npost_months + 1, 12)
    npost_max = int(npost_vec[-1])
    level = sim["level"]
    trend_control = sim["trend_control"]
    sigma = sim["sigma"]
    phi = sim["phi"]
    delay_set = np.arange(1, sim["delay_max"] + 1)
    alpha = stats["alpha"]
    Nsim = 10 if quick else runs["Nsim"]
    simN = 10 if quick else runs["simN"]

    trend_increase = np.round(
        level * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / npost_months, 4
    )
    example_trend = trend_increase[3]

    plots_dir = setup_plots_directory(FOLDER)
    setup_results_directory(FOLDER)

    print("Trend Change AMOC Detection")
    print("=" * 70)
    print(f"Pre-intervention period:          {npre} months")
    print(f"Post-intervention range:          {npost_vec[0]} to {npost_max} months")
    print(f"Number of trend increments:       {len(trend_increase)}")
    print(f"Noise level (sigma):              {sigma}")
    print(f"AR(1) coefficient (phi):          {phi}")
    print(f"Delay range:                      {delay_set[0]}–{delay_set[-1]} months")
    print(
        f"Critical value npost lengths:     {CRITICAL_VALUE_NPOST_SHORT} and {CRITICAL_VALUE_NPOST_LONG} months"
    )
    print()

    loaded = load_simulation_results(FOLDER) or {}

    if plots_only and not loaded:
        print("ERROR: --plots-only requested but no saved results found.")
        raise SystemExit(1)

    critical_values = loaded.get("critical_values")
    detection_results = loaded.get("detection_results", {})
    detection_results_ar = loaded.get("detection_results_ar", {})
    detection_results_iid_ba = loaded.get("detection_results_iid_ba", {})
    test_stats = loaded.get("test_stats", {})

    def _save():
        save_simulation_results(
            FOLDER,
            {
                "critical_values": critical_values,
                "detection_results": detection_results,
                "detection_results_ar": detection_results_ar,
                "detection_results_iid_ba": detection_results_iid_ba,
                "test_stats": test_stats,
            },
        )

    if not plots_only:
        t_total = time.perf_counter()

        sim_data_test = ci_sim(
            seed=42,
            npre=npre,
            npost=npost_max,
            level=level,
            trend=[trend_control, trend_control + example_trend],
            sigma=sigma,
        )
        stats_test = trend_stats(
            y_ctr=sim_data_test["y_ctr"], y_itv=sim_data_test["y_itv"], nt=npre + npost_max
        )
        test_stats = {
            "Tmax": stats_test["Tmax"],
            "cpt": stats_test["cpt"],
            "npre": npre,
            "alpha": alpha,
        }
        print(
            f"Example (seed=42): Tmax={stats_test['Tmax']:.4f}, "
            f"cpt={stats_test['cpt']} months (true: {npre})"
        )
        print()

        if critical_values is not None:
            print("PHASE 1: Critical Values (loaded from cache, skipping)")
            for key, cv_data in critical_values.items():
                print(f"  [{key}] cv = {cv_data['critical_value']:.4f}")
        else:
            print("=" * 70)
            print("PHASE 1: Calculating Critical Values (Null Distribution)")
            print("=" * 70)
            t0 = time.perf_counter()
            critical_values = calculate_critical_values(
                Nsim=Nsim,
                npre=npre,
                level=level,
                trend_control=trend_control,
                sigma=sigma,
                phi=phi,
                alpha=alpha,
            )
            print(f"  Phase 1 total: {fmt_elapsed(time.perf_counter() - t0)}")
            _save()
        print()

        cv_iid = critical_values[f"iid_{CRITICAL_VALUE_NPOST_SHORT}"]["critical_value"]
        cv_ar = critical_values[f"ar1_{CRITICAL_VALUE_NPOST_SHORT}"]["critical_value"]
        cv_iid_ba = critical_values[f"iid_ba_{CRITICAL_VALUE_NPOST_SHORT}"]["critical_value"]

        print("=" * 70)
        print(f"PHASE 2: i.i.d. BACI  ({len(detection_results)}/{len(trend_increase)} cached)")
        print("=" * 70)
        t0 = time.perf_counter()

        def on_done_iid(r):
            nonlocal detection_results
            detection_results = r
            _save()

        detection_results = run_main_simulation(
            simN=simN,
            trend_increase=trend_increase,
            critical_value=cv_iid,
            npre=npre,
            npost_max=npost_max,
            npost_vec=npost_vec,
            level=level,
            trend_control=trend_control,
            sigma=sigma,
            delay_set=delay_set,
            existing_results=detection_results,
            on_trend_done=on_done_iid,
        )
        _save()
        print(f"  Phase 2 total: {fmt_elapsed(time.perf_counter() - t0)}")
        print()

        print("=" * 70)
        print(f"PHASE 3: AR(1)  ({len(detection_results_ar)}/{len(trend_increase)} cached)")
        print("=" * 70)
        t0 = time.perf_counter()

        def on_done_ar(r):
            nonlocal detection_results_ar
            detection_results_ar = r
            _save()

        detection_results_ar = run_main_simulation_ar(
            simN=simN,
            trend_increase=trend_increase,
            critical_value=cv_ar,
            npre=npre,
            npost_max=npost_max,
            npost_vec=npost_vec,
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

        print("=" * 70)
        print(f"PHASE 4: i.i.d. BA  ({len(detection_results_iid_ba)}/{len(trend_increase)} cached)")
        print("=" * 70)
        t0 = time.perf_counter()

        def on_done_ba(r):
            nonlocal detection_results_iid_ba
            detection_results_iid_ba = r
            _save()

        detection_results_iid_ba = run_main_simulation_iid_ba(
            simN=simN,
            trend_increase=trend_increase,
            critical_value=cv_iid_ba,
            npre=npre,
            npost_max=npost_max,
            npost_vec=npost_vec,
            level=level,
            trend_control=trend_control,
            sigma=sigma,
            delay_set=delay_set,
            existing_results=detection_results_iid_ba,
            on_trend_done=on_done_ba,
        )
        _save()
        print(f"  Phase 4 total: {fmt_elapsed(time.perf_counter() - t0)}")
        print()

        print(f"All phases complete in {fmt_elapsed(time.perf_counter() - t_total)}")
        print()

    if plots_only:
        sim_data_test = ci_sim(
            seed=42,
            npre=npre,
            npost=npost_max,
            level=level,
            trend=[trend_control, trend_control + example_trend],
            sigma=sigma,
        )
        stats_test = trend_stats(
            y_ctr=sim_data_test["y_ctr"], y_itv=sim_data_test["y_itv"], nt=npre + npost_max
        )

    _print_summary(
        critical_values,
        detection_results,
        detection_results_ar,
        detection_results_iid_ba,
        trend_increase,
        npost_max,
    )
    _generate_plots(
        plots_dir,
        critical_values,
        detection_results,
        detection_results_ar,
        detection_results_iid_ba,
        sim_data_test,
        stats_test,
        trend_increase,
        npost_vec,
        npre,
    )


def _print_summary(
    critical_values,
    detection_results,
    detection_results_ar,
    detection_results_iid_ba,
    trend_increase,
    npost_max,
):
    has_ba = bool(detection_results_iid_ba)
    col = (
        f"{'Trend':>8}  {'iid detect':>11}  {'iid error':>10}  "
        f"{'ar1 detect':>11}  {'ar1 error':>10}"
        + (f"  {'ba detect':>10}  {'ba error':>9}" if has_ba else "")
    )
    sep = "-" * len(col)
    print("=" * len(col))
    print(f"RESULTS SUMMARY — at npost_max ({npost_max} months)")
    print(
        f"  i.i.d. BACI cv: {critical_values[f'iid_{CRITICAL_VALUE_NPOST_SHORT}']['critical_value']:.4f}"
    )
    print(
        f"  AR(1)  BA   cv: {critical_values[f'ar1_{CRITICAL_VALUE_NPOST_SHORT}']['critical_value']:.4f}"
    )
    if has_ba:
        print(
            f"  i.i.d. BA   cv: {critical_values[f'iid_ba_{CRITICAL_VALUE_NPOST_SHORT}']['critical_value']:.4f}"
        )
    print("=" * len(col))
    print(col)
    print(sep)
    for trend_inc in trend_increase:
        iid = detection_results[trend_inc]
        ar1 = detection_results_ar[trend_inc]
        line = (
            f"{trend_inc:>8.4f}  {iid['detection_rates'][-1]:>11.2%}  {iid['mean_errors'][-1]:>9.2f}m  "
            f"{ar1['detection_rates'][-1]:>11.2%}  {ar1['mean_errors'][-1]:>9.2f}m"
        )
        if has_ba:
            ba = detection_results_iid_ba[trend_inc]
            line += f"  {ba['detection_rates'][-1]:>10.2%}  {ba['mean_errors'][-1]:>8.2f}m"
        print(line)
    print(sep)
    print()


def _generate_plots(
    plots_dir,
    critical_values,
    detection_results,
    detection_results_ar,
    detection_results_iid_ba,
    sim_data_test,
    stats_test,
    trend_increase,
    npost_vec,
    npre,
):
    print("Generating plots...")
    plot_simulation_results(
        [detection_results[t]["detection_rates"][-1] for t in trend_increase],
        trend_increase,
        savefile=f"{plots_dir}/sensitivity_analysis.png",
    )
    plot_time_series(
        sim_data_test, npre, stats=stats_test, savefile=f"{plots_dir}/example_time_series.png"
    )
    plot_power_curves(
        detection_results, trend_increase, npost_vec, savefile=f"{plots_dir}/power_curves_iid.png"
    )
    plot_power_curves(
        detection_results_ar,
        trend_increase,
        npost_vec,
        savefile=f"{plots_dir}/power_curves_ar1.png",
    )
    plot_power_curves_comparison(
        detection_results,
        detection_results_ar,
        trend_increase,
        npost_vec,
        savefile=f"{plots_dir}/power_curves_comparison.png",
    )
    plot_time_to_detection(
        detection_results,
        detection_results_ar,
        trend_increase,
        npost_vec,
        savefile=f"{plots_dir}/time_to_detection.png",
    )
    plot_detection_heatmap(
        detection_results,
        detection_results_ar,
        trend_increase,
        npost_vec,
        savefile=f"{plots_dir}/detection_heatmap.png",
    )
    plot_mean_error_curves(
        detection_results,
        detection_results_ar,
        trend_increase,
        npost_vec,
        savefile=f"{plots_dir}/mean_error_curves.png",
    )
    plot_null_distributions(
        critical_values,
        npost_short=CRITICAL_VALUE_NPOST_SHORT,
        npost_long=CRITICAL_VALUE_NPOST_LONG,
        savefile=f"{plots_dir}/null_distributions.png",
    )
    plot_tmax_signal_vs_noise(
        critical_values,
        detection_results,
        detection_results_ar,
        trend_increase,
        npost_vec,
        npost_short=CRITICAL_VALUE_NPOST_SHORT,
        savefile=f"{plots_dir}/tmax_signal_vs_noise.png",
    )
    plot_detection_by_delay(
        detection_results,
        detection_results_ar,
        trend_increase,
        savefile=f"{plots_dir}/detection_by_delay.png",
    )
    plot_changepoint_bias(
        detection_results,
        detection_results_ar,
        trend_increase,
        npre,
        savefile=f"{plots_dir}/changepoint_bias.png",
    )
    plot_fdr_heatmap(
        detection_results,
        detection_results_ar,
        trend_increase,
        npost_vec,
        npre,
        savefile=f"{plots_dir}/fdr_heatmap.png",
    )
    df_iid = detection_summary_table(detection_results, trend_increase, npre)
    df_ar = detection_summary_table(detection_results_ar, trend_increase, npre)
    print("i.i.d. BACI summary:")
    print(df_iid.to_string(index=False))
    print("\nAR(1) summary:")
    print(df_ar.to_string(index=False))
    print("=" * 70)
    print(f"All plots saved to: {plots_dir}/")
    print("ANALYSIS COMPLETE")
    print("=" * 70)
