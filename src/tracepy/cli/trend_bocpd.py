"""Trend-change detection via BOCPD (Bayesian Online Changepoint Detection).

Python port of the simulation study in
``SimRewilding/Rewild_trend_change_BOCPD.R``.

For each effect-size value in the trend grid this runs ``simN`` Monte Carlo
replications.  Each replication simulates a control/intervention pair with a
random onset *delay*, forms the difference series ``x = y_itv - y_ctr``, and
runs online BOCPD over it, stopping at the first declared changepoint.

Unlike the AMOC/Forecast pipelines, BOCPD is an online detector and needs no
null-distribution critical value, so there is a single simulation phase.  Each
increment is dispatched as a single R call (via ``run_bocpd_increment``) so
that the per-step Python↔R bridge overhead is eliminated.
"""

import time
import warnings

import numpy as np

from tracepy.changepoint.bocpd import init_r, run_bocpd_increment
from tracepy.cli._utils import (
    fmt_elapsed,
    print_complete,
    print_run_header,
    print_stage,
    print_summary_header,
)
from tracepy.params.manager import load_params
from tracepy.plotting.reports import (
    plot_detection_error_distribution,
    plot_simulation_results,
    plot_time_series,
)
from tracepy.simulation.trend import ci_sim
from tracepy.store.persistence import (
    load_simulation_results,
    save_simulation_results,
    setup_plots_directory,
    setup_results_directory,
)

warnings.filterwarnings("ignore")

FOLDER = "trend_bocpd"


def run(quick: bool, plots_only: bool, no_cache: bool = False) -> None:
    cfg = load_params()
    sim = cfg["simulation"]
    runs = cfg["runs"]
    bo = cfg["bocpd"]

    npre = sim["npre"]
    npost_months = 12 * sim["npost_years"]
    npost_max = npost_months
    level = sim["level"]
    trend_control = sim["trend_control"]
    sigma = sim["sigma"]
    delay_set = np.arange(1, sim["delay_max"] + 1)
    simN = 10 if quick else runs["simN"]

    trend_increase = np.round(
        level * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / npost_months, 4
    )
    example_trend = trend_increase[3]

    bocpd_kwargs = dict(
        prior_alpha=(bo["prior_alpha_intercept"], bo["prior_alpha_slope"]),
        prior_sigma2=bo["prior_sigma2"],
        sig_prior=(bo["sig_prior_shape"], bo["sig_prior_rate"]),
        pm=bo["pm"],
        ptr=bo["ptr"],
        maxp=bo["maxp"],
        np_=bo["np"],
        msl=bo["msl"],
    )

    plots_dir = setup_plots_directory(FOLDER)
    setup_results_directory(FOLDER)

    print_run_header(
        "Trend Change BOCPD Detection",
        [
            ("Pre-intervention period", f"{npre} months"),
            ("Post-intervention length", f"{npost_max} months"),
            ("Number of trend increments", len(trend_increase)),
            ("Noise level (sigma)", sigma),
            ("Delay range", f"{delay_set[0]}–{delay_set[-1]} months"),
            ("Min segment length (msl)", bo["msl"]),
            ("Run-length hazard (ptr)", bo["ptr"]),
            ("Replications per increment", simN),
        ],
    )

    loaded = {} if no_cache else (load_simulation_results(FOLDER) or {})
    detection_results = loaded.get("detection_results", {})

    if plots_only and not detection_results:
        print("ERROR: --plots-only requested but no saved results found.")
        raise SystemExit(1)

    if not plots_only:
        print_stage("Detection — BOCPD simulations", len(detection_results), len(trend_increase))
        init_r()
        t_total = time.perf_counter()

        for m, trend_inc in enumerate(trend_increase, start=1):
            if trend_inc in detection_results:
                print(
                    f"  [{m}/{len(trend_increase)}] trend_inc={trend_inc:.4f}  (cached, skipping)"
                )
                continue

            t0 = time.perf_counter()
            print(
                f"  [{m}/{len(trend_increase)}] trend_inc={trend_inc:.4f}  ({simN} sims) ...",
                end="",
                flush=True,
            )
            result = run_bocpd_increment(
                trend_inc=trend_inc,
                trend_idx=m,
                n_trends=len(trend_increase),
                simN=simN,
                npre=npre,
                npost_max=npost_max,
                level=level,
                trend_control=trend_control,
                sigma=sigma,
                delay_max=int(delay_set[-1]),
                prior_alpha=bocpd_kwargs["prior_alpha"],
                prior_sigma2=bocpd_kwargs["prior_sigma2"],
                sig_prior=bocpd_kwargs["sig_prior"],
                pm=bocpd_kwargs["pm"],
                ptr=bocpd_kwargs["ptr"],
                maxp=bocpd_kwargs["maxp"],
                np_=bocpd_kwargs["np_"],
                msl=bocpd_kwargs["msl"],
            )
            detection_results[trend_inc] = result
            save_simulation_results(FOLDER, {"detection_results": detection_results})
            print(
                f" done in {fmt_elapsed(time.perf_counter() - t0)}  "
                f"detect={result['detection_rate']:.2%}  "
                f"err={result['mean_error']:.2f}mo"
            )

        print()
        print(f"Total runtime: {fmt_elapsed(time.perf_counter() - t_total)}")
        print()

    _print_summary(detection_results, trend_increase)
    _generate_plots(
        plots_dir,
        detection_results,
        trend_increase,
        example_trend,
        npre,
        npost_max,
        level,
        trend_control,
        sigma,
    )


def _print_summary(detection_results, trend_increase):
    col = f"{'Trend':>8}  {'detect rate':>12}  {'mean err':>10}  {'mean t_detect':>14}"
    sep = "-" * len(col)
    print_summary_header("Results summary — BOCPD trend-change detection", len(col))
    print(col)
    print(sep)
    for trend_inc in trend_increase:
        r = detection_results.get(trend_inc)
        if r is None:
            continue
        err = "n/a" if np.isnan(r["mean_error"]) else f"{r['mean_error']:>8.2f}mo"
        ttd_val = r["mean_time_to_detect"]
        ttd = "n/a" if np.isnan(ttd_val) else f"{ttd_val:>12.1f}mo"
        print(f"{trend_inc:>8.4f}  {r['detection_rate']:>12.2%}  {err:>10}  {ttd:>14}")
    print(sep)
    print()


def _generate_plots(
    plots_dir,
    detection_results,
    trend_increase,
    example_trend,
    npre,
    npost_max,
    level,
    trend_control,
    sigma,
):
    print("Generating plots...")

    detected = [t for t in trend_increase if t in detection_results]
    plot_simulation_results(
        [detection_results[t]["detection_rate"] for t in detected],
        detected,
        savefile=f"{plots_dir}/sensitivity_analysis.png",
    )

    # Pooled changepoint-location errors across all increments.
    all_errors = [e for t in detected for e in detection_results[t]["errors"]]
    if all_errors:
        plot_detection_error_distribution(
            all_errors, savefile=f"{plots_dir}/changepoint_error_distribution.png"
        )

    # Example difference series for the reference increment.
    sim_data_test = ci_sim(
        seed=42,
        npre=npre,
        npost=npost_max,
        level=level,
        trend=[trend_control, trend_control + example_trend],
        sigma=sigma,
    )
    plot_time_series(sim_data_test, npre, savefile=f"{plots_dir}/example_time_series.png")

    print_complete(plots_dir)
