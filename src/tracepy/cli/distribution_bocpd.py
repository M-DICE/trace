"""Distribution-change detection via BOCPD (Bayesian Online Changepoint Detection).

Python port of the simulation study in
``SimRewilding/Rewild_distribution_change_BOCPD.R``.

For each mean-shift value in the trend grid this runs ``simN`` Monte Carlo
replications.  Each replication simulates a control/intervention pair of
*distribution* time series with a random onset *delay*, forms the Wasserstein
distance series between the two, and runs online BOCPD over it, stopping at the
first declared changepoint.

This mirrors the trend-change BOCPD pipeline exactly; the only difference is the
online series ``x`` fed to ``run_bocpd``: here it is the Wasserstein distance
time series (``wasserstein_distance_baci``), not the scalar difference
``y_itv - y_ctr``.

Like the trend BOCPD pipeline, BOCPD is an online detector and needs no
null-distribution critical value, so there is a single simulation phase.  The
detector is driven through ``rpy2`` and is not picklable across processes, so
the Monte Carlo loop runs serially (matching the R reference).
"""

import time
import warnings

import numpy as np

from tracepy.changepoint.bocpd import (
    init_r,
    run_bocpd,
    run_bocpd_distribution_increment,
)
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
    plot_distance_time_series,
    plot_simulation_results,
)
from tracepy.simulation.distribution import ci_sim_cdf
from tracepy.stats.metrics import wasserstein_distance_baci
from tracepy.store.persistence import (
    load_simulation_results,
    save_simulation_results,
    setup_plots_directory,
    setup_results_directory,
)

warnings.filterwarnings("ignore")

FOLDER = "distribution_bocpd"


def run(quick: bool, plots_only: bool, no_cache: bool = False) -> None:
    cfg = load_params()
    sim = cfg["simulation"]
    runs = cfg["runs"]
    dist = cfg["distribution"]
    bo = cfg["bocpd"]

    npre = sim["npre"]
    npost_max = 12 * sim["npost_years"]
    mu = dist["level_mu"]
    sigma = dist["level_sigma"]
    ns = dist["ns"]
    delay_set = np.arange(1, sim["delay_max"] + 1)
    simN = 10 if quick else runs["simN"]

    trend_increase_mu = np.round(mu * np.concatenate([[0.05], np.arange(0.1, 1.1, 0.1)]) / 120, 4)
    example_trend = trend_increase_mu[3]

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
        "Distribution Change BOCPD Detection",
        [
            ("Pre-intervention period", f"{npre} months"),
            ("Post-intervention length", f"{npost_max} months"),
            ("Number of trend increments", len(trend_increase_mu)),
            ("Distribution mean (mu)", mu),
            ("Distribution std (sigma)", sigma),
            ("Samples per time point", ns),
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
        print_stage("Detection — BOCPD simulations", len(detection_results), len(trend_increase_mu))
        init_r()
        t_total = time.perf_counter()

        for m, trend_inc in enumerate(trend_increase_mu, start=1):
            if trend_inc in detection_results:
                print(
                    f"  [{m}/{len(trend_increase_mu)}] trend_inc={trend_inc:.4f}  "
                    f"(cached, skipping)"
                )
                continue

            t0 = time.perf_counter()
            print(
                f"  [{m}/{len(trend_increase_mu)}] trend_inc={trend_inc:.4f}  ({simN} sims) ...",
                end="",
                flush=True,
            )
            result = run_bocpd_distribution_increment(
                trend_inc=trend_inc,
                trend_idx=m,
                n_trends=len(trend_increase_mu),
                simN=simN,
                npre=npre,
                npost_max=npost_max,
                mu=mu,
                sigma=sigma,
                ns=ns,
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

    _print_summary(detection_results, trend_increase_mu)
    _generate_plots(
        plots_dir,
        detection_results,
        trend_increase_mu,
        example_trend,
        npre,
        npost_max,
        mu,
        sigma,
        ns,
        bocpd_kwargs,
    )


def _print_summary(detection_results, trend_increase_mu):
    col = f"{'Trend':>8}  {'detect rate':>12}  {'mean err':>10}  {'mean t_detect':>14}"
    sep = "-" * len(col)
    print_summary_header("Results summary — BOCPD distribution-change detection", len(col))
    print(col)
    print(sep)
    for trend_inc in trend_increase_mu:
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
    trend_increase_mu,
    example_trend,
    npre,
    npost_max,
    mu,
    sigma,
    ns,
    bocpd_kwargs,
):
    print("Generating plots...")

    detected = [t for t in trend_increase_mu if t in detection_results]
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

    # Example distance series for the reference increment.
    sim_data_test = ci_sim_cdf(
        seed=42,
        npre=npre,
        npost=npost_max,
        level=[mu, sigma],
        trend=[example_trend, 0],
        ns=ns,
    )
    dist_ts = wasserstein_distance_baci(sim_data_test["sample_ctr"], sim_data_test["sample_itv"])
    res = run_bocpd(dist_ts, **bocpd_kwargs)
    ex_stats = {"cpt": res["cpt_est"]}
    plot_distance_time_series(
        sim_data_test,
        npre,
        dist_ts,
        stats=ex_stats,
        dist_measure="wasserstein",
        savefile=f"{plots_dir}/example_distance_time_series.png",
    )

    print_complete(plots_dir)
