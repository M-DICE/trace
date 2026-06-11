"""Trend-change detection via BOCPD (Bayesian Online Changepoint Detection).

Python port of the simulation study in
``SimRewilding/Rewild_trend_change_BOCPD.R``.

For each effect-size value in the trend grid this runs ``simN`` Monte Carlo
replications.  Each replication simulates a control/intervention pair with a
random onset *delay*, forms the difference series ``x = y_itv - y_ctr``, and
runs online BOCPD over it, stopping at the first declared changepoint.

Unlike the AMOC/Forecast pipelines, BOCPD is an online detector and needs no
null-distribution critical value, so there is a single simulation phase.  The
detector is driven through ``rpy2`` and is not picklable across processes, so
the Monte Carlo loop runs serially (matching the R reference).
"""

import time
import warnings

import numpy as np

from tracepy.changepoint.bocpd import run_bocpd
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
        print_stage(
            "Detection — BOCPD simulations", len(detection_results), len(trend_increase)
        )
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
            result = _run_increment(
                trend_inc=trend_inc,
                trend_idx=m,
                n_trends=len(trend_increase),
                simN=simN,
                npre=npre,
                npost_max=npost_max,
                level=level,
                trend_control=trend_control,
                sigma=sigma,
                delay_set=delay_set,
                bocpd_kwargs=bocpd_kwargs,
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


def _run_increment(
    *,
    trend_inc,
    trend_idx,
    n_trends,
    simN,
    npre,
    npost_max,
    level,
    trend_control,
    sigma,
    delay_set,
    bocpd_kwargs,
):
    """Run *simN* BOCPD replications for one trend increment (serial).

    Mirrors the inner ``sapply`` of the R reference: each replication draws a
    random delay, simulates a control/intervention pair, and runs BOCPD on the
    difference series ``y_itv - y_ctr``, stopping at the first changepoint.

    The true changepoint in the difference series sits at ``npre + delay``
    (1-indexed), so the location error is ``cpt_est - (npre + delay)``.
    """
    cpt_est = []
    time_est = []
    delays = []
    errors = []

    for s in range(1, simN + 1):
        seed = s * n_trends + trend_idx
        rng = np.random.default_rng(seed)
        delay = int(rng.choice(delay_set))
        npre_delay = npre + delay
        npost_delay = npost_max - delay

        sim_ts = ci_sim(
            seed=seed,
            npre=npre_delay,
            npost=npost_delay,
            level=level,
            trend=[trend_control, trend_control + trend_inc],
            sigma=sigma,
        )
        x = sim_ts["y_itv"] - sim_ts["y_ctr"]

        res = run_bocpd(x, **bocpd_kwargs)
        cpt = res["cpt_est"]

        cpt_est.append(cpt)
        time_est.append(res["time_est"])
        delays.append(delay)
        if cpt is not None:
            errors.append(cpt - npre_delay)

    n_detected = len(errors)
    detected_times = [t for t in time_est if t is not None]
    return {
        "cpt_est": cpt_est,
        "time_est": time_est,
        "delay": delays,
        "errors": errors,
        "detection_rate": n_detected / simN if simN else 0.0,
        "mean_error": float(np.mean(errors)) if errors else float("nan"),
        "mean_time_to_detect": float(np.mean(detected_times)) if detected_times else float("nan"),
    }


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
