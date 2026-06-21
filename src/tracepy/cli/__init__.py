"""CLI entry point for TRACE simulations.

Usage:
    uv run trace-sim all [--quick] [--plots-only] [--no-cache]
    uv run trace-sim trend-amoc [--quick] [--plots-only] [--no-cache]
    uv run trace-sim trend-forecast [--quick] [--plots-only] [--no-cache]
    uv run trace-sim trend-bocpd [--quick] [--plots-only] [--no-cache]
    uv run trace-sim distribution-amoc [--quick] [--plots-only] [--no-cache]
    uv run trace-sim distribution-bocpd [--quick] [--plots-only] [--no-cache]
    uv run trace-sim distribution-forecast [--quick] [--plots-only] [--no-cache]
"""

import argparse

from tracepy.cli import (
    distribution_amoc,
    distribution_bocpd,
    distribution_forecast,
    trend_amoc,
    trend_bocpd,
    trend_forecast,
)

ALL_SIMULATIONS = [
    "trend-amoc",
    "trend-forecast",
    "trend-bocpd",
    "distribution-amoc",
    "distribution-bocpd",
    "distribution-forecast",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run TRACE changepoint detection simulations.\n"
            "Results are saved to disk after each stage and reused on subsequent runs.\n"
            "Use 'all' to run every simulation in sequence."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run trace-sim -h                               # show this help\n"
            "  uv run trace-sim all                              # run all simulations\n"
            "  uv run trace-sim all --quick                      # fast smoke-test (Nsim=10, simN=10)\n"
            "  uv run trace-sim trend-amoc                       # run one simulation\n"
            "  uv run trace-sim trend-amoc --plots-only          # regenerate plots from saved results\n"
            "  uv run trace-sim trend-amoc --no-cache            # ignore saved results, re-run fresh\n"
        ),
    )
    parser.add_argument(
        "simulation",
        choices=["all", *ALL_SIMULATIONS],
        help="simulation to run; 'all' runs every simulation in sequence",
    )
    parser.add_argument(
        "--quick", action="store_true", help="use Nsim=10, simN=10 for a fast smoke-test"
    )
    parser.add_argument(
        "--plots-only",
        "-p",
        action="store_true",
        help="skip simulation and regenerate plots from saved results",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="ignore saved results and re-run all stages from scratch",
    )
    args = parser.parse_args()

    if args.plots_only and args.no_cache:
        print("ERROR: --no-cache cannot be combined with --plots-only.")
        raise SystemExit(2)

    dispatch = {
        "trend-amoc": trend_amoc.run,
        "trend-bocpd": trend_bocpd.run,
        "distribution-amoc": distribution_amoc.run,
        "distribution-bocpd": distribution_bocpd.run,
        "trend-forecast": trend_forecast.run,
        "distribution-forecast": distribution_forecast.run,
    }

    simulations = ALL_SIMULATIONS if args.simulation == "all" else [args.simulation]
    for sim in simulations:
        dispatch[sim](quick=args.quick, plots_only=args.plots_only, no_cache=args.no_cache)
