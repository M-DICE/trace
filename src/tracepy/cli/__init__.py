"""CLI entry point for TRACE simulations.

Usage:
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRACE simulation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run trace-sim trend-amoc\n"
            "  uv run trace-sim trend-amoc --quick\n"
            "  uv run trace-sim trend-bocpd --quick\n"
            "  uv run trace-sim distribution-amoc --plots-only\n"
            "  uv run trace-sim distribution-bocpd --quick\n"
            "  uv run trace-sim distribution-forecast --quick\n"
            "  uv run trace-sim trend-amoc --no-cache\n"
        ),
    )
    parser.add_argument(
        "simulation",
        choices=[
            "trend-amoc",
            "trend-forecast",
            "trend-bocpd",
            "distribution-amoc",
            "distribution-bocpd",
            "distribution-forecast",
        ],
        help="Which simulation to run",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run with Nsim=10, simN=10 for fast smoke-testing"
    )
    parser.add_argument(
        "--plots-only",
        "-p",
        action="store_true",
        help="Skip simulation phases and generate plots from saved results",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Ignore any saved results and run all simulation phases fresh",
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
    dispatch[args.simulation](quick=args.quick, plots_only=args.plots_only, no_cache=args.no_cache)
