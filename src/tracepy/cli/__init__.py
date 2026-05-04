"""CLI entry point for TRACE simulations.

Usage:
    uv run trace-sim trend-amoc [--quick] [--plots-only]
    uv run trace-sim trend-forecast [--quick] [--plots-only]
    uv run trace-sim distribution-amoc [--quick] [--plots-only]
"""

import argparse

from tracepy.cli import distribution_amoc, trend_amoc, trend_forecast


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TRACE simulation runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run trace-sim trend-amoc\n"
            "  uv run trace-sim trend-amoc --quick\n"
            "  uv run trace-sim distribution-amoc --plots-only\n"
        ),
    )
    parser.add_argument(
        "simulation",
        choices=["trend-amoc", "trend-forecast", "distribution-amoc"],
        help="Which simulation to run",
    )
    parser.add_argument("--quick", action="store_true",
                        help="Run with Nsim=10, simN=10 for fast smoke-testing")
    parser.add_argument("--plots-only", "-p", action="store_true",
                        help="Skip simulation phases and generate plots from saved results")
    args = parser.parse_args()

    dispatch = {
        "trend-amoc":        trend_amoc.run,
        "distribution-amoc": distribution_amoc.run,
        "trend-forecast":    trend_forecast.run,
    }
    dispatch[args.simulation](quick=args.quick, plots_only=args.plots_only)
