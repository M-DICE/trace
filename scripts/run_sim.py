#!/usr/bin/env python3
"""
Unified CLI for TRACE simulations.

Usage:
    python scripts/run_sim.py trend-amoc [--quick] [--plots-only]
    python scripts/run_sim.py trend-forecast [--quick] [--plots-only]
    python scripts/run_sim.py distribution-amoc [--quick] [--plots-only]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(description="TRACE simulation runner")
    parser.add_argument("simulation", choices=["trend-amoc", "trend-forecast", "distribution-amoc"])
    parser.add_argument("--quick", action="store_true", help="Quick run (small Nsim/simN)")
    parser.add_argument("--plots-only", "-p", action="store_true", help="Generate plots from saved results")
    args = parser.parse_args()

    argv = sys.argv[2:]  # pass remaining args through
    if args.simulation == "trend-amoc":
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "__main__",
            Path(__file__).parent.parent / "rewild_trend_change_amoc.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.argv = [str(spec.origin)] + argv
        spec.loader.exec_module(mod)
    elif args.simulation == "trend-forecast":
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "__main__",
            Path(__file__).parent.parent / "rewild_trend_change_forecast.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.argv = [str(spec.origin)] + argv
        spec.loader.exec_module(mod)
    elif args.simulation == "distribution-amoc":
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "__main__",
            Path(__file__).parent.parent / "rewild_distribution_change_amoc.py"
        )
        mod = importlib.util.module_from_spec(spec)
        sys.argv = [str(spec.origin)] + argv
        spec.loader.exec_module(mod)


if __name__ == "__main__":
    main()
