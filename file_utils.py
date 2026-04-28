"""Utilities for saving and loading simulation results."""

import pickle
from pathlib import Path

_ROOT = Path(__file__).parent


def setup_results_directory(folder: str) -> Path:
    """Create and return results/<folder>."""
    path = _ROOT / "results" / folder
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_plots_directory(folder: str) -> Path:
    """Create and return plots/<folder>."""
    path = _ROOT / "plots" / folder
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_directories(folder: str):
    """Create both results and plots directories for folder."""
    setup_results_directory(folder)
    setup_plots_directory(folder)


def save_simulation_results(folder: str, data: dict):
    """Save data dict to results/<folder>/sim_results.pkl."""
    path = _ROOT / "results" / folder / "sim_results.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(data, f)
    print(f"✓ Simulation results saved: {path}")


def load_simulation_results(folder: str):
    """Load and return dict from results/<folder>/sim_results.pkl, or None."""
    path = _ROOT / "results" / folder / "sim_results.pkl"
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    print(f"✓ Simulation results loaded: {path}")
    return data
