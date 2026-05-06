"""Load and merge YAML configuration parameters."""

from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__)
while not (_PROJECT_ROOT / "pyproject.toml").exists():
    _PROJECT_ROOT = _PROJECT_ROOT.parent

_DEFAULT_CONFIG = _PROJECT_ROOT / "config" / "default_params.yaml"


def load_params(path=None):
    """Load params from YAML file. Returns the full nested dict."""
    config_path = Path(path) if path else _DEFAULT_CONFIG
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_simulation_params(path=None):
    """Return the 'simulation' section as a flat dict."""
    return load_params(path).get("simulation", {})
