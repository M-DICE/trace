"""Colour/style constants and shared helpers for TRACE plots."""

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Shared colour / style constants
# ---------------------------------------------------------------------------
CLR_CTR       = '#607D8B'   # grey   — control series
CLR_ITV       = '#4CAF50'   # green  — intervention series
CLR_DIF       = '#FF9800'   # orange — difference series
CLR_BACI      = '#2196F3'   # blue   — i.i.d. BACI noise model
CLR_AR1       = '#FF9800'   # orange — AR(1) noise model
CLR_BA        = '#4CAF50'   # green  — i.i.d. BA noise model
CLR_TAU_TRUE  = '#1A237E'   # navy   — true changepoint τ (dashed)
CLR_TAU_DET   = 'crimson'   # crimson — detected changepoint τ̂ (solid)
CLR_INTV_DATE = 'steelblue' # steelblue — nominal intervention date (dotted)
CLR_THRESH    = '#9E9E9E'   # grey   — power-threshold reference lines
_VIRIDIS_11   = plt.cm.viridis(np.linspace(0, 1, 11))  # 11-level effect-size palette


def _add_power_thresholds(ax, xlim_max=None):
    """Add 80% and 95% power threshold lines with right-edge labels."""
    kw = dict(color=CLR_THRESH, linewidth=1.2, alpha=0.85)
    ax.axhline(y=0.80, linestyle='--', **kw)
    ax.axhline(y=0.95, linestyle=':', **kw)
    ax.text(1.01, 0.80, '80%', va='center', ha='left', fontsize=8,
            color=CLR_THRESH, transform=ax.get_yaxis_transform())
    ax.text(1.01, 0.95, '95%', va='center', ha='left', fontsize=8,
            color=CLR_THRESH, transform=ax.get_yaxis_transform())
