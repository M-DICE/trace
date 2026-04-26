"""
Plotting utilities for SimRewilding AMOC analysis.

Provides functions for visualizing simulated time series,
detected changepoints, and test statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# ---------------------------------------------------------------------------
# Shared colour / style constants
# ---------------------------------------------------------------------------
CLR_CTR       = '#607D8B'   # grey   — control series
CLR_ITV       = '#4CAF50'   # green  — intervention series
CLR_DIF       = '#FF9800'   # orange — difference series
CLR_CIBA      = '#2196F3'   # blue   — i.i.d. CIBA noise model
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


def plot_time_series(sim_data, npre, stats=None, figsize=(12, 6), savefile=None):
    """
    Plot simulated control and intervention time series with changepoint.

    Parameters
    ----------
    sim_data : dict
        Dictionary with 'y_ctr' (control) and 'y_itv' (intervention) arrays
    npre : int
        Number of pre-intervention time points (true changepoint location)
    stats : dict, optional
        Dictionary with 'cpt' (detected changepoint) from trend_stats
    figsize : tuple
        Figure size (width, height)
    savefile : str, optional
        If provided, save figure to this file path

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : numpy.ndarray
        Array of axes objects
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize)

    t = np.arange(len(sim_data['y_ctr']))

    # Plot 1: Control vs Intervention
    axes[0].plot(t, sim_data['y_ctr'], label='Control', color=CLR_CTR, alpha=0.7, linewidth=1.5)
    axes[0].plot(t, sim_data['y_itv'], label='Intervention', color=CLR_ITV, alpha=0.7, linewidth=1.5)
    axes[0].axvspan(0, npre, alpha=0.07, color='cornflowerblue')
    axes[0].axvline(x=npre, color=CLR_TAU_TRUE, linestyle='--', linewidth=2,
                    label=f'True Intervention Start (τ={npre})')

    axes[0].set_xlabel('Time (months)', fontsize=11)
    axes[0].set_ylabel('Value', fontsize=11)
    axes[0].set_title('Simulated Time Series: Control vs Intervention', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Difference series with detected changepoint
    y_dif = sim_data['y_itv'] - sim_data['y_ctr']
    axes[1].plot(t, y_dif, label='Difference (Intervention − Control)',
                 color=CLR_DIF, alpha=0.8, linewidth=1.5)
    axes[1].axvspan(0, npre, alpha=0.07, color='cornflowerblue')
    axes[1].axvline(x=npre, color=CLR_TAU_TRUE, linestyle='--', linewidth=2,
                    label=f'True Changepoint (τ={npre})')

    if stats is not None and 'cpt' in stats:
        detected_cpt = stats['cpt']
        axes[1].axvline(x=detected_cpt, color=CLR_TAU_DET, linestyle='-', linewidth=2,
                       label=f'Detected Changepoint (τ̂={detected_cpt})')
        error = abs(detected_cpt - npre)
        axes[1].text(detected_cpt, y_dif.max() * 0.9, f'Error: {error} months',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)

    axes[1].set_xlabel('Time (months)', fontsize=11)
    axes[1].set_ylabel('Difference', fontsize=11)
    axes[1].set_title('Difference Series with Detected Changepoint', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {savefile}")

    return fig, axes


def plot_test_statistics(test_stats_vec, npre, npost, detected_cpt=None,
                        figsize=(10, 5), savefile=None):
    """
    Plot test statistics across all possible changepoint locations.

    Parameters
    ----------
    test_stats_vec : array-like
        Vector of test statistics for each possible changepoint
    npre : int
        Number of pre-intervention time points (true changepoint)
    npost : int
        Number of post-intervention time points
    detected_cpt : int, optional
        Location of detected changepoint
    figsize : tuple
        Figure size (width, height)
    savefile : str, optional
        If provided, save figure to this file path

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    fig, ax = plt.subplots(figsize=figsize)

    nt = npre + npost
    mint = 12 if npost >= 12 else 2
    maxt = nt - 3
    seg_len = np.arange(mint, maxt + 1)

    ax.plot(seg_len, np.abs(test_stats_vec), marker='o', linewidth=2,
            markersize=4, color=CLR_CIBA, label='|Test Statistic|')

    ax.axvline(x=npre, color=CLR_TAU_TRUE, linestyle='--', linewidth=2,
               label=f'True Changepoint (τ={npre})')

    if detected_cpt is not None:
        ax.axvline(x=detected_cpt, color=CLR_TAU_DET, linestyle='-', linewidth=2,
                  label=f'Detected Changepoint (τ̂={detected_cpt})')
        max_stat = np.abs(test_stats_vec).max()
        ax.plot(detected_cpt, max_stat, 'o', color=CLR_TAU_DET, markersize=10, label='Maximum')

    ax.set_xlabel('Possible Changepoint Location (months)', fontsize=11)
    ax.set_ylabel('|Test Statistic|', fontsize=11)
    ax.set_title('AMOC Test Statistics Across Possible Changepoint Locations',
                fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {savefile}")

    return fig, ax


def plot_simulation_results(detection_rates, trend_increments,
                           figsize=(10, 6), savefile=None):
    """
    Plot detection rates vs trend increment (sensitivity analysis).

    Parameters
    ----------
    detection_rates : array-like
        Detection rates for each trend increment (0-1)
    trend_increments : array-like
        Array of trend increment values
    figsize : tuple
        Figure size (width, height)
    savefile : str, optional
        If provided, save figure to this file path

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(trend_increments, detection_rates, marker='o', linewidth=2.5,
            markersize=8, color=CLR_ITV, label='Detection Rate')
    ax.fill_between(trend_increments, detection_rates, alpha=0.2, color=CLR_ITV)

    ax.set_xlabel('Trend Increment', fontsize=11)
    ax.set_ylabel('Detection Rate', fontsize=11)
    ax.set_title('Sensitivity Analysis: Detection Rate vs Trend Increment',
                fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    _add_power_thresholds(ax)

    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {savefile}")

    return fig, ax


def plot_detection_error_distribution(detection_errors, figsize=(10, 5), savefile=None):
    """
    Plot histogram of changepoint detection errors.

    Parameters
    ----------
    detection_errors : array-like
        Array of errors (detected_cpt - true_cpt) from multiple simulations
    figsize : tuple
        Figure size (width, height)
    savefile : str, optional
        If provided, save figure to this file path

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    ax : matplotlib.axes.Axes
        The axes object
    """
    fig, ax = plt.subplots(figsize=figsize)

    detection_errors = np.array(detection_errors)
    mean_error = np.mean(detection_errors)
    std_error = np.std(detection_errors)

    ax.hist(detection_errors, bins=30, color=CLR_CIBA, alpha=0.7, edgecolor='black')
    ax.axvline(x=mean_error, color=CLR_TAU_DET, linestyle='-', linewidth=2,
              label=f'Mean Error: {mean_error:.2f} months')
    ax.axvline(x=0, color=CLR_TAU_TRUE, linestyle='--', linewidth=2,
               label='Zero Error (Perfect Detection)')

    ax.set_xlabel('Detection Error (months)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title(f'Distribution of Changepoint Detection Errors\n(Mean={mean_error:.2f}, SD={std_error:.2f})',
                fontsize=12, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {savefile}")

    return fig, ax


def plot_power_curves(detection_results, trend_increase, npost_vec,
                      figsize=(12, 7), savefile=None):
    """
    Plot detection rate vs post-intervention length for each trend increment.

    Parameters
    ----------
    detection_results : dict
        Output of run_main_simulation(). Keys are trend increments; each value
        contains 'detection_rates' — a (len(npost_vec),) array.
    trend_increase : array-like
        Ordered array of trend increment values (used for legend labels).
    npost_vec : array-like
        Post-intervention lengths (x-axis).
    figsize : tuple
    savefile : str, optional

    Returns
    -------
    fig, ax
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = _VIRIDIS_11[:len(trend_increase)]

    for color, trend_inc in zip(colors, trend_increase):
        rates = detection_results[trend_inc]['detection_rates']
        ax.plot(npost_vec, rates, color=color, linewidth=1.8,
                label=f'trend = {trend_inc:.4f}')

    ax.set_xlabel('Post-intervention length (months)', fontsize=11)
    ax.set_ylabel('Detection rate', fontsize=11)
    ax.set_title('Power Curves: Detection Rate vs Post-Intervention Length\n'
                 '(i.i.d. noise, growing window)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    _add_power_thresholds(ax)
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {savefile}")
    return fig, ax


def plot_power_curves_comparison(detection_results_iid, detection_results_ar,
                                 trend_increase, npost_vec,
                                 figsize=(16, 7), savefile=None):
    """
    Side-by-side power curves for i.i.d. and AR(1) simulations.

    Parameters
    ----------
    detection_results_iid : dict
        Output of run_main_simulation(). Each value has 'detection_rates' array.
    detection_results_ar : dict
        Output of run_main_simulation_ar(). Same structure.
    trend_increase : array-like
        Ordered trend increment values.
    npost_vec : array-like
        Post-intervention lengths (x-axis).
    figsize : tuple
    savefile : str, optional

    Returns
    -------
    fig, axes
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    colors = _VIRIDIS_11[:len(trend_increase)]

    for ax, results, title in [
        (axes[0], detection_results_iid, 'i.i.d. noise'),
        (axes[1], detection_results_ar,  'AR(1) noise  (φ = 0.5)'),
    ]:
        for color, trend_inc in zip(colors, trend_increase):
            rates = results[trend_inc]['detection_rates']
            ax.plot(npost_vec, rates, color=color, linewidth=1.8,
                    label=f'{trend_inc:.4f}')

        _add_power_thresholds(ax)
        ax.set_xlabel('Post-intervention length (months)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Detection rate', fontsize=11)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Trend increment',
               loc='center right', bbox_to_anchor=(1.0, 0.5),
               fontsize=8, title_fontsize=9)

    fig.suptitle('Power Curves: i.i.d. vs AR(1) Noise\n'
                 '(detection rate vs post-intervention length, growing window)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.88, 1])

    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {savefile}")
    return fig, axes


def plot_time_to_detection(detection_results_iid, detection_results_ar,
                           trend_increase, npost_vec,
                           thresholds=(0.80, 0.95),
                           figsize=(11, 6), savefile=None):
    """
    For each trend increment, the minimum post-intervention length needed to
    reach a given detection power threshold.

    Parameters
    ----------
    detection_results_iid / detection_results_ar : dict
        Each value must contain 'detection_rates' — array of length len(npost_vec).
    trend_increase : array-like
    npost_vec : array-like
    thresholds : tuple of float
        Power thresholds to mark (default 80% and 95%).
    """
    fig, ax = plt.subplots(figsize=figsize)
    npost_vec = np.asarray(npost_vec)

    # colour = noise model; linestyle = power threshold
    threshold_ls = {0.80: '--', 0.95: ':'}
    styles = [
        (detection_results_iid, 'i.i.d.', CLR_CIBA, 'o', '-'),
        (detection_results_ar,  'AR(1)',  CLR_AR1,  's', '--'),
    ]

    for results, label, color, marker, _ in styles:
        for thresh in thresholds:
            months_needed = []
            for trend_inc in trend_increase:
                rates = np.asarray(results[trend_inc]['detection_rates'])
                idx = np.argmax(rates >= thresh)
                months_needed.append(npost_vec[idx] if rates[idx] >= thresh else np.nan)
            ax.plot(trend_increase, months_needed,
                    linestyle=threshold_ls[thresh], marker=marker, markersize=5,
                    color=color,
                    label=f'{label} — {int(thresh*100)}% power',
                    alpha=0.85)

    ax.set_xlabel('Trend increment', fontsize=11)
    ax.set_ylabel('Post-intervention months needed', fontsize=11)
    ax.set_title('Time-to-Detection: Months of Monitoring Needed\nto Reach Target Power',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {savefile}")
    return fig, ax


def plot_detection_heatmap(detection_results_iid, detection_results_ar,
                           trend_increase, npost_vec,
                           figsize=(16, 6), savefile=None):
    """
    2D heatmap of detection rate: trend increment (rows) × npost length (columns).

    Dark = low power, bright = high power. Contour lines mark 80% and 95% power.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    npost_vec     = np.asarray(npost_vec)
    trend_increase = np.asarray(trend_increase)

    for ax, results, title in [
        (axes[0], detection_results_iid, 'i.i.d. noise'),
        (axes[1], detection_results_ar,  'AR(1) noise  (φ = 0.5)'),
    ]:
        matrix = np.array([results[t]['detection_rates'] for t in trend_increase])
        im = ax.imshow(matrix, aspect='auto', origin='lower',
                       extent=[npost_vec[0], npost_vec[-1],
                                -0.5, len(trend_increase) - 0.5],
                       vmin=0, vmax=1, cmap='viridis')
        ax.contour(npost_vec, np.arange(len(trend_increase)), matrix,
                   levels=[0.80, 0.95], colors=['white', 'white'],
                   linewidths=1.5, linestyles=['--', ':'])
        ax.set_xlabel('Post-intervention length (months)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_yticks(np.arange(len(trend_increase)))
        ax.set_yticklabels([f'{t:.4f}' for t in trend_increase], fontsize=7)

    axes[0].set_ylabel('Trend increment', fontsize=11)
    fig.colorbar(im, ax=axes[1], label='Detection rate')
    fig.suptitle('Detection Rate Heatmap: Trend × Post-Intervention Length\n'
                 '(white dashed = 80% power, white dotted = 95% power)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {savefile}")
    return fig, axes


def plot_mean_error_curves(detection_results_iid, detection_results_ar,
                           trend_increase, npost_vec,
                           figsize=(16, 7), savefile=None):
    """
    Mean changepoint detection error (|τ̂ - τ_true|) vs post-intervention length.

    Complements the power curves: power tells you how often you detect;
    error tells you how precisely you locate the change when you do detect.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    colors = _VIRIDIS_11[:len(trend_increase)]

    for ax, results, title in [
        (axes[0], detection_results_iid, 'i.i.d. noise'),
        (axes[1], detection_results_ar,  'AR(1) noise  (φ = 0.5)'),
    ]:
        for color, trend_inc in zip(colors, trend_increase):
            errors = results[trend_inc]['mean_errors']
            ax.plot(npost_vec, errors, color=color, linewidth=1.8,
                    label=f'{trend_inc:.4f}')
        ax.set_xlabel('Post-intervention length (months)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Mean |τ̂ − τ_true| (months)', fontsize=11)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title='Trend increment',
               loc='center right', bbox_to_anchor=(1.0, 0.5),
               fontsize=8, title_fontsize=9)
    fig.suptitle('Changepoint Localisation Error vs Post-Intervention Length',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {savefile}")
    return fig, axes


def plot_null_distributions(critical_values, npost_short=48, npost_long=96,
                            figsize=(14, 6), savefile=None):
    """
    Histograms of Tmax under the null hypothesis for all four combinations:
    iid_{short}, iid_{long}, ar1_{short}, ar1_{long}.
    """
    s, l = npost_short, npost_long
    keys = [f'iid_{s}', f'iid_{l}', f'ar1_{s}', f'ar1_{l}']
    titles = [
        f'i.i.d. noise, npost = {s} mo', f'i.i.d. noise, npost = {l} mo',
        f'AR(1) noise,  npost = {s} mo', f'AR(1) noise,  npost = {l} mo',
    ]
    colors = [CLR_CIBA, CLR_CIBA, CLR_AR1, CLR_AR1]

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=False)

    for ax, key, title, color in zip(axes, keys, titles, colors):
        nd = critical_values[key]['null_dist']
        cv = critical_values[key]['critical_value']
        ax.hist(nd, bins=40, color=color, alpha=0.7, edgecolor='white', linewidth=0.4)
        ax.axvline(cv, color=CLR_TAU_DET, linewidth=2,
                   label=f'CV = {cv:.3f}\n(95th pct)')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlabel('Tmax', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel('Count', fontsize=11)
    fig.suptitle('Null Distribution of Tmax (no trend change)\n'
                 'Crimson line = critical value at α = 95%',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {savefile}")
    return fig, axes


def plot_tmax_signal_vs_noise(critical_values, detection_results_iid,
                               detection_results_ar, trend_increase, npost_vec,
                               npost_short=48, figsize=(16, 5), savefile=None):
    """
    Overlay of null Tmax distribution vs alternative Tmax distribution
    for three representative trend increments (small, medium, large).
    """
    npost_vec = np.asarray(npost_vec)
    idx_short = np.argmin(np.abs(npost_vec - npost_short))

    null_iid = critical_values[f'iid_{npost_short}']['null_dist']
    null_ar  = critical_values[f'ar1_{npost_short}']['null_dist']
    cv_iid   = critical_values[f'iid_{npost_short}']['critical_value']
    cv_ar    = critical_values[f'ar1_{npost_short}']['critical_value']

    trend_increase = np.asarray(trend_increase)
    pick_indices   = [0, len(trend_increase) // 2, -1]
    picked_trends  = trend_increase[pick_indices]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)

    for ax, trend_inc in zip(axes, picked_trends):
        alt_iid = detection_results_iid[trend_inc]['tmax_matrix'][:, idx_short]
        alt_ar  = detection_results_ar[trend_inc]['tmax_matrix'][:, idx_short]

        bins = np.linspace(0, max(alt_iid.max(), alt_ar.max(), null_iid.max()) + 1, 50)

        ax.hist(null_iid, bins=bins, alpha=0.5, color=CLR_CIBA,  label='Null (i.i.d.)')
        ax.hist(alt_iid,  bins=bins, alpha=0.5, color=CLR_BA,    label='Alt (i.i.d.)')
        ax.hist(null_ar,  bins=bins, alpha=0.3, color=CLR_AR1,   label='Null AR(1)',
                linestyle='--', histtype='step', linewidth=1.5)
        ax.axvline(cv_iid, color=CLR_CIBA, linewidth=1.5, linestyle='--',
                   label=f'CV i.i.d. = {cv_iid:.2f}')
        ax.axvline(cv_ar,  color=CLR_AR1,  linewidth=1.5, linestyle=':',
                   label=f'CV AR(1) = {cv_ar:.2f}')

        ax.set_title(f'Trend = {trend_inc:.4f}', fontsize=11, fontweight='bold')
        ax.set_xlabel(f'Tmax (at npost = {npost_short} months)', fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel('Count', fontsize=11)
    fig.suptitle('Tmax Distribution: Null vs Alternative\n'
                 '(overlap = hard to detect; separation = easy to detect)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {savefile}")
    return fig, axes


def plot_detection_by_delay(detection_results_iid, detection_results_ar,
                             trend_increase, figsize=(12, 6), savefile=None):
    """
    Detection rate at npost_max grouped by delay (1–20 months).
    """
    trend_increase = np.asarray(trend_increase)
    pick_indices   = [0, len(trend_increase) // 2, -1]
    picked_trends  = trend_increase[pick_indices]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    for ax, trend_inc in zip(axes, picked_trends):
        for results, label, color, marker in [
            (detection_results_iid, 'i.i.d.', CLR_CIBA, 'o'),
            (detection_results_ar,  'AR(1)',  CLR_AR1,  's'),
        ]:
            delays    = np.array(results[trend_inc]['delays'])
            detected  = results[trend_inc]['detected_matrix'][:, -1]

            delay_vals = np.arange(1, 21)
            rates = [detected[delays == d].mean() if (delays == d).any() else np.nan
                     for d in delay_vals]

            ax.plot(delay_vals, rates, marker=marker, linewidth=1.5,
                    color=color, label=label, markersize=5)

        ax.set_title(f'Trend = {trend_inc:.4f}', fontsize=11, fontweight='bold')
        ax.set_xlabel('Delay (months)', fontsize=10)
        ax.set_ylim([0, 1.05])
        ax.axhline(0.80, color=CLR_THRESH, linestyle='--', linewidth=1, alpha=0.85)
        ax.axhline(0.95, color=CLR_THRESH, linestyle=':', linewidth=1, alpha=0.85)
        ax.text(1.01, 0.80, '80%', va='center', ha='left', fontsize=7,
                color=CLR_THRESH, transform=ax.get_yaxis_transform())
        ax.text(1.01, 0.95, '95%', va='center', ha='left', fontsize=7,
                color=CLR_THRESH, transform=ax.get_yaxis_transform())
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Detection rate at npost_max', fontsize=11)
    fig.suptitle('Effect of Intervention Delay on Detection Rate\n'
                 '(at npost_max; each point = ~50 simulations)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {savefile}")
    return fig, axes


def plot_changepoint_bias(detection_results_iid, detection_results_ar,
                           trend_increase, npre,
                           figsize=(14, 6), savefile=None):
    """
    Distribution of signed changepoint estimation error: τ̂ − τ_true.

    Positive = estimated late, Negative = estimated early.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
    trend_increase = np.asarray(trend_increase)

    for ax, results, title, color in [
        (axes[0], detection_results_iid, 'i.i.d. noise', CLR_CIBA),
        (axes[1], detection_results_ar,  'AR(1) noise  (φ = 0.5)', CLR_AR1),
    ]:
        signed_errors = []
        for trend_inc in trend_increase:
            cpts   = results[trend_inc]['cpt_matrix'][:, -1]
            delays = np.array(results[trend_inc]['delays'])
            true_cpts = npre + delays
            signed_errors.append(cpts - true_cpts)

        parts = ax.violinplot(signed_errors,
                              positions=np.arange(len(trend_increase)),
                              showmedians=True, showmeans=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)

        ax.axhline(0, color=CLR_TAU_TRUE, linewidth=1.5, linestyle='--', label='Zero bias')
        ax.set_xticks(np.arange(len(trend_increase)))
        ax.set_xticklabels([f'{t:.4f}' for t in trend_increase],
                            rotation=45, ha='right', fontsize=7)
        ax.set_xlabel('Trend increment', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    axes[0].set_ylabel('τ̂ − τ_true (months)', fontsize=11)
    fig.suptitle('Changepoint Estimation Bias (signed error at npost_max)\n'
                 'Positive = estimated late, Negative = estimated early',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {savefile}")
    return fig, axes


def compare_methods(results_dict, figsize=(12, 5), savefile=None):
    """
    Compare detection performance across different methods.

    Parameters
    ----------
    results_dict : dict
        Dictionary with method names as keys and detection results as values
        Format: {'method_name': {'detection_rate': float, 'mean_error': float, ...}}
    figsize : tuple
        Figure size (width, height)
    savefile : str, optional
        If provided, save figure to this file path

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    axes : numpy.ndarray
        Array of axes objects
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    methods = list(results_dict.keys())
    detection_rates = [results_dict[m]['detection_rate'] for m in methods]
    mean_errors = [results_dict[m]['mean_error'] for m in methods]

    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))

    # Plot 1: Detection rates
    axes[0].bar(methods, detection_rates, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].axhline(y=0.95, color=CLR_THRESH, linestyle=':', linewidth=1.2, alpha=0.85)
    axes[0].axhline(y=0.80, color=CLR_THRESH, linestyle='--', linewidth=1.2, alpha=0.85)
    axes[0].text(1.01, 0.95, '95%', va='center', ha='left', fontsize=8,
                 color=CLR_THRESH, transform=axes[0].get_yaxis_transform())
    axes[0].text(1.01, 0.80, '80%', va='center', ha='left', fontsize=8,
                 color=CLR_THRESH, transform=axes[0].get_yaxis_transform())
    axes[0].set_ylabel('Detection Rate', fontsize=11)
    axes[0].set_title('Detection Rate by Method', fontsize=12, fontweight='bold')
    axes[0].set_ylim([0, 1.1])
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Mean errors
    axes[1].bar(methods, mean_errors, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].axhline(y=0, color=CLR_TAU_TRUE, linestyle='--', alpha=0.7, label='Zero Error')
    axes[1].set_ylabel('Mean Detection Error (months)', fontsize=11)
    axes[1].set_title('Mean Detection Error by Method', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if savefile:
        plt.savefig(savefile, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {savefile}")

    return fig, axes
