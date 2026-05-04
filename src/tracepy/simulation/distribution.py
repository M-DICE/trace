"""
Simulation functions for distribution-based (CDF) time series (BACI design).
"""

import numpy as np


def ci_sim_cdf(seed, npre, npost, level, trend, ns=200):
    """
    Generate simulated paired distribution time series (BACI design).

    At each time point, ns samples are drawn from a normal distribution. The
    control series keeps constant parameters throughout. The intervention series
    matches the control during the pre-intervention period, then drifts linearly
    in mean and/or SD during the post-intervention period.

    Equivalent to R's CIsim.cdf().

    Parameters
    ----------
    seed : int
        Random seed for reproducibility (passed to np.random.seed).
    npre : int
        Number of time points before the intervention (pre-period).
    npost : int
        Number of time points after the intervention (post-period).
    level : list of float
        [mu, sigma] — baseline mean and standard deviation.
    trend : list of float
        [trend_mu, trend_sigma] — monthly increment applied to mu and sigma
        respectively during the post-intervention period. Set either to 0 to
        hold that parameter fixed.
    ns : int, optional
        Number of samples drawn per time point (default 200).

    Returns
    -------
    dict
        {'sample_ctr': ndarray (ns, nt), 'sample_itv': ndarray (ns, nt)}
        where nt = npre + npost. Columns are time points; rows are samples.
    """
    mu, sigma = level
    trend_mu, trend_sigma = trend

    np.random.seed(seed)
    nt = npre + npost

    sample_ctr = np.zeros((ns, nt))
    sample_itv = np.zeros((ns, nt))

    mus = np.full(nt, mu, dtype=float)
    sigmas = np.full(nt, sigma, dtype=float)

    t_post = np.arange(1, npost + 1)
    mus[npre:] += trend_mu * t_post
    sigmas[npre:] += trend_sigma * t_post

    # Control: constant distribution throughout
    for i in range(nt):
        sample_ctr[:, i] = np.random.normal(mu, sigma, ns)

    # Intervention: pre-period identical to control, post-period drifts
    for i in range(nt):
        sample_itv[:, i] = np.random.normal(mus[i], sigmas[i], ns)

    return {'sample_ctr': sample_ctr, 'sample_itv': sample_itv}
