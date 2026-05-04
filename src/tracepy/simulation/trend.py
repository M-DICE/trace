"""
Simulation functions for trend-change time series (IID and AR(1) noise).
"""

import numpy as np


def ci_sim(seed=10, npre=24, npost=24, level=10, trend=None, sigma=0.05):
    """
    Generate simulated time series with trend change.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    npre : int
        Number of samples before intervention
    npost : int
        Number of samples after intervention
    level : float
        Base level of the time series
    trend : tuple or list
        (trend_control, trend_interv) - slopes before and after intervention
    sigma : float
        Standard deviation of noise

    Returns
    -------
    dict
        Dictionary with 'y_ctr' (control) and 'y_itv' (intervention) time series
    """
    if trend is None:
        trend = [0, 0]

    np.random.seed(seed)
    nt = npre + npost

    trend_control = trend[0]
    trend_interv = trend[1]

    # Generate control time series (no intervention effect)
    t = np.arange(1, nt + 1)
    y_control = level + trend_control * t + np.random.normal(0, sigma, nt)

    # Generate intervention time series (trend changes at tau = npre)
    y_interv = np.zeros(nt)
    y_interv[:npre] = level + trend_control * t[:npre] + np.random.normal(0, sigma, npre)
    y_interv[npre:] = (level + trend_control * npre +
                       trend_interv * np.arange(1, npost + 1) +
                       np.random.normal(0, sigma, npost))

    return {'y_ctr': y_control, 'y_itv': y_interv}


def ci_sim_ar(seed=10, npre=24, npost=24, level=10, trend=None, phi=0.5, sigma=0.05):
    """
    Generate simulated time series with trend change and AR(1) autocorrelated noise.

    Equivalent to R's CIsim.ar(). Generates AR(1) noise via direct recursion,
    matching R's arima.sim(model=list(ar=phi), n=n, sd=sigma).

    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    npre : int
        Number of samples before intervention
    npost : int
        Number of samples after intervention
    level : float
        Base level of the time series
    trend : tuple or list
        (trend_control, trend_interv) - slopes before and after intervention
    phi : float
        AR(1) autocorrelation coefficient
    sigma : float
        Standard deviation of the AR(1) innovations (white noise input)

    Returns
    -------
    dict
        Dictionary with 'y_ctr' (control) and 'y_itv' (intervention) time series
    """
    if trend is None:
        trend = [0, 0]

    np.random.seed(seed)
    nt = npre + npost

    trend_control = trend[0]
    trend_interv = trend[1]

    # Simulate AR(1) noise: X_t = phi * X_{t-1} + eps_t, eps_t ~ N(0, sigma^2)
    # This matches R's arima.sim(model=list(ar=phi), n=n, sd=sigma)
    def ar1_noise(n):
        eps = np.random.normal(0, sigma, n)
        noise = np.zeros(n)
        noise[0] = eps[0]
        for t in range(1, n):
            noise[t] = phi * noise[t - 1] + eps[t]
        return noise

    t = np.arange(1, nt + 1)

    # Control series: constant trend with AR(1) noise over full length
    y_control = level + trend_control * t + ar1_noise(nt)

    # Intervention series: pre-period follows trend_control, post-period switches to trend_interv
    # Each segment gets its own independently seeded AR(1) noise (matching R's two arima.sim calls)
    y_pre = level + trend_control * t[:npre] + ar1_noise(npre)
    y_post = (level + trend_control * npre +
              trend_interv * np.arange(1, npost + 1) +
              ar1_noise(npost))
    y_interv = np.concatenate([y_pre, y_post])

    return {'y_ctr': y_control, 'y_itv': y_interv}
