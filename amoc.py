import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.arima.model import ARIMA


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


def trend_stats(y_ctr=None, y_itv=None, nt=None):
    """
    Calculate test statistics for trend difference (AMOC method).

    Tests for a changepoint in linear trend by evaluating at all possible
    changepoint locations and finding the one that maximizes the test statistic.

    Parameters
    ----------
    y_ctr : array-like, optional
        Control time series
    y_itv : array-like
        Intervention time series
    nt : int
        Total length of time series

    Returns
    -------
    dict
        Dictionary with 'Tmax' (max test statistic) and 'cpt' (changepoint location)
    """
    if y_ctr is not None:
        y_dif = np.array(y_itv) - np.array(y_ctr)
    else:
        y_dif = np.array(y_itv)

    mint = 24  # First two years are pre-intervention
    maxt = nt - 4

    seg_len = np.arange(mint, maxt + 1)
    stats_vec = np.zeros(len(seg_len))

    for m, cp in enumerate(seg_len):
        # Split data at potential changepoint
        seg1 = np.arange(1, cp + 1)   # 1-indexed to match R's 1:seglen[m]
        seg2 = np.arange(cp, nt)

        # Design matrix for trend model
        X = np.column_stack([
            np.ones(nt),
            np.concatenate([seg1, np.full(len(seg2), seg1[-1])]),
            np.concatenate([np.zeros(len(seg1)), np.arange(1, len(seg2) + 1)])
        ])

        # Fit OLS regression
        model = OLS(y_dif, X)
        results = model.fit()

        # Test statistic for trend difference (beta_1 vs beta_2)
        # Use normalized (unscaled) covariance (X'X)^{-1} to match R's cov.unscaled
        vec = np.array([0, -1, 1])
        var_diff = vec @ results.normalized_cov_params @ vec
        stats_vec[m] = (results.params[1] - results.params[2]) / np.sqrt(var_diff)

    Tmax = np.max(np.abs(stats_vec))
    cpt = np.argmax(np.abs(stats_vec)) + mint

    return {'Tmax': Tmax, 'cpt': cpt}


def trend_stats_ar(y_ctr=None, y_itv=None, nt=None):
    """
    Calculate test statistics for trend difference using ARIMA(1,0,0) with external regressors.

    Equivalent to R's trend.stats.ar(). Fits an ARIMA(1,0,0) model with the same design
    matrix as trend_stats(), but accounting for AR(1) autocorrelated residuals.
    Falls back to OLS (same as trend_stats) if ARIMA fit fails.

    Parameters
    ----------
    y_ctr : array-like, optional
        Control time series. If provided, the difference y_itv - y_ctr is analysed.
    y_itv : array-like
        Intervention time series
    nt : int
        Total length of time series

    Returns
    -------
    dict
        Dictionary with 'Tmax' (max test statistic) and 'cpt' (changepoint location)
    """
    if y_ctr is not None:
        y_dif = np.array(y_itv) - np.array(y_ctr)
    else:
        y_dif = np.array(y_itv)

    mint = 24  # First two years are pre-intervention
    maxt = nt - 4

    seg_len = np.arange(mint, maxt + 1)
    stats_vec = np.zeros(len(seg_len))

    for m, cp in enumerate(seg_len):
        seg1 = np.arange(1, cp + 1)   # 1-indexed to match R's 1:seglen[m]
        seg2 = np.arange(cp, nt)

        # Same design matrix as trend_stats
        X = np.column_stack([
            np.ones(nt),
            np.concatenate([seg1, np.full(len(seg2), seg1[-1])]),
            np.concatenate([np.zeros(len(seg1)), np.arange(1, len(seg2) + 1)])
        ])

        try:
            # ARIMA(1,0,0) with external regressors, no intercept (include_mean matches include.mean=FALSE in R)
            # In R: arima(y.dif, xreg=Xmat, order=c(1,0,0), include.mean=FALSE)
            # R defaults to method="CSS-ML": initialises with Conditional Sum of Squares,
            # then refines to full MLE. Python's innovations_mle is NOT equivalent:
            # it estimates via the innovations algorithm (a forward recursion on the
            # Kalman filter), skipping the CSS initialisation step entirely. It was
            # chosen here over the default state-space MLE because it is substantially
            # faster in the inner loop (~nt ARIMA fits per trend_stats_ar call), at
            # the cost of minor numerical differences in parameter estimates vs R.
            # Coefficients: [ar1, beta0, beta1, beta2] → indices [0, 1, 2, 3]
            model = ARIMA(y_dif, exog=X, order=(1, 0, 0), trend='n')
            fit = model.fit(method='innovations_mle', disp=False)

            # Test statistic: (beta1 - beta2) / se(beta1 - beta2)
            # In R: vec = c(0, 0, -1, 1); coef indices 3 and 4 (1-based) = beta1, beta2
            # In Python: exog params are at indices 1, 2, 3 (after AR param at index 0)
            params = fit.params          # [ar1, beta0, beta1, beta2]
            cov = fit.cov_params()       # 4×4 covariance matrix

            # Guard: all diagonal entries must be positive before trusting ARIMA estimates.
            # Matches R's: if (all(diag(armafit$var.coef) > 0))
            if not np.all(np.diag(cov) > 0):
                raise ValueError("non-positive diagonal in ARIMA covariance")

            vec = np.array([0, 0, -1, 1])   # contrast: beta2 - beta1
            var_diff = vec @ cov @ vec
            stats_vec[m] = (params[2] - params[3]) / np.sqrt(var_diff)

        except Exception:
            # OLS fallback if ARIMA fit fails or variance guard triggers
            # Use normalized (unscaled) covariance to match R's cov.unscaled
            ols_model = OLS(y_dif, X)
            ols_fit = ols_model.fit()
            vec_ols = np.array([0, -1, 1])
            var_diff = vec_ols @ ols_fit.normalized_cov_params @ vec_ols
            stats_vec[m] = (ols_fit.params[1] - ols_fit.params[2]) / np.sqrt(var_diff)

    Tmax = np.max(np.abs(stats_vec))
    cpt = np.argmax(np.abs(stats_vec)) + mint

    return {'Tmax': Tmax, 'cpt': cpt}

