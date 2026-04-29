import warnings
import numpy as np
import scipy.stats
import scipy.integrate
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


def wasserstein_distance_baci(sample_ctr, sample_itv):
    """
    Compute Wasserstein distance between control and intervention distributions
    at each time point (BACI design).

    At each time point t the distance is computed between the ns-sample
    empirical distributions of the control and intervention series. This is the
    1-Wasserstein (earth mover's) distance, equivalent to R's
    transport::wasserstein1d(a, b, p=1).

    Equivalent to R's WSdist().

    Parameters
    ----------
    sample_ctr : ndarray (ns, nt)
        Control samples — rows are samples, columns are time points.
    sample_itv : ndarray (ns, nt)
        Intervention samples — same shape as sample_ctr.

    Returns
    -------
    dist_ts : ndarray (nt,)
        Wasserstein distance at each time point.
    """
    nt = sample_ctr.shape[1]
    dist_ts = np.zeros(nt)
    for t in range(nt):
        dist_ts[t] = scipy.stats.wasserstein_distance(sample_ctr[:, t], sample_itv[:, t])
    return dist_ts


def wasserstein_distance_ba(sample_itv, npre):
    """
    Compute Wasserstein distance between each time point and the pooled
    pre-intervention baseline (BA design, no control series required).

    The baseline is formed by pooling all ns × npre samples from the
    pre-intervention columns of the intervention series into a single 1-D
    array. Each time point is then compared against this pooled baseline.

    Equivalent to R's WSdist2(), where y.ctr = sample.itv[, 1:npre] is passed
    as a matrix to wasserstein1d — R's transport package pools the columns,
    matching the flatten() applied here.

    Parameters
    ----------
    sample_itv : ndarray (ns, nt)
        Intervention samples — rows are samples, columns are time points.
    npre : int
        Number of pre-intervention time points used to build the baseline.

    Returns
    -------
    dist_ts : ndarray (nt,)
        Wasserstein distance from the baseline at each time point.
    """
    nt = sample_itv.shape[1]
    baseline = sample_itv[:, :npre].flatten()
    dist_ts = np.zeros(nt)
    for t in range(nt):
        dist_ts[t] = scipy.stats.wasserstein_distance(baseline, sample_itv[:, t])
    return dist_ts


def auc_diff_ts(sample_ctr, sample_itv, bw=0.3, nd=50):
    """
    Compute the AUC of the absolute KDE density difference at each time point.

    For each time point, Gaussian KDEs are fitted to the control and
    intervention samples over a common grid, and the integral of the absolute
    difference |f_itv - f_ctr| is returned. This equals twice the total
    variation distance and matches R's DescTools::AUC(..., absolutearea=TRUE).

    Note on bandwidth: R's density(bw=0.3) uses 0.3 as the kernel standard
    deviation directly. scipy's gaussian_kde multiplies bw_method by the
    sample SD, so we pass bw_method = bw / std(data) per column to match R.

    Equivalent to R's AUCdiff() (uses density() + AUC()).

    Parameters
    ----------
    sample_ctr : ndarray (ns, nt)
        Control samples — rows are samples, columns are time points.
    sample_itv : ndarray (ns, nt)
        Intervention samples — same shape as sample_ctr.
    bw : float, optional
        Kernel standard deviation (default 0.3, matching R's default bw=0.3).
    nd : int, optional
        Number of equally-spaced grid points for density evaluation (default 50).

    Returns
    -------
    dist_ts : ndarray (nt,)
        AUC of |f_itv - f_ctr| at each time point.
    """
    nt = sample_ctr.shape[1]
    dist_ts = np.zeros(nt)

    for t in range(nt):
        data_ctr = sample_ctr[:, t]
        data_itv = sample_itv[:, t]

        vmin = min(np.min(data_ctr), np.min(data_itv))
        vmax = max(np.max(data_ctr), np.max(data_itv))
        grid = np.linspace(vmin, vmax, nd)

        # bw / std converts kernel SD → scipy's scale factor
        bw_ctr = bw / max(np.std(data_ctr), 1e-10)
        bw_itv = bw / max(np.std(data_itv), 1e-10)

        kde_ctr = scipy.stats.gaussian_kde(data_ctr, bw_method=bw_ctr)
        kde_itv = scipy.stats.gaussian_kde(data_itv, bw_method=bw_itv)

        d_ctr = kde_ctr.evaluate(grid)
        d_itv = kde_itv.evaluate(grid)

        dist_ts[t] = scipy.integrate.trapezoid(np.abs(d_itv - d_ctr), x=grid)

    return dist_ts


def trend_stats_cdf(y_dist, nt):
    """
    Apply the AMOC trend-change test to a distance time series.

    Thin wrapper around trend_stats() that treats the distance series as the
    response directly (y_ctr=None), matching the intended behaviour of R's
    trend.stats.cdf(). The distance series replaces the difference series used
    in the scalar time series case.

    Note: R's trend.stats.cdf contains a scoping bug — it references free
    variables y.ctr and y.itv instead of its parameter y, inadvertently
    inheriting values from the enclosing environment. This wrapper correctly
    passes y_dist as the response.

    Equivalent to R's trend.stats.cdf().

    Parameters
    ----------
    y_dist : array-like (nt,)
        Distance time series (Wasserstein or AUC) to test for a trend change.
    nt : int
        Total length of the time series (npre + npost).

    Returns
    -------
    dict
        {'Tmax': float, 'cpt': int} — maximum test statistic and detected
        changepoint location (1-indexed month).
    """
    return trend_stats(y_ctr=None, y_itv=y_dist, nt=nt)


def page_cusum(errors, m, h):
    """
    Two-sided Page-CUSUM detector on standardised forecast errors.

    Equivalent to R's cptForecast(..., detector="PageCUSUM",
    forecastErrorType="Both") from the changepoint.forecast package.

    Maintains separate upper and lower cumulative sums. Detection fires the
    first time either arm exceeds threshold h. The standard deviation is
    estimated from the in-sample (pre-period) residuals.

    Parameters
    ----------
    errors : array-like (m + npost,)
        Residual series: in-sample residuals concatenated with out-of-sample
        forecast errors. Matches R's r.ts = c(lmfit$residuals, lm.predict - y).
    m : int
        Number of in-sample (pre-period) residuals used for variance estimation.
    h : float
        Detection threshold. Calibrated via null simulations (95th percentile
        of max CUSUM statistic under no-change hypothesis).

    Returns
    -------
    int or float
        1-indexed detection time in the post-period (i.e. t - m + 1), or
        np.inf if no detection within the observed window.
    """
    errors = np.asarray(errors, dtype=float)
    sigma_hat = np.std(errors[:m], ddof=1)
    if sigma_hat < 1e-12:
        return np.inf
    z = errors / sigma_hat
    c_upper = 0.0
    c_lower = 0.0
    for t in range(m, len(errors)):
        c_upper = max(0.0, c_upper + z[t])
        c_lower = max(0.0, c_lower - z[t])
        if c_upper > h or c_lower > h:
            return t - m + 1
    return np.inf


def page_cusum_max_stat(errors, m):
    """
    Return the maximum two-sided Page-CUSUM statistic over the post-period.

    Used for threshold calibration: run on null simulations and take the
    alpha-th percentile of the resulting distribution as threshold h.

    Parameters
    ----------
    errors : array-like (m + npost,)
        Residual series (same format as page_cusum).
    m : int
        Number of in-sample residuals.

    Returns
    -------
    float
        max(upper_arm, lower_arm) over all post-period time steps.
    """
    errors = np.asarray(errors, dtype=float)
    sigma_hat = np.std(errors[:m], ddof=1)
    if sigma_hat < 1e-12:
        return 0.0
    z = errors / sigma_hat
    c_upper = 0.0
    c_lower = 0.0
    max_stat = 0.0
    for t in range(m, len(errors)):
        c_upper = max(0.0, c_upper + z[t])
        c_lower = max(0.0, c_lower - z[t])
        if c_upper > max_stat or c_lower > max_stat:
            max_stat = max(c_upper, c_lower)
    return max_stat


def trend_stats_forecast(y_itv, npre, ntt, phi=None, h=5.0):
    """
    Two-stage forecast-based changepoint detection (BA design).

    Equivalent to the inner loop of R's Rewild_trend_change_Forecast.R.

    Stage 1 — fit a linear model (OLS or ARIMA(1,0,0)) on the pre-period,
    produce multi-step-ahead forecasts, build the residual series, and run
    Page-CUSUM to find *when* a change is first declared (time_est).

    Stage 2 — if detection occurred, run AMOC trend_stats on the residuals
    truncated to length npre + time_est to locate *where* the changepoint
    is (cpt_est).

    Sign convention matches R: in-sample residuals are (actual - fitted);
    out-of-sample errors are (predicted - actual). The two-sided CUSUM
    detects shifts in either direction.

    Parameters
    ----------
    y_itv : array-like (ntt,)
        Intervention time series (full length npre + npost_max).
    npre : int
        Training period length (months).
    ntt : int
        Total time series length (npre + npost_max). Used to build the design
        matrix for forecasting all post-period steps at once.
    phi : float or None
        If provided, fit ARIMA(1,0,0) with AR coefficient; if None, use OLS.
    h : float
        Page-CUSUM threshold (default 5.0; calibrate via null simulations).

    Returns
    -------
    dict
        {'time_est': int or inf, 'cpt_est': int or inf}
        time_est: detection time (1-indexed, relative to post-period start).
        cpt_est:  estimated changepoint location (1-indexed absolute month).
    """
    y = np.asarray(y_itv, dtype=float)
    t_idx = np.arange(1, ntt + 1)
    X = np.column_stack([np.ones(ntt), t_idx])

    in_residuals = None
    out_errors = None

    if phi is not None:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model = ARIMA(y[:npre], exog=X[:npre], order=(1, 0, 0), trend='n')
                fit = model.fit(method='innovations_mle', disp=False)
            in_residuals = np.asarray(fit.resid, dtype=float)
            # predicted - actual matches R's arima.predict$pred - y.ts
            forecast_vals = fit.get_forecast(steps=ntt - npre, exog=X[npre:]).predicted_mean
            out_errors = np.asarray(forecast_vals, dtype=float) - y[npre:]
        except Exception:
            in_residuals = None

    if in_residuals is None:
        # OLS fallback (also used when phi is None)
        ols = OLS(y[:npre], X[:npre]).fit()
        in_residuals = np.asarray(ols.resid, dtype=float)
        # predicted - actual matches R's lm.predict - y.ts
        out_errors = X[npre:] @ ols.params - y[npre:]

    r = np.concatenate([in_residuals, out_errors])
    time_est = page_cusum(r, m=npre, h=h)

    if np.isfinite(time_est):
        time_est_int = int(time_est)
        nt = npre + time_est_int
        # trend_stats requires nt >= mint + 4 = 28; fall back to npre if too short
        if nt >= 28:
            stats = trend_stats(y_ctr=None, y_itv=r[:nt], nt=nt)
            cpt_est = stats['cpt']
        else:
            cpt_est = npre
        return {'time_est': time_est_int, 'cpt_est': cpt_est}
    else:
        return {'time_est': np.inf, 'cpt_est': np.inf}

