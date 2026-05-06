"""
Statistical metrics for trend-change detection and distribution comparison.
"""

import numpy as np
import scipy.integrate
import scipy.stats
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.arima.model import ARIMA


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
        seg1 = np.arange(1, cp + 1)  # 1-indexed to match R's 1:seglen[m]
        seg2 = np.arange(cp, nt)

        # Design matrix for trend model
        X = np.column_stack(
            [
                np.ones(nt),
                np.concatenate([seg1, np.full(len(seg2), seg1[-1])]),
                np.concatenate([np.zeros(len(seg1)), np.arange(1, len(seg2) + 1)]),
            ]
        )

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

    return {"Tmax": Tmax, "cpt": cpt}


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
        seg1 = np.arange(1, cp + 1)  # 1-indexed to match R's 1:seglen[m]
        seg2 = np.arange(cp, nt)

        # Same design matrix as trend_stats
        X = np.column_stack(
            [
                np.ones(nt),
                np.concatenate([seg1, np.full(len(seg2), seg1[-1])]),
                np.concatenate([np.zeros(len(seg1)), np.arange(1, len(seg2) + 1)]),
            ]
        )

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
            model = ARIMA(y_dif, exog=X, order=(1, 0, 0), trend="n")
            fit = model.fit(method="innovations_mle", disp=False)

            # Test statistic: (beta1 - beta2) / se(beta1 - beta2)
            # In R: vec = c(0, 0, -1, 1); coef indices 3 and 4 (1-based) = beta1, beta2
            # In Python: exog params are at indices 1, 2, 3 (after AR param at index 0)
            params = fit.params  # [ar1, beta0, beta1, beta2]
            cov = fit.cov_params()  # 4×4 covariance matrix

            # Guard: all diagonal entries must be positive before trusting ARIMA estimates.
            # Matches R's: if (all(diag(armafit$var.coef) > 0))
            if not np.all(np.diag(cov) > 0):
                raise ValueError("non-positive diagonal in ARIMA covariance")

            vec = np.array([0, 0, -1, 1])  # contrast: beta2 - beta1
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

    return {"Tmax": Tmax, "cpt": cpt}


def trend_stats_cdf(y_dist, nt):
    """
    Apply the AMOC trend-change test to a distance time series.

    Thin wrapper around trend_stats() that treats the distance series as the
    response directly (y_ctr=None), matching the intended behaviour of R's
    trend.stats.cdf(). The distance series replaces the difference series used
    in the scalar time series case.

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
