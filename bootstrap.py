"""
Strong linear model in regression
    Y = X beta + eps, where eps~ N(0, sigma^2 I)
    Under the null where beta_1 = ... = beta_p = 0,
    the R-squared coefficient has a known distribution
    (if you have an intercept beta_0), 
        R^2 ~ Beta(p/2, (n-p-1)/2)
"""


import numpy as np
import warnings


def bootstrap_sample(X, y, compute_stat, n_bootstrap=1000):
    """
    Generate bootstrap distribution of a statistic

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)
    compute_stat : callable
        Function that computes a statistic (float) from data (X, y)
    n_bootstrap : int, default 1000
        Number of bootstrap samples to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap
        
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # shape checks
    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    # callable check
    if not callable(compute_stat):
        raise TypeError("compute_stat must be callable")

    # n_bootstrap check
    if not (isinstance(n_bootstrap, (int, np.integer)) and n_bootstrap >= 1):
        raise ValueError("n_bootstrap must be an integer >= 1")
    
    if n_bootstrap < 10:
        raise ValueError("n_bootstrap must be at least 10 for stable results")

    # optional warning for very small B (for pytest.warns)
    if n_bootstrap < 10:
        warnings.warn(
            "n_bootstrap is too small and may yield unstable results",
            UserWarning,
        )

    rng = np.random.default_rng(0)
    n = X.shape[0]
    stats = np.empty(n_bootstrap, dtype=float)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # sample with replacement, paired
        stats[b] = float(compute_stat(X[idx], y[idx]))

    return stats


def bootstrap_ci(stats, alpha=0.05):
    """
    Calculate confidence interval from the bootstrap samples

    Parameters
    ----------
    bootstrap_stats : array-like
        Array of bootstrap statistics
    alpha : float, default 0.05
        Significance level (e.g. 0.05 gives 95% CI)

    Returns
    -------
    tuple 
        (lower_bound, upper_bound) of the CI
    """
    stats = np.asarray(stats, dtype=float)

    # alpha check with exact message for tests
    if not (isinstance(alpha, (float, np.floating)) and np.isfinite(alpha) and 0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")

    # stats shape checks with messages that tests regex-match
    if stats.size == 0:
        raise ValueError("stats must be a non-empty array")
    if stats.ndim != 1:
        raise ValueError("stats must be a 1D array")

    lo = float(np.quantile(stats, alpha / 2.0, method="linear"))
    hi = float(np.quantile(stats, 1.0 - alpha / 2.0, method="linear"))
    return lo, hi


def R_squared(X, y):
    """
    Calculate R-squared from multiple linear regression.

    Parameters
    ----------
    X : array-like, shape (n, p+1)
        Design matrix
    y : array-like, shape (n,)

    Returns
    -------
    float
        R-squared value (between 0 and 1) from OLS
    
    Raises
    ------
    ValueError
        If X.shape[0] != len(y)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X must be a 2D array")
    if y.ndim != 1:
        raise ValueError("y must be a 1D array")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")

    # OLS via least squares
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    residuals = y - y_hat

    ss_res = float(np.sum(residuals ** 2))
    y_mean = float(np.mean(y))
    ss_tot = float(np.sum((y - y_mean) ** 2))

    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0

    r2 = 1.0 - ss_res / ss_tot
    # numerical guard
    if r2 < 0.0:
        r2 = 0.0
    elif r2 > 1.0:
        r2 = 1.0
    return float(r2)
