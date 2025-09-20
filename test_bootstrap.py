import pytest
import numpy as np
from bootstrap import bootstrap_sample, bootstrap_ci, R_squared


def _make_linear_data(n=120, p=2, noise_std=0.5, seed=0):
    rng = np.random.default_rng(seed)
    Xn = rng.normal(size=(n, p))
    X = np.c_[np.ones(n), Xn]  # (n, p+1) with intercept
    beta_true = np.array([1.0, 2.0, -1.0])[: p + 1]
    y = X @ beta_true + rng.normal(scale=noise_std, size=n)
    return X, y, beta_true

def test_smoke_happy_path_runs():
    X, y, _ = _make_linear_data(n=80, p=2, noise_std=1.0, seed=0)
    stats = bootstrap_sample(X, y, compute_stat=R_squared, n_bootstrap=25)
    bootstrap_ci(stats, alpha=0.1)  # should not raise


def test_r_squared_mismatched_rows_raises():
    X, y, _ = _make_linear_data(n=60, p=1, seed=0)
    with pytest.raises(ValueError, match=r"(rows|shape|same number)"):
        R_squared(X[:-1], y)

def test_bootstrap_sample_invalid_inputs_raise():
    X, y, _ = _make_linear_data(n=50, p=1, seed=0)
    # n_bootstrap invalid
    with pytest.raises(ValueError, match=r"n_bootstrap.*(>=\s*1|positive|integer)"):
        bootstrap_sample(X, y, compute_stat=R_squared, n_bootstrap=0)
    # compute_stat must be callable
    with pytest.raises(TypeError, match=r"compute_stat.*callable"):
        bootstrap_sample(X, y, compute_stat=None, n_bootstrap=10)
    # X,y size mismatch
    with pytest.raises(ValueError, match=r"(rows|shape|same number)"):
        bootstrap_sample(X[:-1], y, compute_stat=R_squared, n_bootstrap=5)

def test_bootstrap_ci_invalid_alpha_and_empty_raise():
    rng = np.random.default_rng(0)
    stats = rng.random(100)
    # alpha out of (0,1)
    for bad_alpha in (-0.1, 0.0, 1.0, 2.0, np.nan):
        with pytest.raises(ValueError, match=r"alpha.*\(0, *1\)"):
            bootstrap_ci(stats, alpha=bad_alpha)
    # empty stats
    with pytest.raises(ValueError, match=r"stats.*(non[- ]?empty|at least one)"):
        bootstrap_ci(np.array([]), alpha=0.1)
    # wrong shape 
    with pytest.raises(ValueError, match=r"(1D|one[- ]?dimensional|shape)"):
        bootstrap_ci(stats.reshape(20, 5), alpha=0.1)


def test_bootstrap_sample_error_when_B_small():
    X, y, _ = _make_linear_data(n=40, p=1, seed=0)
    with pytest.raises(ValueError, match=r"n_bootstrap.*at least 10"):
        bootstrap_sample(X, y, compute_stat=R_squared, n_bootstrap=5)





