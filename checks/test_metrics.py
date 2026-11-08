"""Independent numerical checks of scores, intervals and the Bayesian posterior."""
import numpy as np
import pandas as pd
from scipy import stats

import models as M


def test_mae_rmse_three_row_example():
    y, f = [10, 20, 30], [12, 17, 31]
    assert M.mae(y, f) == 2
    assert abs(M.rmse(y, f) - np.sqrt(14 / 3)) < 1e-12
    assert abs(M.rmse(y, f) - 2.160) < 1e-3


def test_coverage_inclusive_endpoints_two_of_three():
    y = [5, 10, 20]
    lo = [5, 0, 0]       # 5 equals the lower endpoint -> covered
    hi = [8, 12, 15]     # 20 above 15 -> not covered
    cov, width = M.coverage_width(y, lo, hi)
    assert abs(cov - 2 / 3) < 1e-12
    assert abs(width - (3 + 12 + 15) / 3) < 1e-12


def test_poisson_quantiles_against_cdf():
    mu = np.array([3.7, 100.0])
    lo, hi = M.poisson_interval(mu, 0.80)
    for m, l, h in zip(mu, lo, hi):
        assert stats.poisson.cdf(l, m) >= 0.10 and stats.poisson.cdf(l - 1, m) < 0.10
        assert stats.poisson.cdf(h, m) >= 0.90 and stats.poisson.cdf(h - 1, m) < 0.90


def test_poisson_log_score_matches_formula():
    y, mu = np.array([0, 3, 12]), np.array([1.5, 2.0, 10.0])
    from math import lgamma, log
    manual = np.mean([m - k * log(m) + lgamma(k + 1) for k, m in zip(y, mu)])
    assert abs(M.poisson_log_score(y, mu) - manual) < 1e-10


def test_bayesian_small_example():
    Phi = np.array([[1.0, 0.0], [1.0, 1.0]])
    b = M.BayesianLinear(alpha=1.0, sigma2=1.0).fit(Phi, [2.0, 4.0], add_intercept=False)
    np.testing.assert_allclose(b.m, [1.6, 1.2], atol=1e-12)
    np.testing.assert_allclose(b.S, [[0.4, -0.2], [-0.2, 0.6]], atol=1e-12)
    assert np.allclose(b.S, b.S.T)
    mean, noise, coef = b.predict_phi(np.array([[1.0, 2.0]]))
    assert abs(mean[0] - 4) < 1e-12 and abs(coef[0] - 2) < 1e-12 and noise[0] == 1
    lo, hi = M.gaussian_interval(mean, noise + coef, 0.80)
    assert abs(lo[0] - 1.78) < 0.005 and abs(hi[0] - 6.22) < 0.005


def test_bayesian_matches_independent_inverse_and_nonnegative_variance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 4))
    y = X @ [1, -2, 0.5, 0] + rng.normal(size=50)
    b = M.BayesianLinear(alpha=0.5, sigma2=2.0).fit(X, y)
    Phi = b.design(X)
    S_ref = np.linalg.inv(0.5 * np.eye(5) + Phi.T @ Phi / 2.0)
    np.testing.assert_allclose(b.S, S_ref, atol=1e-10)
    np.testing.assert_allclose(b.m, S_ref @ Phi.T @ y / 2.0, atol=1e-10)
    _, _, coef = b.predict_phi(Phi)
    assert (coef >= 0).all()


def test_gaussian_interval_illustration():
    lo, hi = M.gaussian_interval(np.array([100.0]), np.array([100.0]), 0.80)
    assert abs(lo[0] - 87.18) < 0.01 and abs(hi[0] - 112.82) < 0.01
    lo, hi = M.gaussian_interval(np.array([100.0]), np.array([100.0]), 0.95)
    assert abs(lo[0] - 80.40) < 0.01 and abs(hi[0] - 119.60) < 0.01


def test_seasonal_baseline_toy_and_fallback():
    raw = pd.DataFrame({"hour": [8, 8, 8, 8, 8], "workingday": [1, 1, 1, 0, 0]})
    b = M.SeasonalBaseline().fit(raw, np.array([100, 120, 140, 20, 40.0]))
    p = b.predict_dist(pd.DataFrame({"hour": [8, 8, 9], "workingday": [1, 0, 1]}))
    np.testing.assert_allclose(p["point"], [120, 30, 84])   # 84 = overall mean fallback
    assert np.isnan(p["lo80"]).all()


def test_clipping_keeps_raw_mean():
    X = np.array([[0.0], [1.0]])
    m = M.LinearPenalised("ridge", 1e-9).fit(X, np.array([-5.0, 5.0]))
    d = m.predict_dist(None, X)
    assert d["raw_mean"][0] < 0 and d["point"][0] == 0
