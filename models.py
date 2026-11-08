"""The five forecasting methods.

Every model exposes ``predict_dist(raw, X) -> dict`` with:
  raw_mean  model mean before clipping
  point     scored point forecast (clipped at 0 for Ridge/Lasso/Bayesian)
  lo80, hi80, lo95, hi95   interval endpoints (NaN when unavailable)
  noise_var, coef_var, total_var  (Bayesian only; NaN otherwise)
``raw`` is the raw feature frame (baseline uses hour/workingday); ``X`` the
shared transformed matrix (without intercept).
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Lasso, PoissonRegressor, Ridge

Z80 = 1.2815515655446004   # Phi^{-1}(0.90)
Z95 = 1.959963984540054    # Phi^{-1}(0.975)
NAN = np.nan

METHODS = ["baseline", "ridge", "lasso", "poisson", "bayes"]
METHOD_LABELS = {
    "baseline": "Seasonal baseline",
    "ridge": "Ridge regression",
    "lasso": "Lasso regression",
    "poisson": "Poisson regression",
    "bayes": "Bayesian linear regression",
}
HAS_INTERVALS = {"poisson", "bayes"}


def _empty(n):
    return {k: np.full(n, NAN) for k in
            ["lo80", "hi80", "lo95", "hi95", "noise_var", "coef_var", "total_var"]}


class SeasonalBaseline:
    method = "baseline"

    def __init__(self):
        self.table: dict[str, tuple[float, int]] = {}
        self.overall_mean = NAN
        self.n_train = 0

    @staticmethod
    def _key(h, w):
        return f"{int(h)}|{int(w)}"

    def fit(self, raw: pd.DataFrame, y: np.ndarray):
        g = pd.DataFrame({"h": raw["hour"].values, "w": raw["workingday"].values, "y": y})
        agg = g.groupby(["h", "w"])["y"].agg(["mean", "size"])
        self.table = {self._key(h, w): (float(r["mean"]), int(r["size"])) for (h, w), r in agg.iterrows()}
        self.overall_mean = float(np.mean(y))
        self.n_train = len(y)
        return self

    def lookup(self, h, w):
        return self.table.get(self._key(h, w), (self.overall_mean, 0))

    def predict_dist(self, raw: pd.DataFrame, X=None) -> dict:
        mu = np.array([self.lookup(h, w)[0] for h, w in zip(raw["hour"], raw["workingday"])])
        d = _empty(len(mu))
        d.update(raw_mean=mu, point=mu.copy())
        return d

    def settings(self):
        return {"groups": "hour x workingday", "fallback": "overall training mean"}

    def table_frame(self):
        rows = [dict(hour=int(k.split("|")[0]), workingday=int(k.split("|")[1]), mean_cnt=v[0], n=v[1])
                for k, v in self.table.items()]
        return pd.DataFrame(rows).sort_values(["workingday", "hour"])


class LinearPenalised:
    """Ridge or Lasso on the shared representation; point forecast clipped at 0."""

    def __init__(self, method: str, alpha: float, max_iter: int = 50000):
        self.method, self.alpha = method, float(alpha)
        if method == "ridge":
            self.est = Ridge(alpha=self.alpha)
        else:
            self.est = Lasso(alpha=self.alpha, max_iter=max_iter, tol=1e-6, selection="cyclic")

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def predict_dist(self, raw, X) -> dict:
        mu = self.est.predict(X)
        d = _empty(len(mu))
        d.update(raw_mean=mu, point=np.clip(mu, 0, None))
        return d

    @property
    def coef_(self):
        return self.est.coef_

    @property
    def intercept_(self):
        return float(self.est.intercept_)

    def settings(self):
        return {"alpha": self.alpha}


class PoissonModel:
    """Poisson GLM, log link, L2 penalty alpha (scikit-learn PoissonRegressor)."""
    method = "poisson"

    def __init__(self, alpha: float):
        self.alpha = float(alpha)
        self.est = PoissonRegressor(alpha=self.alpha, max_iter=5000, tol=1e-8)

    def fit(self, X, y):
        self.est.fit(X, y)
        return self

    def predict_dist(self, raw, X) -> dict:
        mu = self.est.predict(X)
        d = _empty(len(mu))
        d.update(raw_mean=mu, point=mu.copy())
        d["lo80"], d["hi80"] = poisson_interval(mu, 0.80)
        d["lo95"], d["hi95"] = poisson_interval(mu, 0.95)
        return d

    def settings(self):
        return {"alpha": self.alpha}


def poisson_interval(mu, level):
    a = (1 - level) / 2
    return stats.poisson.ppf(a, mu), stats.poisson.ppf(1 - a, mu)


def gaussian_interval(mean, var, level):
    z = {0.80: Z80, 0.95: Z95}[level]
    s = np.sqrt(var)
    return mean - z * s, mean + z * s


class BayesianLinear:
    """beta ~ N(0, alpha^{-1} I) (intercept included), y | beta ~ N(Phi beta, sigma2 I).

    A = alpha I + Phi^T Phi / sigma2,  A m = Phi^T y / sigma2,  A S = I.
    Predictive: mean phi^T m, variance sigma2 + phi^T S phi.
    """
    method = "bayes"

    def __init__(self, alpha: float, sigma2: float | None = None, sigma2_iters: int = 20):
        self.alpha = float(alpha)
        self.sigma2 = None if sigma2 is None else float(sigma2)
        self.sigma2_iters = sigma2_iters
        self.sigma2_rule = "given" if sigma2 is not None else "training residual fixed point"
        self.m = None
        self.S = None

    @staticmethod
    def design(X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _posterior(self, Phi, y, sigma2):
        d = Phi.shape[1]
        A = self.alpha * np.eye(d) + Phi.T @ Phi / sigma2
        m = np.linalg.solve(A, Phi.T @ y / sigma2)
        S = np.linalg.solve(A, np.eye(d))
        S = 0.5 * (S + S.T)
        return m, S

    def fit(self, X, y, add_intercept: bool = True):
        Phi = self.design(X) if add_intercept else X
        y = np.asarray(y, float)
        if self.sigma2 is None:
            # Training-only estimate: start from var(y); alternate between the
            # posterior mean and the mean squared training residual.
            s2 = float(np.var(y))
            for _ in range(self.sigma2_iters):
                m, _ = self._posterior(Phi, y, s2)
                s2_new = float(np.mean((y - Phi @ m) ** 2))
                if abs(s2_new - s2) < 1e-9 * s2:
                    s2 = s2_new
                    break
                s2 = s2_new
            self.sigma2 = s2
        self.m, self.S = self._posterior(Phi, y, self.sigma2)
        return self

    def predict_phi(self, Phi):
        mean = Phi @ self.m
        coef_var = np.einsum("ij,jk,ik->i", Phi, self.S, Phi)
        coef_var = np.maximum(coef_var, 0.0)
        return mean, np.full(len(mean), self.sigma2), coef_var

    def predict_dist(self, raw, X) -> dict:
        mean, noise, coef = self.predict_phi(self.design(X))
        total = noise + coef
        d = _empty(len(mean))
        d.update(raw_mean=mean, point=np.clip(mean, 0, None), noise_var=noise, coef_var=coef, total_var=total)
        d["lo80"], d["hi80"] = gaussian_interval(mean, total, 0.80)
        d["lo95"], d["hi95"] = gaussian_interval(mean, total, 0.95)
        return d

    def settings(self):
        return {"alpha_prior_precision": self.alpha, "sigma2": self.sigma2, "sigma2_rule": self.sigma2_rule}


def make_model(method: str, setting: dict):
    if method == "baseline":
        return SeasonalBaseline()
    if method in ("ridge", "lasso"):
        return LinearPenalised(method, setting["alpha"])
    if method == "poisson":
        return PoissonModel(setting["alpha"])
    if method == "bayes":
        return BayesianLinear(setting["alpha_prior_precision"], setting.get("sigma2"))
    raise ValueError(method)


def fit_model(model, raw, X, y):
    if isinstance(model, SeasonalBaseline):
        return model.fit(raw, y)
    return model.fit(X, y)


def save_model(model, path):
    with open(path, "wb") as fh:
        pickle.dump(model, fh)


def load_model(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def poisson_log_score(y, mu):
    """Mean of -log Pr(Y = y | mu) under Poisson(mu), natural logs."""
    return float(np.mean(-stats.poisson.logpmf(np.asarray(y, int), np.asarray(mu, float))))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y, float) - np.asarray(yhat, float))))


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y, float) - np.asarray(yhat, float)) ** 2)))


def coverage_width(y, lo, hi):
    y, lo, hi = (np.asarray(a, float) for a in (y, lo, hi))
    if len(y) == 0:
        return NAN, NAN
    return float(np.mean((lo <= y) & (y <= hi))), float(np.mean(hi - lo))


def to_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=float)
