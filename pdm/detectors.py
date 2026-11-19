"""Unsupervised detectors, fitted on the first 400 s (normal operation) of each experiment, then run on the rest.
Each returns a score per test second and the alarm limit computed on the training part only."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import Ridge


def _scale(train, test):
    mu, sd = train.mean(0), train.std(0)
    keep = sd > 1e-9                              # constant sensors (e.g. flow fixed during warm-up) are dropped
    return (train[:, keep] - mu[keep]) / sd[keep], (test[:, keep] - mu[keep]) / sd[keep]


def smooth(x, w):
    if w <= 1:
        return x
    c = np.cumsum(np.r_[0.0, x])                  # causal moving average: only the past w seconds
    i = np.arange(len(x)); j = np.maximum(0, i - w + 1)
    return (c[i + 1] - c[j]) / (i + 1 - j)


class T2:
    """Hotelling's T-squared with the F-distribution control limit."""
    name = "Hotelling T²"

    def __init__(self, p=0.999):
        self.p = p

    def fit(self, X):
        Z, _ = _scale(X, X)
        self.X0 = X
        self.inv = np.linalg.pinv(np.cov(Z.T))
        n, m = Z.shape
        self.ucl = m * (n - 1) * (n + 1) / (n * (n - m)) * stats.f.ppf(self.p, m, n - m)
        self.train_score = np.einsum("ij,jk,ik->i", Z, self.inv, Z)
        return self

    def score(self, X):
        _, Z = _scale(self.X0, X)
        return np.einsum("ij,jk,ik->i", Z, self.inv, Z)


class PCAQ:
    """PCA on the training part (85% of the variance): T² in the model space and Q (squared prediction error) in the
    residual space, each against its limit; the score is the larger of the two ratios."""
    name = "PCA T² + Q"

    def __init__(self, var=0.85, p=0.999):
        self.var, self.p = var, p

    def fit(self, X):
        Z, _ = _scale(X, X)
        self.X0 = X
        self.pca = PCA(n_components=self.var, svd_solver="full").fit(Z)
        k = self.pca.n_components_; n = len(Z)
        self.lam = self.pca.explained_variance_
        self.t2_ucl = k * (n - 1) * (n + 1) / (n * (n - k)) * stats.f.ppf(self.p, k, n - k)
        q = self._q(Z)
        g = q.var() / (2 * q.mean()); h = 2 * q.mean() ** 2 / q.var()     # Box's approximation of the Q limit
        self.q_ucl = g * stats.chi2.ppf(self.p, h)
        self.ucl = 1.0
        self.train_score = self._s(Z)
        return self

    def _q(self, Z):
        R = Z - self.pca.inverse_transform(self.pca.transform(Z))
        return (R ** 2).sum(1)

    def _s(self, Z):
        T = self.pca.transform(Z)
        t2 = (T ** 2 / self.lam).sum(1)
        return np.maximum(t2 / self.t2_ucl, self._q(Z) / self.q_ucl)

    def score(self, X):
        _, Z = _scale(self.X0, X)
        return self._s(Z)


class IForest:
    name = "Isolation forest"

    def fit(self, X):
        Z, _ = _scale(X, X)
        self.X0 = X
        self.m = IsolationForest(n_estimators=200, random_state=0).fit(Z)
        self.train_score = -self.m.score_samples(Z)
        self.ucl = np.quantile(self.train_score, 0.999)
        return self

    def score(self, X):
        _, Z = _scale(self.X0, X)
        return -self.m.score_samples(Z)


class VirtualSensors:
    """Each sensor is predicted from the others by a ridge regression fitted on normal operation; a fault breaks
    the physical relations between flow, pressure, current and vibration. Score: mean squared standardised residual."""
    name = "Virtual sensors (cross-sensor residuals)"

    def fit(self, X):
        Z, _ = _scale(X, X)
        self.X0 = X
        d = Z.shape[1]
        self.models = [Ridge(alpha=1.0).fit(np.delete(Z, j, 1), Z[:, j]) for j in range(d)]
        R = self._res(Z)
        self.rsd = R.std(0) + 1e-6
        self.train_score = ((R / self.rsd) ** 2).mean(1)
        self.ucl = np.quantile(self.train_score, 0.999)
        return self

    def _res(self, Z):
        return np.column_stack([Z[:, j] - m.predict(np.delete(Z, j, 1)) for j, m in enumerate(self.models)])

    def score(self, X):
        _, Z = _scale(self.X0, X)
        return ((self._res(Z) / self.rsd) ** 2).mean(1)


DETECTORS = {c.name: c for c in (T2, PCAQ, IForest, VirtualSensors)}


def raw(det, X_train, X_test):
    d = det().fit(X_train)
    return d.score(X_test), d.ucl


def alarms_from(score, ucl, k: float = 1.0, w: int = 1):
    """Smooth the score over the last w seconds, alarm when above k times the training limit."""
    s = smooth(score, w)
    return s > k * ucl, s / ucl


def alarms(det, X_train, X_test, k: float = 1.0, w: int = 1):
    sc, ucl = raw(det, X_train, X_test)
    a, r = alarms_from(sc, ucl, k, w)
    return a, r, sc
