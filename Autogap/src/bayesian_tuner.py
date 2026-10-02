"""Bayesian optimization with Expected Improvement for gap minimization."""
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel


class BayesianGapTuner:
    def __init__(self, bounds, param_names, length_scale=1.0, seed=42):
        self.bounds = np.asarray(bounds, dtype=float)
        self.param_names = param_names
        self.dim = len(param_names)
        kernel = ConstantKernel(1.0) * Matern(
            length_scale=length_scale * np.ones(self.dim),
            nu=2.5,
        )
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-4,
            normalize_y=True,
            n_restarts_optimizer=2,
            random_state=seed,
        )
        self.X_obs = []
        self.y_obs = []
        self.rng = np.random.default_rng(seed)

    def _to_norm(self, theta):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (np.asarray(theta) - lo) / (hi - lo)

    def _to_raw(self, theta_norm):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return lo + np.clip(theta_norm, 0, 1) * (hi - lo)

    def expected_improvement(self, X_cand, xi=0.01):
        if len(self.y_obs) == 0:
            return np.ones(len(X_cand))
        mu, sigma = self.gp.predict(X_cand, return_std=True)
        y_best = np.min(self.y_obs)
        with np.errstate(divide="warn"):
            Z = (y_best - mu - xi) / (sigma + 1e-12)
            ei = (y_best - mu - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma < 1e-9] = 0.0
        return ei

    def suggest(self, n_candidates=512):
        X_cand = self.rng.uniform(0, 1, size=(n_candidates, self.dim))
        if len(self.y_obs) < 2:
            idx = self.rng.integers(n_candidates)
        else:
            ei = self.expected_improvement(X_cand)
            idx = int(np.argmax(ei))
        return self._to_raw(X_cand[idx])

    def observe(self, theta, gap):
        self.X_obs.append(self._to_norm(theta))
        self.y_obs.append(float(gap))
        self.gp.fit(np.array(self.X_obs), np.array(self.y_obs))

    def best(self):
        if not self.y_obs:
            return None, np.inf
        i = int(np.argmin(self.y_obs))
        return self._to_raw(self.X_obs[i]), float(self.y_obs[i])