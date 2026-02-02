from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray


@dataclass
class PosteriorSummary:
    mean: float
    var: float
    precision: float


class CompositeGaussianPairwise:
    """Gaussian process mean inference with pairwise composite likelihood."""

    def __init__(
        self,
        k: int = 20,
        coord_max: float = 20.0,
        tau: float = 1.0,
        omega: float = 3.0,
        nugget: float = 1e-6,
        mu_true: float = 0.0,
        coords: Optional[NDArray[np.float64]] = None,
    ) -> None:
        self.k = k
        self.coord_max = coord_max
        self.tau = tau
        self.omega = omega
        self.nugget = nugget
        self.mu_true = mu_true

        if coords is None:
            self.coords = np.linspace(0.0, coord_max, k)
        else:
            self.coords = np.asarray(coords, dtype=float)
            if self.coords.shape[0] != k:
                raise ValueError("coords must have length k")

        self._build_covariance()
        self._build_pairs()

    def _build_covariance(self) -> None:
        dist = np.abs(self.coords[:, None] - self.coords[None, :])
        self.Sigma = self.tau * np.exp(-dist / self.omega)
        self.Sigma += self.nugget * np.eye(self.k)
        self.Sigma_inv = np.linalg.inv(self.Sigma)
        self.ones = np.ones(self.k)
        self.ones_cov_inv = self.Sigma_inv @ self.ones
        self.full_weight = float(self.ones @ self.ones_cov_inv)

    def _build_pairs(self) -> None:
        pair_i = []
        pair_j = []
        pair_a0 = []
        pair_a1 = []
        pair_w = []

        ones2 = np.ones(2)
        for i in range(self.k - 1):
            for j in range(i + 1, self.k):
                cov_2 = self.Sigma[np.ix_([i, j], [i, j])]
                inv_2 = np.linalg.inv(cov_2)
                a_vec = inv_2 @ ones2
                w_val = float(ones2 @ a_vec)
                pair_i.append(i)
                pair_j.append(j)
                pair_a0.append(float(a_vec[0]))
                pair_a1.append(float(a_vec[1]))
                pair_w.append(w_val)

        self.pair_i = np.array(pair_i, dtype=int)
        self.pair_j = np.array(pair_j, dtype=int)
        self.pair_a0 = np.array(pair_a0, dtype=float)
        self.pair_a1 = np.array(pair_a1, dtype=float)
        self.pair_w = np.array(pair_w, dtype=float)
        self.pair_weight_sum = float(np.sum(self.pair_w))

    def generate_data(self, n_rep: int, seed: Optional[int] = None) -> NDArray[np.float64]:
        rng = np.random.default_rng(seed)
        mean = self.mu_true * self.ones
        return rng.multivariate_normal(mean, self.Sigma, size=n_rep)

    def compute_full_stats(self, y: NDArray[np.float64]) -> Tuple[float, float]:
        n_rep = y.shape[0]
        b_full = float(np.sum(y @ self.ones_cov_inv))
        h_full = float(n_rep * self.full_weight)
        return b_full, h_full

    def compute_composite_stats(self, y: NDArray[np.float64]) -> Tuple[float, float]:
        n_rep = y.shape[0]
        y_i = y[:, self.pair_i]
        y_j = y[:, self.pair_j]
        b_comp = float(np.sum(y_i * self.pair_a0 + y_j * self.pair_a1))
        h_comp = float(n_rep * self.pair_weight_sum)
        return b_comp, h_comp

    def posterior_composite(
        self,
        y: NDArray[np.float64],
        eta: float,
        prior_mean: float,
        prior_var: float,
    ) -> PosteriorSummary:
        b_comp, h_comp = self.compute_composite_stats(y)
        prior_prec = 1.0 / prior_var
        precision = eta * h_comp + prior_prec
        mean = (eta * b_comp + prior_mean * prior_prec) / precision
        var = 1.0 / precision
        return PosteriorSummary(mean=float(mean), var=float(var), precision=float(precision))

    def posterior_full(
        self,
        y: NDArray[np.float64],
        prior_mean: float,
        prior_var: float,
    ) -> PosteriorSummary:
        b_full, h_full = self.compute_full_stats(y)
        prior_prec = 1.0 / prior_var
        precision = h_full + prior_prec
        mean = (b_full + prior_mean * prior_prec) / precision
        var = 1.0 / precision
        return PosteriorSummary(mean=float(mean), var=float(var), precision=float(precision))

    def one_step_newton_full(self, mu_start: float, y: NDArray[np.float64]) -> float:
        b_full, h_full = self.compute_full_stats(y)
        score = b_full - mu_start * h_full
        return float(mu_start + score / h_full)

    def calibrate_samples(
        self,
        samples: NDArray[np.float64],
        mu_work: float,
        mu_location: float,
        target_var: float,
    ) -> Tuple[NDArray[np.float64], float]:
        work_var = float(np.var(samples, ddof=1))
        scale = float(np.sqrt(target_var / work_var))
        calibrated = mu_location + scale * (samples - mu_work)
        return calibrated, scale
