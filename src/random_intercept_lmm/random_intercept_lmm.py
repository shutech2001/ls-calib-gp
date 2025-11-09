from __future__ import annotations

from collections import deque
import logging
from typing import Tuple, Deque, Optional

import numpy as np
from numpy.typing import NDArray
import pandas as pd  # type: ignore
import statsmodels.formula.api as smf  # type: ignore


class RandomInterceptLMM:
    """
    Random Intercept Linear Mixed Model (LMM) with fixed Gaussian prior (for Bayesian inference).
    """

    def __init__(
        self,
        J: int = 60,
        n_per_group: int = 5,
        p: int = 1,
        beta_true: Optional[NDArray[np.float64]] = None,
        tau: float = 1.0,
        sigma: float = 1.0,
        eta: float = 1.0,
    ) -> None:
        """Initialize the RandomInterceptLMM model.

        Args:
            J (int, optional): Number of groups. Defaults to 60.
            n_per_group (int, optional): Number of observations per group. Defaults to 5.
            p (int, optional): Number of covariates. Defaults to 1.
            beta_true (Optional[NDArray[np.float64]], optional): True fixed effects. Defaults to None.
            tau (float, optional): True random intercept variance. Defaults to 1.0.
            sigma (float, optional): True residual variance. Defaults to 1.0.
            eta (float, optional): Prior precision. Defaults to 1.0.
        """
        self.J: int = J
        self.n_per_group: int = n_per_group
        self.p: int = p
        self.beta_true: NDArray[np.float64] = beta_true if beta_true is not None else np.ones(p)
        self.tau: float = tau
        self.sigma: float = sigma
        self.eta: float = eta
        self.s_n = J  # effective scale

        # Prior parameters (fixed Gaussian prior)
        self.mu_0: NDArray[np.float64] = np.zeros(p)
        self.Sigma_0: NDArray[np.float64] = 10.0 * np.eye(p)

    def generate_data(self, seed: Optional[int] = None) -> Tuple[Deque, Deque]:
        """Generate synthetic data from the RandomInterceptLMM model.

        Args:
            seed (Optional[int], optional): Random seed. Defaults to None.

        Returns:
            Tuple[Deque, Deque]: A tuple of two deques containing the response and design matrices.
        """
        if seed is not None:
            np.random.seed(seed)

        y_deque: Deque = deque()
        X_deque: Deque = deque()

        for j in range(self.J):
            # Generate design matrix
            X_i = np.random.randn(self.n_per_group, self.p)

            # Generate random intercept
            b_i = np.random.randn() * self.tau

            # Generate response
            epsilon_i = np.random.randn(self.n_per_group) * self.sigma
            y_i = X_i @ self.beta_true + b_i + epsilon_i

            y_deque.append(y_i)
            X_deque.append(X_i)

        return y_deque, X_deque

    def compute_Sigma_i(self) -> NDArray:
        """Compute the covariance matrix of the random intercepts.

        Returns:
            NDArray: The covariance matrix of the random intercepts.
        """
        n = self.n_per_group
        Sigma_i = self.tau**2 * np.ones((n, n)) + self.sigma**2 * np.eye(n)
        return Sigma_i

    def compute_posterior(self, y_deque: Deque, X_deque: Deque) -> Tuple[NDArray, NDArray]:
        """Compute the posterior mean and covariance matrix.

        Args:
            y_deque (Deque): A deque containing the response vectors.
            X_deque (Deque): A deque containing the design matrices.

        Returns:
            Tuple[NDArray, NDArray]: A tuple containing the posterior mean and covariance matrix.
        """
        Sigma_i = self.compute_Sigma_i()
        Sigma_i_inv = np.linalg.inv(Sigma_i)

        # Compute posterior precision
        Lambda_post = np.linalg.inv(self.Sigma_0)
        for j in range(self.J):
            Lambda_post += self.eta * X_deque[j].T @ Sigma_i_inv @ X_deque[j]

        m_post_term = np.linalg.inv(self.Sigma_0) @ self.mu_0
        for j in range(self.J):
            m_post_term += self.eta * X_deque[j].T @ Sigma_i_inv @ y_deque[j]

        Lambda_post_inv = np.linalg.inv(Lambda_post)
        m_post = Lambda_post_inv @ m_post_term

        return m_post, Lambda_post_inv

    def sample_posterior(self, m_post: NDArray, Lambda_post_inv: NDArray, M: int = 1000) -> NDArray:
        """Sample from the posterior distribution.

        Args:
            m_post (NDArray): Posterior mean.
            Lambda_post_inv (NDArray): Posterior precision matrix.
            M (int, optional): Number of samples. Defaults to 1000.

        Returns:
            NDArray: Samples from the posterior distribution.
        """
        samples = np.random.multivariate_normal(m_post, Lambda_post_inv, size=M)
        return samples

    def compute_U_n(self, beta: NDArray, y_deque: Deque, X_deque: Deque) -> NDArray:
        """Compute the U_n matrix.

        Args:
            beta (NDArray): Fixed effects.
            y_deque (Deque): A deque containing the response vectors.
            X_deque (Deque): A deque containing the design matrices.

        Returns:
            NDArray: The U_n matrix.
        """
        Sigma_i = self.compute_Sigma_i()
        Sigma_i_inv = np.linalg.inv(Sigma_i)

        U_n = np.zeros(self.p)
        for j in range(self.J):
            residual = y_deque[j] - X_deque[j] @ beta
            U_n += -X_deque[j].T @ Sigma_i_inv @ residual

        U_n /= self.J
        return U_n

    def compute_J_n(self, beta: NDArray, X_deque: Deque) -> NDArray:
        """Compute the J_n matrix.

        Args:
            beta (NDArray): Fixed effects.
            X_deque (Deque): A deque containing the design matrices.

        Returns:
            NDArray: The J_n matrix.
        """
        Sigma_i = self.compute_Sigma_i()
        Sigma_i_inv = np.linalg.inv(Sigma_i)

        J_n = np.zeros((self.p, self.p))
        for j in range(self.J):
            J_n += X_deque[j].T @ Sigma_i_inv @ X_deque[j]

        J_n /= self.J
        return J_n

    def compute_one_step_center(self, beta_GB: NDArray, y_deque: Deque, X_deque: Deque) -> NDArray:
        """Compute the one-step center (admissible center).

        Args:
            beta_GB (NDArray): Posterior mean of the fixed effects.
            y_deque (Deque): A deque containing the response vectors.
            X_deque (Deque): A deque containing the design matrices.

        Returns:
            NDArray: The one-step center.
        """
        U_n = self.compute_U_n(beta_GB, y_deque, X_deque)
        J_n = self.compute_J_n(beta_GB, X_deque)

        beta_dagger = beta_GB - np.linalg.solve(J_n, U_n)
        return beta_dagger

    def compute_calibration_matrix(
        self,
        beta_GB: NDArray,
        beta_dagger: NDArray,
        samples: NDArray,
        y_deque: Deque,
        X_deque: Deque,
    ) -> Tuple[NDArray, NDArray]:
        """Compute the calibration matrix.

        Args:
            beta_GB (NDArray): Posterior mean of the fixed effects.
            beta_dagger (NDArray): One-step center.
            samples (NDArray): Samples from the posterior distribution.
            y_deque (Deque): A deque containing the response vectors.
            X_deque (Deque): A deque containing the design matrices.

        Returns:
            Tuple[NDArray, NDArray]: A tuple containing the calibration matrix and the target variance.
        """
        # Compute posterior covariance
        if self.p == 1:
            Sigma_post = np.var(samples, axis=0, ddof=1).reshape(1, 1)
        else:
            Sigma_post = np.cov(samples.T)
        H_0_inv = self.s_n * Sigma_post

        # Compute J_hat
        J_hat = self.compute_J_n(beta_GB, X_deque)

        # Compute K_hat
        Sigma_i = self.compute_Sigma_i()
        Sigma_i_inv = np.linalg.inv(Sigma_i)

        U_deque: Deque = deque()
        for j in range(self.J):
            residual = y_deque[j] - X_deque[j] @ beta_dagger
            U_i = -X_deque[j].T @ Sigma_i_inv @ residual
            U_deque.append(U_i)

        U_array = np.array(list(U_deque))
        if self.p == 1:
            K_hat = np.var(U_array, ddof=1).reshape(1, 1)
        else:
            K_hat = np.cov(U_array.T)

        # Compute target variance
        J_hat_inv = np.linalg.inv(J_hat)
        V_target_hat = J_hat_inv @ K_hat @ J_hat_inv.T

        # Compute calibration matrix \Omega = V_target^{1/2} H_0^{1/2}
        H_0 = np.linalg.inv(H_0_inv)
        V_target_sqrt = self._matrix_sqrt(V_target_hat)
        H_0_sqrt = self._matrix_sqrt(H_0)
        Omega = V_target_sqrt @ H_0_sqrt

        return Omega, V_target_hat

    def _matrix_sqrt(self, A: NDArray) -> NDArray:
        """Compute the square root of a matrix.

        Args:
            A (NDArray): The matrix to compute the square root of.

        Returns:
            NDArray: The square root of the matrix.
        """
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-16)  # ensure positive definite
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def calibrate_samples(
        self,
        samples: NDArray,
        beta_GB: NDArray,
        beta_dagger: NDArray,
        Omega: NDArray,
    ) -> NDArray:
        """Calibrate the samples.

        Args:
            samples (NDArray): Samples from the posterior distribution.
            beta_GB (NDArray): Posterior mean of the fixed effects.
            beta_dagger (NDArray): One-step center.
            Omega (NDArray): Calibration matrix.

        Returns:
            NDArray: Calibrated samples.
        """
        M = len(samples)
        calibrated_samples = np.zeros_like(samples)

        if self.p == 1:
            for m in range(M):
                calibrated_samples[m] = beta_dagger + Omega[0, 0] * (samples[m] - beta_GB)
        else:
            for m in range(M):
                calibrated_samples[m] = beta_dagger + Omega @ (samples[m] - beta_GB)

        return calibrated_samples

    def fit_frequentist_lmm(
        self,
        y_deque: Deque,
        X_deque: Deque,
        alpha: float = 0.05,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Fit a frequentist LMM.

        Args:
            y_deque (Deque): A deque containing the response vectors.
            X_deque (Deque): A deque containing the design matrices.
            alpha (float, optional): Significance level. Defaults to 0.05.

        Returns:
            Tuple[NDArray, NDArray, NDArray]:
                A tuple containing the fixed effects estimates, the lower and upper bounds of the confidence interval.
            If the fit fails, returns NaN for all three values.
        """
        y_flat = np.concatenate(list(y_deque))
        X_flat = np.vstack(list(X_deque))
        group = np.repeat(np.arange(self.J), self.n_per_group)

        data_dict = {"y": y_flat, "group": group}
        for j in range(self.p):
            data_dict[f"x{j}"] = X_flat[:, j]

        data = pd.DataFrame(data_dict)

        if self.p == 1:
            formula = "y ~ x0 - 1"
        else:
            x_terms = " + ".join([f"x{j}" for j in range(self.p)])
            formula = f"y ~ {x_terms} - 1"

        try:
            model = smf.mixedlm(formula, data, groups=data["group"])
            result = model.fit(reml=True, method="powell")

            # Extract fixed effects estimates
            beta_freq = result.params.values

            # Get confidence intervals
            fe_ci = result.conf_int(alpha=alpha)
            ci_lower = fe_ci.iloc[:, 0].values
            ci_upper = fe_ci.iloc[:, 1].values

            return beta_freq, ci_lower, ci_upper

        except Exception as e:
            logging.warning(f"Frequentist fit failed: {e}. Returning NaN.")
            # Return NaN if fit fails
            return np.full(self.p, np.nan), np.full(self.p, np.nan), np.full(self.p, np.nan)
