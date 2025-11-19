from __future__ import annotations

from collections import deque
import logging
from typing import Tuple, Deque, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize  # type: ignore
from scipy.stats import invgauss  # type: ignore


class HuberRandomIntercept:
    """
    Random Intercept Linear Mixed Model (LMM) with Huber loss and MCMC inference.
    """

    def __init__(
        self,
        G: int = 60,
        n_per_group: int = 5,
        p: int = 1,
        beta_true: Optional[NDArray[np.float64]] = None,
        tau: float = 1.0,
        sigma: float = 1.0,
        c: float = 1.0,
        eta: float = 1.0,
        lam: float = 1.0,
        mu_prior: Optional[NDArray[np.float64]] = None,
    ) -> None:
        """Initialize the HuberRandomInterceptLMM model.

        Args:
            G (int, optional): Number of groups. Defaults to 60.
            n_per_group (int, optional): Number of observations per group. Defaults to 5.
            p (int, optional): Number of covariates. Defaults to 1.
            beta_true (Optional[NDArray[np.float64]], optional): True fixed effects. Defaults to None.
            tau (float, optional): True random intercept standard deviation. Defaults to 1.0.
            sigma (float, optional): True residual standard deviation. Defaults to 1.0.
            c (float, optional): Huber tuning constant. Defaults to 1.0.
            eta (float, optional): Learning rate. Defaults to 1.0.
            lam (float, optional): Ridge penalty parameter. Defaults to 1.0.
            mu_prior (Optional[NDArray[np.float64]], optional): Prior mean for beta. Defaults to None.
        """
        self.G: int = G
        self.n_per_group: int = n_per_group
        self.p: int = p
        self.beta_true: NDArray[np.float64] = beta_true if beta_true is not None else np.ones(p)
        self.tau: float = tau
        self.sigma: float = sigma
        self.c: float = c
        self.eta: float = eta
        self.lam: float = lam
        self.mu_prior: NDArray[np.float64] = mu_prior if mu_prior is not None else np.zeros(p)

        # Effective scale s_n = n = G * n_per_group
        self.s_n = G * n_per_group

        # Prior precision matrix Q (identity for ridge)
        self.Q: NDArray[np.float64] = np.eye(p)

    def generate_data(
        self,
        seed: Optional[int] = None,
        outlier_prob: float = 0.0,
        outlier_scale: float = 10.0,
    ) -> Tuple[Deque, Deque]:
        """Generate synthetic data from the RandomInterceptLMM model.

        Args:
            seed (Optional[int], optional): Random seed. Defaults to None.
            outlier_prob (float, optional): Probability of outliers. Defaults to 0.0.
            outlier_scale (float, optional): Scale of outlier noise. Defaults to 10.0.

        Returns:
            Tuple[Deque, Deque]: A tuple of two deques containing the response and design matrices.
        """
        if seed is not None:
            np.random.seed(seed)

        y_deque: Deque = deque()
        X_deque: Deque = deque()

        for i in range(self.G):
            # Generate design matrix
            X_i = np.random.randn(self.n_per_group, self.p)

            # Generate random intercept
            b_i = np.random.randn() * self.tau

            # Generate response with potential outliers
            epsilon_i = np.random.randn(self.n_per_group) * self.sigma

            # Add outliers
            if outlier_prob > 0:
                outlier_mask = np.random.uniform(0, 1, size=self.n_per_group) < outlier_prob
                epsilon_i[outlier_mask] += np.random.randn(outlier_mask.sum()) * outlier_scale

            y_i = X_i @ self.beta_true + b_i + epsilon_i

            y_deque.append(y_i)
            X_deque.append(X_i)

        return y_deque, X_deque

    def compute_pseudo_true_beta_lambda(
        self,
        G_large: int = 1000,
        large_replication: int = 500,
        outlier_prob: float = 0.1,
        outlier_scale: float = 10.0,
        seed: int = 98765,
        verbose: bool = True,
    ) -> NDArray[np.float64]:
        """Numerically approximate the pseudo-true value beta_lambda^*.

        We define beta_lambda^* as the population limit of the penalized
        estimating-equation estimator under the data-generating model
        specified by (beta_true, tau, sigma, c, eta, lam, outliers, ...).

        This routine:
          1. Constructs a 'large' model with G_large groups.
          2. Repeats:
                - generate a large dataset,
                - fit the penalized EE estimator beta_hat,
             for n_rep_large replications.
          3. Returns the average of beta_hat across replications as a
             Monte Carlo approximation to beta_lambda^*.

        Args:
            G_large (int, optional):
                Number of groups in the 'large-n' model.
                Total n_large = G_large * n_per_group.
            n_rep_large (int, optional):
                Number of Monte Carlo replications R.
            outlier_prob (float, optional):
                Outlier probability for the pseudo-true target
                (should match the simulation setting).
            outlier_scale (float, optional):
                Outlier scale (should match the simulation setting).
            seed (int, optional):
                Base random seed.
            verbose (bool, optional):
                If True, print simple progress and Monte Carlo SE.

        Returns:
            NDArray[np.float64]:
                Approximate pseudo-true value beta_lambda^* (vector of length p),
                also stored in self.beta_pseudo_true.
        """
        rng = np.random.default_rng(seed)

        # Build a "large-n" model with the same hyperparameters,
        # but many more groups G_large.
        large_model = HuberRandomIntercept(
            G=G_large,
            n_per_group=self.n_per_group,
            p=self.p,
            beta_true=self.beta_true,
            tau=self.tau,
            sigma=self.sigma,
            c=self.c,
            eta=self.eta,
            lam=self.lam,
            mu_prior=self.mu_prior,
        )

        # Storage for beta_hat in each replication
        beta_hats = np.zeros((large_replication, self.p))

        for r in range(large_replication):
            # Each replication uses a fresh seed derived from the base RNG
            seed_r = int(rng.integers(0, 2**31 - 1))
            # Generate one large dataset
            y_large, X_large = large_model.generate_data(
                seed=seed_r,
                outlier_prob=outlier_prob,
                outlier_scale=outlier_scale,
            )
            # Fit penalized EE estimator on the large dataset
            beta_hat_r, _, _ = large_model.fit_frequentist_penalized_ee(
                y_deque=y_large,
                X_deque=X_large,
                alpha=0.05,  # CI not used here; only point estimate
            )
            beta_hats[r, :] = beta_hat_r

            if verbose and (r + 1) % max(1, large_replication // 10) == 0:
                print(f"  replication {r + 1}/{large_replication} done")

        # Monte Carlo approximation to beta_lambda^*
        beta_pseudo = beta_hats.mean(axis=0)
        # MC standard error of this approximation
        beta_mc_se = beta_hats.std(axis=0, ddof=1) / np.sqrt(large_replication)

        if verbose:
            print("\nApproximate pseudo-true value (beta_lambda^*):")
            for j in range(self.p):
                print(f"  component {j}: {beta_pseudo[j]:.6f} " f"(MC SE ≈ {beta_mc_se[j]:.6f})")
            print("=" * 80)

        # Store and return
        self.beta_pseudo_true = beta_pseudo
        return beta_pseudo

    def compute_Sigma_i(self) -> NDArray:
        """Compute the covariance matrix Sigma_i = tau^2 * 1_n * 1_n^T + sigma^2 * I_n.

        Returns:
            NDArray: The covariance matrix of group i.
        """
        n = self.n_per_group
        Sigma_i = self.tau**2 * np.ones((n, n)) + self.sigma**2 * np.eye(n)
        return Sigma_i

    def compute_L_i(self) -> NDArray:
        """Compute the symmetric square root L_i of Sigma_i.

        Returns:
            NDArray: The matrix L_i such that L_i @ L_i.T = Sigma_i.
        """
        Sigma_i = self.compute_Sigma_i()
        eigvals, eigvecs = np.linalg.eigh(Sigma_i)
        eigvals = np.maximum(eigvals, 1e-16)  # ensure positive definite
        L_i = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        return L_i

    def whiten_data(self, y_deque: Deque, X_deque: Deque) -> Tuple[Deque, Deque]:
        """Apply whitening transformation to data.

        Args:
            y_deque (Deque): Response vectors.
            X_deque (Deque): Design matrices.

        Returns:
            Tuple[Deque, Deque]: Whitened response vectors and design matrices.
        """
        L_i = self.compute_L_i()
        L_i_inv = np.linalg.inv(L_i)

        y_tilde_deque: Deque = deque()
        X_tilde_deque: Deque = deque()

        for i in range(self.G):
            y_tilde_i = L_i_inv @ y_deque[i]
            X_tilde_i = L_i_inv @ X_deque[i]
            y_tilde_deque.append(y_tilde_i)
            X_tilde_deque.append(X_tilde_i)

        return y_tilde_deque, X_tilde_deque

    def huber_rho(self, u: NDArray) -> NDArray:
        """Huber loss function.

        Args:
            u (NDArray): Input values.

        Returns:
            NDArray: Huber loss values.
        """
        abs_u = np.abs(u)
        quad = 0.5 * u**2
        lin = self.c * (abs_u - 0.5 * self.c)
        return np.where(abs_u <= self.c, quad, lin)

    def huber_psi(self, u: NDArray) -> NDArray:
        """Huber score function (derivative of rho).

        Args:
            u (NDArray): Input values.

        Returns:
            NDArray: Huber score values.
        """
        abs_u = np.abs(u)
        return np.where(abs_u <= self.c, u, self.c * np.sign(u))

    def huber_psi_prime(self, u: NDArray) -> NDArray:
        """Derivative of Huber score function.

        Args:
            u (NDArray): Input values.

        Returns:
            NDArray: Derivative values (indicator that |u| <= c).
        """
        abs_u = np.abs(u)
        return (abs_u <= self.c).astype(float)

    def gibbs_sampler(
        self,
        y_deque: Deque,
        X_deque: Deque,
        n_iter: int = 2000,
        n_burn_in: int = 1000,
        seed: Optional[int] = None,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Run Gibbs sampler for the Huber LMM.

        Args:
            y_deque (Deque): Response vectors.
            X_deque (Deque): Design matrices.
            n_iter (int, optional): Number of MCMC iterations. Defaults to 2000.
            n_burn_in (int, optional): Number of burn-in iterations. Defaults to 1000.
            seed (Optional[int], optional): Random seed. Defaults to None.

        Returns:
            Tuple[NDArray, NDArray, NDArray]: Samples of beta, t, and s after burn-in.
        """
        if seed is not None:
            np.random.seed(seed)

        # Whiten the data
        y_tilde_deque, X_tilde_deque = self.whiten_data(y_deque, X_deque)

        # Initialize
        beta = np.zeros(self.p)
        t = np.zeros((self.G, self.n_per_group))
        s = np.ones((self.G, self.n_per_group))

        # Storage for samples
        beta_samples = np.zeros((n_iter - n_burn_in, self.p))
        t_samples = np.zeros((n_iter - n_burn_in, self.G, self.n_per_group))
        s_samples = np.zeros((n_iter - n_burn_in, self.G, self.n_per_group))

        # Precompute lambda_n * s_n for efficiency
        lam_n_s_n = self.eta * self.lam * self.s_n

        for iter_idx in range(n_iter):
            # Step 1: Sample s_ij | t_ij
            for i in range(self.G):
                for j in range(self.n_per_group):
                    # s_ij ~ IG(mu = |t_ij| / (eta * c), lambda = 1)
                    # Using scipy's invgauss parameterization: mu, scale
                    abs_t = np.abs(t[i, j])
                    if abs_t < 1e-10:
                        s[i, j] = 1.0  # Default value for near-zero t
                    else:
                        mu_s = abs_t / (self.eta * self.c)
                        s[i, j] = invgauss.rvs(mu=mu_s, scale=1.0)

            # Step 2: Sample t_ij | beta, s_ij, data
            for i in range(self.G):
                y_tilde_i = y_tilde_deque[i]
                X_tilde_i = X_tilde_deque[i]
                r_tilde_i = y_tilde_i - X_tilde_i @ beta

                for j in range(self.n_per_group):
                    # t_ij | rest ~ N(mu_t_ij, sigma_t_ij^2)
                    sigma_t_sq = 1.0 / (self.eta + 1.0 / s[i, j])
                    mu_t = sigma_t_sq * self.eta * r_tilde_i[j]
                    t[i, j] = np.random.normal(mu_t, np.sqrt(sigma_t_sq))

            # Step 3: Sample beta | t, s, data
            # Posterior precision: Lambda_post = eta * sum_i X_tilde_i^T X_tilde_i + lam_n * s_n * Q
            Lambda_post = np.zeros((self.p, self.p))
            for i in range(self.G):
                X_tilde_i = X_tilde_deque[i]
                Lambda_post += self.eta * X_tilde_i.T @ X_tilde_i
            Lambda_post += lam_n_s_n * self.Q

            # Posterior mean term
            m_post_term = np.zeros(self.p)
            for i in range(self.G):
                X_tilde_i = X_tilde_deque[i]
                y_tilde_i = y_tilde_deque[i]
                t_i = t[i, :]
                m_post_term += self.eta * X_tilde_i.T @ (y_tilde_i - t_i)
            m_post_term += lam_n_s_n * self.Q @ self.mu_prior

            # Sample beta
            Lambda_post_inv = np.linalg.inv(Lambda_post)
            m_post = Lambda_post_inv @ m_post_term

            if self.p == 1:
                beta = np.random.normal(m_post[0], np.sqrt(Lambda_post_inv[0, 0]), size=1)
            else:
                beta = np.random.multivariate_normal(m_post, Lambda_post_inv)

            # Store samples after burn-in
            if iter_idx >= n_burn_in:
                idx = iter_idx - n_burn_in
                beta_samples[idx, :] = beta
                t_samples[idx, :, :] = t
                s_samples[idx, :, :] = s

        return beta_samples, t_samples, s_samples

    def compute_U_n(self, beta: NDArray, y_deque: Deque, X_deque: Deque) -> NDArray:
        """Compute the rescaled score U_n(beta).

        Args:
            beta (NDArray): Fixed effects.
            y_deque (Deque): Response vectors.
            X_deque (Deque): Design matrices.

        Returns:
            NDArray: The U_n vector.
        """
        y_tilde_deque, X_tilde_deque = self.whiten_data(y_deque, X_deque)

        U_n = np.zeros(self.p)
        for i in range(self.G):
            y_tilde_i = y_tilde_deque[i]
            X_tilde_i = X_tilde_deque[i]
            r_tilde_i = y_tilde_i - X_tilde_i @ beta
            psi_r = self.huber_psi(r_tilde_i)
            U_n += -X_tilde_i.T @ psi_r

        U_n /= self.s_n
        return U_n

    def _compute_J_n_with_data(
        self,
        beta: NDArray,
        X_deque: Deque,
        y_deque: Optional[Deque] = None,
    ) -> NDArray:
        """Compute J_n with actual data for residuals.

        Args:
            beta (NDArray): Fixed effects.
            X_deque (Deque): Design matrices.
            y_deque (Optional[Deque], optional): Response vectors. If None, uses zeros.

        Returns:
            NDArray: The J_n matrix.
        """
        if y_deque is None:
            y_deque = deque([np.zeros(self.n_per_group) for _ in range(self.G)])

        y_tilde_deque, X_tilde_deque = self.whiten_data(y_deque, X_deque)

        J_n = np.zeros((self.p, self.p))
        for i in range(self.G):
            y_tilde_i = y_tilde_deque[i]
            X_tilde_i = X_tilde_deque[i]
            r_tilde_i = y_tilde_i - X_tilde_i @ beta

            # W_i(beta) = diag(psi_c'(r_tilde_ij))
            W_i = np.diag(self.huber_psi_prime(r_tilde_i))
            J_n += X_tilde_i.T @ W_i @ X_tilde_i

        J_n /= self.s_n
        return J_n

    def fit_frequentist_penalized_ee(
        self,
        y_deque: Deque,
        X_deque: Deque,
        alpha: float = 0.05,
        maxiter: int = 1000,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """Fit the model using penalized estimating equations.

        Args:
            y_deque (Deque): Response vectors.
            X_deque (Deque): Design matrices.
            alpha (float, optional): Significance level. Defaults to 0.05.
            maxiter (int, optional): Maximum iterations for optimization. Defaults to 1000.

        Returns:
            Tuple[NDArray, NDArray, NDArray]:
                Estimated beta, lower CI bound, upper CI bound.
        """

        # Objective function: M_n(beta) + lambda * s_n * rho(beta)
        def objective(beta: NDArray) -> float:
            y_tilde_deque, X_tilde_deque = self.whiten_data(y_deque, X_deque)

            M_n = 0.0
            for i in range(self.G):
                y_tilde_i = y_tilde_deque[i]
                X_tilde_i = X_tilde_deque[i]
                r_tilde_i = y_tilde_i - X_tilde_i @ beta
                M_n += np.sum(self.huber_rho(r_tilde_i))

            # Ridge penalty
            penalty = 0.5 * (beta - self.mu_prior).T @ self.Q @ (beta - self.mu_prior)
            # Convert to scalar float (penalty may be a 1x1 array when p=1)
            penalty_val: float = float(np.asarray(penalty).item())

            return M_n + self.lam * self.s_n * penalty_val

        # Gradient
        def gradient(beta: NDArray) -> NDArray:
            U_n = self.compute_U_n(beta, y_deque, X_deque)
            penalty_grad = self.Q @ (beta - self.mu_prior)
            return self.s_n * U_n + self.lam * self.s_n * penalty_grad

        # Optimize
        result = minimize(
            fun=objective, x0=np.zeros(self.p), jac=gradient, method="L-BFGS-B", options={"maxiter": maxiter}
        )

        if not result.success:
            logging.warning(f"Optimization did not converge: {result.message}")

        beta_hat = result.x

        # Compute sandwich variance
        J_n = self._compute_J_n_with_data(beta_hat, X_deque, y_deque)
        J_lambda = J_n + self.lam * self.Q

        # Compute K_n
        y_tilde_deque, X_tilde_deque = self.whiten_data(y_deque, X_deque)
        U_i_list = []
        for i in range(self.G):
            y_tilde_i = y_tilde_deque[i]
            X_tilde_i = X_tilde_deque[i]
            r_tilde_i = y_tilde_i - X_tilde_i @ beta_hat
            psi_r = self.huber_psi(r_tilde_i)
            U_i = -X_tilde_i.T @ psi_r
            U_i_list.append(U_i)

        U_array = np.array(U_i_list)  # Shape: (G, p)
        K_n = (U_array.T @ U_array) / self.s_n

        # Sandwich variance: V_target = J_lambda^{-1} K_n J_lambda^{-T}
        J_lambda_inv = np.linalg.inv(J_lambda)
        V_target = J_lambda_inv @ K_n @ J_lambda_inv.T

        # Asymptotic variance of sqrt(s_n) * (beta_hat - beta_true)
        # is V_target, so variance of beta_hat is V_target / s_n
        var_beta_hat = V_target / self.s_n

        # Compute confidence intervals
        from scipy.stats import norm

        z_alpha = norm.ppf(1 - alpha / 2)

        if self.p == 1:
            se = np.sqrt(var_beta_hat[0, 0])
            ci_lower = beta_hat - z_alpha * se
            ci_upper = beta_hat + z_alpha * se
        else:
            se = np.sqrt(np.diag(var_beta_hat))
            ci_lower = beta_hat - z_alpha * se
            ci_upper = beta_hat + z_alpha * se

        return beta_hat, ci_lower, ci_upper

    def compute_one_step_center(self, beta_GB: NDArray, y_deque: Deque, X_deque: Deque) -> NDArray:
        """Compute the one-step center (admissible center).

        Args:
            beta_GB (NDArray): Posterior mean of the fixed effects.
            y_deque (Deque): Response vectors.
            X_deque (Deque): Design matrices.

        Returns:
            NDArray: The one-step center.
        """
        U_n = self.compute_U_n(beta_GB, y_deque, X_deque)
        J_n = self._compute_J_n_with_data(beta_GB, X_deque, y_deque)

        # One-step update with penalty
        J_lambda = J_n + self.lam * self.Q
        penalty_term = self.lam * self.Q @ (beta_GB - self.mu_prior)

        beta_dagger = beta_GB - np.linalg.solve(J_lambda, U_n + penalty_term)
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
            y_deque (Deque): Response vectors.
            X_deque (Deque): Design matrices.

        Returns:
            Tuple[NDArray, NDArray]: Calibration matrix Omega and target variance.
        """
        # Compute posterior covariance
        if self.p == 1:
            Sigma_post = np.var(samples, axis=0, ddof=1).reshape(1, 1)
        else:
            Sigma_post = np.cov(samples.T)
        H_0_inv = self.s_n * Sigma_post

        # Compute J_hat
        J_hat = self._compute_J_n_with_data(beta_GB, X_deque, y_deque)
        J_lambda_hat = J_hat + self.lam * self.Q

        # Compute K_hat
        y_tilde_deque, X_tilde_deque = self.whiten_data(y_deque, X_deque)

        U_i_list = []
        for i in range(self.G):
            y_tilde_i = y_tilde_deque[i]
            X_tilde_i = X_tilde_deque[i]
            r_tilde_i = y_tilde_i - X_tilde_i @ beta_dagger
            psi_r = self.huber_psi(r_tilde_i)
            U_i = -X_tilde_i.T @ psi_r
            U_i_list.append(U_i)

        U_array = np.array(U_i_list)
        K_hat = (U_array.T @ U_array) / self.s_n

        # Compute target variance
        J_lambda_inv = np.linalg.inv(J_lambda_hat)
        V_target_hat = J_lambda_inv @ K_hat @ J_lambda_inv.T

        # Compute calibration matrix Omega = V_target^{1/2} H_0^{1/2}
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
