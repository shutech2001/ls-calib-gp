from __future__ import annotations

from collections import deque
import logging
from typing import Tuple, Deque, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar  # type: ignore
from scipy.stats import invgauss  # type: ignore


class HuberRandomInterceptLaplace:
    """
    Random Intercept Linear Mixed Model (LMM) with Huber loss and Laplace prior.

    The Laplace prior corresponds to an L1 penalty, implemented via a
    normal-exponential scale-mixture representation.
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

    def generate_data(
        self,
        seed: Optional[int] = None,
        outlier_prob: float = 0.0,
        outlier_scale: float = 10.0,
    ) -> Tuple[Deque, Deque]:
        if seed is not None:
            np.random.seed(seed)

        y_deque: Deque = deque()
        X_deque: Deque = deque()

        for _ in range(self.G):
            X_i = np.random.randn(self.n_per_group, self.p)
            b_i = np.random.randn() * self.tau

            epsilon_i = np.random.randn(self.n_per_group) * self.sigma
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
        rng = np.random.default_rng(seed)

        large_model = HuberRandomInterceptLaplace(
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

        beta_hats = np.zeros((large_replication, self.p))

        for r in range(large_replication):
            seed_r = int(rng.integers(0, 2**31 - 1))
            y_large, X_large = large_model.generate_data(
                seed=seed_r,
                outlier_prob=outlier_prob,
                outlier_scale=outlier_scale,
            )
            beta_hat_r, _, _ = large_model.fit_frequentist_penalized_ee(
                y_deque=y_large,
                X_deque=X_large,
                alpha=0.05,
            )
            beta_hats[r, :] = beta_hat_r

            if verbose and (r + 1) % max(1, large_replication // 10) == 0:
                print(f"  replication {r + 1}/{large_replication} done")

        beta_pseudo = beta_hats.mean(axis=0)
        beta_mc_se = beta_hats.std(axis=0, ddof=1) / np.sqrt(large_replication)

        if verbose:
            print("\nApproximate pseudo-true value (beta_lambda^*):")
            for j in range(self.p):
                print(f"  component {j}: {beta_pseudo[j]:.6f} " f"(MC SE approx {beta_mc_se[j]:.6f})")
            print("=" * 80)

        self.beta_pseudo_true = beta_pseudo
        return beta_pseudo

    def compute_Sigma_i(self) -> NDArray:
        n = self.n_per_group
        Sigma_i = self.tau**2 * np.ones((n, n)) + self.sigma**2 * np.eye(n)
        return Sigma_i

    def compute_L_i(self) -> NDArray:
        Sigma_i = self.compute_Sigma_i()
        eigvals, eigvecs = np.linalg.eigh(Sigma_i)
        eigvals = np.maximum(eigvals, 1e-16)
        L_i = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        return L_i

    def whiten_data(self, y_deque: Deque, X_deque: Deque) -> Tuple[Deque, Deque]:
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
        abs_u = np.abs(u)
        quad = 0.5 * u**2
        lin = self.c * (abs_u - 0.5 * self.c)
        return np.where(abs_u <= self.c, quad, lin)

    def huber_psi(self, u: NDArray) -> NDArray:
        abs_u = np.abs(u)
        return np.where(abs_u <= self.c, u, self.c * np.sign(u))

    def huber_psi_prime(self, u: NDArray) -> NDArray:
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
        if seed is not None:
            np.random.seed(seed)

        y_tilde_deque, X_tilde_deque = self.whiten_data(y_deque, X_deque)

        beta = np.zeros(self.p)
        t = np.zeros((self.G, self.n_per_group))
        s = np.ones((self.G, self.n_per_group))
        tau_beta = np.ones(self.p)

        beta_samples = np.zeros((n_iter - n_burn_in, self.p))
        t_samples = np.zeros((n_iter - n_burn_in, self.G, self.n_per_group))
        s_samples = np.zeros((n_iter - n_burn_in, self.G, self.n_per_group))

        kappa = self.eta * self.lam * self.s_n

        for iter_idx in range(n_iter):
            # Step 1: Sample s_ij | t_ij
            for i in range(self.G):
                for j in range(self.n_per_group):
                    abs_t = np.abs(t[i, j])
                    if abs_t < 1e-10:
                        s[i, j] = 1.0
                    else:
                        mu_s = abs_t / (self.eta * self.c)
                        s[i, j] = invgauss.rvs(mu=mu_s, scale=1.0)

            # Step 2: Sample t_ij | beta, s_ij, data
            for i in range(self.G):
                y_tilde_i = y_tilde_deque[i]
                X_tilde_i = X_tilde_deque[i]
                r_tilde_i = y_tilde_i - X_tilde_i @ beta

                for j in range(self.n_per_group):
                    sigma_t_sq = 1.0 / (self.eta + 1.0 / s[i, j])
                    mu_t = sigma_t_sq * self.eta * r_tilde_i[j]
                    t[i, j] = np.random.normal(mu_t, np.sqrt(sigma_t_sq))

            # Step 3: Sample tau_beta | beta (Laplace prior scale mixture)
            if self.lam > 0:
                for j in range(self.p):
                    diff = np.abs(beta[j] - self.mu_prior[j])
                    if diff < 1e-12:
                        tau_beta[j] = 1.0
                    else:
                        u = np.random.wald(mean=kappa / diff, scale=kappa**2)
                        tau_beta[j] = 1.0 / u
            else:
                tau_beta[:] = 1e6

            # Step 4: Sample beta | t, s, tau_beta
            Lambda_post = np.zeros((self.p, self.p))
            for i in range(self.G):
                X_tilde_i = X_tilde_deque[i]
                Lambda_post += self.eta * X_tilde_i.T @ X_tilde_i
            Lambda_post += np.diag(1.0 / tau_beta)

            m_post_term = np.zeros(self.p)
            for i in range(self.G):
                X_tilde_i = X_tilde_deque[i]
                y_tilde_i = y_tilde_deque[i]
                t_i = t[i, :]
                m_post_term += self.eta * X_tilde_i.T @ (y_tilde_i - t_i)
            m_post_term += (1.0 / tau_beta) * self.mu_prior

            Lambda_post_inv = np.linalg.inv(Lambda_post)
            m_post = Lambda_post_inv @ m_post_term

            if self.p == 1:
                beta = np.random.normal(m_post[0], np.sqrt(Lambda_post_inv[0, 0]), size=1)
            else:
                beta = np.random.multivariate_normal(m_post, Lambda_post_inv)

            if iter_idx >= n_burn_in:
                idx = iter_idx - n_burn_in
                beta_samples[idx, :] = beta
                t_samples[idx, :, :] = t
                s_samples[idx, :, :] = s

        return beta_samples, t_samples, s_samples

    def compute_U_n(self, beta: NDArray, y_deque: Deque, X_deque: Deque) -> NDArray:
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
        if y_deque is None:
            y_deque = deque([np.zeros(self.n_per_group) for _ in range(self.G)])

        y_tilde_deque, X_tilde_deque = self.whiten_data(y_deque, X_deque)

        J_n = np.zeros((self.p, self.p))
        for i in range(self.G):
            y_tilde_i = y_tilde_deque[i]
            X_tilde_i = X_tilde_deque[i]
            r_tilde_i = y_tilde_i - X_tilde_i @ beta
            W_i = np.diag(self.huber_psi_prime(r_tilde_i))
            J_n += X_tilde_i.T @ W_i @ X_tilde_i

        J_n /= self.s_n
        return J_n

    def _soft_threshold(self, z: NDArray, threshold: float) -> NDArray:
        return np.sign(z) * np.maximum(np.abs(z) - threshold, 0.0)

    def fit_frequentist_penalized_ee(
        self,
        y_deque: Deque,
        X_deque: Deque,
        alpha: float = 0.05,
        maxiter: int = 1000,
        tol: float = 1e-6,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        y_tilde_deque, X_tilde_deque = self.whiten_data(y_deque, X_deque)

        gram = np.zeros((self.p, self.p))
        for i in range(self.G):
            X_tilde_i = X_tilde_deque[i]
            gram += X_tilde_i.T @ X_tilde_i
        eigvals = np.linalg.eigvalsh(gram)
        lipschitz = float(np.max(eigvals))
        lipschitz = max(lipschitz, 1e-8)
        step = 1.0 / lipschitz

        beta = np.zeros(self.p)
        for _ in range(maxiter):
            U_n = self.compute_U_n(beta, y_deque, X_deque)
            grad = self.s_n * U_n
            z = beta - step * grad
            beta_new = self.mu_prior + self._soft_threshold(z - self.mu_prior, step * self.lam * self.s_n)
            if np.linalg.norm(beta_new - beta) < tol:
                beta = beta_new
                break
            beta = beta_new

        # If p == 1, refine with a 1D derivative-free search for stability.
        if self.p == 1:
            def objective_1d(x: float) -> float:
                y_tilde_d, X_tilde_d = y_tilde_deque, X_tilde_deque
                M_n = 0.0
                for i in range(self.G):
                    y_tilde_i = y_tilde_d[i]
                    X_tilde_i = X_tilde_d[i]
                    r_tilde_i = y_tilde_i - X_tilde_i @ np.array([x])
                    M_n += np.sum(self.huber_rho(r_tilde_i))
                penalty = self.lam * self.s_n * np.abs(x - float(self.mu_prior[0]))
                return M_n + penalty

            res = minimize_scalar(objective_1d, method="brent")
            if res.success:
                beta = np.array([res.x])
            else:
                logging.warning("Scalar optimization did not converge; using proximal solution.")

        beta_hat = beta

        J_n = self._compute_J_n_with_data(beta_hat, X_deque, y_deque)
        J_lambda = J_n

        y_tilde_deque, X_tilde_deque = self.whiten_data(y_deque, X_deque)
        U_i_list = []
        for i in range(self.G):
            y_tilde_i = y_tilde_deque[i]
            X_tilde_i = X_tilde_deque[i]
            r_tilde_i = y_tilde_i - X_tilde_i @ beta_hat
            psi_r = self.huber_psi(r_tilde_i)
            U_i = -X_tilde_i.T @ psi_r
            U_i_list.append(U_i)

        U_array = np.array(U_i_list)
        K_n = (U_array.T @ U_array) / self.s_n

        J_lambda_inv = np.linalg.inv(J_lambda)
        V_target = J_lambda_inv @ K_n @ J_lambda_inv.T

        var_beta_hat = V_target / self.s_n

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
        U_n = self.compute_U_n(beta_GB, y_deque, X_deque)
        J_n = self._compute_J_n_with_data(beta_GB, X_deque, y_deque)

        diff = beta_GB - self.mu_prior
        subgrad = np.sign(diff)
        subgrad[np.abs(diff) < 1e-12] = 0.0

        penalty_term = self.lam * subgrad

        beta_dagger = beta_GB - np.linalg.solve(J_n, U_n + penalty_term)
        return beta_dagger

    def compute_calibration_matrix(
        self,
        beta_GB: NDArray,
        beta_dagger: NDArray,
        samples: NDArray,
        y_deque: Deque,
        X_deque: Deque,
    ) -> Tuple[NDArray, NDArray]:
        if self.p == 1:
            Sigma_post = np.var(samples, axis=0, ddof=1).reshape(1, 1)
        else:
            Sigma_post = np.cov(samples.T)
        H_0_inv = self.s_n * Sigma_post

        J_hat = self._compute_J_n_with_data(beta_GB, X_deque, y_deque)
        J_lambda_hat = J_hat

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

        J_lambda_inv = np.linalg.inv(J_lambda_hat)
        V_target_hat = J_lambda_inv @ K_hat @ J_lambda_inv.T

        H_0 = np.linalg.inv(H_0_inv)
        V_target_sqrt = self._matrix_sqrt(V_target_hat)
        H_0_sqrt = self._matrix_sqrt(H_0)
        Omega = V_target_sqrt @ H_0_sqrt

        return Omega, V_target_hat

    def _matrix_sqrt(self, A: NDArray) -> NDArray:
        eigvals, eigvecs = np.linalg.eigh(A)
        eigvals = np.maximum(eigvals, 1e-16)
        return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    def calibrate_samples(
        self,
        samples: NDArray,
        beta_GB: NDArray,
        beta_dagger: NDArray,
        Omega: NDArray,
    ) -> NDArray:
        M = len(samples)
        calibrated_samples = np.zeros_like(samples)

        if self.p == 1:
            for m in range(M):
                calibrated_samples[m] = beta_dagger + Omega[0, 0] * (samples[m] - beta_GB)
        else:
            for m in range(M):
                calibrated_samples[m] = beta_dagger + Omega @ (samples[m] - beta_GB)

        return calibrated_samples
