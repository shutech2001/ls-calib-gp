from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import chi2, norm


Array = NDArray[np.float64]


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    G: int
    n_per_group: int
    p: int
    beta_true: tuple[float, ...]
    tau: float
    sigma: float
    c: float
    lam: float
    mu: tuple[float, ...]
    q_matrix: tuple[tuple[float, ...], ...]
    outlier_prob: float
    outlier_scale: float
    x_corr: float
    alpha: float = 0.05

    @property
    def n(self) -> int:
        return self.G * self.n_per_group

    @property
    def beta_true_array(self) -> Array:
        return np.asarray(self.beta_true, dtype=float)

    @property
    def mu_array(self) -> Array:
        return np.asarray(self.mu, dtype=float)

    @property
    def q_array(self) -> Array:
        return np.asarray(self.q_matrix, dtype=float)


@dataclass(frozen=True)
class RunConfig:
    n_sim: int
    etas: tuple[float, ...]
    n_iter: int
    n_burn_in: int
    thin: int
    proposal_scale: float
    pseudo_groups: int
    pseudo_replications: int
    nominal_levels: tuple[float, ...]
    jobs: int
    seed: int
    output_root: str


def _huber_rho(u: Array, c: float) -> Array:
    abs_u = np.abs(u)
    return np.where(abs_u <= c, 0.5 * u * u, c * (abs_u - 0.5 * c))


def _huber_psi(u: Array, c: float) -> Array:
    return np.clip(u, -c, c)


def _huber_weight(u: Array, c: float) -> Array:
    return (np.abs(u) < c).astype(float)


def _matrix_sqrt(a: Array) -> Array:
    vals, vecs = np.linalg.eigh(0.5 * (a + a.T))
    vals = np.maximum(vals, 1e-12)
    return vecs @ np.diag(np.sqrt(vals)) @ vecs.T


def _matrix_inv_safely(a: Array) -> Array:
    jitter = 1e-10
    eye = np.eye(a.shape[0])
    for _ in range(6):
        try:
            return np.linalg.inv(a + jitter * eye)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    return np.linalg.pinv(a)


def _covariance_ar1(p: int, rho: float) -> Array:
    idx = np.arange(p)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _ridge_matrix(p: int, seed: int, non_diagonal: bool) -> Array:
    if not non_diagonal or p == 1:
        return np.eye(p)
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(p, p))
    orth, _ = np.linalg.qr(raw)
    eig = np.linspace(0.4, 1.8, p)
    return orth @ np.diag(eig) @ orth.T


def _as_tuple_matrix(a: Array) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(x) for x in row) for row in a)


def scalar_scenario() -> ScenarioConfig:
    p = 1
    return ScenarioConfig(
        name="scalar_lmm",
        G=100,
        n_per_group=5,
        p=p,
        beta_true=(2.0,),
        tau=1.0,
        sigma=1.0,
        c=1.0,
        lam=0.5,
        mu=(0.0,),
        q_matrix=_as_tuple_matrix(np.eye(p)),
        outlier_prob=0.1,
        outlier_scale=10.0,
        x_corr=0.0,
    )


def multivariate_scenario(p: int = 5) -> ScenarioConfig:
    beta = np.linspace(1.0, 0.25, p)
    q = _ridge_matrix(p, seed=20260511, non_diagonal=True)
    return ScenarioConfig(
        name=f"multivariate_p{p}_lmm",
        G=100,
        n_per_group=5,
        p=p,
        beta_true=tuple(float(x) for x in beta),
        tau=1.0,
        sigma=1.0,
        c=1.0,
        lam=0.5,
        mu=tuple(0.0 for _ in range(p)),
        q_matrix=_as_tuple_matrix(q),
        outlier_prob=0.1,
        outlier_scale=10.0,
        x_corr=0.55,
    )


def location_scenario() -> ScenarioConfig:
    base = scalar_scenario()
    return replace(
        base,
        name="location_contribution_lmm",
        G=40,
        lam=1.5,
        outlier_prob=0.2,
        outlier_scale=12.0,
    )


def generate_data(cfg: ScenarioConfig, seed: int) -> tuple[Array, Array]:
    rng = np.random.default_rng(seed)
    cov_x = _covariance_ar1(cfg.p, cfg.x_corr)
    x = rng.multivariate_normal(np.zeros(cfg.p), cov_x, size=(cfg.G, cfg.n_per_group))
    b = rng.normal(0.0, cfg.tau, size=cfg.G)
    eps = rng.normal(0.0, cfg.sigma, size=(cfg.G, cfg.n_per_group))
    if cfg.outlier_prob > 0:
        mask = rng.uniform(size=(cfg.G, cfg.n_per_group)) < cfg.outlier_prob
        eps[mask] += rng.normal(0.0, cfg.outlier_scale, size=int(mask.sum()))
    y = np.einsum("gmp,p->gm", x, cfg.beta_true_array) + b[:, None] + eps
    return y.astype(float), x.astype(float)


def whiten_data(cfg: ScenarioConfig, y: Array, x: Array) -> tuple[Array, Array]:
    m = cfg.n_per_group
    sigma_i = cfg.tau**2 * np.ones((m, m)) + cfg.sigma**2 * np.eye(m)
    vals, vecs = np.linalg.eigh(sigma_i)
    l_inv = vecs @ np.diag(1.0 / np.sqrt(np.maximum(vals, 1e-14))) @ vecs.T
    y_tilde = np.einsum("ab,gb->ga", l_inv, y)
    x_tilde = np.einsum("ab,gbp->gap", l_inv, x)
    return y_tilde, x_tilde


def neglog_loss(cfg: ScenarioConfig, beta: Array, y_tilde: Array, x_tilde: Array) -> float:
    r = y_tilde - np.einsum("gmp,p->gm", x_tilde, beta)
    diff = beta - cfg.mu_array
    penalty = 0.5 * cfg.lam * cfg.n * float(diff.T @ cfg.q_array @ diff)
    return float(np.sum(_huber_rho(r, cfg.c)) + penalty)


def grad_neglog(cfg: ScenarioConfig, beta: Array, y_tilde: Array, x_tilde: Array) -> Array:
    r = y_tilde - np.einsum("gmp,p->gm", x_tilde, beta)
    score_loss = -np.einsum("gmp,gm->p", x_tilde, _huber_psi(r, cfg.c))
    penalty = cfg.lam * cfg.n * (cfg.q_array @ (beta - cfg.mu_array))
    return score_loss + penalty


def scaled_hessian_loss(cfg: ScenarioConfig, beta: Array, y_tilde: Array, x_tilde: Array) -> Array:
    r = y_tilde - np.einsum("gmp,p->gm", x_tilde, beta)
    w = _huber_weight(r, cfg.c)
    return np.einsum("gmp,gm,gmq->pq", x_tilde, w, x_tilde) / cfg.n


def fit_penalized_huber(cfg: ScenarioConfig, y_tilde: Array, x_tilde: Array, start: Array | None = None) -> Array:
    x0 = np.zeros(cfg.p) if start is None else np.asarray(start, dtype=float)
    result = minimize(
        fun=lambda b: neglog_loss(cfg, np.asarray(b), y_tilde, x_tilde),
        x0=x0,
        jac=lambda b: grad_neglog(cfg, np.asarray(b), y_tilde, x_tilde),
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-10, "gtol": 1e-8},
    )
    if not result.success:
        result = minimize(
            fun=lambda b: neglog_loss(cfg, np.asarray(b), y_tilde, x_tilde),
            x0=result.x,
            method="BFGS",
            options={"maxiter": 1000, "gtol": 1e-7},
        )
    return np.asarray(result.x, dtype=float)


def one_step_center(cfg: ScenarioConfig, beta_start: Array, y_tilde: Array, x_tilde: Array) -> Array:
    r = y_tilde - np.einsum("gmp,p->gm", x_tilde, beta_start)
    u_n = -np.einsum("gmp,gm->p", x_tilde, _huber_psi(r, cfg.c)) / cfg.n
    j_lambda = scaled_hessian_loss(cfg, beta_start, y_tilde, x_tilde) + cfg.lam * cfg.q_array
    f_n = u_n + cfg.lam * (cfg.q_array @ (beta_start - cfg.mu_array))
    return beta_start - np.linalg.solve(j_lambda + 1e-10 * np.eye(cfg.p), f_n)


def sandwich_components(cfg: ScenarioConfig, beta: Array, y_tilde: Array, x_tilde: Array) -> tuple[Array, Array, Array]:
    j_lambda = scaled_hessian_loss(cfg, beta, y_tilde, x_tilde) + cfg.lam * cfg.q_array
    r = y_tilde - np.einsum("gmp,p->gm", x_tilde, beta)
    group_scores = -np.einsum("gmp,gm->gp", x_tilde, _huber_psi(r, cfg.c))
    centered = group_scores - np.mean(group_scores, axis=0, keepdims=True)
    k_hat = centered.T @ centered / cfg.n
    j_inv = _matrix_inv_safely(j_lambda)
    v_target = j_inv @ k_hat @ j_inv.T
    return j_lambda, k_hat, 0.5 * (v_target + v_target.T)


def frequentist_summary(cfg: ScenarioConfig, beta_hat: Array, v_target: Array) -> dict[str, Any]:
    cov = v_target / cfg.n
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    z = norm.ppf(1.0 - cfg.alpha / 2.0)
    lower = beta_hat - z * se
    upper = beta_hat + z * se
    chi = chi2.ppf(1.0 - cfg.alpha, cfg.p)
    diff_true = beta_hat - beta_hat
    return {
        "center": beta_hat,
        "lower": lower,
        "upper": upper,
        "cov": cov,
        "v_target": v_target,
        "joint_radius": chi,
        "placeholder": diff_true,
    }


def log_mvn_density(x: Array, mean: Array, chol: Array) -> float:
    z = np.linalg.solve(chol, x - mean)
    logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
    p = mean.size
    return -0.5 * (p * math.log(2.0 * math.pi) + logdet + float(z.T @ z))


def sample_huber_posterior(
    cfg: ScenarioConfig,
    eta: float,
    beta_map: Array,
    j_lambda_map: Array,
    y_tilde: Array,
    x_tilde: Array,
    run_cfg: RunConfig,
    seed: int,
) -> tuple[Array, float]:
    """Sample the exact Huber generalized posterior with an MH-corrected Gaussian block.

    The proposal is the Gaussian block one obtains from the local quadratic
    approximation to the ridge-penalized Huber loss.  The Metropolis correction
    is what makes the invariant density the exact Huber generalized posterior.
    """
    rng = np.random.default_rng(seed)
    precision = eta * cfg.n * j_lambda_map
    cov_prop = run_cfg.proposal_scale**2 * _matrix_inv_safely(precision)
    cov_prop = 0.5 * (cov_prop + cov_prop.T) + 1e-12 * np.eye(cfg.p)
    chol = np.linalg.cholesky(cov_prop)

    def log_target(beta: Array) -> float:
        return -eta * neglog_loss(cfg, beta, y_tilde, x_tilde)

    beta = beta_map.copy()
    logp = log_target(beta)
    logq_beta = log_mvn_density(beta, beta_map, chol)
    draws: list[Array] = []
    accepted = 0
    total = 0
    for iteration in range(run_cfg.n_iter):
        proposal = beta_map + chol @ rng.normal(size=cfg.p)
        logp_prop = log_target(proposal)
        logq_prop = log_mvn_density(proposal, beta_map, chol)
        log_accept = logp_prop - logp + logq_beta - logq_prop
        total += 1
        if math.log(rng.uniform()) < min(0.0, log_accept):
            beta = proposal
            logp = logp_prop
            logq_beta = logq_prop
            accepted += 1
        if iteration >= run_cfg.n_burn_in and (iteration - run_cfg.n_burn_in) % run_cfg.thin == 0:
            draws.append(beta.copy())
    return np.asarray(draws, dtype=float), accepted / max(total, 1)


def interval_from_samples(samples: Array, alpha: float) -> tuple[Array, Array]:
    return (
        np.quantile(samples, alpha / 2.0, axis=0),
        np.quantile(samples, 1.0 - alpha / 2.0, axis=0),
    )


def ellipsoid_volume(cov: Array, alpha: float) -> float:
    p = cov.shape[0]
    radius = chi2.ppf(1.0 - alpha, p)
    unit_ball = math.pi ** (p / 2.0) / math.exp(gammaln(p / 2.0 + 1.0))
    det_cov = max(float(np.linalg.det(0.5 * (cov + cov.T))), 0.0)
    return unit_ball * radius ** (p / 2.0) * math.sqrt(det_cov)


def sample_joint_contains(samples: Array, center: Array, target: Array, nominal: float) -> bool:
    cov = np.cov(samples.T) if samples.shape[1] > 1 else np.asarray([[float(np.var(samples[:, 0], ddof=1))]])
    inv_cov = _matrix_inv_safely(np.atleast_2d(cov))
    diffs = samples - center
    sample_dist = np.einsum("np,pq,nq->n", diffs, inv_cov, diffs)
    cutoff = float(np.quantile(sample_dist, nominal))
    target_diff = target - center
    target_dist = float(target_diff.T @ inv_cov @ target_diff)
    return target_dist <= cutoff


def method_summaries(
    cfg: ScenarioConfig,
    eta: float,
    beta_pseudo: Array,
    samples: Array,
    beta_map: Array,
    beta_one_step: Array,
    v_target_calib: Array,
    v_target_freq: Array,
) -> dict[str, dict[str, Any]]:
    post_mean = np.mean(samples, axis=0)
    post_cov = np.cov(samples.T) if cfg.p > 1 else np.asarray([[float(np.var(samples[:, 0], ddof=1))]])
    h0_inv = cfg.n * np.atleast_2d(post_cov)
    h0 = _matrix_inv_safely(h0_inv)
    omega = _matrix_sqrt(v_target_calib) @ _matrix_sqrt(h0)

    centered = samples - post_mean
    ofs = post_mean + centered @ omega.T
    ls_map = beta_map + centered @ omega.T
    ls_one = beta_one_step + centered @ omega.T

    freq_cov = v_target_freq / cfg.n
    z = norm.ppf(1.0 - cfg.alpha / 2.0)
    freq_se = np.sqrt(np.maximum(np.diag(freq_cov), 0.0))
    freq_lower = beta_map - z * freq_se
    freq_upper = beta_map + z * freq_se

    methods: dict[str, dict[str, Any]] = {
        "frequentist": {"center": beta_map, "lower": freq_lower, "upper": freq_upper, "samples": None, "cov": freq_cov},
        "uncalibrated": {"center": post_mean, "samples": samples, "cov": post_cov},
        "open-faced sandwich": {"center": np.mean(ofs, axis=0), "samples": ofs, "cov": np.cov(ofs.T) if cfg.p > 1 else np.asarray([[float(np.var(ofs[:, 0], ddof=1))]])},
        "location-scale map": {"center": np.mean(ls_map, axis=0), "samples": ls_map, "cov": np.cov(ls_map.T) if cfg.p > 1 else np.asarray([[float(np.var(ls_map[:, 0], ddof=1))]])},
        "location-scale one-step": {"center": np.mean(ls_one, axis=0), "samples": ls_one, "cov": np.cov(ls_one.T) if cfg.p > 1 else np.asarray([[float(np.var(ls_one[:, 0], ddof=1))]])},
    }
    for name, info in methods.items():
        if info["samples"] is not None:
            lower, upper = interval_from_samples(np.asarray(info["samples"]), cfg.alpha)
            info["lower"] = lower
            info["upper"] = upper
        lower = np.asarray(info["lower"])
        upper = np.asarray(info["upper"])
        center = np.asarray(info["center"])
        info["marginal_cover"] = ((lower <= beta_pseudo) & (beta_pseudo <= upper)).astype(float)
        info["width"] = upper - lower
        info["bias"] = center - beta_pseudo
        if name == "frequentist":
            diff = beta_map - beta_pseudo
            info["joint_cover"] = float(
                cfg.n * diff.T @ _matrix_inv_safely(v_target_freq) @ diff <= chi2.ppf(1.0 - cfg.alpha, cfg.p)
            )
        else:
            info["joint_cover"] = float(sample_joint_contains(np.asarray(info["samples"]), center, beta_pseudo, 1.0 - cfg.alpha))
        info["volume"] = ellipsoid_volume(np.asarray(info["cov"]), cfg.alpha)
    return methods


def nominal_coverage(
    cfg: ScenarioConfig,
    beta_pseudo: Array,
    method_info: dict[str, dict[str, Any]],
    nominal_levels: Iterable[float],
) -> dict[str, dict[str, list[float]]]:
    output: dict[str, dict[str, list[float]]] = {}
    for method, info in method_info.items():
        center = np.asarray(info["center"])
        output[method] = {"marginal": [], "joint": []}
        for nominal in nominal_levels:
            alpha = 1.0 - nominal
            if info["samples"] is None:
                cov = np.asarray(info["cov"])
                se = np.sqrt(np.maximum(np.diag(cov), 0.0))
                z = norm.ppf(1.0 - alpha / 2.0)
                lower = center - z * se
                upper = center + z * se
                diff = center - beta_pseudo
                v_target = cov * cfg.n
                joint = float(cfg.n * diff.T @ _matrix_inv_safely(v_target) @ diff <= chi2.ppf(nominal, cfg.p))
            else:
                samples = np.asarray(info["samples"])
                lower, upper = interval_from_samples(samples, alpha)
                joint = float(sample_joint_contains(samples, center, beta_pseudo, nominal))
            output[method]["marginal"].append(float(lower[0] <= beta_pseudo[0] <= upper[0]))
            output[method]["joint"].append(joint)
    return output


def estimate_pseudo_true(cfg: ScenarioConfig, run_cfg: RunConfig) -> tuple[Array, Array]:
    estimates = np.zeros((run_cfg.pseudo_replications, cfg.p))
    for rep in range(run_cfg.pseudo_replications):
        large_cfg = replace(cfg, G=run_cfg.pseudo_groups, name=f"{cfg.name}_pseudo")
        y, x = generate_data(large_cfg, run_cfg.seed + 900_000 + rep * 101)
        y_tilde, x_tilde = whiten_data(large_cfg, y, x)
        estimates[rep] = fit_penalized_huber(large_cfg, y_tilde, x_tilde)
    pseudo = np.mean(estimates, axis=0)
    mc_se = np.std(estimates, axis=0, ddof=1) / math.sqrt(max(run_cfg.pseudo_replications, 1))
    return pseudo, mc_se


def _run_cell(args: tuple[ScenarioConfig, RunConfig, float, int, tuple[float, ...]]) -> dict[str, Any]:
    cfg, run_cfg, eta, sim_idx, beta_pseudo_tuple = args
    beta_pseudo = np.asarray(beta_pseudo_tuple, dtype=float)
    data_seed = run_cfg.seed + sim_idx * 1009
    y, x = generate_data(cfg, data_seed)
    y_tilde, x_tilde = whiten_data(cfg, y, x)
    beta_map = fit_penalized_huber(cfg, y_tilde, x_tilde)
    j_lambda_map, _, v_target_freq = sandwich_components(cfg, beta_map, y_tilde, x_tilde)
    samples, accept_rate = sample_huber_posterior(
        cfg=cfg,
        eta=eta,
        beta_map=beta_map,
        j_lambda_map=j_lambda_map,
        y_tilde=y_tilde,
        x_tilde=x_tilde,
        run_cfg=run_cfg,
        seed=data_seed + int(round(eta * 10_000)) + 17,
    )
    beta_one = one_step_center(cfg, np.mean(samples, axis=0), y_tilde, x_tilde)
    _, _, v_target_calib = sandwich_components(cfg, beta_one, y_tilde, x_tilde)
    summaries = method_summaries(
        cfg,
        eta,
        beta_pseudo,
        samples,
        beta_map,
        beta_one,
        v_target_calib,
        v_target_freq,
    )
    nominal = nominal_coverage(cfg, beta_pseudo, summaries, run_cfg.nominal_levels)
    compact: dict[str, Any] = {"eta": eta, "sim_idx": sim_idx, "accept_rate": accept_rate, "methods": {}, "nominal": nominal}
    for method, info in summaries.items():
        compact["methods"][method] = {
            "marginal_cover": np.asarray(info["marginal_cover"], dtype=float).tolist(),
            "joint_cover": float(info["joint_cover"]),
            "width": np.asarray(info["width"], dtype=float).tolist(),
            "bias": np.asarray(info["bias"], dtype=float).tolist(),
            "volume": float(info["volume"]),
        }
    return compact


def aggregate_cells(cells: list[dict[str, Any]], cfg: ScenarioConfig, run_cfg: RunConfig) -> dict[str, Any]:
    methods = list(cells[0]["methods"].keys())
    etas = np.asarray(run_cfg.etas, dtype=float)
    by_eta: dict[float, list[dict[str, Any]]] = {float(eta): [] for eta in etas}
    for cell in cells:
        by_eta[float(cell["eta"])].append(cell)

    out: dict[str, Any] = {
        "etas": etas.tolist(),
        "methods": methods,
        "accept_rate": [],
        "metrics": {},
        "nominal_levels": list(run_cfg.nominal_levels),
        "nominal": {},
    }
    for method in methods:
        out["metrics"][method] = {
            "coverage": [],
            "joint_coverage": [],
            "width": [],
            "volume": [],
            "bias": [],
            "bias_sd": [],
            "bias_l2": [],
        }
        out["nominal"][method] = {"marginal": [], "joint": []}

    for eta in etas:
        eta_cells = by_eta[float(eta)]
        out["accept_rate"].append(float(np.mean([c["accept_rate"] for c in eta_cells])))
        for method in methods:
            cover = np.asarray([c["methods"][method]["marginal_cover"] for c in eta_cells], dtype=float)
            width = np.asarray([c["methods"][method]["width"] for c in eta_cells], dtype=float)
            bias = np.asarray([c["methods"][method]["bias"] for c in eta_cells], dtype=float)
            joint = np.asarray([c["methods"][method]["joint_cover"] for c in eta_cells], dtype=float)
            volume = np.asarray([c["methods"][method]["volume"] for c in eta_cells], dtype=float)
            out["metrics"][method]["coverage"].append(float(np.mean(cover[:, 0])))
            out["metrics"][method]["joint_coverage"].append(float(np.mean(joint)))
            out["metrics"][method]["width"].append(float(np.mean(width[:, 0]) if cfg.p == 1 else np.mean(width)))
            out["metrics"][method]["volume"].append(float(np.mean(volume)))
            out["metrics"][method]["bias"].append(float(np.mean(bias[:, 0])))
            out["metrics"][method]["bias_sd"].append(float(np.std(bias[:, 0], ddof=1)))
            out["metrics"][method]["bias_l2"].append(float(np.mean(np.linalg.norm(bias, axis=1))))
            marg_nom = np.asarray([c["nominal"][method]["marginal"] for c in eta_cells], dtype=float)
            joint_nom = np.asarray([c["nominal"][method]["joint"] for c in eta_cells], dtype=float)
            out["nominal"][method]["marginal"].append(np.mean(marg_nom, axis=0).tolist())
            out["nominal"][method]["joint"].append(np.mean(joint_nom, axis=0).tolist())
    return out


def run_experiment(cfg: ScenarioConfig, run_cfg: RunConfig) -> dict[str, Any]:
    beta_pseudo, pseudo_mc_se = estimate_pseudo_true(cfg, run_cfg)
    tasks = [
        (cfg, run_cfg, float(eta), sim_idx, tuple(float(x) for x in beta_pseudo))
        for eta in run_cfg.etas
        for sim_idx in range(run_cfg.n_sim)
    ]
    cells: list[dict[str, Any]] = []
    if run_cfg.jobs == 1:
        for task in tasks:
            cells.append(_run_cell(task))
    else:
        with ProcessPoolExecutor(max_workers=run_cfg.jobs) as executor:
            futures = [executor.submit(_run_cell, task) for task in tasks]
            for future in as_completed(futures):
                cells.append(future.result())
    result = aggregate_cells(cells, cfg, run_cfg)
    result["scenario"] = cfg.name
    result["pseudo_true"] = beta_pseudo.tolist()
    result["pseudo_true_mc_se"] = pseudo_mc_se.tolist()
    result["config"] = {
        "G": cfg.G,
        "n_per_group": cfg.n_per_group,
        "p": cfg.p,
        "beta_true": list(cfg.beta_true),
        "lambda": cfg.lam,
        "outlier_prob": cfg.outlier_prob,
        "outlier_scale": cfg.outlier_scale,
        "n_sim": run_cfg.n_sim,
        "n_iter": run_cfg.n_iter,
        "n_burn_in": run_cfg.n_burn_in,
        "thin": run_cfg.thin,
        "proposal_scale": run_cfg.proposal_scale,
        "pseudo_groups": run_cfg.pseudo_groups,
        "pseudo_replications": run_cfg.pseudo_replications,
    }
    return result


def coordinate_block(x: Iterable[float], y: Iterable[float]) -> str:
    return "{" + " ".join(f"({float(xi):.12g},{float(yi):.12g})" for xi, yi in zip(x, y)) + "}"


def write_metric_tex(path: Path, x: Iterable[float], method_values: dict[str, Iterable[float]]) -> None:
    lines: list[str] = []
    for method, values in method_values.items():
        lines.append(f"{method}:")
        lines.append(coordinate_block(x, values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric_pdf(
    path: Path,
    x: Array,
    method_values: dict[str, Iterable[float]],
    y_label: str,
    x_label: str,
    log_x: bool,
    nominal_line: float | None = None,
    show_legend: bool = True,
    tick_labelsize: int = 13,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 28,
            "axes.labelsize": 34,
            "axes.titlesize": 34,
            "axes.linewidth": 2.2,
            "xtick.labelsize": tick_labelsize,
            "ytick.labelsize": tick_labelsize,
            "legend.fontsize": 24,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "frequentist": "#111111",
        "uncalibrated": "#D55E00",
        "open-faced sandwich": "#0072B2",
        "location-scale map": "#009E73",
        "location-scale one-step": "#CC79A7",
    }
    markers = {
        "frequentist": "s",
        "uncalibrated": "o",
        "open-faced sandwich": "^",
        "location-scale map": "D",
        "location-scale one-step": "X",
    }
    line_styles = {
        "frequentist": (0, (1, 1)),
        "uncalibrated": "-",
        "open-faced sandwich": "--",
        "location-scale map": "-.",
        "location-scale one-step": (0, (5, 2, 1, 2)),
    }
    zorders = {
        "uncalibrated": 2,
        "open-faced sandwich": 3,
        "frequentist": 4,
        "location-scale map": 5,
        "location-scale one-step": 6,
    }
    fig, ax = plt.subplots(figsize=(15.6, 11.0), constrained_layout=True)
    for method, values in method_values.items():
        color = colors.get(method, None)
        ax.plot(
            x,
            np.asarray(list(values), dtype=float),
            label=method,
            color=color,
            marker=markers.get(method, "o"),
            linewidth=5.0,
            markersize=16.0,
            markerfacecolor="none",
            markeredgewidth=3.0,
            linestyle=line_styles.get(method, "-"),
            alpha=0.78,
            dash_capstyle="round",
            solid_capstyle="round",
            zorder=zorders.get(method, 2),
        )
    if nominal_line is not None:
        ax.axhline(nominal_line, color="#555555", linewidth=3.0, linestyle=(0, (1, 2)), alpha=0.9, zorder=1)
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True, color="#CFCFCF", linewidth=1.5, alpha=0.7)
    ax.margins(x=0.03, y=0.08)
    if show_legend:
        ax.legend(frameon=False, ncol=2, handlelength=3.6, columnspacing=1.1, handletextpad=0.7)
    fig.savefig(path)
    plt.close(fig)


def plot_legend_pdf(path: Path, methods: list[str]) -> None:
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 46,
            "legend.fontsize": 50,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "frequentist": "#111111",
        "uncalibrated": "#D55E00",
        "open-faced sandwich": "#0072B2",
        "location-scale map": "#009E73",
        "location-scale one-step": "#CC79A7",
    }
    markers = {
        "frequentist": "s",
        "uncalibrated": "o",
        "open-faced sandwich": "^",
        "location-scale map": "D",
        "location-scale one-step": "X",
    }
    line_styles = {
        "frequentist": (0, (1, 1)),
        "uncalibrated": "-",
        "open-faced sandwich": "--",
        "location-scale map": "-.",
        "location-scale one-step": (0, (5, 2, 1, 2)),
    }
    fig, ax = plt.subplots(figsize=(46.8, 5.2), constrained_layout=True)
    handles = []
    for method in methods:
        handle, = ax.plot(
            [],
            [],
            label=method,
            color=colors.get(method, None),
            marker=markers.get(method, "o"),
            linewidth=9.0,
            markersize=30.0,
            markerfacecolor="none",
            markeredgewidth=5.5,
            linestyle=line_styles.get(method, "-"),
            alpha=0.78,
        )
        handles.append(handle)
    ax.axis("off")
    ax.legend(
        handles=handles,
        labels=methods,
        loc="center",
        ncol=len(methods),
        frameon=False,
        handlelength=4.6,
        columnspacing=1.6,
        handletextpad=0.8,
    )
    fig.savefig(path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def write_outputs(result: dict[str, Any], run_cfg: RunConfig) -> None:
    out_dir = Path(run_cfg.output_root) / f"{result['scenario']}_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    etas = np.asarray(result["etas"], dtype=float)
    metrics = result["metrics"]
    methods = result["methods"]
    plot_legend_pdf(out_dir / "legend_output.pdf", methods)
    metric_labels = {
        "coverage": ("coverage_output.tex", "coverage_output.pdf", "Empirical coverage", 1.0 - scalar_scenario().alpha),
        "joint_coverage": ("joint_coverage_output.tex", "joint_coverage_output.pdf", "Joint ellipsoid coverage", 1.0 - scalar_scenario().alpha),
        "width": ("width_output.tex", "width_output.pdf", "Average interval width", None),
        "volume": ("volume_output.tex", "volume_output.pdf", "Average ellipsoid volume", None),
        "bias": ("bias_output.tex", "bias_output.pdf", "Bias", 0.0),
        "bias_sd": ("bias_sd_output.tex", "bias_sd_output.pdf", "Bias SD", None),
        "bias_l2": ("bias_l2_output.tex", "bias_l2_output.pdf", "Mean L2 bias", None),
    }
    for metric, (tex_name, pdf_name, y_label, ref_line) in metric_labels.items():
        values = {method: metrics[method][metric] for method in methods}
        write_metric_tex(out_dir / tex_name, etas, values)
        plot_metric_pdf(
            out_dir / pdf_name,
            etas,
            values,
            y_label=y_label,
            x_label="Learning rate",
            log_x=True,
            nominal_line=ref_line,
        )
        plot_metric_pdf(
            out_dir / pdf_name.replace(".pdf", "_no_ylabel.pdf"),
            etas,
            values,
            y_label="",
            x_label="",
            log_x=True,
            nominal_line=ref_line,
            show_legend=False,
            tick_labelsize=36,
        )

    nominal_levels = np.asarray(result["nominal_levels"], dtype=float)
    selected_eta_index = int(np.argmin(np.abs(etas - 1.0)))
    for kind, pdf_label in [("marginal", "Marginal coverage"), ("joint", "Joint coverage")]:
        values = {
            method: result["nominal"][method][kind][selected_eta_index]
            for method in methods
        }
        write_metric_tex(out_dir / f"nominal_{kind}_coverage_output.tex", nominal_levels, values)
        plot_metric_pdf(
            out_dir / f"nominal_{kind}_coverage_output.pdf",
            nominal_levels,
            values,
            y_label=pdf_label,
            x_label="Nominal level",
            log_x=False,
            nominal_line=None,
        )
        plot_metric_pdf(
            out_dir / f"nominal_{kind}_coverage_output_no_ylabel.pdf",
            nominal_levels,
            values,
            y_label="",
            x_label="",
            log_x=False,
            nominal_line=None,
            show_legend=False,
            tick_labelsize=36,
        )

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def parse_etas(raw: str) -> tuple[float, ...]:
    if "," in raw:
        return tuple(float(x.strip()) for x in raw.split(",") if x.strip())
    parts = raw.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("etas must be comma-separated or min:max:count")
    lo, hi, count = float(parts[0]), float(parts[1]), int(parts[2])
    return tuple(float(x) for x in np.logspace(np.log10(lo), np.log10(hi), count))


def build_run_config(args: argparse.Namespace) -> RunConfig:
    if args.smoke:
        n_sim = args.n_sim if args.n_sim is not None else 8
        n_iter = args.n_iter if args.n_iter is not None else 700
        n_burn = args.n_burn_in if args.n_burn_in is not None else 250
        pseudo_groups = args.pseudo_groups if args.pseudo_groups is not None else 800
        pseudo_rep = args.pseudo_replications if args.pseudo_replications is not None else 5
        etas = args.etas if args.etas is not None else tuple(float(x) for x in np.logspace(-2, 2, 5))
    else:
        n_sim = args.n_sim if args.n_sim is not None else 200
        n_iter = args.n_iter if args.n_iter is not None else 2500
        n_burn = args.n_burn_in if args.n_burn_in is not None else 1000
        pseudo_groups = args.pseudo_groups if args.pseudo_groups is not None else 5000
        pseudo_rep = args.pseudo_replications if args.pseudo_replications is not None else 50
        etas = args.etas if args.etas is not None else tuple(float(x) for x in np.logspace(-2, 2, 20))
    return RunConfig(
        n_sim=n_sim,
        etas=tuple(float(x) for x in etas),
        n_iter=n_iter,
        n_burn_in=n_burn,
        thin=args.thin,
        proposal_scale=args.proposal_scale,
        pseudo_groups=pseudo_groups,
        pseudo_replications=pseudo_rep,
        nominal_levels=tuple(float(x) for x in np.r_[np.arange(0.50, 1.00, 0.10), 0.95, 0.99]),
        jobs=min(max(1, args.jobs), 12),
        seed=args.seed,
        output_root=args.output_root,
    )


def scenario_list(name: str, p: int) -> list[ScenarioConfig]:
    if name == "scalar":
        return [scalar_scenario()]
    if name == "multivariate":
        return [multivariate_scenario(p=p)]
    if name == "location":
        return [location_scenario()]
    if name == "all":
        return [scalar_scenario(), multivariate_scenario(p=p), location_scenario()]
    raise ValueError(f"Unknown scenario: {name}")


def render_saved_outputs(output_root: str, scenario_name: str, p: int) -> None:
    run_cfg = RunConfig(
        n_sim=0,
        etas=(),
        n_iter=0,
        n_burn_in=0,
        thin=1,
        proposal_scale=1.0,
        pseudo_groups=0,
        pseudo_replications=0,
        nominal_levels=(),
        jobs=1,
        seed=0,
        output_root=output_root,
    )
    for scenario in scenario_list(scenario_name, p):
        summary_path = Path(output_root) / f"{scenario.name}_output" / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file: {summary_path}")
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        write_outputs(result, run_cfg)
        print(f"Redrew {summary_path.parent}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Section 7 empirical illustrations for location-scale calibration.")
    parser.add_argument("--scenario", choices=["scalar", "multivariate", "location", "all"], default="all")
    parser.add_argument("--p", type=int, default=5, help="Dimension for the multivariate scenario.")
    parser.add_argument("--n_sim", type=int, default=None)
    parser.add_argument("--etas", type=parse_etas, default=None, help="Comma list or min:max:count, e.g. 0.01:100:20.")
    parser.add_argument("--n_iter", type=int, default=None)
    parser.add_argument("--n_burn_in", type=int, default=None)
    parser.add_argument("--thin", type=int, default=1)
    parser.add_argument("--proposal_scale", type=float, default=1.0)
    parser.add_argument("--pseudo_groups", type=int, default=None)
    parser.add_argument("--pseudo_replications", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--output_root", type=str, default="empirical_output")
    parser.add_argument("--smoke", action="store_true", help="Use a quick configuration for verification.")
    parser.add_argument("--render_only", action="store_true", help="Redraw figures from existing summary.json files.")
    args = parser.parse_args()

    if args.render_only:
        render_saved_outputs(args.output_root, args.scenario, args.p)
        return

    run_cfg = build_run_config(args)
    for scenario in scenario_list(args.scenario, args.p):
        print(f"Running {scenario.name} with n_sim={run_cfg.n_sim}, etas={len(run_cfg.etas)}, jobs={run_cfg.jobs}")
        result = run_experiment(scenario, run_cfg)
        write_outputs(result, run_cfg)
        out_dir = Path(run_cfg.output_root) / f"{scenario.name}_output"
        print(f"Wrote {out_dir}")
        print(f"Pseudo-true: {result['pseudo_true']} (MC SE {result['pseudo_true_mc_se']})")
        print(f"Mean MH acceptance by eta: {result['accept_rate']}")


if __name__ == "__main__":
    main()
