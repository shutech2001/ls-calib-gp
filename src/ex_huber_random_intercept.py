from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from huber_random_intercept import HuberRandomIntercept


def run_simulation_one_scenario(
    n_sim: int = 200,
    G: int = 100,
    n_per_group: int = 5,
    p: int = 1,
    beta_true: Optional[NDArray[np.float64]] = None,
    G_large: int = 5000,
    large_replication: int = 1000,
    beta_pseudo_true: Optional[NDArray[np.float64]] = None,
    tau: float = 1.0,
    sigma: float = 1.0,
    c: float = 1.0,
    eta: float = 1.0,
    lam: float = 0.1,
    outlier_prob: float = 0.1,
    outlier_scale: float = 10.0,
    n_iter: int = 1000,
    n_burn_in: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    do_freq: bool = True,
    verbose: bool = True,
) -> Dict:
    if beta_true is None:
        beta_true = np.array([2.0])

    np.random.seed(seed)

    # Storage for results
    coverage_uncalib = np.zeros(n_sim)
    coverage_calib = np.zeros(n_sim)
    coverage_freq = np.zeros(n_sim)

    width_uncalib = np.zeros(n_sim)
    width_calib = np.zeros(n_sim)
    width_freq = np.zeros(n_sim)

    bias_uncalib = np.zeros(n_sim)
    bias_calib = np.zeros(n_sim)
    bias_freq = np.zeros(n_sim)

    model = HuberRandomIntercept(
        G=G,
        n_per_group=n_per_group,
        p=p,
        beta_true=beta_true,
        tau=tau,
        sigma=sigma,
        c=c,
        eta=eta,
        lam=lam,
    )

    if beta_pseudo_true is None:
        beta_pseudo_true = model.compute_pseudo_true_beta_lambda(
            G_large=G_large,
            large_replication=large_replication,
            outlier_prob=outlier_prob,
            outlier_scale=outlier_scale,
            seed=seed,
            verbose=verbose,
        )
    else:
        beta_pseudo_true = np.asarray(beta_pseudo_true)
    if verbose:
        logging.info(f"True beta: {beta_true}")
        logging.info(f"Pseudo-true beta: {beta_pseudo_true}")

    for sim in range(n_sim):
        if verbose:
            logging.info(f"Running simulation {sim + 1} of {n_sim}...")

        # Generate data
        y_deque, X_deque = model.generate_data(
            seed=seed + sim * 1000,
            outlier_prob=outlier_prob,
            outlier_scale=outlier_scale,
        )

        # Run MCMC
        beta_samples, _, _ = model.gibbs_sampler(
            y_deque=y_deque,
            X_deque=X_deque,
            n_iter=n_iter,
            n_burn_in=n_burn_in,
            seed=seed + sim * 1000 + 1,
        )

        # Uncalibrated posterior
        beta_GB = np.mean(beta_samples, axis=0)
        uncalib_lower = np.quantile(beta_samples, alpha / 2, axis=0)
        uncalib_upper = np.quantile(beta_samples, 1 - alpha / 2, axis=0)

        # Compute one-step center and calibrate
        beta_dagger, _, _ = model.fit_frequentist_penalized_ee(y_deque, X_deque, alpha)
        Omega, _ = model.compute_calibration_matrix(beta_GB, beta_dagger, beta_samples, y_deque, X_deque)
        beta_samples_calib = model.calibrate_samples(beta_samples, beta_GB, beta_dagger, Omega)

        # Calibrated posterior
        beta_calib_mean = np.mean(beta_samples_calib, axis=0)
        calib_lower = np.quantile(beta_samples_calib, alpha / 2, axis=0)
        calib_upper = np.quantile(beta_samples_calib, 1 - alpha / 2, axis=0)

        # Frequentist estimation
        if do_freq:
            beta_freq, freq_lower, freq_upper = model.fit_frequentist_penalized_ee(
                y_deque=y_deque,
                X_deque=X_deque,
                alpha=alpha,
            )
        else:
            beta_freq = None
            freq_lower = None
            freq_upper = None

        # Compute metrics (for first component)
        idx = 0

        # Coverage
        coverage_uncalib[sim] = uncalib_lower[idx] <= beta_pseudo_true[idx] <= uncalib_upper[idx]
        coverage_calib[sim] = calib_lower[idx] <= beta_pseudo_true[idx] <= calib_upper[idx]
        if do_freq:
            assert freq_lower is not None and freq_upper is not None
            coverage_freq[sim] = freq_lower[idx] <= beta_pseudo_true[idx] <= freq_upper[idx]
        else:
            coverage_freq[sim] = np.nan

        # Width
        width_uncalib[sim] = uncalib_upper[idx] - uncalib_lower[idx]
        width_calib[sim] = calib_upper[idx] - calib_lower[idx]
        if do_freq:
            assert freq_lower is not None and freq_upper is not None
            width_freq[sim] = freq_upper[idx] - freq_lower[idx]
        else:
            width_freq[sim] = np.nan

        # Bias
        bias_uncalib[sim] = beta_GB[idx] - beta_pseudo_true[idx]
        bias_calib[sim] = beta_calib_mean[idx] - beta_pseudo_true[idx]
        if do_freq:
            assert beta_freq is not None
            bias_freq[sim] = beta_freq[idx] - beta_pseudo_true[idx]
        else:
            bias_freq[sim] = np.nan

    return {
        "beta_pseudo_true": beta_pseudo_true,
        "coverage_uncalibrated": np.mean(coverage_uncalib),
        "coverage_calibrated": np.mean(coverage_calib),
        "coverage_frequentist": np.mean(coverage_freq),
        "width_uncalibrated": np.mean(width_uncalib),
        "width_calibrated": np.mean(width_calib),
        "width_frequentist": np.mean(width_freq),
        "bias_uncalibrated": np.mean(bias_uncalib),
        "bias_calibrated": np.mean(bias_calib),
        "bias_frequentist": np.mean(bias_freq),
        "bias_sd_uncalibrated": np.std(bias_uncalib, ddof=1),
        "bias_sd_calibrated": np.std(bias_calib, ddof=1),
        "bias_sd_frequentist": np.std(bias_freq, ddof=1),
    }


def run_simulation_varying_learning_rate(
    min_learning_rate: float = 0.01,
    max_learning_rate: float = 100.0,
    n_learning_rates: int = 20,
    n_sim: int = 200,
    G: int = 100,
    n_per_group: int = 5,
    p: int = 1,
    beta_true: Optional[NDArray[np.float64]] = None,
    G_large: int = 5000,
    large_replication: int = 1000,
    tau: float = 1.0,
    sigma: float = 1.0,
    c: float = 1.0,
    lam: float = 0.1,
    outlier_prob: float = 0.1,
    outlier_scale: float = 10.0,
    n_iter: int = 1000,
    n_burn_in: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    etas = np.logspace(np.log10(min_learning_rate), np.log10(max_learning_rate), n_learning_rates)

    coverage_uncalib = np.zeros(n_learning_rates)
    coverage_calib = np.zeros(n_learning_rates)
    width_uncalib = np.zeros(n_learning_rates)
    width_calib = np.zeros(n_learning_rates)
    bias_uncalib = np.zeros(n_learning_rates)
    bias_calib = np.zeros(n_learning_rates)
    bias_sd_uncalib = np.zeros(n_learning_rates)
    bias_sd_calib = np.zeros(n_learning_rates)

    # results for frequentist
    coverage_freq = 0
    width_freq = 0
    bias_freq = 0
    bias_sd_freq = 0

    # storage for beta_pseudo_true
    beta_pseudo_true = None

    for i, eta in enumerate(etas):
        if verbose:
            logging.info(f"Running simulation with learning rate {eta}...")
        if i == 0:
            results = run_simulation_one_scenario(
                n_sim=n_sim,
                G=G,
                n_per_group=n_per_group,
                p=p,
                beta_true=beta_true,
                G_large=G_large,
                large_replication=large_replication,
                tau=tau,
                sigma=sigma,
                c=c,
                eta=eta,
                lam=lam,
                outlier_prob=outlier_prob,
                outlier_scale=outlier_scale,
                n_iter=n_iter,
                n_burn_in=n_burn_in,
                alpha=alpha,
                seed=seed,
                do_freq=True,
                verbose=verbose,
            )
            beta_pseudo_true = results["beta_pseudo_true"]
            coverage_freq = results["coverage_frequentist"]
            width_freq = results["width_frequentist"]
            bias_freq = results["bias_frequentist"]
            bias_sd_freq = results["bias_sd_frequentist"]
        else:
            results = run_simulation_one_scenario(
                n_sim=n_sim,
                G=G,
                n_per_group=n_per_group,
                p=p,
                beta_true=beta_true,
                beta_pseudo_true=beta_pseudo_true,
                tau=tau,
                sigma=sigma,
                c=c,
                eta=eta,
                lam=lam,
                outlier_prob=outlier_prob,
                outlier_scale=outlier_scale,
                n_iter=n_iter,
                n_burn_in=n_burn_in,
                alpha=alpha,
                seed=seed,
                do_freq=False,
                verbose=verbose,
            )

        coverage_uncalib[i] = results["coverage_uncalibrated"]
        coverage_calib[i] = results["coverage_calibrated"]
        width_uncalib[i] = results["width_uncalibrated"]
        width_calib[i] = results["width_calibrated"]
        bias_uncalib[i] = results["bias_uncalibrated"]
        bias_calib[i] = results["bias_calibrated"]
        bias_sd_uncalib[i] = results["bias_sd_uncalibrated"]
        bias_sd_calib[i] = results["bias_sd_calibrated"]

    return {
        "etas": etas,
        "coverage_uncalibrated": coverage_uncalib,
        "coverage_calibrated": coverage_calib,
        "coverage_frequentist": coverage_freq,
        "width_uncalibrated": width_uncalib,
        "width_calibrated": width_calib,
        "width_frequentist": width_freq,
        "bias_uncalibrated": bias_uncalib,
        "bias_calibrated": bias_calib,
        "bias_frequentist": bias_freq,
        "bias_sd_uncalibrated": bias_sd_uncalib,
        "bias_sd_calibrated": bias_sd_calib,
        "bias_sd_frequentist": bias_sd_freq,
    }


def print_results(results: Dict) -> None:
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS - HUBER RANDOM INTERCEPT")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Uncalibrated':<12} {'Calibrated':<12} {'Frequentist':<12}")
    print("-" * 80)
    print(
        f"{'Coverage Probability':<30} "
        f"{results['coverage_uncalibrated']:.4f}{'':<6} "
        f"{results['coverage_calibrated']:.4f}{'':<6} "
        f"{results['coverage_frequentist']:.4f}"
    )
    print(
        f"{'Average Interval Width':<30} "
        f"{results['width_uncalibrated']:.4f}{'':<6} "
        f"{results['width_calibrated']:.4f}{'':<6} "
        f"{results['width_frequentist']:.4f}"
    )
    print(
        f"{'Mean Bias':<30} "
        f"{results['bias_uncalibrated']:.4f}{'':<6} "
        f"{results['bias_calibrated']:.4f}{'':<6} "
        f"{results['bias_frequentist']:.4f}"
    )
    print(
        f"{'Bias SD':<30} "
        f"{results['bias_sd_uncalibrated']:.4f}{'':<6} "
        f"{results['bias_sd_calibrated']:.4f}{'':<6} "
        f"{results['bias_sd_frequentist']:.4f}"
    )
    print("=" * 80)
    print("\nMethods:")
    print("  - Uncalibrated             : Bayesian posterior with learning rate eta")
    print("  - Location-Scale Calibrated: Bayesian posterior + location-scale calibration")
    print("  - Frequentist              : Frequentist LMM")
    print("=" * 80)


def parse_beta(beta_str: str) -> NDArray:
    """Generate an NDArray from a comma-separated string

    Args:
        beta_str (str): comma-separated parameter (e.g., 5.0,3.5)

    Returns:
        NDArray: true coefficient vector
    """
    try:
        beta_list: list[float] = [float(b.strip()) for b in beta_str.split(",")]
    except ValueError as e:
        raise argparse.ArgumentTypeError("beta_true must be a comma-separated list of numbers. (e.g., 5.0,3.5)") from e
    return np.array(beta_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a simulation of the Huber random intercept model with n-dependent Gaussian prior."
    )
    parser.add_argument("--n_sim", "--N", type=int, default=200, help="Number of simulations (default: 200).")
    parser.add_argument("--G", type=int, default=100, help="Number of groups (default: 100).")
    parser.add_argument(
        "--n_per_group", "--NG", type=int, default=5, help="Number of observations per group (default: 5)."
    )
    parser.add_argument("--p", type=int, default=1, help="Dimension of covariates (default: 1).")
    parser.add_argument("--beta_true", type=parse_beta, default="2.0", help="True coefficients (default: 2.0).")
    parser.add_argument(
        "--G_large", type=int, default=5000, help="Number of groups for pseudo-true beta (default: 5000)."
    )
    parser.add_argument(
        "--large_replication",
        type=int,
        default=1000,
        help="Number of replications for pseudo-true beta (default: 1000).",
    )
    parser.add_argument("--tau", type=float, default=1.0, help="True random intercept variance (default: 1.0).")
    parser.add_argument("--sigma", type=float, default=1.0, help="True residual variance (default: 1.0).")
    parser.add_argument("--c", type=float, default=1.0, help="Huber tuning parameter (default: 1.0).")
    parser.add_argument("--lam", type=float, default=0.1, help="Ridge penalty parameter (default: 0.1).")
    parser.add_argument("--outlier_prob", type=float, default=0.1, help="Outlier probability (default: 0.1).")
    parser.add_argument("--outlier_scale", type=float, default=10.0, help="Outlier scale (default: 10.0).")
    parser.add_argument("--n_iter", type=int, default=1000, help="Number of Gibbs iterations (default: 1000).")
    parser.add_argument("--n_burn_in", type=int, default=500, help="Number of burn-in iterations (default: 500).")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--verbose", type=bool, default=False, help="Whether to print verbose output (default: False).")

    parser.add_argument(
        "--one_scenario", "--OS", type=bool, default=False, help="Whether to run a single scenario (default: False)."
    )
    parser.add_argument("--eta", "--LR", type=float, default=1.0, help="Learning rate (default: 1.0).")

    parser.add_argument(
        "--min_learning_rate", "--minLR", type=float, default=0.01, help="Minimum learning rate (default: 0.01)."
    )
    parser.add_argument(
        "--max_learning_rate", "--maxLR", type=float, default=100.0, help="Maximum learning rate (default: 100.0)."
    )
    parser.add_argument(
        "--n_learning_rates", "--NLR", type=int, default=20, help="Number of learning rates (default: 20)."
    )

    args = parser.parse_args()

    if args.one_scenario:
        results = run_simulation_one_scenario(
            n_sim=args.n_sim,
            G=args.G,
            n_per_group=args.n_per_group,
            p=args.p,
            beta_true=args.beta_true,
            G_large=args.G_large,
            large_replication=args.large_replication,
            tau=args.tau,
            sigma=args.sigma,
            c=args.c,
            eta=args.eta,
            lam=args.lam,
            outlier_prob=args.outlier_prob,
            outlier_scale=args.outlier_scale,
            n_iter=args.n_iter,
            n_burn_in=args.n_burn_in,
            alpha=args.alpha,
            seed=args.seed,
            verbose=args.verbose,
        )
        print_results(results)
    else:
        results = run_simulation_varying_learning_rate(
            min_learning_rate=args.min_learning_rate,
            max_learning_rate=args.max_learning_rate,
            n_learning_rates=args.n_learning_rates,
            n_sim=args.n_sim,
            G=args.G,
            n_per_group=args.n_per_group,
            p=args.p,
            beta_true=args.beta_true,
            G_large=args.G_large,
            large_replication=args.large_replication,
            tau=args.tau,
            sigma=args.sigma,
            c=args.c,
            lam=args.lam,
            outlier_prob=args.outlier_prob,
            outlier_scale=args.outlier_scale,
            n_iter=args.n_iter,
            n_burn_in=args.n_burn_in,
            alpha=args.alpha,
            seed=args.seed,
            verbose=args.verbose,
        )
        # output for tex plots (coordinates are (eta, value))
        os.makedirs("huber_random_intercept_output", exist_ok=True)
        with open("huber_random_intercept_output/coverage_output.tex", "w", encoding="utf-8") as f:
            # frequentist
            f.write("frequentist:\n")
            out_freq = (
                "{"
                + " ".join(
                    f"({xi},{yi})"
                    for xi, yi in zip(results["etas"], results["coverage_frequentist"] * np.ones(len(results["etas"])))
                )
                + "}"
            )
            f.write(out_freq + "\n")
            # uncalibrated
            f.write("uncalibrated:\n")
            out_uncalib = (
                "{"
                + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["coverage_uncalibrated"]))
                + "}"
            )
            f.write(out_uncalib + "\n")
            # location-scale calibrated
            f.write("location-scale calibrated:\n")
            out_calib = (
                "{" + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["coverage_calibrated"])) + "}"
            )
            f.write(out_calib + "\n")
        with open("huber_random_intercept_output/width_output.tex", "w", encoding="utf-8") as f:
            # frequentist
            f.write("frequentist:\n")
            out_freq = (
                "{"
                + " ".join(
                    f"({xi},{yi})"
                    for xi, yi in zip(results["etas"], results["width_frequentist"] * np.ones(len(results["etas"])))
                )
                + "}"
            )
            f.write(out_freq + "\n")
            # uncalibrated
            f.write("uncalibrated:\n")
            out_uncalib = (
                "{" + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["width_uncalibrated"])) + "}"
            )
            f.write(out_uncalib + "\n")
            # location-scale calibrated
            f.write("location-scale calibrated:\n")
            out_calib = (
                "{" + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["width_calibrated"])) + "}"
            )
            f.write(out_calib + "\n")
        with open("huber_random_intercept_output/bias_output.tex", "w", encoding="utf-8") as f:
            # frequentist
            f.write("frequentist:\n")
            out_freq = (
                "{"
                + " ".join(
                    f"({xi},{yi})"
                    for xi, yi in zip(results["etas"], results["bias_frequentist"] * np.ones(len(results["etas"])))
                )
                + "}"
            )
            f.write(out_freq + "\n")
            # uncalibrated
            f.write("uncalibrated:\n")
            out_uncalib = (
                "{" + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["bias_uncalibrated"])) + "}"
            )
            f.write(out_uncalib + "\n")
            # location-scale calibrated
            f.write("location-scale calibrated:\n")
            out_calib = (
                "{" + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["bias_calibrated"])) + "}"
            )
            f.write(out_calib + "\n")
        with open("huber_random_intercept_output/bias_sd_output.tex", "w", encoding="utf-8") as f:
            # frequentist
            f.write("frequentist:\n")
            out_freq = (
                "{"
                + " ".join(
                    f"({xi},{yi})"
                    for xi, yi in zip(results["etas"], results["bias_sd_frequentist"] * np.ones(len(results["etas"])))
                )
                + "}"
            )
            f.write(out_freq + "\n")
            # uncalibrated
            f.write("uncalibrated:\n")
            out_uncalib = (
                "{" + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["bias_sd_uncalibrated"])) + "}"
            )
            f.write(out_uncalib + "\n")
            # location-scale calibrated
            f.write("location-scale calibrated:\n")
            out_calib = (
                "{" + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["bias_sd_calibrated"])) + "}"
            )
            f.write(out_calib + "\n")
        logging.info(
            "Results written to huber_random_intercept_output/coverage_output.tex, huber_random_intercept_output/width_output.tex, huber_random_intercept_output/bias_output.tex, and huber_random_intercept_output/bias_sd_output.tex"  # noqa: E501
        )
