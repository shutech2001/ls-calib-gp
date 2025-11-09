from __future__ import annotations

import argparse
import logging
import os
from typing import Dict

import numpy as np

from random_intercept_lmm import RandomInterceptLMM


def run_simulation_one_scenario(
    n_sim: int = 500,
    J: int = 60,
    n_per_group: int = 5,
    p: int = 1,
    eta: float = 1.0,
    M_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    do_freq: bool = True,
    verbose: bool = False,
) -> Dict:
    """Run a simulation for a given scenario.

    Args:
        n_sim (int, optional): Number of simulations. Defaults to 500.
        J (int, optional): Number of groups. Defaults to 60.
        n_per_group (int, optional): Number of observations per group. Defaults to 5.
        p (int, optional): Number of covariates. Defaults to 1.
        eta (float, optional): Learning rate. Defaults to 1.0.
        M_samples (int, optional): Number of samples from the posterior. Defaults to 1000.
        alpha (float, optional): Significance level. Defaults to 0.05.
        seed (int, optional): Random seed. Defaults to 42.
        do_freq (bool, optional): Whether to fit a frequentist LMM. Defaults to True.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        Dict: A dictionary containing the simulation results.
            - coverage_uncalibrated (float): Coverage probability of the uncalibrated intervals.
            - coverage_calibrated (float): Coverage probability of the calibrated intervals.
            - coverage_frequentist (float): Coverage probability of the frequentist intervals.
            - width_uncalibrated (float): Average width of the uncalibrated intervals.
            - width_calibrated (float): Average width of the calibrated intervals.
            - width_frequentist (float): Average width of the frequentist intervals.
            - bias_uncalibrated (float): Average bias of the uncalibrated intervals.
            - bias_calibrated (float): Average bias of the calibrated intervals.
            - bias_frequentist (float): Average bias of the frequentist intervals.
            - bias_sd_uncalibrated (float): Standard deviation of the bias of the uncalibrated intervals.
            - bias_sd_calibrated (float): Standard deviation of the bias of the calibrated intervals.
            - bias_sd_frequentist (float): Standard deviation of the bias of the frequentist intervals.
    """
    np.random.seed(seed)

    # True parameters
    beta_true = np.ones(p)
    tau = 1.0
    sigma = 1.0

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

    for sim in range(n_sim):
        if verbose:
            logging.info(f"Running simulation {sim + 1} of {n_sim}...")

        # Initialize model
        model = RandomInterceptLMM(
            J=J, n_per_group=n_per_group, p=p, beta_true=beta_true, tau=tau, sigma=sigma, eta=eta
        )

        # Generate data
        y_deque, X_deque = model.generate_data(seed=seed + sim * 10)

        # Compute posterior
        m_post, Lambda_post_inv = model.compute_posterior(y_deque, X_deque)

        # Sample from posterior
        samples = model.sample_posterior(m_post, Lambda_post_inv, M_samples)

        # Compute admissible center (here, one-step center)
        beta_GB = m_post
        beta_dagger = model.compute_one_step_center(beta_GB, y_deque, X_deque)

        # Compute calibration matrix
        Omega, V_target_hat = model.compute_calibration_matrix(beta_GB, beta_dagger, samples, y_deque, X_deque)

        # Calibrate samples
        samples_calib = model.calibrate_samples(samples, beta_GB, beta_dagger, Omega)

        # Fit frequentist LMM
        if do_freq:
            beta_freq, freq_lower, freq_upper = model.fit_frequentist_lmm(y_deque, X_deque, alpha)
        else:
            beta_freq = None
            freq_lower = None
            freq_upper = None

        # Compute intervals
        # # Uncalibrated
        uncalib_mean = np.mean(samples, axis=0)
        uncalib_lower = np.quantile(samples, alpha / 2, axis=0)
        uncalib_upper = np.quantile(samples, 1 - alpha / 2, axis=0)

        # # Locaiton-Scale Calibrated
        calib_mean = np.mean(samples_calib, axis=0)
        calib_lower = np.quantile(samples_calib, alpha / 2, axis=0)
        calib_upper = np.quantile(samples_calib, 1 - alpha / 2, axis=0)

        # Compute metrics (for first component when p=1)
        idx = 0
        coverage_uncalib[sim] = np.mean(uncalib_lower[idx] <= beta_true[idx] <= uncalib_upper[idx])
        coverage_calib[sim] = np.mean(calib_lower[idx] <= beta_true[idx] <= calib_upper[idx])
        if do_freq:
            assert freq_lower is not None and freq_upper is not None
            coverage_freq[sim] = np.mean(freq_lower[idx] <= beta_true[idx] <= freq_upper[idx])
        else:
            coverage_freq[sim] = np.nan

        width_uncalib[sim] = uncalib_upper[idx] - uncalib_lower[idx]
        width_calib[sim] = calib_upper[idx] - calib_lower[idx]
        if do_freq:
            assert freq_lower is not None and freq_upper is not None
            width_freq[sim] = freq_upper[idx] - freq_lower[idx]
        else:
            width_freq[sim] = np.nan

        bias_uncalib[sim] = uncalib_mean[idx] - beta_true[idx]
        bias_calib[sim] = calib_mean[idx] - beta_true[idx]
        if do_freq:
            assert beta_freq is not None
            bias_freq[sim] = beta_freq[idx] - beta_true[idx]
        else:
            bias_freq[sim] = np.nan

    return {
        "coverage_uncalibrated": np.mean(coverage_uncalib),
        "coverage_calibrated": np.mean(coverage_calib),
        "coverage_frequentist": np.mean(coverage_freq),
        "width_uncalibrated": np.mean(width_uncalib),
        "width_calibrated": np.mean(width_calib),
        "width_frequentist": np.mean(width_freq),
        "bias_uncalibrated": np.mean(bias_uncalib),
        "bias_calibrated": np.mean(bias_calib),
        "bias_frequentist": np.mean(bias_freq),
        "bias_sd_uncalibrated": np.std(bias_uncalib),
        "bias_sd_calibrated": np.std(bias_calib),
        "bias_sd_frequentist": np.std(bias_freq),
    }


def run_simulation_varying_learning_rate(
    min_learning_rate: float = 0.01,
    max_learning_rate: float = 100.0,
    n_learning_rates: int = 20,
    n_sim: int = 500,
    J: int = 60,
    n_per_group: int = 5,
    p: int = 1,
    eta: float = 1.0,
    M_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    verbose: bool = False,
) -> Dict:
    """Run a simulation varying the learning rate.

    Args:
        min_learning_rate (float, optional): Minimum learning rate. Defaults to 0.01.
        max_learning_rate (float, optional): Maximum learning rate. Defaults to 100.0.
        n_learning_rates (int, optional): Number of learning rates. Defaults to 20.
        n_sim (int, optional): Number of simulations. Defaults to 500.
        J (int, optional): Number of groups. Defaults to 60.
        n_per_group (int, optional): Number of observations per group. Defaults to 5.
        p (int, optional): Number of covariates. Defaults to 1.
        eta (float, optional): Learning rate. Defaults to 1.0.
        M_samples (int, optional): Number of samples from the posterior. Defaults to 1000.
        alpha (float, optional): Significance level. Defaults to 0.05.
        seed (int, optional): Random seed. Defaults to 42.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        Dict: A dictionary containing the simulation results.
            - etas (List[float]): List of learning rates.
            - coverage_uncalibrated (List[float]):
                Coverage probability of the uncalibrated intervals for each learning rate.
            - coverage_calibrated (List[float]):
                Coverage probability of the calibrated intervals for each learning rate.
            - coverage_frequentist (float): Coverage probability of the frequentist intervals.
            - width_uncalibrated (List[float]): Average width of the uncalibrated intervals for each learning rate.
            - width_calibrated (List[float]): Average width of the calibrated intervals for each learning rate.
            - width_frequentist (float): Average width of the frequentist intervals.
            - bias_uncalibrated (List[float]): Average bias of the uncalibrated intervals for each learning rate.
            - bias_calibrated (List[float]): Average bias of the calibrated intervals for each learning rate.
            - bias_frequentist (float): Average bias of the frequentist intervals.
            - bias_sd_uncalibrated (List[float]):
                Standard deviation of the bias of the uncalibrated intervals for each learning rate.
            - bias_sd_calibrated (List[float]):
                Standard deviation of the bias of the calibrated intervals for each learning rate.
            - bias_sd_frequentist (float): Standard deviation of the bias of the frequentist intervals.
    """
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

    for i, eta in enumerate(etas):
        if verbose:
            logging.info(f"Running simulation with learning rate {eta}...")
        if i == 0:
            results = run_simulation_one_scenario(
                n_sim=n_sim,
                J=J,
                n_per_group=n_per_group,
                p=p,
                eta=eta,
                M_samples=M_samples,
                alpha=alpha,
                seed=seed,
                do_freq=True,
                verbose=verbose,
            )
            coverage_freq = results["coverage_frequentist"]
            width_freq = results["width_frequentist"]
            bias_freq = results["bias_frequentist"]
            bias_sd_freq = results["bias_sd_frequentist"]
        else:
            results = run_simulation_one_scenario(
                n_sim=n_sim,
                J=J,
                n_per_group=n_per_group,
                p=p,
                eta=eta,
                M_samples=M_samples,
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
    """Print the simulation results.

    Args:
        results (Dict): A dictionary containing the simulation results.
            - coverage_uncalibrated (float): Coverage probability of the uncalibrated intervals.
            - coverage_calibrated (float): Coverage probability of the calibrated intervals.
            - coverage_frequentist (float): Coverage probability of the frequentist intervals.
            - width_uncalibrated (float): Average width of the uncalibrated intervals.
            - width_calibrated (float): Average width of the calibrated intervals.
            - width_frequentist (float): Average width of the frequentist intervals.
            - bias_uncalibrated (float): Average bias of the uncalibrated intervals.
            - bias_calibrated (float): Average bias of the calibrated intervals.
            - bias_frequentist (float): Average bias of the frequentist intervals.
            - bias_sd_uncalibrated (float): Standard deviation of the bias of the uncalibrated intervals.
            - bias_sd_calibrated (float): Standard deviation of the bias of the calibrated intervals.
            - bias_sd_frequentist (float): Standard deviation of the bias of the frequentist intervals.
    """
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Uncalib':<12} {'Calibrated':<12} {'Frequentist':<12}")
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
    print("\nNote: Target coverage probability is 0.95 (alpha=0.05)")
    print("\nMethods:")
    print("  - Uncalibrated: Bayesian posterior with learning rate eta")
    print("  - Calibrated  : Bayesian posterior + sandwich variance calibration")
    print("  - Frequentist : Frequentist LMM")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a simulation of the random intercept LMM with Bayesian posterior and sandwich variance calibration."  # noqa: E501
    )
    parser.add_argument("--n_sim", type=int, default=500, help="Number of simulations.")
    parser.add_argument("--J", type=int, default=60, help="Number of groups.")
    parser.add_argument("--n_per_group", type=int, default=5, help="Number of observations per group.")
    parser.add_argument("--p", type=int, default=1, help="Dimension of covariates.")
    parser.add_argument("--M_samples", type=int, default=1000, help="Number of samples from the posterior.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose output.")

    parser.add_argument("--one_scenario", action="store_true", help="Whether to run a single scenario.")
    parser.add_argument("--eta", type=float, default=1.0, help="Learning rate for running a single scenario.")

    parser.add_argument("--min_learning_rate", type=float, default=0.01, help="Minimum learning rate.")
    parser.add_argument("--max_learning_rate", type=float, default=100.0, help="Maximum learning rate.")
    parser.add_argument("--n_learning_rates", type=int, default=20, help="Number of learning rates.")
    args = parser.parse_args()

    if args.one_scenario:
        results = run_simulation_one_scenario(
            n_sim=args.n_sim,
            J=args.J,
            n_per_group=args.n_per_group,
            p=args.p,
            eta=args.eta,
            M_samples=args.M_samples,
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
            J=args.J,
            n_per_group=args.n_per_group,
            p=args.p,
            eta=args.eta,
            M_samples=args.M_samples,
            alpha=args.alpha,
            seed=args.seed,
            verbose=args.verbose,
        )
        # output for tex plots (coordinates are (eta, value))
        os.makedirs("lmm_output", exist_ok=True)
        with open("lmm_output/coverage_output.tex", "w", encoding="utf-8") as f:
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
        with open("lmm_output/width_output.tex", "w", encoding="utf-8") as f:
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
        with open("lmm_output/bias_output.tex", "w", encoding="utf-8") as f:
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
        with open("lmm_output/bias_sd_output.tex", "w", encoding="utf-8") as f:
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
        print(
            "Results written to lmm_output/coverage_output.tex, lmm_output/width_output.tex, lmm_output/bias_output.tex, and lmm_output/bias_sd_output.tex"  # noqa: E501
        )
