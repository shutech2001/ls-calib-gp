from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray

from composite_gaussian_pairwise import CompositeGaussianPairwise


def generate_coords(k: int, coord_max: float, seed: int) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    coords = np.sort(rng.uniform(0.0, coord_max, size=k))
    return coords


def run_simulation_one_scenario(
    n_sim: int = 200,
    n_rep: int = 50,
    k: int = 20,
    coord_max: float = 20.0,
    mu_true: float = 0.0,
    tau: float = 1.0,
    omega: float = 3.0,
    nugget: float = 1e-6,
    eta: float = 1.0,
    prior_mean: float = 0.0,
    prior_var: float = 100.0,
    n_draws: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    coords: Optional[NDArray[np.float64]] = None,
    do_full: bool = True,
    verbose: bool = True,
) -> Dict:
    if coords is None:
        coords = generate_coords(k, coord_max, seed)

    model = CompositeGaussianPairwise(
        k=k,
        coord_max=coord_max,
        tau=tau,
        omega=omega,
        nugget=nugget,
        mu_true=mu_true,
        coords=coords,
    )

    coverage_uncalib = np.zeros(n_sim)
    coverage_calib = np.zeros(n_sim)
    coverage_full = np.zeros(n_sim)

    width_uncalib = np.zeros(n_sim)
    width_calib = np.zeros(n_sim)
    width_full = np.zeros(n_sim)

    bias_uncalib = np.zeros(n_sim)
    bias_calib = np.zeros(n_sim)
    bias_full = np.zeros(n_sim)

    for sim in range(n_sim):
        if verbose:
            logging.info(f"Running simulation {sim + 1} of {n_sim}...")

        data_seed = seed + sim * 1000
        y = model.generate_data(n_rep=n_rep, seed=data_seed)

        comp_post = model.posterior_composite(y, eta=eta, prior_mean=prior_mean, prior_var=prior_var)
        full_post = model.posterior_full(y, prior_mean=prior_mean, prior_var=prior_var)

        rng = np.random.default_rng(data_seed + 1)
        comp_samples = rng.normal(comp_post.mean, np.sqrt(comp_post.var), size=n_draws)
        full_samples = rng.normal(full_post.mean, np.sqrt(full_post.var), size=n_draws)

        mu_loc = model.one_step_newton_full(comp_post.mean, y)
        calib_samples, _ = model.calibrate_samples(comp_samples, comp_post.mean, mu_loc, full_post.var)

        uncalib_lower = np.quantile(comp_samples, alpha / 2)
        uncalib_upper = np.quantile(comp_samples, 1 - alpha / 2)

        calib_lower = np.quantile(calib_samples, alpha / 2)
        calib_upper = np.quantile(calib_samples, 1 - alpha / 2)

        if do_full:
            full_lower = np.quantile(full_samples, alpha / 2)
            full_upper = np.quantile(full_samples, 1 - alpha / 2)
        else:
            full_lower = np.float64(np.nan)
            full_upper = np.float64(np.nan)

        coverage_uncalib[sim] = uncalib_lower <= mu_true <= uncalib_upper
        coverage_calib[sim] = calib_lower <= mu_true <= calib_upper
        if do_full:
            coverage_full[sim] = full_lower <= mu_true <= full_upper
        else:
            coverage_full[sim] = np.nan

        width_uncalib[sim] = uncalib_upper - uncalib_lower
        width_calib[sim] = calib_upper - calib_lower
        if do_full:
            width_full[sim] = full_upper - full_lower
        else:
            width_full[sim] = np.nan

        bias_uncalib[sim] = comp_post.mean - mu_true
        bias_calib[sim] = float(np.mean(calib_samples)) - mu_true
        if do_full:
            bias_full[sim] = full_post.mean - mu_true
        else:
            bias_full[sim] = np.nan

    return {
        "coverage_uncalibrated": np.mean(coverage_uncalib),
        "coverage_calibrated": np.mean(coverage_calib),
        "coverage_full": np.mean(coverage_full),
        "width_uncalibrated": np.mean(width_uncalib),
        "width_calibrated": np.mean(width_calib),
        "width_full": np.mean(width_full),
        "bias_uncalibrated": np.mean(bias_uncalib),
        "bias_calibrated": np.mean(bias_calib),
        "bias_full": np.mean(bias_full),
        "bias_sd_uncalibrated": np.std(bias_uncalib, ddof=1),
        "bias_sd_calibrated": np.std(bias_calib, ddof=1),
        "bias_sd_full": np.std(bias_full, ddof=1),
    }


def run_simulation_varying_learning_rate(
    min_learning_rate: float = 0.01,
    max_learning_rate: float = 100.0,
    n_learning_rates: int = 20,
    n_sim: int = 200,
    n_rep: int = 50,
    k: int = 20,
    coord_max: float = 20.0,
    mu_true: float = 0.0,
    tau: float = 1.0,
    omega: float = 3.0,
    nugget: float = 1e-6,
    prior_mean: float = 0.0,
    prior_var: float = 100.0,
    n_draws: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
    coord_seed: Optional[int] = None,
    verbose: bool = True,
) -> Dict:
    etas = np.logspace(np.log10(min_learning_rate), np.log10(max_learning_rate), n_learning_rates)
    coord_seed = seed if coord_seed is None else coord_seed
    coords = generate_coords(k, coord_max, coord_seed)

    coverage_uncalib = np.zeros(n_learning_rates)
    coverage_calib = np.zeros(n_learning_rates)
    width_uncalib = np.zeros(n_learning_rates)
    width_calib = np.zeros(n_learning_rates)
    bias_uncalib = np.zeros(n_learning_rates)
    bias_calib = np.zeros(n_learning_rates)
    bias_sd_uncalib = np.zeros(n_learning_rates)
    bias_sd_calib = np.zeros(n_learning_rates)

    coverage_full = 0.0
    width_full = 0.0
    bias_full = 0.0
    bias_sd_full = 0.0

    for i, eta in enumerate(etas):
        if verbose:
            logging.info(f"Running simulation with learning rate {eta}...")
        if i == 0:
            results = run_simulation_one_scenario(
                n_sim=n_sim,
                n_rep=n_rep,
                k=k,
                coord_max=coord_max,
                mu_true=mu_true,
                tau=tau,
                omega=omega,
                nugget=nugget,
                eta=eta,
                prior_mean=prior_mean,
                prior_var=prior_var,
                n_draws=n_draws,
                alpha=alpha,
                seed=seed,
                coords=coords,
                do_full=True,
                verbose=verbose,
            )
            coverage_full = results["coverage_full"]
            width_full = results["width_full"]
            bias_full = results["bias_full"]
            bias_sd_full = results["bias_sd_full"]
        else:
            results = run_simulation_one_scenario(
                n_sim=n_sim,
                n_rep=n_rep,
                k=k,
                coord_max=coord_max,
                mu_true=mu_true,
                tau=tau,
                omega=omega,
                nugget=nugget,
                eta=eta,
                prior_mean=prior_mean,
                prior_var=prior_var,
                n_draws=n_draws,
                alpha=alpha,
                seed=seed,
                coords=coords,
                do_full=False,
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
        "coverage_full": coverage_full,
        "width_uncalibrated": width_uncalib,
        "width_calibrated": width_calib,
        "width_full": width_full,
        "bias_uncalibrated": bias_uncalib,
        "bias_calibrated": bias_calib,
        "bias_full": bias_full,
        "bias_sd_uncalibrated": bias_sd_uncalib,
        "bias_sd_calibrated": bias_sd_calib,
        "bias_sd_full": bias_sd_full,
    }


def print_results(results: Dict) -> None:
    print("\n" + "=" * 80)
    print("SIMULATION RESULTS - COMPOSITE GAUSSIAN PAIRWISE")
    print("=" * 80)
    print(f"\n{'Metric':<30} {'Uncalibrated':<12} {'Calibrated':<12} {'Full':<12}")
    print("-" * 80)
    print(
        f"{'Coverage Probability':<30} "
        f"{results['coverage_uncalibrated']:.4f}{'':<6} "
        f"{results['coverage_calibrated']:.4f}{'':<6} "
        f"{results['coverage_full']:.4f}"
    )
    print(
        f"{'Average Interval Width':<30} "
        f"{results['width_uncalibrated']:.4f}{'':<6} "
        f"{results['width_calibrated']:.4f}{'':<6} "
        f"{results['width_full']:.4f}"
    )
    print(
        f"{'Mean Bias':<30} "
        f"{results['bias_uncalibrated']:.4f}{'':<6} "
        f"{results['bias_calibrated']:.4f}{'':<6} "
        f"{results['bias_full']:.4f}"
    )
    print(
        f"{'Bias SD':<30} "
        f"{results['bias_sd_uncalibrated']:.4f}{'':<6} "
        f"{results['bias_sd_calibrated']:.4f}{'':<6} "
        f"{results['bias_sd_full']:.4f}"
    )
    print("=" * 80)
    print("\nMethods:")
    print("  - Uncalibrated             : General Bayes with composite likelihood")
    print("  - Location-Scale Calibrated: One-step Newton location + scale to full posterior")
    print("  - Full                      : Bayes with full likelihood")
    print("=" * 80)


def parse_mu(mu_str: str) -> float:
    try:
        return float(mu_str.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError("mu_true must be a number (e.g., 0.0)") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Composite likelihood general Bayes with location-scale calibration to full likelihood."
    )
    parser.add_argument("--n_sim", "--N", type=int, default=200, help="Number of simulations (default: 200).")
    parser.add_argument("--n_rep", type=int, default=50, help="Number of replicates per dataset (default: 50).")
    parser.add_argument("--k", type=int, default=20, help="Number of locations (default: 20).")
    parser.add_argument("--coord_max", type=float, default=20.0, help="Maximum coordinate value (default: 20.0).")
    parser.add_argument("--mu_true", type=parse_mu, default=0.0, help="True mean (default: 0.0).")
    parser.add_argument("--tau", type=float, default=1.0, help="Sill parameter (default: 1.0).")
    parser.add_argument("--omega", type=float, default=3.0, help="Range parameter (default: 3.0).")
    parser.add_argument("--nugget", type=float, default=1e-6, help="Nugget for stability (default: 1e-6).")
    parser.add_argument("--prior_mean", type=float, default=0.0, help="Prior mean (default: 0.0).")
    parser.add_argument("--prior_var", type=float, default=100.0, help="Prior variance (default: 100.0).")
    parser.add_argument("--n_draws", type=int, default=2000, help="Posterior draws (default: 2000).")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--coord_seed", type=int, default=None, help="Seed for locations (default: seed).")
    parser.add_argument("--verbose", type=bool, default=False, help="Whether to print verbose output.")

    parser.add_argument("--one_scenario", "--OS", type=bool, default=False, help="Whether to run a single scenario.")
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
            n_rep=args.n_rep,
            k=args.k,
            coord_max=args.coord_max,
            mu_true=args.mu_true,
            tau=args.tau,
            omega=args.omega,
            nugget=args.nugget,
            eta=args.eta,
            prior_mean=args.prior_mean,
            prior_var=args.prior_var,
            n_draws=args.n_draws,
            alpha=args.alpha,
            seed=args.seed,
            coords=None,
            do_full=True,
            verbose=args.verbose,
        )
        print_results(results)
    else:
        results = run_simulation_varying_learning_rate(
            min_learning_rate=args.min_learning_rate,
            max_learning_rate=args.max_learning_rate,
            n_learning_rates=args.n_learning_rates,
            n_sim=args.n_sim,
            n_rep=args.n_rep,
            k=args.k,
            coord_max=args.coord_max,
            mu_true=args.mu_true,
            tau=args.tau,
            omega=args.omega,
            nugget=args.nugget,
            prior_mean=args.prior_mean,
            prior_var=args.prior_var,
            n_draws=args.n_draws,
            alpha=args.alpha,
            seed=args.seed,
            coord_seed=args.coord_seed,
            verbose=args.verbose,
        )

        os.makedirs("composite_gaussian_pairwise_output", exist_ok=True)
        with open("composite_gaussian_pairwise_output/coverage_output.tex", "w", encoding="utf-8") as f:
            f.write("full:\n")
            out_full = (
                "{"
                + " ".join(
                    f"({xi},{yi})"
                    for xi, yi in zip(results["etas"], results["coverage_full"] * np.ones(len(results["etas"])))
                )
                + "}"
            )
            f.write(out_full + "\n")
            f.write("uncalibrated:\n")
            out_uncalib = (
                "{"
                + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["coverage_uncalibrated"]))
                + "}"
            )
            f.write(out_uncalib + "\n")
            f.write("location-scale calibrated:\n")
            out_calib = (
                "{" + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["coverage_calibrated"])) + "}"
            )
            f.write(out_calib + "\n")

        with open("composite_gaussian_pairwise_output/width_output.tex", "w", encoding="utf-8") as f:
            f.write("full:\n")
            out_full = (
                "{"
                + " ".join(
                    f"({xi},{yi})"
                    for xi, yi in zip(results["etas"], results["width_full"] * np.ones(len(results["etas"])))
                )
                + "}"
            )
            f.write(out_full + "\n")
            f.write("uncalibrated:\n")
            out_uncalib = (
                "{" + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["width_uncalibrated"])) + "}"
            )
            f.write(out_uncalib + "\n")
            f.write("location-scale calibrated:\n")
            out_calib = (
                "{" + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["width_calibrated"])) + "}"
            )
            f.write(out_calib + "\n")

        with open("composite_gaussian_pairwise_output/bias_output.tex", "w", encoding="utf-8") as f:
            f.write("full:\n")
            out_full = (
                "{"
                + " ".join(
                    f"({xi},{yi})"
                    for xi, yi in zip(results["etas"], results["bias_full"] * np.ones(len(results["etas"])))
                )
                + "}"
            )
            f.write(out_full + "\n")
            f.write("uncalibrated:\n")
            out_uncalib = (
                "{" + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["bias_uncalibrated"])) + "}"
            )
            f.write(out_uncalib + "\n")
            f.write("location-scale calibrated:\n")
            out_calib = (
                "{" + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["bias_calibrated"])) + "}"
            )
            f.write(out_calib + "\n")

        with open("composite_gaussian_pairwise_output/bias_sd_output.tex", "w", encoding="utf-8") as f:
            f.write("full:\n")
            out_full = (
                "{"
                + " ".join(
                    f"({xi},{yi})"
                    for xi, yi in zip(results["etas"], results["bias_sd_full"] * np.ones(len(results["etas"])))
                )
                + "}"
            )
            f.write(out_full + "\n")
            f.write("uncalibrated:\n")
            out_uncalib = (
                "{" + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["bias_sd_uncalibrated"])) + "}"
            )
            f.write(out_uncalib + "\n")
            f.write("location-scale calibrated:\n")
            out_calib = (
                "{" + " ".join(f"({xi},{yi})" for xi, yi in zip(results["etas"], results["bias_sd_calibrated"])) + "}"
            )
            f.write(out_calib + "\n")

        logging.info(
            "Results written to composite_gaussian_pairwise_output/coverage_output.tex, composite_gaussian_pairwise_output/width_output.tex, composite_gaussian_pairwise_output/bias_output.tex, and composite_gaussian_pairwise_output/bias_sd_output.tex"  # noqa: E501
        )
