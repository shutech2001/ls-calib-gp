from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np

from export_tex import DEFAULT_OUTPUT, export_tex
from model import (
    ETAS,
    SEED,
    C,
    FloatArray,
    G,
    M,
    N,
    generate_dataset,
    make_pieces,
    population_target,
)
from model import LAM as RIDGE_PENALTY
from sampler import EnergyGibbs

METHODS = ("wald", "raw", "cal")
SUMMARY_COLUMNS = (
    "eta",
    "method",
    "coverage",
    "coverage_mcse",
    "mean_width",
    "mean_width_mcse",
    "mean_bias",
    "sd_bias",
    "mean_bias_mcse",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Compute budget and output destination for the experiment.

    Attributes:
        replications (int): Number of independently generated data sets.
        kept (int): Retained Gibbs draws per chain.
        burn (int): Discarded Gibbs draws per chain.
        pilot (bool): Use two data sets and three learning rates.
        output (Path): Destination for the summary and TeX figure.
    """

    replications: int = 200
    kept: int = 10000
    burn: int = 1000
    pilot: bool = False
    output: Path = DEFAULT_OUTPUT


@dataclass(frozen=True)
class IntervalResult:
    """One method's interval and estimation error for a data set and rate.

    Attributes:
        eta (float): Learning rate.
        method (str): Interval method identifier.
        coverage (bool): Whether the interval contains the population target.
        width (float): Interval width.
        bias (float): Point estimate minus the population target.
    """

    eta: float
    method: str
    coverage: bool
    width: float
    bias: float


def replication_results(
    sampler: EnergyGibbs,
    config: ExperimentConfig,
    seed: np.random.SeedSequence,
    etas: FloatArray,
    target: float,
) -> list[IntervalResult]:
    """Compute the three interval methods for one common data set.

    Args:
        sampler (EnergyGibbs): Compiled Gibbs sampler.
        config (ExperimentConfig): Retained and discarded draw counts.
        seed (np.random.SeedSequence): Data set and chain random stream.
        etas (FloatArray): Learning rates to evaluate.
        target (float): Penalized population target.

    Returns:
        list[IntervalResult]: Interval outcomes for each rate and method.
    """
    rng = np.random.default_rng(seed)
    x, y = generate_dataset(rng)
    pieces = make_pieces(x, y)
    beta = pieces.mode
    residuals = y - x * beta
    scores = -(x * np.clip(residuals, -C, C)).sum(axis=1)
    jlam = float(np.sum(x * x * (np.abs(residuals) < C)) / N + RIDGE_PENALTY)
    khat = float(np.sum((scores - scores.mean()) ** 2) / (M * (G - 1)))
    vtarget = khat / jlam**2
    standard_error = math.sqrt(vtarget / N)
    wald = (
        beta - 1.959963984540054 * standard_error,
        beta + 1.959963984540054 * standard_error,
    )
    results = []
    for rate in etas:
        eta = float(rate)
        sd = 1 / math.sqrt(eta * N * jlam)
        chain_seeds = [int(rng.integers(0, 2**63, dtype=np.uint64)) for _ in range(2)]
        chains = np.array(
            [
                sampler.sample(
                    pieces,
                    eta,
                    config.burn,
                    config.kept,
                    chain_seed,
                    beta + sign * 4 * sd,
                )
                for chain_seed, sign in zip(chain_seeds, (-1, 1), strict=True)
            ]
        )
        draws = chains.reshape(-1)
        posterior_mean = float(draws.mean())
        posterior_variance = float(draws.var())
        omega = math.sqrt(vtarget / (N * posterior_variance))
        calibrated = beta + omega * (draws - posterior_mean)
        raw_ci = np.quantile(draws, [0.025, 0.975])
        cal_ci = np.quantile(calibrated, [0.025, 0.975])
        intervals = (
            ("wald", beta, wald),
            ("raw", posterior_mean, (float(raw_ci[0]), float(raw_ci[1]))),
            ("cal", float(calibrated.mean()), (float(cal_ci[0]), float(cal_ci[1]))),
        )
        for method, center, (lower, upper) in intervals:
            results.append(
                IntervalResult(
                    eta,
                    method,
                    lower <= target <= upper,
                    upper - lower,
                    center - target,
                )
            )
    return results


def summarize(
    results: list[IntervalResult], replications: int
) -> list[dict[str, float | str]]:
    """Aggregate coverage, interval widths, bias, and Monte Carlo errors.

    Args:
        results (list[IntervalResult]): Per-data-set interval outcomes.
        replications (int): Number of independent data sets.

    Returns:
        list[dict[str, float | str]]: CSV rows ordered by rate and method.
    """
    summary: list[dict[str, float | str]] = []
    for eta in sorted({result.eta for result in results}):
        for method in METHODS:
            part = [
                result
                for result in results
                if result.eta == eta and result.method == method
            ]
            coverage = float(np.mean([result.coverage for result in part]))
            widths = np.array([result.width for result in part])
            biases = np.array([result.bias for result in part])
            sd_bias = float(biases.std(ddof=1))
            summary.append(
                {
                    "eta": eta,
                    "method": method,
                    "coverage": coverage,
                    "coverage_mcse": math.sqrt(
                        coverage * (1 - coverage) / replications
                    ),
                    "mean_width": float(widths.mean()),
                    "mean_width_mcse": float(widths.std(ddof=1))
                    / math.sqrt(replications),
                    "mean_bias": float(biases.mean()),
                    "sd_bias": sd_bias,
                    "mean_bias_mcse": sd_bias / math.sqrt(replications),
                }
            )
    return summary


def run(config: ExperimentConfig) -> tuple[Path, Path]:
    """Run the study and write its two deliverables.

    Args:
        config (ExperimentConfig): Compute budget and output destination.

    Returns:
        tuple[Path, Path]: Summary CSV and four-panel TeX figure paths.
    """
    started = monotonic()
    sampler = EnergyGibbs()
    target, _ = population_target()
    replications = 2 if config.pilot else config.replications
    etas = ETAS[[0, 10, 19]] if config.pilot else ETAS
    seeds = np.random.SeedSequence(SEED).spawn(replications + 1)
    results = []
    for index in range(replications):
        results.extend(replication_results(sampler, config, seeds[index], etas, target))
        if (index + 1) % 10 == 0 or config.pilot:
            print(
                f"Completed {index + 1}/{replications} data sets "
                f"in {monotonic() - started:.2f}s",
                flush=True,
            )
    config.output.mkdir(parents=True, exist_ok=True)
    summary_path = config.output / "summary_results.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summarize(results, replications))
    return summary_path, export_tex(config.output)


def main() -> None:
    """Parse command-line options and run the experiment.

    Args:
        None.

    Returns:
        None: Writes the summary and figure and prints their paths.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--replications", type=int, default=200)
    parser.add_argument("--kept", type=int, default=10000)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.replications < 2 or args.kept < 2 or args.burn < 0:
        parser.error(
            "replications and kept must be at least 2; burn must be nonnegative"
        )
    config = ExperimentConfig(
        replications=args.replications,
        kept=args.kept,
        burn=args.burn,
        pilot=args.pilot,
        output=args.output,
    )
    for path in run(config):
        print(path)


if __name__ == "__main__":
    main()
