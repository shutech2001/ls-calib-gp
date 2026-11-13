from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import ndtr

FloatArray = NDArray[np.float64]

G = 100
M = 5
N = G * M
C = 1.0
LAM = 0.5
TRUE_BETA = 2.0
WHITEN = np.eye(M) + (1 / math.sqrt(1 + 2 * M) - 1) * np.ones((M, M)) / M
ETAS = np.logspace(-2, 2, 20)
SEED = 42


def population_target() -> tuple[float, float]:
    """Compute the penalized population target under the contamination model.

    Args:
        None.

    Returns:
        tuple[float, float]: Target coefficient and population score at the target.
    """
    patterns = np.array(list(itertools.product([0, 1], repeat=M)))
    probability = np.prod(np.where(patterns == 1, 0.1, 0.9), axis=1)
    vx = np.diag(WHITEN @ WHITEN.T)
    noise_variance = 1 + 100 * patterns @ (WHITEN**2).T

    def score(beta: float) -> float:
        """Evaluate the penalized population estimating equation.

        Args:
            beta (float): Candidate coefficient.

        Returns:
            float: Expected Huber score plus the ridge penalty derivative.
        """
        difference = TRUE_BETA - beta
        sd = np.sqrt(difference * difference * vx[None, :] + noise_variance)
        band = 2 * ndtr(C / sd) - 1
        return float(
            -difference * np.sum(probability[:, None] * vx[None, :] * band) / M
            + LAM * beta
        )

    root = float(brentq(score, 0, TRUE_BETA, xtol=1e-14, rtol=1e-14))
    return root, score(root)


def huber(residual: FloatArray) -> FloatArray:
    """Evaluate the Huber loss elementwise.

    Args:
        residual (FloatArray): Residual values.

    Returns:
        FloatArray: Loss values with the same shape as the residuals.
    """
    absolute = np.abs(residual)
    return np.where(absolute <= C, 0.5 * residual**2, C * absolute - 0.5 * C**2)


@dataclass
class Pieces:
    """Continuous quadratic loss pieces measured relative to the global mode.

    Attributes:
        breaks (FloatArray): Sorted finite boundaries between quadratic pieces.
        heights (FloatArray): Loss relative to the mode at each boundary.
        slopes (FloatArray): Loss derivative at each boundary.
        curvature (FloatArray): Second derivative within each quadratic piece.
        mode (float): Minimizer of the penalized sample loss.
        mode_piece (int): Index of the quadratic piece containing the mode.
    """

    breaks: FloatArray
    heights: FloatArray
    slopes: FloatArray
    curvature: FloatArray
    mode: float
    mode_piece: int


def make_pieces(x: FloatArray, y: FloatArray) -> Pieces:
    """Construct quadratic loss pieces by integrating outwards from the mode.

    Args:
        x (FloatArray): Whitened covariates, with one row per group.
        y (FloatArray): Whitened responses with the same shape as the covariates.

    Returns:
        Pieces: Contiguous arrays describing the penalized sample loss.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    nonzero = x != 0
    xx, yy = x[nonzero], y[nonzero]
    lower = np.minimum((yy - C) / xx, (yy + C) / xx)
    upper = np.maximum((yy - C) / xx, (yy + C) / xx)
    breakpoints = np.concatenate([lower, upper])
    order = np.argsort(breakpoints, kind="stable")
    breaks, unique_start = np.unique(breakpoints[order], return_index=True)
    delta_curvature = np.add.reduceat(
        np.concatenate([xx * xx, -xx * xx])[order], unique_start
    )
    delta_linear = np.add.reduceat(
        np.concatenate([xx * yy - C * np.abs(xx), -xx * yy - C * np.abs(xx)])[order],
        unique_start,
    )
    curvature = np.asarray(
        np.r_[LAM * N, LAM * N + np.cumsum(delta_curvature, dtype=np.longdouble)],
        dtype=float,
    )
    initial_linear = C * np.abs(xx).sum()
    linear = np.asarray(
        np.r_[
            initial_linear,
            initial_linear + np.cumsum(delta_linear, dtype=np.longdouble),
        ],
        dtype=float,
    )

    def score(beta: float) -> float:
        """Evaluate the derivative of the penalized sample loss.

        Args:
            beta (float): Candidate coefficient.

        Returns:
            float: Sum of Huber score contributions and ridge derivative.
        """
        return float(-np.sum(x * np.clip(y - x * beta, -C, C)) + LAM * N * beta)

    mode = float(brentq(score, -100, 100, xtol=1e-13, rtol=1e-14))
    mode_piece = int(np.searchsorted(breaks, mode, side="right"))
    mode = float(linear[mode_piece] / curvature[mode_piece])
    heights = np.empty(len(breaks))
    slopes = np.empty(len(breaks))
    anchor, energy, slope = mode, 0.0, 0.0
    for index in range(mode_piece, len(breaks)):
        distance = breaks[index] - anchor
        energy += slope * distance + 0.5 * curvature[index] * distance**2
        slope += curvature[index] * distance
        heights[index], slopes[index] = energy, slope
        anchor = breaks[index]
    anchor, energy, slope = mode, 0.0, 0.0
    for index in range(mode_piece - 1, -1, -1):
        distance = anchor - breaks[index]
        energy += slope * distance + 0.5 * curvature[index + 1] * distance**2
        slope += curvature[index + 1] * distance
        heights[index], slopes[index] = energy, -slope
        anchor = breaks[index]
    return Pieces(
        np.ascontiguousarray(breaks),
        np.ascontiguousarray(heights),
        np.ascontiguousarray(slopes),
        np.ascontiguousarray(curvature),
        mode,
        mode_piece,
    )


def generate_dataset(rng: np.random.Generator) -> tuple[FloatArray, FloatArray]:
    """Generate and whiten grouped observations with contaminated errors.

    Args:
        rng (np.random.Generator): Random generator for all dataset draws.

    Returns:
        tuple[FloatArray, FloatArray]: Whitened covariates and responses.
    """
    x = rng.normal(size=(G, M))
    intercept = rng.normal(scale=math.sqrt(2), size=(G, 1))
    errors = rng.normal(size=(G, M))
    contamination = rng.random((G, M)) < 0.1
    errors += contamination * rng.normal(scale=10, size=(G, M))
    y = TRUE_BETA * x + intercept + errors
    return x @ WHITEN.T, y @ WHITEN.T
