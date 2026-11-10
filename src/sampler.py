from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

from model import FloatArray, Pieces


class EnergyGibbs:
    """Compile and expose the C++ implementation of the Gibbs sampler."""

    def __init__(self) -> None:
        """Compile the sampler when needed and configure its C interface.

        Args:
            None.

        Returns:
            None: The sampler is ready to generate draws.
        """
        source = Path(__file__).with_name("energy_gibbs.cpp")
        cache = Path(__file__).resolve().parents[1] / ".cache" / "lscalibgb"
        cache.mkdir(parents=True, exist_ok=True)
        binary = cache / "energy_gibbs.so"
        if not binary.exists() or binary.stat().st_mtime < source.stat().st_mtime:
            subprocess.run(
                [
                    "c++",
                    "-O3",
                    "-std=c++17",
                    "-shared",
                    "-fPIC",
                    str(source),
                    "-o",
                    str(binary),
                ],
                check=True,
            )
        self._library = ctypes.CDLL(str(binary))
        self._sample = self._library.sample_energy_gibbs
        array_type = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self._sample.argtypes = [
            ctypes.c_int,
            array_type,
            array_type,
            array_type,
            array_type,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_double,
            array_type,
        ]
        self._sample.restype = ctypes.c_int

    def sample(
        self,
        pieces: Pieces,
        eta: float,
        burn: int,
        kept: int,
        seed: int,
        initial: float,
    ) -> FloatArray:
        """Generate retained draws from one Gibbs chain.

        Args:
            pieces (Pieces): Continuous quadratic representation of the loss.
            eta (float): Positive learning rate of the generalized posterior.
            burn (int): Number of initial updates to discard.
            kept (int): Number of posterior draws to retain.
            seed (int): Random number generator initialization value.
            initial (float): Initial coefficient value for the chain.

        Returns:
            FloatArray: Retained posterior draws in sampling order.
        """
        output = np.empty(kept, dtype=np.float64)
        status = int(
            self._sample(
                len(pieces.breaks),
                pieces.breaks,
                pieces.heights,
                pieces.slopes,
                pieces.curvature,
                pieces.mode_piece,
                pieces.mode,
                eta,
                burn,
                kept,
                seed,
                initial,
                output,
            )
        )
        if status:
            raise RuntimeError(f"Gibbs sampling failed with status {status}")
        return output
