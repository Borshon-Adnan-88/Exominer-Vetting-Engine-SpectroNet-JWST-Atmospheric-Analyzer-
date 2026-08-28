"""Per-quarter normalization with no population-fitted statistics."""

import numpy as np

from .schemas import LightCurveError


def normalize_quarter(flux: np.ndarray, flux_err: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    flux = np.asarray(flux, dtype=np.float64)
    flux_err = np.asarray(flux_err, dtype=np.float64)
    if flux.shape != flux_err.shape or flux.ndim != 1:
        raise LightCurveError("flux and flux_err must be matching one-dimensional arrays")
    median = float(np.median(flux))
    if not np.isfinite(median) or median <= 0:
        raise LightCurveError("quarter median must be finite and positive")
    return flux / median, flux_err / median, median
