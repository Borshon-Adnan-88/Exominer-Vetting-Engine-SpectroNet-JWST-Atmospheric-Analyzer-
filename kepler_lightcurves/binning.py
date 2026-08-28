"""Deterministic median binning and auditable interpolation."""

import numpy as np

from .schemas import BinnedView, LightCurveError


def _longest_missing(mask: np.ndarray, circular: bool) -> int:
    missing = ~mask
    if not missing.any():
        return 0
    values = np.concatenate([missing, missing]) if circular else missing
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return min(longest, len(mask))


def _fill(values: np.ndarray, observed: np.ndarray, circular: bool) -> np.ndarray:
    if not observed.any():
        raise LightCurveError("view has no observed bins")
    x = np.arange(len(values), dtype=float)
    known = x[observed]
    known_values = values[observed]
    if circular and len(known) > 1:
        xp = np.concatenate([known - len(values), known, known + len(values)])
        fp = np.tile(known_values, 3)
        return np.interp(x, xp, fp)
    return np.interp(x, known, known_values)


def median_bin(coordinate: np.ndarray, flux: np.ndarray, minimum: float, maximum: float, bins: int, *, circular: bool) -> BinnedView:
    coordinate = np.asarray(coordinate, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)
    valid = np.isfinite(coordinate) & np.isfinite(flux) & (coordinate >= minimum) & (coordinate < maximum)
    coordinate, flux = coordinate[valid], flux[valid]
    edges = np.linspace(minimum, maximum, bins + 1)
    indices = np.floor((coordinate - minimum) / (maximum - minimum) * bins).astype(int)
    output = np.full(bins, np.nan)
    count = np.bincount(indices, minlength=bins).astype(np.int32)
    for index in np.flatnonzero(count):
        output[index] = np.median(flux[indices == index])
    observed = count > 0
    filled = _fill(output, observed, circular)
    centers = (edges[:-1] + edges[1:]) / 2
    return BinnedView(
        flux=filled.astype(np.float32), observed_mask=observed,
        count=count, centers=centers.astype(np.float32),
        observed_fraction=float(observed.mean()),
        longest_missing_run_fraction=float(_longest_missing(observed, circular) / bins),
    )
