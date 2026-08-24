"""Deterministic synthetic transit profiles used by the prototype UI.

This module does not download, phase-fold, or classify observed light curves.
"""

from numbers import Integral, Real

import numpy as np


def _positive_finite(value: Real, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and greater than zero")
    return value


def _positive_integer(value: Integral, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def generate_folded_views(
    period: Real,
    duration_hours: Real,
    global_bins: Integral = 2000,
    local_bins: Integral = 200,
    seed: Integral | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate toy global and local transit-shaped relative-flux arrays.

    These arrays are synthetic illustrations, not phase-folded observations.
    Epoch is intentionally absent because it has no role in this generator; it
    will return when genuine Kepler/TESS light curves are phase-folded.
    """
    period = _positive_finite(period, "period")
    duration_hours = _positive_finite(duration_hours, "duration_hours")
    global_bins = _positive_integer(global_bins, "global_bins")
    local_bins = _positive_integer(local_bins, "local_bins")
    if duration_hours >= period * 24.0:
        raise ValueError("duration_hours must be shorter than the orbital period")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, Integral)):
        raise TypeError("seed must be an integer or None")

    rng = np.random.default_rng(None if seed is None else int(seed))
    phase = np.linspace(-0.5, 0.5, global_bins, endpoint=False)
    duration_phase = duration_hours / (period * 24.0)
    sigma = max(duration_phase / 3.5, 0.5 / global_bins)
    global_flux = 1.0 - 0.01 * np.exp(-0.5 * (phase / sigma) ** 2)
    global_flux += rng.normal(0.0, 0.001, global_bins)

    local_phase = np.linspace(-2.5, 2.5, local_bins)
    edge_softness = 0.12
    transit_window = 0.5 * (
        np.tanh((local_phase + 1.0) / edge_softness)
        - np.tanh((local_phase - 1.0) / edge_softness)
    )
    local_flux = 1.0 - 0.01 * transit_window
    local_flux += rng.normal(0.0, 0.0005, local_bins)

    return global_flux.astype(np.float32), local_flux.astype(np.float32)
