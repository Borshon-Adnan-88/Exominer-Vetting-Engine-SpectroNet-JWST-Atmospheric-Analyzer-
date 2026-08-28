"""Explicit BKJD phase-folding conventions."""

import numpy as np

from .schemas import LightCurveError


def validate_ephemeris(period_days: float, epoch_bkjd: float, duration_hours: float) -> None:
    if not np.isfinite(period_days) or period_days <= 0:
        raise LightCurveError("invalid_period")
    if not np.isfinite(epoch_bkjd):
        raise LightCurveError("invalid_epoch")
    if not np.isfinite(duration_hours) or duration_hours <= 0:
        raise LightCurveError("invalid_duration")
    if duration_hours >= period_days * 24:
        raise LightCurveError("duration_not_shorter_than_period")


def phase_fold(time_bkjd: np.ndarray, period_days: float, epoch_bkjd: float) -> np.ndarray:
    if not np.isfinite(period_days) or period_days <= 0 or not np.isfinite(epoch_bkjd):
        raise LightCurveError("impossible phase-folding inputs")
    time = np.asarray(time_bkjd, dtype=np.float64)
    return ((time - epoch_bkjd + 0.5 * period_days) % period_days) / period_days - 0.5


def local_coordinate(phase: np.ndarray, period_days: float, duration_hours: float) -> np.ndarray:
    return np.asarray(phase) * period_days * 24.0 / duration_hours
