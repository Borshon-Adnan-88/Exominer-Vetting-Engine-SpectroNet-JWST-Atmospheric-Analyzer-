"""Shared Stage 2 data structures and errors."""

from dataclasses import dataclass

import numpy as np


class LightCurveError(ValueError):
    """Raised when an object is technically unusable."""


@dataclass(frozen=True)
class QuarterData:
    kepid: int
    quarter: int
    data_release: int
    time: np.ndarray
    pdcsap_flux: np.ndarray
    pdcsap_flux_err: np.ndarray
    sap_flux: np.ndarray
    quality: np.ndarray
    cadenceno: np.ndarray
    actual_cadence_seconds: float
    filename: str
    sha256: str


@dataclass(frozen=True)
class BinnedView:
    flux: np.ndarray
    observed_mask: np.ndarray
    count: np.ndarray
    centers: np.ndarray
    observed_fraction: float
    longest_missing_run_fraction: float
