"""Explicit Kepler cadence quality masking."""

import numpy as np

from . import QUALITY_MASK


def quality_keep_mask(quality: np.ndarray, bitmask: int = QUALITY_MASK) -> np.ndarray:
    values = np.asarray(quality)
    if values.ndim != 1:
        raise ValueError("quality must be one-dimensional")
    return (values.astype(np.int64) & int(bitmask)) == 0


def quality_summary(quality: np.ndarray, bitmask: int = QUALITY_MASK) -> dict[str, int]:
    keep = quality_keep_mask(quality, bitmask)
    return {"cadences_total": int(keep.size), "cadences_kept": int(keep.sum()), "cadences_rejected": int((~keep).sum())}
