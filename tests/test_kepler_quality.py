import numpy as np

from kepler_lightcurves.quality import quality_keep_mask, quality_summary


def test_frozen_bitmask_is_applied_explicitly():
    quality = np.array([0, 1, 128, 2, 1130799], dtype=np.int64)
    keep = quality_keep_mask(quality)
    assert keep.tolist() == [True, False, True, False, False]
    assert quality_summary(quality) == {"cadences_total": 5, "cadences_kept": 2, "cadences_rejected": 3}
