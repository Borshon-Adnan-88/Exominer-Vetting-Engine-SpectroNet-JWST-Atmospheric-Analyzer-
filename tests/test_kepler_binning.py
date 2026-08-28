import numpy as np
import pytest

from kepler_lightcurves.binning import median_bin


def test_median_binning_counts_masks_and_shape():
    view = median_bin(np.array([-0.4, -0.4, 0.1]), np.array([1.0, 0.8, 1.1]), -0.5, 0.5, 10, circular=True)
    assert view.flux.shape == view.count.shape == view.observed_mask.shape == (10,)
    assert view.count.sum() == 3
    assert view.flux[0] == pytest.approx(0.9)
    assert np.isfinite(view.flux).all()


def test_input_order_does_not_change_bins():
    x = np.array([-0.4, 0.1, -0.4]); y = np.array([1.0, 1.1, 0.8])
    assert np.array_equal(median_bin(x, y, -0.5, 0.5, 10, circular=False).flux, median_bin(x[::-1], y[::-1], -0.5, 0.5, 10, circular=False).flux)
