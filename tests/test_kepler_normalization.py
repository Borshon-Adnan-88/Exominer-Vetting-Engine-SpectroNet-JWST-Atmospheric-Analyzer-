import numpy as np
import pytest

from kepler_lightcurves.normalize import normalize_quarter
from kepler_lightcurves.schemas import LightCurveError


def test_normalization_preserves_fractional_depth_and_scales_error():
    flux, error, median = normalize_quarter(np.array([100.0, 100.0, 99.0]), np.ones(3))
    assert median == 100
    assert np.median(flux) == 1
    assert flux[-1] == pytest.approx(0.99)
    assert np.allclose(error, 0.01)


def test_invalid_median_fails():
    with pytest.raises(LightCurveError):
        normalize_quarter(np.array([-1.0, 0.0]), np.ones(2))
