import numpy as np
import pytest

from kepler_lightcurves.fold import local_coordinate, phase_fold, validate_ephemeris
from kepler_lightcurves.schemas import LightCurveError


def test_epoch_alignment_wrapping_and_epoch_sign():
    phase = phase_fold(np.array([-5.0, 0.0, 5.0, 9.999]), 10.0, -5.0)
    assert phase[:3].tolist() == pytest.approx([0.0, -0.5, 0.0])
    assert -0.5 <= phase[-1] < 0.5
    validate_ephemeris(10.0, -5.0, 3.0)


def test_local_duration_coordinate_and_invalid_inputs():
    assert local_coordinate(np.array([0.01]), 10, 24)[0] == pytest.approx(0.1)
    with pytest.raises(LightCurveError, match="invalid_period"):
        validate_ephemeris(0, 1, 2)
    with pytest.raises(LightCurveError, match="invalid_epoch"):
        validate_ephemeris(2, np.nan, 2)
