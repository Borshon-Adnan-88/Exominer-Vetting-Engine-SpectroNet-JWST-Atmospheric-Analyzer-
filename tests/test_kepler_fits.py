import pytest

from kepler_lightcurves.fits import embedded_checksum_status, read_kepler_quarter
from kepler_lightcurves.schemas import LightCurveError


def test_programmatic_fits_identity_and_actual_cadence(kepler_fits_factory):
    path = kepler_fits_factory(kepid=42, quarter=7)
    quarter = read_kepler_quarter(path, 42)
    assert quarter.kepid == 42 and quarter.quarter == 7 and quarter.data_release == 25
    assert quarter.actual_cadence_seconds == pytest.approx(1765.47, rel=1e-3)


def test_fits_identity_and_release_mismatch_fail(kepler_fits_factory):
    path = kepler_fits_factory(kepid=42)
    with pytest.raises(LightCurveError, match="identity"):
        read_kepler_quarter(path, 43)
    old = kepler_fits_factory(kepid=44, data_release=24, filename="kplr000000044-old_llc.fits")
    with pytest.raises(LightCurveError, match="data_release"):
        read_kepler_quarter(old, 44)


def test_programmatic_fixture_has_explicit_embedded_checksum_status(kepler_fits_factory):
    path = kepler_fits_factory(kepid=42)
    status = embedded_checksum_status(path)
    assert isinstance(status["hdu_results"], list)
    assert status["embedded_checksums_supplied"] is True
    assert status["embedded_checksums_valid"] is True
