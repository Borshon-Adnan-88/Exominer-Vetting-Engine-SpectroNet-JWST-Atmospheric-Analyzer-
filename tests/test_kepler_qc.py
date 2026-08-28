import json

from kepler_lightcurves.fits import read_kepler_quarter
from kepler_lightcurves.preprocess import process_object


def test_temporal_baseline_fraction_uses_documented_raw_and_usable_spans(kepler_fits_factory):
    path = kepler_fits_factory(kepid=77, quarter=4)
    quarter = read_kepler_quarter(path, 77)
    config = json.load(open("configs/kepler_lightcurve_preprocessing_v1.json", encoding="utf-8"))
    _, metadata = process_object([quarter], 2.0, 201.0, 2.0, config)
    assert metadata["temporal_baseline_fraction"] == 1.0
    definition = config["pilot_qc_flags"]["temporal_baseline_fraction_definition"]
    assert "maximum usable" in definition["numerator"]
    assert "maximum finite raw" in definition["denominator"]
