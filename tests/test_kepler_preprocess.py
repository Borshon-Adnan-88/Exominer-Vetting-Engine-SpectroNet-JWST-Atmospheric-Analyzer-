import json

import numpy as np

from kepler_lightcurves.fits import read_kepler_quarter
from kepler_lightcurves.preprocess import process_object


def test_constructs_exact_global_local_arrays(kepler_fits_factory):
    quarters = [read_kepler_quarter(kepler_fits_factory(kepid=55, quarter=q, filename=f"kplr000000055-q{q}_llc.fits"), 55) for q in range(1, 4)]
    config = json.load(open("configs/kepler_lightcurve_preprocessing_v1.json", encoding="utf-8"))
    arrays, metadata = process_object(quarters, 2.0, 201.0, 2.0, config)
    assert arrays["global_flux"].shape == (2000,)
    assert arrays["local_flux"].shape == (200,)
    assert arrays["global_count"].dtype == np.int32
    assert metadata["processing_status"] == "constructed"
    assert "valid_quarters_below_8" in metadata["qc_reasons"]
