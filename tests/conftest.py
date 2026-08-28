from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits


@pytest.fixture
def kepler_fits_factory(tmp_path):
    def create(*, kepid=1001, quarter=4, data_release=25, rows=500, cadence_days=0.02043365, filename=None):
        time = 200.0 + np.arange(rows) * cadence_days
        phase = ((time - 201.0 + 1.0) % 2.0) / 2.0
        transit = np.minimum(np.abs(phase), np.abs(phase - 1.0)) < 0.02
        pdcsap = np.full(rows, 10000.0); pdcsap[transit] -= 100.0
        sap = pdcsap * 1.01
        columns = [
            fits.Column(name="TIME", format="D", array=time),
            fits.Column(name="PDCSAP_FLUX", format="E", array=pdcsap.astype("f4")),
            fits.Column(name="PDCSAP_FLUX_ERR", format="E", array=np.full(rows, 10, dtype="f4")),
            fits.Column(name="SAP_FLUX", format="E", array=sap.astype("f4")),
            fits.Column(name="SAP_QUALITY", format="J", array=np.zeros(rows, dtype="i4")),
            fits.Column(name="CADENCENO", format="J", array=np.arange(rows, dtype="i4")),
        ]
        primary = fits.PrimaryHDU()
        primary.header["KEPLERID"] = kepid
        primary.header["QUARTER"] = quarter
        primary.header["DATA_REL"] = data_release
        path = tmp_path / (filename or f"kplr{kepid:09d}-test_llc.fits")
        fits.HDUList([primary, fits.BinTableHDU.from_columns(columns)]).writeto(path, checksum=True)
        return path
    return create
