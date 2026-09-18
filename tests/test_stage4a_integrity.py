"""Detect payload damage even for valid CHECKSUM-only FITS headers."""
import numpy as np
from astropy.io import fits

from kepler_scale.integrity import inspect_fits_bytes, ones_complement_sum


def test_end_around_carry():
    assert ones_complement_sum(bytes.fromhex("ffffffff00000001")) == 1


def test_checksum_without_datasum_and_payload_corruption(tmp_path):
    path = tmp_path / "sample.fits"
    hdus = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.arange(64, dtype=np.int16))])
    # override_datasum creates a valid full-HDU CHECKSUM without a DATASUM card.
    for hdu in hdus:
        hdu.add_checksum(override_datasum=True)
    hdus.writeto(path)
    good = inspect_fits_bytes(path)
    assert all(hdu["datasum_keyword"] is None for hdu in good["hdus"])
    assert all(hdu["raw_checksum_valid"] for hdu in good["hdus"])
    with fits.open(path) as reopened:
        offset = reopened[1].fileinfo()["datLoc"]
    damaged = bytearray(path.read_bytes()); damaged[offset] ^= 1; path.write_bytes(damaged)
    bad = inspect_fits_bytes(path)
    assert bad["hdus"][0]["raw_checksum_valid"]
    assert not bad["hdus"][1]["raw_checksum_valid"]


def test_checksums_with_datasum(tmp_path):
    path = tmp_path / "sample.fits"
    fits.PrimaryHDU(np.arange(128, dtype=np.int32)).writeto(path, checksum=True)
    result = inspect_fits_bytes(path)
    assert result["hdus"][0]["raw_checksum_valid"]
    assert result["hdus"][0]["raw_datasum_valid"]
