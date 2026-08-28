"""Strict reading and identity validation for Kepler DR25 LLC FITS files."""

import hashlib
from pathlib import Path

import numpy as np
from astropy.io import fits

from .schemas import LightCurveError, QuarterData


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def embedded_checksum_status(path: str | Path) -> dict[str, object]:
    """Verify every FITS HDU that supplies CHECKSUM/DATASUM metadata."""
    results: list[dict[str, object]] = []
    with fits.open(path, memmap=False, checksum=False) as hdul:
        for index, hdu in enumerate(hdul):
            has_checksum = "CHECKSUM" in hdu.header
            has_datasum = "DATASUM" in hdu.header
            checksum_valid = bool(hdu.verify_checksum()) if has_checksum else None
            datasum_valid = bool(hdu.verify_datasum()) if has_datasum else None
            results.append({
                "hdu": index,
                "checksum_present": has_checksum,
                "datasum_present": has_datasum,
                "checksum_valid": checksum_valid,
                "datasum_valid": datasum_valid,
            })
    supplied = [item for item in results if item["checksum_present"] or item["datasum_present"]]
    valid = all(
        (not item["checksum_present"] or item["checksum_valid"])
        and (not item["datasum_present"] or item["datasum_valid"])
        for item in supplied
    )
    return {"embedded_checksums_supplied": bool(supplied), "embedded_checksums_valid": valid, "hdu_results": results}


def read_kepler_quarter(path: str | Path, expected_kepid: int) -> QuarterData:
    path = Path(path)
    if not path.name.lower().endswith("_llc.fits"):
        raise LightCurveError("unsupported_cadence")
    with fits.open(path, memmap=False, checksum=False) as hdul:
        primary, table_header = hdul[0].header, hdul[1].header
        kepid = int(primary.get("KEPLERID", table_header.get("KEPLERID", -1)))
        quarter = int(primary.get("QUARTER", table_header.get("QUARTER", -1)))
        data_release = int(primary.get("DATA_REL", table_header.get("DATA_REL", -1)))
        if kepid != int(expected_kepid):
            raise LightCurveError("fits_identity_mismatch")
        if data_release != 25:
            raise LightCurveError("unsupported_data_release")
        if not 1 <= quarter <= 17:
            raise LightCurveError("quarter outside Q1-Q17")
        names = set(hdul[1].columns.names)
        required = {"TIME", "PDCSAP_FLUX", "PDCSAP_FLUX_ERR", "SAP_FLUX", "SAP_QUALITY", "CADENCENO"}
        missing = sorted(required - names)
        if missing:
            raise LightCurveError(f"missing FITS columns: {missing}")
        data = hdul[1].data
        time = np.asarray(data["TIME"], dtype=np.float64)
        finite_time = np.sort(time[np.isfinite(time)])
        cadence = float(np.median(np.diff(finite_time)) * 86400) if len(finite_time) > 1 else float("nan")
        if np.isfinite(cadence) and not 1000 <= cadence <= 2500:
            raise LightCurveError("unsupported_cadence")
        return QuarterData(
            kepid=kepid, quarter=quarter, data_release=data_release, time=time,
            pdcsap_flux=np.asarray(data["PDCSAP_FLUX"], dtype=np.float64),
            pdcsap_flux_err=np.asarray(data["PDCSAP_FLUX_ERR"], dtype=np.float64),
            sap_flux=np.asarray(data["SAP_FLUX"], dtype=np.float64),
            quality=np.asarray(data["SAP_QUALITY"], dtype=np.int64),
            cadenceno=np.asarray(data["CADENCENO"], dtype=np.int64),
            actual_cadence_seconds=cadence, filename=path.name, sha256=sha256_file(path),
        )
