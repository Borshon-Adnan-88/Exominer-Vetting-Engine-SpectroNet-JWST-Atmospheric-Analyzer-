"""Read-only FITS byte checks independent of Astropy's DATASUM shortcut.

FITS checksum convention: https://fits.gsfc.nasa.gov/registry/checksum.html
Stage 2 is deliberately unchanged; this is a Stage 4A certification audit.
"""

import hashlib
from pathlib import Path

import numpy as np
from astropy.io import fits


def ones_complement_sum(payload: bytes) -> int:
    """Sum big-endian 32-bit words with end-around carry, without overflow."""
    if len(payload) % 4:
        raise ValueError("FITS checksum input is not a multiple of four bytes")
    total = 0
    for offset in range(0, len(payload), 1024 * 1024):
        words = np.frombuffer(payload[offset:offset + 1024 * 1024], dtype=">u4")
        total += int(words.sum(dtype=np.uint64))
        while total >> 32:
            total = (total & 0xffffffff) + (total >> 32)
    return total


def inspect_fits_bytes(path: str | Path) -> dict:
    """Check untouched on-disk HDU records; never rewrite or repair a header."""
    path = Path(path)
    payload = path.read_bytes()
    results = []
    with fits.open(path, mode="readonly", memmap=False, checksum=False) as hdus:
        hdus.verify("exception")
        for index, hdu in enumerate(hdus):
            info = hdu.fileinfo()
            start, data_start, data_span = info["hdrLoc"], info["datLoc"], info["datSpan"]
            end = data_start + data_span
            if end > len(payload) or (end - start) % 2880:
                raise ValueError("truncated or unaligned FITS HDU")
            raw_sum = ones_complement_sum(payload[start:end])
            data_sum = ones_complement_sum(payload[data_start:end])
            checksum, datasum = hdu.header.get("CHECKSUM"), hdu.header.get("DATASUM")
            results.append({
                "hdu": index, "name": hdu.name,
                "checksum_keyword": checksum, "datasum_keyword": datasum,
                "checksum_card": str(hdu.header.cards["CHECKSUM"]) if checksum is not None else None,
                "astropy_checksum_status": int(hdu.verify_checksum()),
                "astropy_datasum_status": int(hdu.verify_datasum()),
                "raw_hdu_sum_hex": f"{raw_sum:08x}",
                "raw_checksum_valid": raw_sum == 0xffffffff if checksum is not None else None,
                "raw_data_sum": data_sum,
                "raw_datasum_valid": data_sum == int(datasum) if datasum is not None and str(datasum).strip() else None,
            })
        if end != len(payload):
            raise ValueError("unaccounted trailing bytes in FITS file")
        header = hdus[0].header
        extension_header = hdus[1].header if len(hdus) > 1 else header
        # PROCVER embeds an archive-internal machine URI; it is not an identity
        # requirement and must not leak into portable scientific metadata.
        identity = {key: header.get(key, extension_header.get(key)) for key in ("KEPLERID", "QUARTER", "DATA_REL", "CREATOR")}
        columns = list(hdus[1].columns.names) if len(hdus) > 1 and hasattr(hdus[1], "columns") else []
        # Force all HDU payloads to be read, not just their headers.
        for hdu in hdus:
            if hdu.data is not None:
                _ = hdu.data.shape
    return {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            "fits_readable": True, "structural_validation": "passed",
            "identity": identity, "science_columns": columns, "hdus": results}
