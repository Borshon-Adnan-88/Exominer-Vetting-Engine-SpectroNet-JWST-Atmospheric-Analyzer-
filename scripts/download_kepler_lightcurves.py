"""Download only the frozen MAST inventory and validate each FITS identity."""

import json
from pathlib import Path

import pandas as pd

from kepler_lightcurves.fits import embedded_checksum_status, read_kepler_quarter
from kepler_lightcurves.mast import download_product
from kepler_lightcurves.provenance import sha256_file, write_json


def main() -> None:
    inventory_path = Path("data/lightcurve_inventory/kepler_pilot_mast_products.csv")
    inventory = pd.read_csv(inventory_path)
    raw_dir = Path("data/lightcurves/raw")
    records = []
    for row in inventory.itertuples(index=False):
        destination = raw_dir / str(int(row.kepid)) / row.product_filename
        result = download_product(row.data_uri, destination) if not destination.exists() else {
            "path": str(destination), "bytes": destination.stat().st_size,
            "sha256": __import__("hashlib").sha256(destination.read_bytes()).hexdigest(),
        }
        quarter = read_kepler_quarter(destination, int(row.kepid))
        embedded = embedded_checksum_status(destination)
        records.append({
            **row._asdict(), **result, "local_path": str(destination).replace("\\", "/"),
            "quarter": quarter.quarter, "data_release": quarter.data_release,
            "actual_cadence_seconds": quarter.actual_cadence_seconds,
            "embedded_checksums_supplied": embedded["embedded_checksums_supplied"],
            "embedded_checksums_valid": embedded["embedded_checksums_valid"],
        })
        print(f"KIC {int(row.kepid)} Q{quarter.quarter}: {destination.name}")
    downloaded = pd.DataFrame(records).sort_values(["kepid", "quarter"], kind="stable")
    if downloaded.duplicated(["kepid", "quarter"]).any():
        raise RuntimeError("multiple frozen products resolved to one target-quarter")
    downloaded.to_csv(inventory_path, index=False, lineterminator="\n", float_format="%.15g")
    checksum_path = Path("data/lightcurve_inventory/kepler_pilot_fits_checksums.sha256")
    checksum_path.write_text("".join(f"{row.sha256}  {row.local_path}\n" for row in downloaded.itertuples()), encoding="utf-8")
    provenance_path = Path("data/lightcurve_inventory/kepler_pilot_retrieval_provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance.update({
        "downloaded_inventory_sha256": sha256_file(inventory_path),
        "fits_checksum_manifest_sha256": sha256_file(checksum_path),
        "downloaded_bytes": int(downloaded["bytes"].sum()),
        "embedded_checksum_failures": int((~downloaded["embedded_checksums_valid"]).sum()),
    })
    write_json(provenance, provenance_path)
    print(json.dumps({"files": len(downloaded), "bytes": int(downloaded["bytes"].sum()), "targets": int(downloaded["kepid"].nunique())}, indent=2))


if __name__ == "__main__":
    main()
