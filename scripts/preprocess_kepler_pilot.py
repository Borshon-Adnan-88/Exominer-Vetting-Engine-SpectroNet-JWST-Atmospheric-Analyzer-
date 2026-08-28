"""Build independent per-KOI model-ready views from frozen FITS products."""

import hashlib
import json
from pathlib import Path

import pandas as pd

from kepler_lightcurves.fits import read_kepler_quarter
from kepler_lightcurves.preprocess import process_object, save_arrays
from kepler_lightcurves.provenance import write_json
from kepler_lightcurves.schemas import LightCurveError


def main() -> None:
    selection = pd.read_csv("data/pilot/kepler_pilot_selection.csv")
    inventory = pd.read_csv("data/lightcurve_inventory/kepler_pilot_mast_products.csv")
    config = json.loads(Path("configs/kepler_lightcurve_preprocessing_v1.json").read_text(encoding="utf-8"))
    arrays_dir = Path("data/lightcurves/processed")
    records, checksums = [], []
    for row in selection.sort_values("pilot_order").itertuples(index=False):
        products = inventory.loc[inventory["kepid"].eq(row.kepid)].sort_values("quarter")
        base = {"pilot_order": row.pilot_order, "kepid": row.kepid, "kepoi_name": row.kepoi_name}
        try:
            quarters = [read_kepler_quarter(path, int(row.kepid)) for path in products["local_path"]]
            arrays, metadata = process_object(quarters, row.koi_period, row.koi_time0bk, row.koi_duration, config)
            filename = f"{row.kepoi_name.replace('.', '_')}.npz"
            array_hash = save_arrays(arrays_dir / filename, arrays)
            checksums.append((array_hash, f"data/lightcurves/processed/{filename}"))
            records.append({**base, "array_filename": filename, "array_sha256": array_hash, **metadata})
        except (LightCurveError, OSError, ValueError) as exc:
            records.append({**base, "processing_status": "hard_failure", "qc_pass": False, "qc_reasons": "[]", "hard_failure_reason": str(exc)})
    manifest = pd.DataFrame(records)
    for field in ("qc_reasons", "valid_quarters", "quarter_stats"):
        if field in manifest:
            manifest[field] = manifest[field].map(lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value)
    output = Path("data/lightcurve_processing"); output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "kepler_pilot_processing_manifest.csv"
    manifest.to_csv(manifest_path, index=False, lineterminator="\n", float_format="%.15g")
    summary = {
        "objects": int(len(manifest)), "constructed": int(manifest["processing_status"].eq("constructed").sum()),
        "hard_failures": int(manifest["processing_status"].eq("hard_failure").sum()),
        "qc_pass": int(manifest["qc_pass"].eq(True).sum()), "qc_fail": int(manifest["qc_pass"].eq(False).sum()),
        "processing_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    write_json(summary, output / "kepler_pilot_processing_summary.json")
    (output / "kepler_pilot_arrays_checksums.sha256").write_text("".join(f"{digest}  {path}\n" for digest, path in checksums), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
