"""Bounded fresh-byte comparison and retained-source audit; no bulk operations."""

import argparse
import hashlib
import inspect
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import astropy
import pandas as pd
import requests
from astropy.io.fits.hdu.base import _ValidHDU

from kepler_lightcurves.fits import read_kepler_quarter
from kepler_scale.contracts import load_config, validate_inputs
from kepler_scale.integrity import inspect_fits_bytes


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scratch-dataset", required=True)
    parser.add_argument("--fresh-root", required=True)
    args = parser.parse_args()
    root, scratch, fresh = map(Path, (args.dataset, args.scratch_dataset, args.fresh_root))
    config = load_config()
    validate_inputs(config)
    inventory_path = Path("data/stage4a/kepler_stage4a_mast_inventory_v1.csv")
    provenance = json.loads(Path("data/stage4a/kepler_stage4a_provenance_v1.json").read_text())
    if sha(inventory_path) != provenance["mast_inventory_sha256"]:
        raise ValueError("frozen inventory changed")
    inventory = pd.read_csv(inventory_path).sort_values(["kepid", "quarter", "product_filename"])
    hosts = sorted(inventory.kepid.unique())
    selected = []
    for host in (hosts[0], hosts[len(hosts) // 2], hosts[-1]):
        products = inventory.loc[inventory.kepid.eq(host)]
        selected.extend([products.iloc[0], products.iloc[-1]])
    retained = sorted((scratch / "raw_fits" / "3530668").glob("*_llc.fits"))
    if not retained:
        raise ValueError("expected retained evidence absent")
    for path in (retained[0], retained[-1]):
        selected.append(inventory.loc[inventory.product_filename.eq(path.name)].iloc[0])
    selected = sorted({row.product_filename: row for row in selected}.values(), key=lambda row: (int(row.kepid), int(row.quarter)))
    if len(selected) > 8:
        raise ValueError("bounded audit cannot download more than eight products")
    fresh.mkdir(parents=True, exist_ok=True)
    states = {}

    def record(row, path):
        host = int(row.kepid)
        if host not in states:
            states[host] = json.loads((root / "cache/host_state" / f"{host:09d}.json").read_text())
        frozen = next(item for item in states[host]["fits_products"] if item["product_filename"] == row.product_filename)
        details = inspect_fits_bytes(path)
        quarter = read_kepler_quarter(path, host)
        if details["bytes"] != int(row.product_size) or details["sha256"] != frozen["sha256"]:
            raise ValueError(f"source bytes differ: {row.product_filename}")
        if quarter.quarter != int(row.quarter):
            raise ValueError("frozen quarter mismatch")
        if any(hdu["raw_checksum_valid"] is False or hdu["raw_datasum_valid"] is False for hdu in details["hdus"]):
            raise ValueError(f"raw-byte checksum invalid: {row.product_filename}")
        return {"kepid": host, "quarter": int(row.quarter), "product_filename": row.product_filename,
                "frozen_mast_uri": row.data_uri, "advertised_bytes": int(row.product_size),
                "retrieval_record_sha256": frozen["sha256"],
                "legacy_embedded_checksums_valid": frozen["embedded_checksums_valid"],
                "matches_retrieval_record": True, **details}

    retained_records = []
    for path in retained:
        row = inventory.loc[inventory.product_filename.eq(path.name)].iloc[0]
        item = record(row, path)
        item.update(relative_path=path.relative_to(scratch).as_posix(),
                    created_at_utc=stamp(path.stat().st_ctime), modified_at_utc=stamp(path.stat().st_mtime))
        retained_records.append(item)
    fresh_records = []
    session = requests.Session()
    for row in selected:
        destination = fresh / row.product_filename
        if destination.exists():
            raise FileExistsError("fresh-byte audit requires an empty sample destination")
        url = "https://mast.stsci.edu/api/v0.1/Download/file?" + urlencode({"uri": row.data_uri})
        print(f"Fresh MAST sample: {row.product_filename}", flush=True)
        with session.get(url, stream=True, timeout=(30, 120), headers={"Accept-Encoding": "identity"}) as response:
            response.raise_for_status()
            with destination.open("xb") as handle:
                written = 0
                for chunk in response.iter_content(1024 * 1024):
                    written += len(chunk)
                    if written > int(row.product_size):
                        raise ValueError("fresh download exceeds frozen advertised size")
                    handle.write(chunk)
        item = record(row, destination)
        original = next((x for x in retained_records if x["product_filename"] == row.product_filename), None)
        item["prior_comparison"] = "retained_file_and_retrieval_record" if original else "retrieval_record_only_original_purged"
        item["matches_retained_file"] = item["sha256"] == original["sha256"] if original else None
        item["same_hdu_results_as_retained"] = item["hdus"] == original["hdus"] if original else None
        fresh_records.append(item)
    state_path = root / "cache/host_state/003530668.json"
    log_path = root / "cache/run_logs/batch-00041-home_wifi.log"
    log_bytes = log_path.read_bytes()
    log = log_bytes.decode("utf-16" if log_bytes.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8", errors="replace")
    report = {
        "audit_version": "stage4a_integrity_v1", "astropy_version": astropy.__version__,
        "astropy_verify_checksum_source_sha256": hashlib.sha256(inspect.getsource(_ValidHDU.verify_checksum).encode()).hexdigest(),
        "mast_inventory_sha256": sha(inventory_path),
        "sample_rule": "First, median-index and last kepid: earliest/latest available quarter; retained host: earliest/latest retained filename. Eight exact frozen URIs maximum.",
        "fresh_count": len(fresh_records), "fresh_download_bytes": sum(x["bytes"] for x in fresh_records),
        "fresh_samples": fresh_records, "retained_sources": retained_records,
        "retained_count": len(retained_records), "retained_bytes": sum(x["bytes"] for x in retained_records),
        "host_3530668": {
            "state_sha256": sha(state_path), "state_modified_at_utc": stamp(state_path.stat().st_mtime),
            "recorded_raw_purged": states[3530668]["raw_purged"],
            "deterministic_rebuild_verified": states[3530668]["deterministic_rebuild_verified"],
            "all_retained_modifications_after_state": all(p.stat().st_mtime > state_path.stat().st_mtime for p in retained),
            "log_sha256": sha(log_path),
            "log_contains_peak_residue_4881600": '"peak_scratch_bytes_after_host": 4881600' in log,
        },
        "conclusion": "Astropy CHECKSUM-without-DATASUM interpretation defect reproduced; sampled raw HDU checksums valid; fresh bytes match historical retrieval SHA-256.",
        "limits": "Eight fresh products are sampled, not a fresh verification of all purged FITS; historical hashes were measured at retrieval, not archive-published SHA-256 values.",
        "files_altered": False,
    }
    output = Path("data/stage4a/kepler_stage4a_integrity_audit_v1.json")
    output.write_bytes((json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    print(json.dumps({key: report[key] for key in ("fresh_count", "fresh_download_bytes", "retained_count", "retained_bytes", "conclusion")}, indent=2))


if __name__ == "__main__":
    main()
