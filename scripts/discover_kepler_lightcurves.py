"""Freeze official MAST LLC product discovery for selected pilot targets."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from kepler_lightcurves.mast import discover_target
from kepler_lightcurves.provenance import sha256_file, write_json


def main() -> None:
    selection_path = Path("data/pilot/kepler_pilot_selection.csv")
    selection = pd.read_csv(selection_path)
    frames = []
    for kepid in selection.sort_values("pilot_order")["kepid"]:
        result = discover_target(int(kepid))
        if result.empty:
            print(f"WARNING: no LLC products found for KIC {int(kepid)}")
        else:
            frames.append(result)
    inventory = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if inventory.empty:
        raise RuntimeError("MAST discovery returned no pilot products")
    if inventory.duplicated("data_uri").any():
        raise RuntimeError("duplicate MAST data URI in frozen inventory")
    if inventory.duplicated(["kepid", "quarter"]).any():
        raise RuntimeError("duplicate target-quarter in frozen inventory")
    output = Path("data/lightcurve_inventory"); output.mkdir(parents=True, exist_ok=True)
    inventory_path = output / "kepler_pilot_mast_products.csv"
    inventory.to_csv(inventory_path, index=False, lineterminator="\n")
    provenance = {
        "archive": "MAST", "mission": "Kepler", "author": "Kepler",
        "cadence": "long", "quarters_requested": "Q1-Q17",
        "discovered_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": sha256_file(selection_path),
        "inventory_sha256": sha256_file(inventory_path),
        "target_count": int(selection["kepid"].nunique()), "product_count": int(len(inventory)),
    }
    write_json(provenance, output / "kepler_pilot_retrieval_provenance.json")
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
