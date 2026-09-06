"""Official MAST discovery and frozen-product download helpers."""

import hashlib
import re
from pathlib import Path

import pandas as pd
from astroquery.mast import Observations


def discover_target(kepid: int) -> pd.DataFrame:
    canonical_target = f"kplr{int(kepid):09d}"
    observations = Observations.query_criteria(obs_collection="Kepler", target_name=canonical_target)
    if len(observations):
        observations = observations[observations["target_name"] == canonical_target]
    if len(observations) == 0:
        return pd.DataFrame()
    products = Observations.get_product_list(observations)
    rows = []
    for row in products:
        filename = str(row["productFilename"])
        if not filename.lower().endswith("_llc.fits"):
            continue
        uri = str(row["dataURI"])
        description = str(row["description"])
        match = re.search(r"\bQ(\d+)\b", description)
        if match is None:
            raise RuntimeError(f"cannot determine Kepler quarter from MAST description: {description}")
        quarter = int(match.group(1))
        if not 1 <= quarter <= 17:
            continue
        # Kepler LLC timestamp identifies a single quarter; quarter is verified from FITS after download.
        rows.append({
            "kepid": int(kepid), "obsid": str(row["obsID"]), "product_filename": filename,
            "data_uri": uri, "product_size": int(row["size"]) if row["size"] is not None else pd.NA,
            "quarter": quarter, "description": description,
        })
    result = pd.DataFrame(rows).drop_duplicates(subset=["data_uri"])
    if not result.empty:
        if result.duplicated(["kepid", "quarter"]).any():
            raise RuntimeError(f"multiple official LLC products found for KIC {kepid} in one quarter")
        result = result.sort_values(["quarter", "product_filename"], kind="stable").reset_index(drop=True)
    return result


def discover_targets(kepids: list[int]) -> dict[int, pd.DataFrame]:
    """Discover a deterministic small batch while representing every target once."""
    requested = {f"kplr{int(k):09d}": int(k) for k in kepids}
    observations = Observations.query_criteria(obs_collection="Kepler", target_name=list(requested))
    if len(observations) == 0:
        return {int(k): pd.DataFrame() for k in kepids}
    observations = observations[[str(value) in requested for value in observations["target_name"]]]
    obs_to_kepid = {str(row["obsid"]): requested[str(row["target_name"])] for row in observations}
    products = Observations.get_product_list(observations)
    rows = {int(k): [] for k in kepids}
    for row in products:
        filename = str(row["productFilename"])
        if not filename.lower().endswith("_llc.fits"):
            continue
        kepid = obs_to_kepid.get(str(row["obsID"]))
        if kepid is None:
            continue
        description = str(row["description"]); match = re.search(r"\bQ(\d+)\b", description)
        if match is None:
            raise RuntimeError(f"cannot determine Kepler quarter from MAST description: {description}")
        quarter = int(match.group(1))
        if 1 <= quarter <= 17:
            rows[kepid].append({"kepid":kepid,"obsid":str(row["obsID"]),"product_filename":filename,"data_uri":str(row["dataURI"]),"product_size":int(row["size"]) if row["size"] is not None else pd.NA,"quarter":quarter,"description":description})
    result={}
    for kepid,items in rows.items():
        frame=pd.DataFrame(items).drop_duplicates(subset=["data_uri"])
        if not frame.empty:
            if frame.duplicated(["kepid","quarter"]).any(): raise RuntimeError(f"multiple official LLC products found for KIC {kepid} in one quarter")
            frame=frame.sort_values(["quarter","product_filename"],kind="stable").reset_index(drop=True)
        result[kepid]=frame
    return result


def download_product(data_uri: str, destination: str | Path) -> dict[str, object]:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_suffix(destination.suffix + ".part")
    status = Observations.download_file(data_uri, local_path=str(part), cache=False)
    status_name = status[0] if isinstance(status, tuple) else status
    if str(status_name).upper() not in {"COMPLETE", "LOCAL"}:
        raise RuntimeError(f"MAST download failed for {data_uri}: {status}")
    part.replace(destination)
    data = destination.read_bytes()
    return {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), "path": str(destination)}
