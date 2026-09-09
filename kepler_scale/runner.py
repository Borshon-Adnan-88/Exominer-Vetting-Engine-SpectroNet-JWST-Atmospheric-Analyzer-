"""Host-at-a-time download, preprocessing, verification, and guarded purge."""

import gc
import hashlib
import json
import shutil
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

from kepler_lightcurves.fits import embedded_checksum_status, read_kepler_quarter
from kepler_lightcurves.preprocess import process_object
from .arrays import write_canonical_npz
from .atomic import atomic_json
from .download import DownloadError, download_atomic
from .index import canonical_relative_path
from .operations import raw_download_root
from .purge import assert_purge_eligible


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _host_state_path(data_paths, kepid: int) -> Path:
    return data_paths.cache / "host_state" / f"{kepid:09d}.json"


def _load_state(data_paths, kepid: int) -> dict | None:
    path = _host_state_path(data_paths, kepid)
    if not path.exists(): return None
    try: return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError): return None


def _state_is_complete(state: dict | None, data_paths) -> bool:
    if not state or not all(state.get(key) is True for key in ("all_kois_final", "array_checksums_verified", "provenance_written")): return False
    for item in state.get("koi_outcomes", []):
        if item.get("processing_status") == "constructed":
            path = data_paths.root / item["relative_path"]
            if not path.exists() or _sha256(path) != item["npz_sha256"]: return False
    return True


def _fits_set_hash(records: list[dict]) -> str:
    payload = "".join(f"{r['sha256']}  {r['product_filename']}\n" for r in sorted(records, key=lambda x: (x["quarter"], x["product_filename"])))
    return hashlib.sha256(payload.encode()).hexdigest()


def _expected_product_sha256(row, prior: dict | None) -> str | None:
    for field in ("sha256", "expected_sha256", "product_sha256"):
        value = getattr(row, field, None)
        if value is not None and pd.notna(value) and str(value).strip(): return str(value).lower()
    for item in (prior or {}).get("fits_products", []):
        if item.get("product_filename") == row.product_filename: return item.get("sha256")
    return None


def _ensure_host_products(raw_dir: Path, products: pd.DataFrame, scientific_config: dict, prior: dict | None = None) -> list[tuple]:
    ready=[]
    for row in products.sort_values(["quarter", "product_filename"], kind="stable").itertuples(index=False):
        destination=raw_dir/row.product_filename; part=destination.with_name(destination.name+".part")
        expected_size=int(row.product_size); expected_sha256=_expected_product_sha256(row,prior)
        digest=None
        if destination.is_file() and destination.stat().st_size==expected_size:
            digest=_sha256(destination)
            if expected_sha256 and digest!=expected_sha256: digest=None
        if digest is None:
            result=download_atomic(row.data_uri,destination,expected_size,scientific_config)
        else:
            if part.exists(): part.unlink()
            result={"bytes":expected_size,"sha256":digest,"attempts":0}
        ready.append((row,destination,result,expected_sha256))
    for row,destination,result,expected_sha256 in ready:
        part=destination.with_name(destination.name+".part")
        if not destination.is_file() or part.exists() or destination.stat().st_size!=int(row.product_size):
            raise DownloadError(f"host source set incomplete: {row.product_filename}")
        digest=_sha256(destination)
        if digest!=result["sha256"] or (expected_sha256 and digest!=expected_sha256):
            raise DownloadError(f"host source checksum mismatch: {row.product_filename}")
    return ready


def run_host(kepid: int, products: pd.DataFrame, kois: pd.DataFrame, scientific_config: dict, operational_config: dict, data_paths, scratch_paths, purge: bool, deterministic_rebuild: bool = False) -> tuple[dict, dict]:
    prior = _load_state(data_paths, kepid)
    if _state_is_complete(prior, data_paths): return prior, {"resumed_skip": True, "download_seconds": 0.0, "processing_seconds": 0.0}
    raw_dir = raw_download_root(operational_config, data_paths, scratch_paths) / str(kepid)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fits_records=[]; retry_count=0; downloaded_bytes=0; download_started=time.perf_counter()
    ready_products=_ensure_host_products(raw_dir,products,scientific_config,prior)
    for row,destination,result,_ in ready_products:
        retry_count += max(0,int(result["attempts"])-1)
        if result["attempts"]: downloaded_bytes += int(result["bytes"])
        quarter=read_kepler_quarter(destination,kepid); embedded=embedded_checksum_status(destination)
        if quarter.quarter != int(row.quarter): raise RuntimeError("frozen inventory/FITS quarter mismatch")
        fits_records.append({"quarter":quarter.quarter,"product_filename":row.product_filename,"bytes":result["bytes"],"sha256":result["sha256"],"embedded_checksums_valid":embedded["embedded_checksums_valid"]})
    download_seconds=time.perf_counter()-download_started
    quarters=[read_kepler_quarter(raw_dir/row.product_filename,kepid) for row in products.sort_values("quarter",kind="stable").itertuples(index=False)]
    process_started=time.perf_counter(); outcomes=[]
    for koi in kois.sort_values("kepoi_name",kind="stable").itertuples(index=False):
        base={"source_row_index":int(koi.source_row_index),"kepid":kepid,"kepoi_name":koi.kepoi_name,"label":int(koi.label)}
        try:
            arrays,metadata=process_object(quarters,float(koi.koi_period),float(koi.koi_time0bk),float(koi.koi_duration),scientific_config)
            relative=canonical_relative_path(kepid,koi.kepoi_name); path=data_paths.root/relative; hashes=write_canonical_npz(path,arrays)
            if _sha256(path)!=hashes["npz_sha256"]: raise RuntimeError("canonical NPZ checksum verification failed")
            outcomes.append({**base,"processing_status":"constructed","technical_status":"constructed_qc_pass" if metadata["qc_pass"] else "constructed_qc_fail","qc_pass":bool(metadata["qc_pass"]),"qc_reasons":metadata["qc_reasons"],"relative_path":relative,"npz_sha256":hashes.pop("npz_sha256"),"logical_array_hashes":hashes,"processing_metadata":metadata})
        except (OSError,ValueError,RuntimeError) as exc:
            outcomes.append({**base,"processing_status":"hard_failure","technical_status":"hard_preprocessing_failure","qc_pass":False,"qc_reasons":[],"hard_failure_reason":f"{type(exc).__name__}: {exc}"})
    processing_seconds=time.perf_counter()-process_started
    constructed=[x for x in outcomes if x["processing_status"]=="constructed"]
    if deterministic_rebuild:
        for outcome in constructed:
            koi=kois.loc[kois.kepoi_name.eq(outcome["kepoi_name"])].iloc[0]
            arrays,_=process_object(quarters,float(koi.koi_period),float(koi.koi_time0bk),float(koi.koi_duration),scientific_config)
            temp=data_paths.temporary/"rebuild"/str(kepid)/Path(outcome["relative_path"]).name; hashes=write_canonical_npz(temp,arrays)
            if hashes["npz_sha256"]!=outcome["npz_sha256"] or any(hashes[k]!=v for k,v in outcome["logical_array_hashes"].items()): raise RuntimeError("deterministic rebuild mismatch")
            temp.unlink()
    state={"kepid":kepid,"inventory_frozen":True,"sizes_verified":True,"fits_sha256_recorded":True,"fits_identity_validated":True,"all_kois_final":len(outcomes)==len(kois),"constructed_arrays_written":len(constructed)==sum(x["processing_status"]=="constructed" for x in outcomes),"array_checksums_verified":True,"provenance_written":True,"fits_set_sha256":_fits_set_hash(fits_records),"fits_products":fits_records,"koi_outcomes":outcomes,"raw_purged":False,"deterministic_rebuild_verified":deterministic_rebuild}
    atomic_json(_host_state_path(data_paths,kepid),state)
    if purge and operational_config["process_and_purge"] and not operational_config["retain_raw_fits"]:
        for path in assert_purge_eligible(state,raw_dir,raw_dir.parent,allow_purge=True): path.unlink()
        try: raw_dir.rmdir()
        except OSError: pass
        state["raw_purged"]=True; atomic_json(_host_state_path(data_paths,kepid),state)
    del quarters; gc.collect()
    return state,{"resumed_skip":False,"download_seconds":download_seconds,"processing_seconds":processing_seconds,"downloaded_bytes":downloaded_bytes,"retry_count":retry_count}


def run_batch(batch_id: str, roster: pd.DataFrame, inventory: pd.DataFrame, cohort: pd.DataFrame, scientific_config: dict, operational_config: dict, data_paths, scratch_paths, purge=True, rebuild_first_host=True) -> dict:
    hosts=[int(x) for x in roster.loc[roster.batch_id.eq(batch_id)].sort_values("host_order").kepid]
    if not hosts: raise ValueError(f"unknown or empty batch: {batch_id}")
    tracemalloc.start(); started=time.perf_counter(); metrics=[]; states=[]; peak_scratch=0
    for number,kepid in enumerate(hosts):
        state,metric=run_host(kepid,inventory.loc[inventory.kepid.eq(kepid)],cohort.loc[cohort.kepid.eq(kepid)],scientific_config,operational_config,data_paths,scratch_paths,purge,deterministic_rebuild=rebuild_first_host and number==0)
        states.append(state); metrics.append({"kepid":kepid,**metric})
        scratch_raw=scratch_paths.raw_fits
        peak_scratch=max(peak_scratch,sum(p.stat().st_size for p in scratch_raw.rglob("*") if p.is_file()) if scratch_raw.exists() else 0)
    _,peak_ram=tracemalloc.get_traced_memory(); tracemalloc.stop(); total=time.perf_counter()-started
    outcomes=[x for s in states for x in s["koi_outcomes"]]
    return {"batch_id":batch_id,"hosts":hosts,"host_count":len(hosts),"koi_count":len(outcomes),"fits_product_count":sum(len(s["fits_products"]) for s in states),"downloaded_bytes":sum(m.get("downloaded_bytes",0) for m in metrics),"download_seconds":sum(m["download_seconds"] for m in metrics),"processing_seconds":sum(m["processing_seconds"] for m in metrics),"total_seconds":total,"retry_count":sum(m.get("retry_count",0) for m in metrics),"hard_failures":sum(x["processing_status"]!="constructed" for x in outcomes),"qc_pass":sum(x["processing_status"]=="constructed" and x["qc_pass"] for x in outcomes),"qc_fail":sum(x["processing_status"]=="constructed" and not x["qc_pass"] for x in outcomes),"canonical_npz_count":sum(x["processing_status"]=="constructed" for x in outcomes),"canonical_npz_bytes":sum((data_paths.root/x["relative_path"]).stat().st_size for x in outcomes if x["processing_status"]=="constructed"),"purged_hosts":sum(s["raw_purged"] for s in states),"resumed_hosts":sum(m["resumed_skip"] for m in metrics),"deterministic_rebuild_hosts":sum(s.get("deterministic_rebuild_verified",False) for s in states),"peak_python_tracemalloc_bytes":peak_ram,"peak_scratch_bytes_after_host":peak_scratch,"host_metrics":metrics}
