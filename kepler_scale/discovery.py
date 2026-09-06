"""Resumable, deterministic per-host MAST discovery."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from astroquery.mast import conf as mast_conf

from kepler_lightcurves.mast import discover_targets
from kepler_lightcurves.provenance import sha256_file
from .checkpoints import read_checkpoint, write_checkpoint

PRODUCT_COLUMNS=["kepid","quarter","obsid","product_filename","data_uri","product_size","description"]


def _record(kepid:int,result:pd.DataFrame,attempt:int,config_hash:str)->dict:
    products=[] if result.empty else result[PRODUCT_COLUMNS].sort_values(["quarter","product_filename"],kind="stable").to_dict(orient="records")
    return {"kepid":kepid,"status":"products_found" if products else "no_archive_product","attempts":attempt,"products":products,"stage4a_config_sha256":config_hash,"error":None}

def _discover_batch(kepids:list[int], cache:Path, config:dict) -> list[dict]:
    config_hash=sha256_file("configs/kepler_stage4a_scale_v1.json")
    attempts=config["retry"]["maximum_attempts"]; backoff=config["retry"]["backoff_seconds"]
    for attempt in range(1,attempts+1):
        try:
            results=discover_targets(kepids); records=[]
            for kepid in kepids:
                record=_record(kepid,results[kepid],attempt,config_hash); write_checkpoint(cache,kepid,record); records.append(record)
            return records
        except RuntimeError as exc:
            message=str(exc); ambiguous="multiple official LLC products" in message or "cannot determine Kepler quarter" in message
            if ambiguous:
                records=[]
                for kepid in kepids:
                    record={"kepid":kepid,"status":"ambiguous_products","attempts":attempt,"products":[],"stage4a_config_sha256":config_hash,"error":message}; write_checkpoint(cache,kepid,record); records.append(record)
                return records
            if attempt<attempts: time.sleep(backoff[min(attempt-1,len(backoff)-1)])
            else:
                records=[]
                for kepid in kepids:
                    record={"kepid":kepid,"status":"retrieval_failure","attempts":attempt,"products":[],"stage4a_config_sha256":config_hash,"error":message}; write_checkpoint(cache,kepid,record); records.append(record)
                return records
        except Exception as exc:
            if attempt<attempts: time.sleep(backoff[min(attempt-1,len(backoff)-1)])
            else:
                records=[]
                for kepid in kepids:
                    record={"kepid":kepid,"status":"retrieval_failure","attempts":attempt,"products":[],"stage4a_config_sha256":config_hash,"error":f"{type(exc).__name__}: {exc}"}; write_checkpoint(cache,kepid,record); records.append(record)
                return records
    raise AssertionError("unreachable")


def discover_hosts(roster:pd.DataFrame,cache:Path,config:dict,concurrency:int,retry_failures=False,progress=None)->list[dict]:
    mast_conf.timeout=config["retry"]["read_timeout_seconds"]
    kepids=[int(x) for x in roster.sort_values("host_order").kepid]
    config_hash=sha256_file("configs/kepler_stage4a_scale_v1.json"); records=[]; pending=[]
    for kepid in kepids:
        prior=read_checkpoint(cache,kepid)
        reusable=prior and prior.get("stage4a_config_sha256")==config_hash and (prior.get("status") in {"products_found","no_archive_product"} or (not retry_failures and prior.get("status") in {"ambiguous_products","retrieval_failure"}))
        if reusable: records.append(prior)
        else: pending.append(kepid)
    batch_size=config["profiles"]["laptop_safe"]["host_batch_size"]
    batches=[pending[i:i+batch_size] for i in range(0,len(pending),batch_size)]
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        done=len(records)
        for batch_records in executor.map(lambda batch:_discover_batch(batch,cache,config),batches):
            records.extend(batch_records); done+=len(batch_records)
            if progress: progress(done,len(kepids),batch_records[-1])
    return sorted(records,key=lambda x:x["kepid"])


def freeze_discovery(records:list[dict])->tuple[pd.DataFrame,pd.DataFrame]:
    if not records or any("status" not in r for r in records): raise RuntimeError("discovery is incomplete")
    outcomes=pd.DataFrame([{k:v for k,v in r.items() if k!="products"} for r in records]).sort_values("kepid",kind="stable")
    products=[p for r in records for p in r.get("products",[])]
    inventory=pd.DataFrame(products,columns=PRODUCT_COLUMNS).sort_values(["kepid","quarter","product_filename"],kind="stable").reset_index(drop=True)
    if inventory.duplicated("data_uri").any(): raise RuntimeError("duplicate MAST URI in discovery inventory")
    if inventory.duplicated(["kepid","quarter"]).any(): raise RuntimeError("ambiguous target-quarter in discovery inventory")
    return inventory,outcomes
