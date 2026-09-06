"""Read-only Stage 4A capacity and completion report."""

import shutil
from pathlib import Path
import pandas as pd


def build_preflight(config:dict,profile_name:str,paths,roster:pd.DataFrame,batches:pd.DataFrame,inventory:pd.DataFrame,outcomes:pd.DataFrame)->dict:
    profile=config["profiles"][profile_name]; cached=0; cached_bytes=0
    for row in inventory.itertuples(index=False):
        candidate=paths.raw_fits/str(int(row.kepid))/row.product_filename
        if candidate.exists() and candidate.stat().st_size==int(row.product_size): cached+=1; cached_bytes+=candidate.stat().st_size
    total=int(inventory.product_size.fillna(0).sum()); remaining=total-cached_bytes
    max_host=int(inventory.groupby("kepid").product_size.sum().max()) if len(inventory) else 0
    processed_estimate=150*1024*1024
    if profile["retain_raw_fits"]: required=remaining+processed_estimate+profile["safety_margin_bytes"]
    else: required=max_host+processed_estimate+profile["safety_margin_bytes"]
    usage=shutil.disk_usage(paths.root)
    return {"pipeline_id":config["pipeline_id"],"profile":profile_name,"configured_data_root":"<EXOMINER_DATA_ROOT>/kepler_dr25","kois":int(roster.koi_count.sum()),"hosts":len(roster),"batches":len(batches),"expected_mast_products":len(inventory),"hosts_with_products":int(inventory.kepid.nunique()),"hosts_no_products":int((outcomes.status=="no_archive_product").sum()),"ambiguous_hosts":int((outcomes.status=="ambiguous_products").sum()),"retrieval_failure_hosts":int((outcomes.status=="retrieval_failure").sum()),"advertised_total_bytes":total,"cached_size_matching_products":cached,"cached_bytes":cached_bytes,"missing_products":len(inventory)-cached,"remaining_download_bytes":remaining,"free_disk_bytes":usage.free,"required_free_bytes":required,"free_space_pass":usage.free>=required,"estimated_peak_temporary_bytes":max_host,"cpu_workers":profile["cpu_workers"],"download_concurrency":profile["download_concurrency"]}
