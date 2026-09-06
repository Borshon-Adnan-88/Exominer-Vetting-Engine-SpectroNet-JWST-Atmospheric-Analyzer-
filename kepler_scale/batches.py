"""Deterministic label-blind host roster and batches."""

import pandas as pd


def build_roster(cohort: pd.DataFrame, batch_size: int) -> tuple[pd.DataFrame,pd.DataFrame]:
    if batch_size<1: raise ValueError("batch_size must be positive")
    counts=cohort.groupby("kepid",sort=True).size().rename("koi_count").reset_index()
    counts["host_order"]=range(len(counts)); counts["batch_number"]=counts.host_order//batch_size
    counts["batch_id"]=counts.batch_number.map(lambda x:f"batch-{x:05d}")
    batches=counts.groupby(["batch_number","batch_id"],sort=True).agg(first_kepid=("kepid","min"),last_kepid=("kepid","max"),host_count=("kepid","size"),koi_count=("koi_count","sum")).reset_index()
    return counts,batches
