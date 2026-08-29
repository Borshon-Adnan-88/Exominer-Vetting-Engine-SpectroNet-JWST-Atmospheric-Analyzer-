"""Acceptance and descriptive distribution audits for split assignments."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from .schemas import SplitValidationError


def _stats(values: pd.Series, full: pd.Series) -> dict[str, float | int]:
    clean = values.dropna().astype(float); population = full.dropna().astype(float)
    if clean.empty:
        return {"count": 0, "missing": int(values.isna().sum()), "minimum": np.nan, "p10": np.nan, "median": np.nan, "p90": np.nan, "maximum": np.nan, "standardized_median_difference": np.nan, "ks_statistic": np.nan}
    scale = float(population.std(ddof=0)); median_difference = (float(clean.median()) - float(population.median())) / scale if scale > 0 else 0.0
    return {"count": int(clean.size), "missing": int(values.isna().sum()), "minimum": float(clean.min()), "p10": float(clean.quantile(.1)), "median": float(clean.median()), "p90": float(clean.quantile(.9)), "maximum": float(clean.max()), "standardized_median_difference": median_difference, "ks_statistic": float(ks_2samp(clean, population).statistic)}


def build_audit(assignments: pd.DataFrame, cohort: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    source = cohort.set_index("kepoi_name"); full_prevalence = float(cohort["label"].mean()); total = len(cohort)
    host_sizes = cohort.groupby("kepid").size(); mixed = cohort.groupby("kepid")["label"].nunique().loc[lambda x: x > 1].index
    records=[]; repeats=[]; failures=[]
    for (strategy, repeat_id), repeat in assignments.groupby(["strategy", "repeat_id"], sort=True):
        if len(repeat) != total or repeat["kepoi_name"].nunique() != total or set(repeat["kepoi_name"]) != set(cohort["kepoi_name"]):
            raise SplitValidationError(f"row loss, duplication, or cohort mismatch: {strategy} repeat {repeat_id}")
        part_hosts={p:set(x["kepid"]) for p,x in repeat.groupby("partition")}
        overlap=len((part_hosts["train"]&part_hosts["validation"]) | (part_hosts["train"]&part_hosts["test"]) | (part_hosts["validation"]&part_hosts["test"]))
        per_host=repeat.groupby("kepid")["partition"].nunique(); fragmented=int((per_host>1).sum()); fragmented_multi=int(((per_host>1)&(host_sizes>1)).sum())
        mixed_intact=int(sum(repeat.loc[repeat["kepid"].isin(mixed)].groupby("kepid")["partition"].nunique()==1))
        repeats.append({"strategy":strategy,"repeat_id":int(repeat_id),"seed":int(repeat["seed"].iat[0]),"koi_partition_overlap":0,"missing_or_additional_kois":0,"grouped_host_overlap":overlap,"fragmented_hosts":fragmented,"fragmented_multi_koi_hosts":fragmented_multi,"mixed_label_hosts_intact":mixed_intact})
        if strategy == "host_group_stratified" and overlap:
            failures.append(f"{strategy} repeat {repeat_id}: {overlap} overlapping hosts")
        for partition in ("train","validation","test"):
            block=repeat.loc[repeat["partition"]==partition]; joined=block.join(source[config["audit_fields"]],on="kepoi_name")
            fraction=len(block)/total; prevalence=float(block["label"].mean()); target=config["target_fractions"][partition]
            row={"strategy":strategy,"repeat_id":int(repeat_id),"seed":int(block["seed"].iat[0]),"partition":partition,"rows":len(block),"target_fraction":target,"row_fraction":fraction,"row_fraction_deviation":fraction-target,"negative_count":int((block.label==0).sum()),"positive_count":int((block.label==1).sum()),"positive_prevalence":prevalence,"prevalence_deviation":prevalence-full_prevalence,"hosts":int(block.kepid.nunique()),"single_koi_hosts":int(sum(host_sizes.loc[list(set(block.kepid))]==1)),"multi_koi_hosts":int(sum(host_sizes.loc[list(set(block.kepid))]>1)),"multi_koi_rows":int(block.kepid.map(host_sizes).gt(1).sum()),"mixed_label_hosts":int(len(set(block.kepid)&set(mixed)))}
            for field in config["audit_fields"]:
                for key,value in _stats(joined[field],cohort[field]).items(): row[f"{field}_{key}"]=value
            records.append(row)
            tol=config["tolerances"]
            if abs(fraction-target)>tol["maximum_absolute_row_fraction_deviation"]+1e-12: failures.append(f"{strategy} repeat {repeat_id} {partition}: row fraction tolerance")
            if abs(prevalence-full_prevalence)>tol["maximum_absolute_class_prevalence_deviation"]+1e-12: failures.append(f"{strategy} repeat {repeat_id} {partition}: prevalence tolerance")
            if block.label.nunique()!=2: failures.append(f"{strategy} repeat {repeat_id} {partition}: both classes absent")
    audit=pd.DataFrame(records); repeat_audit=pd.DataFrame(repeats)
    paired=[]
    for repeat_id, block in assignments.groupby("repeat_id", sort=True):
        pivot=block.pivot(index="kepoi_name",columns="strategy",values="partition")
        paired.append({"repeat_id":int(repeat_id),"same_partition_rows":int((pivot["object_stratified"]==pivot["host_group_stratified"]).sum()),"same_partition_fraction":float((pivot["object_stratified"]==pivot["host_group_stratified"]).mean())})
    summary={"cohort_rows":total,"cohort_hosts":int(cohort.kepid.nunique()),"negative_count":int((cohort.label==0).sum()),"positive_count":int((cohort.label==1).sum()),"positive_prevalence":full_prevalence,"multi_koi_hosts":int((host_sizes>1).sum()),"mixed_label_hosts":int(len(mixed)),"assignment_rows":len(assignments),"strategy_repeats":repeat_audit.to_dict(orient="records"),"paired_strategy_agreement":paired,"tolerance_failures":failures,"acceptance_pass":not failures}
    return audit, summary


def enforce_acceptance(summary: dict) -> None:
    if summary["tolerance_failures"]:
        raise SplitValidationError("; ".join(summary["tolerance_failures"]))
