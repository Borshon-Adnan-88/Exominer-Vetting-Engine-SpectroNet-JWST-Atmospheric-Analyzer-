"""Join immutable Stage 3 denominators to Stage 4A outcomes."""

import pandas as pd


TECHNICAL_STATUSES={"constructed_qc_pass","constructed_qc_fail","retrieval_failure","no_archive_product","invalid_source_product","hard_preprocessing_failure","other_technical_failure"}


def validate_outcomes(outcomes:pd.DataFrame,cohort:pd.DataFrame)->None:
    if outcomes.kepoi_name.duplicated().any() or set(outcomes.kepoi_name)!=set(cohort.kepoi_name): raise ValueError("every KOI must have exactly one outcome")
    if not set(outcomes.technical_status)<=TECHNICAL_STATUSES: raise ValueError("unknown technical status")


def build_attrition(assignments:pd.DataFrame,outcomes:pd.DataFrame)->pd.DataFrame:
    joined=assignments.merge(outcomes,on=["source_row_index","kepid","kepoi_name","label"],how="left",validate="many_to_one")
    if joined.technical_status.isna().any(): raise ValueError("missing Stage 4A outcome")
    return joined.groupby(["strategy","repeat_id","partition","label","technical_status","model_data_available"],dropna=False).size().rename("rows").reset_index()
