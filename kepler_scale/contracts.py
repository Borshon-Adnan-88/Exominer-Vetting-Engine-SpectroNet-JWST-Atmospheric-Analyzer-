"""Frozen input validation and Stage 3 immutability checks."""

import json
from pathlib import Path

import pandas as pd

from kepler_lightcurves.provenance import sha256_file


class ContractError(RuntimeError): pass


def load_config(path="configs/kepler_stage4a_scale_v1.json") -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_inputs(config: dict) -> dict:
    expected=config["expected_inputs"]
    hashes={}
    for key in ("stage1_manifest","stage2_config","stage3_assignments"):
        actual=sha256_file(expected[key]); hashes[key+"_sha256"]=actual
        if actual != expected[key+"_sha256"]: raise ContractError(f"{key} SHA-256 mismatch: {actual}")
    manifest=pd.read_csv(expected["stage1_manifest"])
    cohort=manifest[(manifest.inclusion_status=="included") & (~manifest.label_conflict.astype(bool))]
    assignments=pd.read_csv(expected["stage3_assignments"])
    if len(cohort)!=expected["expected_kois"] or cohort.kepid.nunique()!=expected["expected_hosts"]: raise ContractError("scientific cohort size mismatch")
    if len(assignments)!=expected["expected_assignment_rows"]: raise ContractError("assignment row count mismatch")
    if assignments.groupby(["strategy","repeat_id","kepoi_name"]).size().ne(1).any(): raise ContractError("Stage 3 assignments are incomplete or duplicated")
    if set(assignments.kepoi_name)!=set(cohort.kepoi_name): raise ContractError("Stage 3 cohort membership mismatch")
    return {**hashes,"kois":len(cohort),"hosts":int(cohort.kepid.nunique()),"assignment_rows":len(assignments)}


def load_cohort(config: dict) -> pd.DataFrame:
    validate_inputs(config); frame=pd.read_csv(config["expected_inputs"]["stage1_manifest"])
    return frame[(frame.inclusion_status=="included") & (~frame.label_conflict.astype(bool))].sort_values(["kepid","kepoi_name","source_row_index"],kind="stable").reset_index(drop=True)
